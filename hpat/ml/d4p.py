# HPAT support for daal4py - an easy-to-use ML API (to Intel(R) DAAL).
#
# We provide a factory which creates all numba/HPAT code needed to compile/distribute daal4py code.
# Given a algorithm specification (see list at end of the file) it generates numba types
# and lowering stuff for function calls (construction and compute) and member accesses
# (to attributes of Result/Model).
#
# Algorithm/Result/Model objects simply get lowered to opaque pointers.
# Attribute access gets redirected to DAAL's actual accessor methods.
#
# FIXME:
#   - sub-classing: parameters of '__interface__' types must accept derived types
#   - boxing/unboxing of algorithms
#   - GC: result/model objects returned by daal4py wrappers are newly allocated shared pointers, need to get gc'ed
#   - float32 tables, input type selection etc.
#   - key-word/optional input arguments
#   - see fixme's below

import numpy as np
from numpy import nan
from numba import types, cgutils
from numba.extending import (intrinsic, typeof_impl, overload, overload_method,
                             overload_attribute, box, unbox, make_attribute_wrapper,
                             type_callable, models, register_model, lower_builtin, lower_getattr,
                             NativeValue, lower_cast, get_cython_function_address, typeof_impl)
from numba.targets.imputils import impl_ret_new_ref
from numba.typing.templates import (signature, AbstractTemplate, infer, infer_getattr,
                                    ConcreteTemplate, AttributeTemplate, bound_function, infer_global)
from collections import namedtuple
import warnings
from hpat.str_ext import string_type
from hpat.distributed_analysis import Distribution as DType
from llvmlite import ir as lir
from numba.targets.arrayobj import _empty_nd_impl
import ctypes


##############################################################################
##############################################################################
import daal4py
from daal4py import NAN32, NAN64, hpat_spec
import llvmlite.binding as ll

def open_daal4py():
    '''open daal4py library and load C-symbols'''
    import os
    import glob

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(daal4py.__file__))), '_daal4py.c*')
    lib = glob.glob(path)
    assert len(lib) == 1

    # just load the whole thing
    ll.load_library_permanently(lib[0])

##############################################################################
##############################################################################

# short-cut for Array type. FIXME: we currently only support 2d-double arrays
dtable_type = types.Array(types.float64, 2, 'C')
ftable_type = types.Array(types.float32, 2, 'C')
itable_type = types.Array(types.intc, 2, 'C')

d4ptypes = {
    dtable_type: 0,
    ftable_type: 1,
    itable_type: 2,
}

d4p_dtypes = {
    'REP'     : DType.REP,
    'Thread'  : DType.Thread,
    'TwoD'    : DType.TwoD,
    'OneD_Var': DType.OneD_Var,
    'OneD'    : DType.OneD,
}

def get_lir_type(context, typ):
    '''Return llvm IR type for given numba type'''
    # some types have no or an incorrect built-in mapping
    lirtypes = {
        string_type:   lir.IntType(8).as_pointer(),
        types.boolean: lir.IntType(1),  # FREQ
        dtable_type:   [lir.DoubleType().as_pointer(), lir.IntType(64), lir.IntType(64)],
        ftable_type:   [lir.FloatType().as_pointer(), lir.IntType(64), lir.IntType(64)],
        itable_type:   [lir.IntType(32).as_pointer(), lir.IntType(64), lir.IntType(64)], # FIXME ILP
    }
    if isinstance(typ, str):
        typ = algo_factory.all_nbtypes[typ]
    return lirtypes[typ] if typ in lirtypes else context.get_data_type(typ)


def nt2nd(context, builder, ptr, ary_type):
    '''Generate ir code to convert a pointer-to-daal-numeric-table to a ndarray'''

    # we need to prepare the shape array and a pointer
    shape_type = lir.ArrayType(lir.IntType(64), 2)
    shape = cgutils.alloca_once(builder, shape_type)
    data = cgutils.alloca_once(builder, lir.DoubleType().as_pointer())
    assert(ary_type in [dtable_type, ftable_type, itable_type,])
    # we also need indicate the type of the array (e.g. what we expect)
    d4ptype = context.get_constant(types.byte, d4ptypes[ary_type])
    # we can now declare and call our conversion function
    fnty = lir.FunctionType(lir.VoidType(),
                            [lir.IntType(8).as_pointer(), # actually pointer to numeric table
                             lir.DoubleType().as_pointer().as_pointer(),
                             shape_type.as_pointer(),
                             lir.IntType(8)])
    fn = builder.module.get_or_insert_function(fnty, name='to_c_array')
    builder.call(fn, [ptr, data, shape, d4ptype])
    # convert to ndarray
    shape = cgutils.unpack_tuple(builder, builder.load(shape))
    ary = _empty_nd_impl(context, builder, ary_type, shape)
    cgutils.raw_memcpy(builder, ary.data, builder.load(data), ary.nitems, ary.itemsize, align=1)
    # we are done!
    return impl_ret_new_ref(context, builder, ary_type, ary._getvalue())


##############################################################################
##############################################################################
# Class configs.
# A specification defines a daal4py class, which can be an algorithm, a model or a result.
# The following information is needed:
#    - spec.pyclass is expected to be the actual daal4py class.
#    - spec.c_name provides the name of the class name as used in C.
#    - spec.params list of tuples (name, numba type, default) for the algo parameters (constructor) [algo only]
#    - spec.input_types: list of pairs for input arguments to compute: (numba type, distribution type) [algo only]
#    - spec.attrs: list of tuple (name, numba-type) representing result or model attributes [model/result only]
#    - spec.result_dist: distribution type of result [algo only]
# Note: input/attribute types can be actual numba types or strings.
#       In the latter case, the type is looked up in the list of 'own' factory-created types
#       At this point this requires that we can put the list in a canonical order...

D4PSpec = namedtuple('D4PSpec',
                     'pyclass c_name params input_types result_dist attrs')
# default values, only name is required
D4PSpec.__new__.__defaults__ = (None, None, None, DType.REP, None)

##############################################################################
##############################################################################
class algo_factory(object):
    '''
    This factory class accepts a configuration for a daal4py class.
    Providing all the numba/lowering stuff needed to compile the given algo:
      - algo construction
      - algo computation
      - attribute access (results and models)
    FIXME: GC for shared pointers of result/model objects
    '''

    # list of types, so that we can reference them when dealing others
    all_nbtypes = {
        'dtable_type' : dtable_type,
        'ftable_type' : ftable_type,
        'itable_type' : itable_type,
        'size_t'      : types.uint64,
        'double'      : types.float64,
        'float'       : types.float32,
        'bool'        : types.boolean,
        'std::string&': string_type,  # TODO: fix and test Numba unicode_type
    }

    def from_d4p(self, spec):
        '''
        Import the raw daal4py spec and convert it to our needs
        '''
        assert any(x in spec for x in ['pyclass', 'c_name']), 'Missing required attribute in daal4py specification: ' + str(spec)
        assert 'attrs' in spec or any(x in spec for x in ['params', 'input_types']) or '__iface__' in spec['c_name'], 'invalid daal4py specification: ' + str(spec)
        if 'params' in spec:
            return D4PSpec(spec['pyclass'],
                           spec['c_name'],
                           params = spec['params'],
                           input_types = [(x[0], x[1].rstrip('*'), d4p_dtypes[x[2]]) for x in spec['input_types']],
                           result_dist =  d4p_dtypes[spec['result_dist']])
        elif 'attrs' in spec:
            # filter out (do not support) properties for which we do not know the numba type
            attrs = []
            for x in spec['attrs']:
                typ = x[1].rstrip('*')
                if typ in self.all_nbtypes:
                    attrs.append((x[0], typ))
                else:
                    warnings.warn("Warning: couldn't find numba type for '" + x[1] +"'. Ignored.")
            return D4PSpec(spec['pyclass'],
                           spec['c_name'],
                           attrs = attrs)
        return None

    def __init__(self, spec): #algo, c_name, params, input_types, result_attrs, result_dist):
        '''
        See D4PSpec for input specification.
        Defines numba type. To make it usable also call activate().
        '''
        if 'alias' not in spec:
            self.mk_type(spec)
        else:
            self.name = spec['c_name']
        self.spec = spec


    def activate(self):
        '''Bring class to life'''
        if 'alias' in self.spec:
            return
        self.spec = self.from_d4p(self.spec)
        self.mk_ctor()
        self.mk_attrs()
        self.mk_boxing()


    def mk_type(self, spec):
        '''Make numba type and register opaque model'''
        assert 'pyclass' in spec, "Missing required attribute 'pyclass' in daal4py spec: " + str(spec)
        self.name = spec['pyclass'].__name__
        def mk_simple(name):
            class NbType(types.Opaque):
                '''Our numba type for given algo class'''
                def __init__(self):
                    super(NbType, self).__init__(name=name)
            return NbType

        # make type and type instance for algo, its result or possibly model
        # also register their opaque data model
        self.NbType = mk_simple(self.name + '_nbtype')
        self.all_nbtypes[self.name] = self.NbType()

        register_model(self.NbType)(models.OpaqueModel)

        # some of the classes can be parameters to others and have a default NULL/None
        # We need to cast Python None to C NULL
        @lower_cast(types.none, self.NbType())
        def none_to_nbtype(context, builder, fromty, toty, val):
            zero = context.get_constant(types.intp, 0)
            return builder.inttoptr(zero, context.get_value_type(toty))


    def mk_ctor(self):
        '''
        Declare type and lowering code for constructing an algo object
        Lowers algo's constructor: we just call the C-function.
        We provide an @intrinsic which calls the C-function and an @overload which calls the former.
        '''

        if not self.spec or not self.spec.params:
            if self.spec:
                # this must be a result or model, not an algo class
                # we don't want a constructor but Numba must know ts model for boxing/unboxing
                @typeof_impl.register(self.spec.pyclass)
                def typeof_index(val, c):
                    return self.all_nbtypes[self.name]

            return

        # FIXME: check args

        # PR numba does not support kwargs when lowering/typing, so we need to fully expand arguments.
        # We can't do this with 'static' code because we do not know the argument names in advance,
        # they are propvided and the D4PSpec. Hence we generate a function def as a string and python-exec it
        # unfortunately this needs to be done for the @intrinsic and the @overload.
        # The @intrinsic is evaluated lazily, which is probably why we cannot really bind variables here, we need to
        # expand everything to global names (hence the format(...) below).
        # What a drag.
        cmm_string = '''
@intrinsic
def _cmm_{0}(typingctx, {2}):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.IntType(8).as_pointer(), # ctors always just return an opaque pointer
                                [{3}])
        fn = builder.module.get_or_insert_function(fnty, name='mk_{1}')
        return builder.call(fn, args)
    return algo_factory.all_nbtypes['{0}']({4}), codegen
'''.format(self.name,
           self.spec.c_name,
           ', '.join([x[0] for x in self.spec.params]),
           ', '.join(['get_lir_type(context, "' + x[1] + '")' for x in self.spec.params]),
           ', '.join(['algo_factory.all_nbtypes["' + x[1] + '"]' for x in self.spec.params]))

        loc_vars = {}
        exec(cmm_string, globals(), loc_vars)
        call_maker_maker = loc_vars['_cmm_'+self.name]

        @overload(self.spec.pyclass)
        def _ovld(*args, **kwargs):
            assert len(self.spec.params) >= len(args) + len(kwargs), 'Invalid number of arguments to ' + str(self.spec.pyclass)
            gstr = 'def _ovld_impl(' + ', '.join([x[0] + ('=' + ('"{}"'.format(x[2]) if algo_factory.all_nbtypes[x[1]] == string_type else str(x[2])) if len(x) > 2 else '') for x in self.spec.params]) + '):\n'
            gstr += '    return _cmm_' + self.name + '(' + ', '.join([x[0] for x in self.spec.params]) + ')'
            loc_vars = {}
            exec(gstr, {'nan': nan, 'intrinsic': intrinsic, '_cmm_'+self.name: call_maker_maker}, loc_vars)
            impl = loc_vars['_ovld_impl']
            return impl

    def gen_call(context, builder, sig, args, c_func):
        '''
        This is generating the llvm code for calling a d4p C function.
        May also convert to ndarray.
        Used by our dynamically generated/exec'ed @lower_builtin/@lower_getattr functions below.
        '''
        lir_types = [lir.IntType(8).as_pointer()]  # the first arg is always our algo object (shrd_ptr)
        c_args = [args[0]]                         # the first arg is always our algo object (shrd_ptr)
        # prepare our args (usually none for most get_* attribuutes/properties)
        for i in range(1, len(args)):
            lirt = get_lir_type(context, sig.args[i])
            if isinstance(lirt, list):  # Array!
                # generate lir code to extract actual arguments
                # collect args/types in list
                lir_types += lirt
                in_arrtype = sig.args[i]
                in_array = context.make_array(in_arrtype)(context, builder, args[i])
                in_shape = cgutils.unpack_tuple(builder, in_array.shape)
                c_args += [in_array.data, in_shape[0], in_shape[1]]
            else:
                lir_types.append(lirt)
                c_args.append(args[i])
        #ret_typ = sig if c_func.startswith('get_') else sig.return_type
        # Our getter might return an array, which needs special handling
        ret_is_array = isinstance(sig.return_type, types.Array)
        # define our llvm return type
        c_func_ret_type = lir.IntType(8).as_pointer() if ret_is_array else context.get_data_type(sig.return_type)
        # Now we can define the signature
        fnty = lir.FunctionType(c_func_ret_type, lir_types)
        # Get function
        fn = builder.module.get_or_insert_function(fnty, name=c_func)
        # and finally generate the call
        ptr = builder.call(fn, c_args)
        return nt2nd(context, builder, ptr, sig.return_type) if ret_is_array else ptr


    def mk_attrs(self):
        '''
        Provide the typing and lowering for getters and calling compute
        Again, we can't use static code for our infer_getarr class, so we need
        to generate code that we exec.
        Apparently the @infer_getattr/@bound_function and @lower_builtin need to
        be in the same context, so we need to include the @lower_builtin stuff
        there, too.
        '''

        if not self.spec:
            return

        # Initing the code we want to exec
        lower_code = ''
        infer_code = '''# This is our class providing the resolve_* methods
@infer_getattr
class AlgoAttributes_{0}(AttributeTemplate):
    "declares numba signatures of attributes/methods of {0} objects"
    key = algo_factory.all_nbtypes['{0}'].__class__
'''.format(self.spec.c_name)

        # The python name stub (module + class)
        name_stub = '_'+'.'.join([self.spec.pyclass.__module__.strip('_'), self.spec.pyclass.__name__])
        result_type = None

        # First handle properties (result/model classes)
        if self.spec.attrs:
            for a in self.spec.attrs:
                c_func = '_'.join(['get', self.spec.c_name, a[0]])
                full_name = '.'.join([name_stub, a[0]])
                # for typing we simply provide resolve_* method for each attribute
                infer_code += '''
    def resolve_{0}(self, obj):
        return algo_factory.all_nbtypes['{1}']
'''.format(a[0], a[1])

                # lowering code for properties through @lower_getattr
                lower_code += '''
@lower_getattr(algo_factory.all_nbtypes['{0}'], '{1}') # getters have no args (other than self)
def lower_{2}(context, builder, typ, val):
    return algo_factory.gen_call(context, builder, signature(algo_factory.all_nbtypes['{3}']), [val], '{2}')
'''.format(self.name, a[0], c_func, a[1])

        # Now we handle compute method algorithms classes
        if self.spec.input_types:
            compute_name = '.'.join([name_stub, 'compute'])

            # using bound_function for typing
            infer_code += '''
    @bound_function("{0}")
    def resolve_compute(self, dict, args, kws):
        # FIXME: keyword args
        # FIXME: check args
        return signature(algo_factory.all_nbtypes['{1}_result'], *args)
'''.format(compute_name, self.name)

            # lowering methods provided with lower_builtin
            lower_code += '''
@lower_builtin('{0}', algo_factory.all_nbtypes['{1}'], *[{2}])
def lower_compute(context, builder, sig, args):
    return algo_factory.gen_call(context, builder, sig, args, 'compute_{3}')
'''.format(compute_name,
           self.name,
           ', '.join(["algo_factory.all_nbtypes['{}']".format(x[1]) for x in self.spec.input_types]), #if isinstance(x[1], str) else "'{}'")
            self.spec.c_name)

        loc_vars = {}
        exec(infer_code+lower_code, globals(), loc_vars)


    def cy_unboxer(self):
        '''
        Return the address of the cython generated C(++) function which
        provides the actual DAAL C++ pointer for our result/model Python object.
        '''
        addr = get_cython_function_address("_daal4py", 'unbox_'+self.spec.c_name)
        return addr


    def mk_boxing(self):
        '''provide boxing and unboxing'''

        if not self.spec or self.spec.input_types:
            # we only support models/results at this point
            return

        ll.add_symbol('unbox_'+self.spec.c_name, self.cy_unboxer())

        @box(self.NbType)
        def box_me(typ, val, c):
            'Call Cythons contructor with the C++ pointer and return Python object.'
            ll_intp = c.context.get_value_type(types.uintp)
            addr = c.builder.ptrtoint(val, ll_intp)
            v = c.box(types.uintp, addr)
            py_obj = c.pyapi.unserialize(c.pyapi.serialize_object(self.spec.pyclass))
            res = c.pyapi.call_function_objargs(py_obj, (v,))
            return res

        @unbox(self.NbType)
        def unbox_me(typ, obj, c):
            'Call C++ unboxing function and return as NativeValue'
            # Define the signature
            fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer(),])
            # Get function
            fn = c.builder.module.get_or_insert_function(fnty, name='unbox_'+self.spec.c_name)
            # finally generate the call
            ptr = c.builder.call(fn, [obj])
            return NativeValue(ptr, is_error=c.pyapi.c_api_error())



##############################################################################
##############################################################################
##############################################################################
# finally let the factory do its job

open_daal4py()

# first define types
algos = [algo_factory(x) for x in hpat_spec]
# then setup aliases
for s in hpat_spec:
    if 'alias' in s:
        # for assume we have no recurring aliasing
        assert s['alias'] in algo_factory.all_nbtypes, "Recurring aliasing not supported"
        algo_factory.all_nbtypes[s['c_name']] =  algo_factory.all_nbtypes[s['alias']]
# now bring life to the classes
for a in algos:
    a.activate()

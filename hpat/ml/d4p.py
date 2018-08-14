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
#   - boxing/unboxing
#   - GC: result/model objects returned by daal4py wrappers are newly allocated shared pointers, need to get gc'ed
#   - float32 tables, input type selection etc.
#   - key-word/optional input arguments
#   - see fixme's below

import numpy as np
from numpy import nan
from numba import types, cgutils
from numba.extending import intrinsic, typeof_impl, overload, overload_method, overload_attribute, box, unbox, make_attribute_wrapper, type_callable, models, register_model, lower_builtin, NativeValue, lower_cast
from numba.targets.imputils import impl_ret_new_ref
from numba.typing.templates import (signature, AbstractTemplate, infer, infer_getattr,
                                    ConcreteTemplate, AttributeTemplate, bound_function, infer_global)
from collections import namedtuple
from hpat.str_ext import string_type
from hpat.distributed_analysis import Distribution as DType
from llvmlite import ir as lir
from numba.targets.arrayobj import _empty_nd_impl

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
        'bool'        : types.boolean,
        'std::string&': string_type,
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
                           input_types = [(x[0], self.all_nbtypes[x[1].rstrip('*')], d4p_dtypes[x[2]]) for x in spec['input_types']])
        elif 'attrs' in spec:
            # filter out (do not support) properties for which we do not know the numba type
            attrs = []
            for x in spec['attrs']:
                typ = x[1].rstrip('*')
                if typ in self.all_nbtypes:
                    attrs.append((x[0], self.all_nbtypes[typ]))
                else:
                    print("Warning: couldn't find numba type for '" + x[1] +"'. Ignored.")
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
        self.mk_compute()
        self.mk_attrs()


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

        # make type and type instance for algo, its result and possibly model
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
            # this must be a result or model, not an algo class
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


    def mk_compute(self):
        '''provide the typing and lowering for calling compute on an algo object'''

        if not self.spec or not self.spec.input_types:
            # this must be a result or model, not an algo class
            return

        algo_type = self.NbType
        result_type = self.all_nbtypes[self.name+'_result']
        compute_name = '.'.join([self.spec.pyclass.__module__.strip('_'), self.spec.pyclass.__name__, 'compute'])

        @infer_getattr
        class AlgoAttributes(AttributeTemplate):
            '''declares numba signatures of attributes/methods of algo objects'''
            key = algo_type

            @bound_function(compute_name)
            def resolve_compute(self, dict, args, kws):
                # FIXME: keyword args
                # FIXME: check args
                return signature(result_type, *args)


        @lower_builtin(compute_name, self.all_nbtypes[self.name], *[self.all_nbtypes[x[1]] if isinstance(x[1], str) else x[1] for x in self.spec.input_types])
        def lower_compute(context, builder, sig, args):
            '''lowers compute method algo objects'''
            # First prepare list of argument types
            lir_types = [lir.IntType(8).as_pointer()]  # the first arg is always our algo object (shrd_ptr)
            c_args = [args[0]]                         # the first arg is always our algo object (shrd_ptr)
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
            # Now we can define the signature and call the C-function
            fnty = lir.FunctionType(lir.IntType(8).as_pointer(), lir_types)
            fn = builder.module.get_or_insert_function(fnty, name='compute_' + self.spec.c_name)
            # finally we call the function
            return builder.call(fn, c_args)


    def add_attr(self, NbType, attr, attr_type, c_func):
        '''
        Generate getter for attribute 'attr' on objects of numba result/model type.
        Calls c_func to retrieve the attribute from the given result/model object.
        Converts to ndarray if attr_type is Array
        '''

        if isinstance(attr_type, str):
            attr_type = self.all_nbtypes[attr_type]
        is_array = isinstance(attr_type, types.Array)

        @intrinsic
        def get_attr_impl(typingctx, obj):
            '''
            This is creating the llvm wrapper calling the C function.
            May also convert to ndarray.
            '''
            def codegen(context, builder, sig, args):
                assert(len(args) == 1)
                c_func_ret_type = lir.IntType(8).as_pointer() if is_array else context.get_data_type(attr_type)
                # First call the getter
                fnty = lir.FunctionType(c_func_ret_type, [lir.IntType(8).as_pointer()])
                fn = builder.module.get_or_insert_function(fnty, name=c_func)
                ptr = builder.call(fn, args)
                return nt2nd(context, builder, ptr, sig.return_type) if is_array else ptr

            return attr_type(obj), codegen

        @overload_attribute(NbType, attr)
        def get_attr(obj):
            '''declaring getter for attribute 'attr' of objects of type "NbType"'''
            def getter(obj):
                return get_attr_impl(obj)
            return getter


    def mk_attrs(self):
        '''Make attributes of result and model known to numba and how to get their values.'''

        if not self.spec or not self.spec.attrs:
            # this must be algo class, which does not have attributes
            # or it is an alias only
            return

        for a in self.spec.attrs:
            self.add_attr(self.NbType, a[0], a[1], '_'.join(['get', self.spec.c_name, a[0]]))

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

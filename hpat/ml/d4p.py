# HPAT support for daal4py - an easy-to-use ML API (to Intel(R) DAAL).
#
# We provide a factory whcih creates all numba/HPAT code needed to compile/distribute daal4py code.
# Given a algorithm specification (see list at end of the file) it generates numba types
# and lowering stuff for function calls (construction and compute) and member accesses
# (to attributes of Result/Model).
#
# Algorithm/Result/Model objects simply get lowered to opaque pointers.
# Attribute access gets redirected to DAAL's actual accessor methods.
#
# FIXME:
#   - boxing/unboxing
#   - GC: objects returned by daal4py wrappers are newly allocated shared pointers, need to get gc'ed
#   - float32 tables, input type selection etc.
#   - key-word/optional arguments/paramters
#   - see fixme's below

import numpy as np
from numba import types, cgutils
from numba.extending import intrinsic, typeof_impl, overload, overload_method, overload_attribute, box, unbox, make_attribute_wrapper, type_callable, models, register_model, lower_builtin, NativeValue
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
import llvmlite.binding as ll

def open_daal4py():
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

def get_lir_type(context, typ):
    "Return llvm ir type for given numba type"
    # some types have no or an incorrect built-in mapping
    lirtypes = {
        string_type:   lir.IntType(8).as_pointer(),
        types.boolean: lir.IntType(1),  # FREQ
        dtable_type:   [lir.DoubleType().as_pointer(), lir.IntType(64), lir.IntType(64)],
        ftable_type:   [lir.FloatType().as_pointer(), lir.IntType(64), lir.IntType(64)],
        itable_type:   [lir.IntType(32).as_pointer(), lir.IntType(64), lir.IntType(64)], # FIXME ILP
    }
    return lirtypes[typ] if typ in lirtypes else context.get_data_type(typ)


def nt2nd(context, builder, ptr, ary_type):
    """
    Generate ir code to convert a pointer-to-daal-numeric-table to a ndarray
    FIXME: handle NULL pointer/table
    """

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
    fn = builder.module.get_or_insert_function(fnty, name="to_c_array")
    builder.call(fn, [ptr, data, shape, d4ptype])
    # convert to ndarray
    shape = cgutils.unpack_tuple(builder, builder.load(shape))
    ary = _empty_nd_impl(context, builder, ary_type, shape)
    cgutils.raw_memcpy(builder, ary.data, builder.load(data), ary.nitems, ary.itemsize, align=1)
    # we are done!
    return impl_ret_new_ref(context, builder, ary_type, ary._getvalue())


class algo_factory(object):
    """
    This factory class accepts a configuration for a daal4py algorithm
    and provides all the numba/lowering stuff needed to compile the given algo:
      - algo construction
      - algo computation
      - result attribute access
    FIXME: GC for shared pointers (result/model objects)
    """

    "list of types, so that we can reference them when dealing others"
    all_nbtypes = {}

    def __init__(self, spec): #algo, c_name, param_types, input_types, result_attrs, result_dist):
        """
        factory to build the numba types and lowering stuff.
        See D4PSpec for input specification.
        """
        self.spec = spec
        self.mk_types()
        self.mk_ctor()
        self.mk_compute()
        self.mk_attrs()


    def mk_types(self):
        """Make numba types (algo and result) and register opaque model"""
        self.name = self.spec.algo.__name__
        def mk_simple(name):
            class NbType(types.Opaque):
                """Our numba type for given algo class"""
                def __init__(self):
                    super(NbType, self).__init__(name=name)
            return NbType

        # make type and type instance for algo, its result and possibly model
        # also register their opaque data model
        self.NbType_algo = mk_simple(self.name + '_nbtype')
        self.all_nbtypes[self.name] = self.NbType_algo()
        register_model(self.NbType_algo)(models.OpaqueModel)

        self.NbType_res = mk_simple(self.name + '_result_nbtype')
        self.all_nbtypes[self.name + '_result'] = self.NbType_res()
        register_model(self.NbType_res)(models.OpaqueModel)

        if self.spec.model_attrs:
            self.NbType_model = mk_simple(self.name + '_model_nbtype')
            self.all_nbtypes[self.name + '_model'] = self.NbType_model()
            register_model(self.NbType_model)(models.OpaqueModel)


    def mk_ctor(self):
        """declare type and lowering code for constructing an algo object"""

        @type_callable(self.spec.algo)
        def ctor_decl(context):
            """declare numba type for constructing the algo object"""
            # FREQ: *args, **kwargs doesn't work
            def typer(a=1, b=1, c=1, d=1, e=1, f=1, g=1, h=1, i=1, j=1, k=1, l=1):
                # FIXME: check types
                # FIXME: keyword args
                return self.all_nbtypes[self.name]
            return typer

        @lower_builtin(self.spec.algo, *self.spec.param_types)
        def ctor_impl(context, builder, sig, args):
            """
            Lowers algo's constructor.
            We just call the C-function.
            We need to add the extra boolean argument "distributed", it's not really used but
            daal4py's code generation would become too complicated without.
            FIXME: keyword args
            """
            fls = context.get_constant(types.boolean, False)
            fnty = lir.FunctionType(lir.IntType(8).as_pointer(), # ctors always just return an opaque pointer
                                    [get_lir_type(context, x) for x in self.spec.param_types + [types.boolean]])
            fn = builder.module.get_or_insert_function(fnty, name=self.spec.c_name + '_ptr')
            return builder.call(fn, args + (fls,))


    def mk_compute(self):
        algo_type = self.NbType_algo
        result_type = self.all_nbtypes[self.name+'_result']
        compute_name = '.'.join([self.spec.algo.__module__.strip('_'), self.spec.algo.__name__, 'compute'])

        @infer_getattr
        class AlgoAttributes(AttributeTemplate):
            """declares numba signatures of attributes/methods of algo objects"""
            key = algo_type

            @bound_function(compute_name)
            def resolve_compute(self, dict, args, kws):
                # FIXME: keyword args
                # FIXME: check args
                return signature(result_type, *args)


        @lower_builtin(compute_name, self.all_nbtypes[self.name], *[self.all_nbtypes[x[0]] if isinstance(x[0], str) else x[0] for x in self.spec.input_types])
        def lower_compute(context, builder, sig, args):
            """lowers compute method algo objects"""
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
        """
        Generate getter for attribute 'attr' on objects of numba result/model type.
        Calls c_func to retrieve the attribute from the given result/model object.
        Converts to ndarray if attr_type is Array
        """
        if isinstance(attr_type, str):
            attr_type = self.all_nbtypes[attr_type]
        is_array = isinstance(attr_type, types.Array)

        @intrinsic
        def get_attr_impl(typingctx, obj):
            """
            This is creating the llvm wrapper calling the C function.
            May also convert to ndarray.
            """
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
            """declaring getter for attribute 'attr' of objects of type 'NbType'"""
            def getter(obj):
                return get_attr_impl(obj)
            return getter


    def mk_attrs(self):
        """Make attributes of result and model known to numba and how to get their values."""
        for a in self.spec.result_attrs:
            self.add_attr(self.NbType_res, a[0], a[1], '_'.join(['get', self.spec.c_name, 'Result', a[0]]))
        if self.spec.model_attrs:
            for a in self.spec.model_attrs:
                self.add_attr(self.NbType_model, a[0], a[1], '_'.join(['get', self.spec.model_base, a[0]]))

##############################################################################
##############################################################################

open_daal4py()

##############################################################################
##############################################################################
# algorithm configs
# A specification lists the following attributes of an algorithms:
#    - spec.algo is expected to be the actual daal4py function.
#    - spec.c_name provides the name of the algorithm name as used in C.
#    - spec.param_types: list of numba types for the algo construction paramters
#    - spec.input_types: list of pairs for input arguments to compute: (numba type, distribution type)
#    - spec.result_attrs: list of tuple (name, numba-type) representing result attributes
#    - spec.model_attrs: [optional] list of tuple (name, numba-type) representing model attributes
#    - spec.model_base: [optional] base string for C lookup function, FIXME: should not be necessary
# Note: input/attribute types can be actual numby types or strings.
#       In the latter case, the type is looked up in the list of "own" factory-created types


D4PSpec = namedtuple('D4PSpec',
                     'algo c_name param_types input_types result_attrs result_dist model_attrs model_base')
D4PSpec.__new__.__defaults__ = (None, None) # the model data is optional

# calling the factory for every algorithm.
# See algo_factory.__init__ for arguments
algo_specs = [
    # K-Means
    D4PSpec(algo         = daal4py.kmeans_init,
            c_name       = 'kmeans_init',
            param_types  = [types.intp, string_type, string_type, types.intp, types.float64, types.intp],
            input_types  = [(dtable_type, DType.OneD)],
            result_attrs = [('centroids', dtable_type)],
            result_dist  = DType.REP,
    ),
    D4PSpec(algo         = daal4py.kmeans,
            c_name       = 'kmeans',
            param_types  = [types.intp, types.intp, string_type, string_type, types.float64, types.float64, string_type],
            input_types  = [(dtable_type, DType.OneD), (dtable_type, DType.REP)],
            result_attrs = [('centroids', dtable_type),
                            ('assignments', itable_type),
                            ('objectiveFunction', dtable_type),
                            ('goalFunction', dtable_type),
                            ('nIterations', itable_type)],
            result_dist  = DType.REP,
    ),
    # Linear Regression
    D4PSpec(algo         = daal4py.linear_regression_training,
            c_name       = 'linear_regression_training',
            param_types  = [string_type, string_type, string_type],
            input_types  = [(dtable_type, DType.OneD), (dtable_type, DType.OneD)],
            result_attrs = [('model', 'linear_regression_training_model')],
            result_dist  = DType.REP,
            model_attrs  = [('NumberOfBetas', types.uint64),
                            ('NumberOfResponses', types.uint64),
                            ('InterceptFlag', types.boolean),
                            ('Beta', dtable_type),
                            ('NumberOfFeatures', types.uint64)],
            model_base   = 'linear_regression_Model',
    ),
    D4PSpec(algo         = daal4py.linear_regression_prediction,
            c_name       = 'linear_regression_prediction',
            param_types  = [string_type, string_type],
            input_types  = [(dtable_type, DType.OneD), ('linear_regression_training_model', DType.REP)],
            result_attrs = [('prediction', dtable_type)],
            result_dist  = DType.REP,
    ),
]

# finally let the factory do its job
algos = [algo_factory(x) for x in algo_specs]

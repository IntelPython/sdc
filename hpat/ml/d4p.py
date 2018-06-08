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
    """
    def __init__(self, algo, c_name, param_types, input_types, result_attrs, result_dist):
        """
        factory to build the numba types and lowering stuff.
          - algo is expected to be the actual daal4py function.
          - c_name provides the name of the algorithm name as used in C.
          - param_types: list of numba types for the algo construction paramters
          - input_types: list of pairs for input arguments to compute: (numba type, distribution type)
          - result_attrs: list of tuple (name, numba-type) representing result attributes
        """
        self.algo = algo
        self.c_name = c_name
        self.param_types = param_types
        self.input_types = input_types
        self.result_attrs = result_attrs
        self.result_dist = result_dist
        self.mk_types()
        self.mk_ctor()
        self.mk_compute()
        self.mk_res_attrs()

    def mk_types(self):
        """Make numba types (algo and result) and register opaque model"""
        name = self.algo.__name__
        def mk_simple(name):
            class NbType(types.Opaque):
                """Our numba type for given algo class"""
                def __init__(self):
                    super(NbType, self).__init__(name=name)
            return NbType
        # make type and type instance for algo and its result
        self.NbType_algo = mk_simple(name + '_nbtype')
        self.nbtype_algo = self.NbType_algo()
        self.NbType_res = mk_simple(name + '_result_nbtype')
        self.nbtype_res = self.NbType_res()
        # register opaque model for both (we actually get pointers/shared pointers)
        register_model(self.NbType_algo)(models.OpaqueModel)
        register_model(self.NbType_res)(models.OpaqueModel)

    def mk_ctor(self):
        """declare type and lowering code for constructing an algo object"""

        @type_callable(self.algo)
        def ctor_decl(context):
            """declare numba type for constructing the algo object"""
            # FREQ: *args, **kwargs doesn't work
            def typer(a=1, b=1, c=1, d=1, e=1, f=1, g=1, h=1, i=1, j=1, k=1, l=1):
                # FIXME: check types
                # FIXME: keyword args
                return self.nbtype_algo
            return typer

        @lower_builtin(self.algo, *self.param_types)
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
                                    [get_lir_type(context, x) for x in self.param_types + [types.boolean]])
            fn = builder.module.get_or_insert_function(fnty, name=self.c_name + '_ptr')
            return builder.call(fn, args + (fls,))

    def mk_compute(self):
        algo_type = self.NbType_algo
        result_type = self.nbtype_res
        compute_name = '.'.join([self.algo.__module__.strip('_'), self.algo.__name__, 'compute'])

        @infer_getattr
        class AlgoAttributes(AttributeTemplate):
            """declares numba signatures of attributes/methods of algo objects"""
            key = algo_type

            @bound_function(compute_name)
            def resolve_compute(self, dict, args, kws):
                # FIXME: keyword args
                # FIXME: check args
                return signature(result_type, *args)


        @lower_builtin(compute_name, self.nbtype_algo, *[x[0] for x in self.input_types])
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
                    # we currently support only arrays as input arguments
                    assert(False)
            # Now we can define the signature and call the C-function
            fnty = lir.FunctionType(lir.IntType(8).as_pointer(), lir_types)
            fn = builder.module.get_or_insert_function(fnty, name='compute_' + self.c_name)
            # finally we call the function
            return builder.call(fn, c_args)


    def add_attr(self, attr, attr_type, c_func):
        """
        Generate getter for attribute 'attr' on objects of numba result type.
        Calls c_func to retrieve the attribute from the given result object.
        Converts to ndarray if attr_type is Array
        """

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

        @overload_attribute(self.NbType_res, attr)
        def get_attr(res):
            """declaring getter for attribute 'attr' of objects of type 'NbType'"""
            def getter(res):
                return get_attr_impl(res)
            return getter


    def mk_res_attrs(self):
        for a in self.result_attrs:
            self.add_attr(a[0], a[1], 'get_' + self.c_name + '_Result_' + a[0])

##############################################################################
##############################################################################

open_daal4py()

##############################################################################
##############################################################################
# algorithm configs
# calling the factory for every algorithm.
# See algo_factory.__init__ for arguments
algos = [
    algo_factory(daal4py.kmeans_init,
                 'kmeans_init',
                 [types.intp, string_type, string_type, types.intp, types.float64, types.intp],
                 [(dtable_type, DType.OneD)],
                 [('centroids', dtable_type)],
                 DType.REP,
    ),
    algo_factory(daal4py.kmeans,
                 'kmeans',
                 [types.intp, types.intp, string_type, string_type, types.float64, types.float64, string_type],
                 [(dtable_type, DType.OneD), (dtable_type, DType.REP)],
                 [('centroids', dtable_type),
                  ('assignments', itable_type),
                  ('objectiveFunction', dtable_type),
                  ('goalFunction', dtable_type),
                  ('nIterations', itable_type)],
                 DType.REP,
    ),
]

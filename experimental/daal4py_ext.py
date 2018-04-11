import numpy as np
from numba import types, cgutils
from numba.extending import intrinsic, typeof_impl, overload, overload_method, overload_attribute, box, unbox, make_attribute_wrapper, type_callable, models, register_model, lower_builtin, NativeValue
from numba.targets.imputils import impl_ret_new_ref
from numba.typing.templates import (signature, AbstractTemplate, infer, infer_getattr,
                                    ConcreteTemplate, AttributeTemplate, bound_function, infer_global)
from collections import namedtuple
from hpat.str_ext import string_type
import hpat
from llvmlite import ir as lir
from numba.targets.arrayobj import _empty_nd_impl

##############################################################################
##############################################################################
# dummy classes
# FIXME: use/import actual daal4py

class kmeans_init(object):
    """Dummy class"""
    def __init__(self, nClusters, fptype, method, seed, oversamplingFactor, nRounds, distributed):
        pass
    def compute(self, data):
        pass
    
class kmeans(object):
    """Dummy class"""
    def __init__(self, nClusters, maxIterations, fptype, method, accuracyThreshold, gamma, assignFlag, distributed):
        pass
    def compute(self, data, centroids):
        pass

##############################################################################
##############################################################################


# FIXME: this needs to become more generic, we need to find the actual so in the python root
so = "/localdisk/work/fschlimb/miniconda3/envs/HPAT/lib/python3.6/site-packages/daal4py-0.2018.20180411-py3.6-linux-x86_64.egg/_daal4py.cpython-36m-x86_64-linux-gnu.so"

import llvmlite.binding as ll
# just load the whole thing
ll.load_library_permanently(so)


##############################################################################
##############################################################################

def nt2nd(context, builder, ptr, ary_type):
    """
    Generate ir code to convert a pointer-to-daal-numeric-table to a ndarray
    """

    # we need to prepare the shape array and a pointer
    shape_type = lir.ArrayType(lir.IntType(64), 2)
    shape = cgutils.alloca_once(builder, shape_type)
    data = cgutils.alloca_once(builder, lir.DoubleType().as_pointer())
    # we can now declare and call our conversion function
    fnty = lir.FunctionType(lir.VoidType(),
                            [lir.IntType(8).as_pointer(), # actually pointer to numeric table
                             lir.DoubleType().as_pointer().as_pointer(),
                             shape_type.as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="to_c_array")
    builder.call(fn, [ptr, data, shape])
    # convert to ndarray
    shape = cgutils.unpack_tuple(builder, builder.load(shape))
    ary = _empty_nd_impl(context, builder, ary_type, shape)
    cgutils.raw_memcpy(builder, ary.data, builder.load(data), ary.nitems, ary.itemsize, align=1)
    # we are done!
    return impl_ret_new_ref(context, builder, ary_type, ary._getvalue())


def add_attr(NbType, attr, attr_type, c_func):
    """
    Generate getter for attribute 'attr' on objects of numba type 'NbType'.
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

    @overload_attribute(NbType, attr)
    def get_attr(res):
        """declaring getter for attribute 'attr' of objects of type 'NbType'"""
        def getter(res):
            return get_attr_impl(res)
        return getter


# short-cut for Array type. FIXME: we currently only support 2d-double arrays
table_type = types.Array(types.float64, 2, 'C')


##############################################################################
##############################################################################
# Kmeans-init

class NbTypeKmeansInit(types.Opaque):
    """Our numba type for kmeans-init"""
    def __init__(self):
        super(NbTypeKmeansInit, self).__init__(name='KmeansInitType')


class NbTypeKmeansInitResult(types.Opaque):
    """Our numba type for kmeans-init Result (returned by kmeans_init.compute)"""
    def __init__(self):
        super(NbTypeKmeansInitResult, self).__init__(name='KmeansInitResultType')


# type instance
NbType_kmeans_init = NbTypeKmeansInit()
NbType_kmeans_init_result = NbTypeKmeansInitResult()


# we treat our class objects as opaque (e.g. pointers)
register_model(NbTypeKmeansInit)(models.OpaqueModel)
register_model(NbTypeKmeansInitResult)(models.OpaqueModel)


@type_callable(kmeans_init)
def type_kmeans_init(context):
    """declare numba type for constructing a kmeans_init"""
    def typer(a, b, c, d, e, f, g):
        # FIXME: check types
        # FIXME: keyword args
        return NbType_kmeans_init
    return typer


@lower_builtin(kmeans_init, types.intp, string_type, string_type, types.intp, types.float64, types.intp, types.boolean)
def ctor_kmeans_init(context, builder, sig, args):
    """
    Lowers kmeans_init constructor.
    We just call the C-function.
    FIXME: keyword args
    """
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(64),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(64),
                             lir.DoubleType(),
                             lir.IntType(64),
                             lir.IntType(1)])
    fn = builder.module.get_or_insert_function(fnty, name="kmeans_init_ptr")
    return builder.call(fn, args)


@infer_getattr
class KMIAttribute(AttributeTemplate):
    """declares numba signatures of attributes/methods of kmeans_init objects"""
    key = NbTypeKmeansInit

    @bound_function("kmeans_init.compute")
    def resolve_compute(self, dict, args, kws):
        # FIXME: keyword args
        assert not kws
        assert len(args) == 1
        return signature(NbType_kmeans_init_result, *args)


@lower_builtin("kmeans_init.compute", NbType_kmeans_init, types.Array(types.float64, 2, 'C'))
def lower_compute_kmeans_init(context, builder, sig, args):
    """lowers compute method of kmeans_init objects, accepting 1 array argument"""
    in_arrtype = sig.args[1]
    in_array = context.make_array(in_arrtype)(context, builder, args[1])
    in_shape = cgutils.unpack_tuple(builder, in_array.shape)
    
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.DoubleType().as_pointer(), lir.IntType(64), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="compute_kmeans_init")
    
    return builder.call(fn, [args[0], in_array.data, in_shape[0], in_shape[1]])


# kmeans_init result has only a single attribute: centroids
add_attr(NbTypeKmeansInitResult, "centroids", table_type, "get_kmeans__init__Result_centroids")


##############################################################################
##############################################################################
# Kmeans (main)

class NbTypeKmeans(types.Opaque):
    """Our numba type for kmeans"""
    def __init__(self):
        super(NbTypeKmeans, self).__init__(name='KmeansType')


class NbTypeKmeansResult(types.Opaque):
    """Our numba type for kmeans Result (returned by kmeans.compute)"""
    def __init__(self):
        super(NbTypeKmeansResult, self).__init__(name='KmeansResultType')


# type instance
NbType_kmeans = NbTypeKmeans()
NbType_kmeans_result = NbTypeKmeansResult()


# we treat our class objects as opaque (e.g. pointers)
register_model(NbTypeKmeans)(models.OpaqueModel)
register_model(NbTypeKmeansResult)(models.OpaqueModel)


@type_callable(kmeans)
def type_kmeans(context):
    """declare numba type for constructing a kmeans"""
    def typer(nClusters, maxIterations, fptype, method, accuracyThreshold, gamma, assignFlag, distributed):
        # FIXME: check types
        # FIXME: keyword args
        return NbType_kmeans
    return typer


@lower_builtin(kmeans, types.intp, types.intp, string_type, string_type, types.float64, types.float64, string_type, types.boolean)
def ctor_kmeans(context, builder, sig, args):
    """
    Lowers kmeans constructor.
    We just call the C-function.
    FIXME: keyword args
    """
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(64),
                             lir.IntType(64),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(8).as_pointer(),
                             lir.DoubleType(),
                             lir.DoubleType(),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(1)])
    fn = builder.module.get_or_insert_function(fnty, name="kmeans_ptr")
    return builder.call(fn, args)


@infer_getattr
class KMAttribute(AttributeTemplate):
    """declares numba signatures of attributes/methods of kmeans objects"""
    key = NbTypeKmeans

    @bound_function("kmeans.compute")
    def resolve_compute(self, dict, args, kws):
        # FIXME: keyword args
        assert not kws
        assert len(args) == 2
        return signature(NbType_kmeans_result, *args)


@lower_builtin("kmeans.compute", NbType_kmeans, types.Array(types.float64, 2, 'C'), types.Array(types.float64, 2, 'C'))
def lower_compute_kmeans(context, builder, sig, args):
    """lowers compute method of kmeans objects, accepting 2 array arguments"""
    data_type = sig.args[1]
    data_array = context.make_array(data_type)(context, builder, args[1])
    data_shape = cgutils.unpack_tuple(builder, data_array.shape)
    centroids_type = sig.args[2]
    centroids_array = context.make_array(centroids_type)(context, builder, args[2])
    centroids_shape = cgutils.unpack_tuple(builder, centroids_array.shape)
    
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.DoubleType().as_pointer(), lir.IntType(64), lir.IntType(64),
                             lir.IntType(8).as_pointer(), lir.DoubleType().as_pointer(), lir.IntType(64), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="compute_kmeans")
    
    return builder.call(fn, [args[0],
                             data_array.data, data_shape[0], data_shape[1],
                             centroids_array.data, centroids_shape[0], centroids_shape[1]])


# FIXME: add attributes "assignments", "objectiveFunction", "goalFunction", "nIterations"
add_attr(NbTypeKmeansResult, "centroids", table_type, "get_kmeans__Result_centroids")


##############################################################################
##############################################################################

def full_kmeans(a, nClusters, fptype, initmethod, seed, oversamplingFactor, nRounds, maxIterations, method, accuracyThreshold, gamma, assignFlag, distributed):
    kmi = kmeans_init(nClusters, fptype, initmethod, seed, oversamplingFactor, nRounds, distributed)
    kmir = kmi.compute(a)
    print(kmir.centroids)
    km = kmeans(nClusters, maxIterations, fptype, method, accuracyThreshold, gamma, assignFlag, distributed)
    # FIXME
    #kmr = km.compute(a, kmir.centroids)
    #print(kmr.centroids)

fkm = hpat.jit(nopython=True)(full_kmeans)

a = np.ones((2,2), dtype=np.float64)
fkm(a, 2, "double", "defaultDense", -1, 2.2, 33, 300, "lloydDense", 0.00001, 0.1, "True", False)

#, fptype, method, seed, oversamplingFactor, nRounds, distributed

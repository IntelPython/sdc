import operator
import numpy as np
from llvmlite import ir as lir

import numba
from numba import types, cgutils
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate, AttributeTemplate, bound_function
from numba.extending import register_model, models, infer_getattr, infer, intrinsic

import hpat
import hpat.io
from hpat.str_ext import string_type
from hpat.utils import unliteral_all

if hpat.config._has_h5py:
    import h5py
    from hpat.io import _hdf5
    import llvmlite.binding as ll
    ll.add_symbol('hpat_h5_read_filter', _hdf5.hpat_h5_read_filter)


# **************** Types ***********************

class H5FileType(types.Opaque):
    def __init__(self):
        super(H5FileType, self).__init__(name='H5FileType')


h5file_type = H5FileType()


class H5DatasetType(types.Opaque):
    def __init__(self):
        super(H5DatasetType, self).__init__(name='H5DatasetType')


h5dataset_type = H5DatasetType()


class H5GroupType(types.Opaque):
    def __init__(self):
        super(H5GroupType, self).__init__(name='H5GroupType')


h5group_type = H5GroupType()


class H5DatasetOrGroupType(types.Opaque):
    def __init__(self):
        super(H5DatasetOrGroupType, self).__init__(name='H5DatasetOrGroupType')


h5dataset_or_group_type = H5DatasetOrGroupType()

h5file_data_type = types.int64

if hpat.config._has_h5py:
    # hid_t is 32bit in 1.8 but 64bit in 1.10
    if h5py.version.hdf5_version_tuple[1] == 8:
        h5file_data_type = types.int32
    else:
        assert h5py.version.hdf5_version_tuple[1] == 10


@register_model(H5FileType)
@register_model(H5DatasetType)
@register_model(H5GroupType)
@register_model(H5DatasetOrGroupType)
class H5FileModel(models.IntegerModel):
    def __init__(self, dmm, fe_type):
        super(H5FileModel, self).__init__(dmm, h5file_data_type)


# type for list of names
string_list_type = types.List(string_type)


#################################################

def _create_dataset_typer(args, kws):
    kwargs = dict(kws)
    name = args[0] if len(args) > 0 else types.unliteral(kwargs['name'])
    shape = args[1] if len(args) > 1 else types.unliteral(kwargs['shape'])
    dtype = args[2] if len(args) > 2 else types.unliteral(kwargs['dtype'])

    def create_dset_stub(name, shape, dtype):
        pass
    pysig = numba.utils.pysignature(create_dset_stub)
    return signature(h5dataset_type, name, shape, dtype).replace(pysig=pysig)


@infer_getattr
class FileAttribute(AttributeTemplate):
    key = h5file_type

    @bound_function("h5file.keys")
    def resolve_keys(self, dict, args, kws):
        assert not kws
        assert not args
        return signature(string_list_type, *args)

    @bound_function("h5file.close")
    def resolve_close(self, f_id, args, kws):
        return signature(types.none, *args)

    @bound_function("h5file.create_dataset")
    def resolve_create_dataset(self, f_id, args, kws):
        return _create_dataset_typer(unliteral_all(args), kws)

    @bound_function("h5file.create_group")
    def resolve_create_group(self, f_id, args, kws):
        return signature(h5group_type, *unliteral_all(args))


@infer_getattr
class GroupOrDatasetAttribute(AttributeTemplate):
    key = h5dataset_or_group_type

    @bound_function("h5group.keys")
    def resolve_keys(self, dict, args, kws):
        assert not kws
        assert not args
        return signature(string_list_type, *args)


@infer_getattr
class GroupAttribute(AttributeTemplate):
    key = h5group_type

    @bound_function("h5group.create_dataset")
    def resolve_create_dataset(self, f_id, args, kws):
        return _create_dataset_typer(unliteral_all(args), kws)


@infer_global(operator.getitem)
class GetItemH5File(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        assert not kws
        (in_f, in_idx) = args
        if in_f == h5file_type:
            assert in_idx == string_type
            return signature(h5dataset_or_group_type, in_f, in_idx)
        if in_f == h5dataset_or_group_type and in_idx == string_type:
            return signature(h5dataset_or_group_type, in_f, in_idx)


@infer_global(operator.setitem)
class SetItemH5Dset(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        if args[0] == h5dataset_type:
            return signature(types.none, *args)


def h5g_get_num_objs():
    return


def h5g_get_objname_by_idx():
    return


def h5size():
    """dummy function for C h5_size"""
    return


def h5read():
    """dummy function for C h5_read"""
    return


def h5close():
    """dummy function for C h5_close"""
    return


def h5create_dset():
    """dummy function for C h5_create_dset"""
    return


def h5create_group():
    """dummy function for C h5create_group"""
    return


def h5write():
    """dummy function for C h5_write"""
    return


def h5_read_dummy():
    return


@infer_global(h5_read_dummy)
class H5ReadType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ndim = args[1].literal_value
        dtype = getattr(types, args[2].literal_value)
        ret_typ = types.Array(dtype, ndim, 'C')
        return signature(ret_typ, *args)


if hpat.config._has_h5py:
    @infer_global(h5py.File)
    class H5File(AbstractTemplate):
        def generic(self, args, kws):
            assert not kws
            return signature(h5file_type, *unliteral_all(args))


@infer_global(h5size)
class H5Size(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.int64, *unliteral_all(args))


@infer_global(h5read)
class H5Read(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 6
        return signature(types.int32, *unliteral_all(args))


@infer_global(h5close)
class H5Close(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(types.none, *args)


@infer_global(h5create_dset)
class H5CreateDSet(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        return signature(h5file_type, *unliteral_all(args))


@infer_global(h5create_group)
class H5CreateGroup(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(h5file_type, *unliteral_all(args))


@infer_global(h5write)
class H5Write(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 6
        return signature(types.int32, *unliteral_all(args))


@infer_global(h5g_get_num_objs)
class H5GgetNobj(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(types.int64, *args)


@infer_global(h5g_get_objname_by_idx)
class H5GgetObjNameByIdx(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(string_type, *args)


sum_op = hpat.distributed_api.Reduce_Type.Sum.value


@numba.njit
def get_filter_read_indices(bool_arr):
    indices = bool_arr.nonzero()[0]
    rank = hpat.distributed_api.get_rank()
    n_pes = hpat.distributed_api.get_size()

    # get number of elements before this processor to align the indices
    # assuming bool_arr can be 1D_Var
    all_starts = np.empty(n_pes, np.int64)
    n_bool = len(bool_arr)
    hpat.distributed_api.allgather(all_starts, n_bool)
    ind_start = all_starts.cumsum()[rank] - n_bool
    #n_arr = hpat.distributed_api.dist_reduce(len(bool_arr), np.int32(sum_op))
    #ind_start = hpat.distributed_api.get_start(n_arr, n_pes, rank)
    indices += ind_start

    # TODO: use prefix-sum and all-to-all
    # all_indices = np.empty(n, indices.dtype)
    # allgatherv(all_indices, indices)
    n = hpat.distributed_api.dist_reduce(len(indices), np.int32(sum_op))
    inds = hpat.distributed_api.gatherv(indices)
    if rank == 0:
        all_indices = inds
    else:
        all_indices = np.empty(n, indices.dtype)
    hpat.distributed_api.bcast(all_indices)

    start = hpat.distributed_api.get_start(n, n_pes, rank)
    end = hpat.distributed_api.get_end(n, n_pes, rank)
    return all_indices[start:end]


@intrinsic
def tuple_to_ptr(typingctx, tuple_tp=None):
    def codegen(context, builder, sig, args):
        ptr = cgutils.alloca_once(builder, args[0].type)
        builder.store(args[0], ptr)
        return builder.bitcast(ptr, lir.IntType(8).as_pointer())
    return signature(types.voidptr, tuple_tp), codegen


_h5read_filter = types.ExternalFunction(
    "hpat_h5_read_filter",
    types.int32(
        h5dataset_or_group_type,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.int32))


@numba.njit
def h5read_filter(dset_id, ndim, starts, counts, is_parallel, out_arr, read_indices):
    starts_ptr = tuple_to_ptr(starts)
    counts_ptr = tuple_to_ptr(counts)
    type_enum = hpat.distributed_api.get_type_enum(out_arr)
    return _h5read_filter(dset_id, ndim, starts_ptr, counts_ptr, is_parallel,
                          out_arr.ctypes, type_enum, read_indices.ctypes, len(read_indices))

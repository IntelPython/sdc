from numba import types
from numba.typing.templates import infer_global, AbstractTemplate, AttributeTemplate, bound_function
from numba.typing import signature
import h5py
from numba.extending import register_model, models, infer_getattr, infer
from hpat.str_ext import string_type


################## Types #######################

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

@infer
class GetItemH5File(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        (in_f, in_idx) = args
        if in_f == h5file_type:
            assert in_idx == string_type
            return signature(h5dataset_or_group_type, in_f, in_idx)
        if in_f == h5dataset_or_group_type and in_idx == string_type:
            return signature(h5dataset_or_group_type, in_f, in_idx)

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
        ndim = args[1].value
        dtype = getattr(types, args[2].value)
        ret_typ = types.Array(dtype, ndim, 'C')
        return signature(ret_typ, *args)

H5ReadType.support_literals = True

@infer_global(h5py.File)
class H5File(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(h5file_type, *args)


@infer_global(h5size)
class H5Size(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.int64, *args)


@infer_global(h5read)
class H5Read(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 6
        return signature(types.int32, *args)


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
        return signature(h5file_type, *args)


@infer_global(h5create_group)
class H5CreateGroup(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(h5file_type, *args)


@infer_global(h5write)
class H5Write(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 7
        return signature(types.int32, *args)


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

from numba import types
from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature
import h5py

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

@infer_global(h5py.File)
class H5File(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==3
        return signature(types.int32, *args)

@infer_global(h5size)
class H5Size(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==3
        return signature(types.int64, *args)

@infer_global(h5read)
class H5Read(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==7
        return signature(types.int32, *args)

@infer_global(h5close)
class H5Close(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==1
        return signature(types.int32, *args)

@infer_global(h5create_dset)
class H5CreateDSet(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==4
        return signature(types.int32, *args)

@infer_global(h5create_group)
class H5CreateGroup(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==2
        return signature(types.int32, *args)

@infer_global(h5write)
class H5Write(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==7
        return signature(types.int32, *args)

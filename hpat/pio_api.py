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

from numba import types
from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature
import time

def get_rank():
    """dummy function for C mpi get_rank"""
    return 0

def get_size():
    """dummy function for C mpi get_size"""
    return 0

def get_end(total_size, div, pes, rank):
    """get end point of range for parfor division"""
    return total_size if rank==pes-1 else (rank+1)*div

def get_node_portion(total_size, div, pes, rank):
    """get portion of size for alloc division"""
    return total_size-div*rank if rank==pes-1 else div

def dist_reduce(value):
    """dummy to implement simple reductions"""
    return value

def dist_arr_reduce(arr):
    """dummy to implement array reductions"""
    return -1

def dist_cumsum(arr):
    """dummy to implement cumsum"""
    return arr

def dist_cumprod(arr):
    """dummy to implement cumprod"""
    return arr

def dist_exscan(value):
    """dummy to implement simple exscan"""
    return value

def irecv():
    return 0

def isend():
    return 0

def wait():
    return 0

@infer_global(get_rank)
class DistRank(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==0
        return signature(types.int32, *args)

@infer_global(get_size)
class DistSize(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==0
        return signature(types.int32, *args)

@infer_global(get_end)
class DistEnd(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==4
        return signature(types.int64, *args)

@infer_global(get_node_portion)
class DistPortion(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==4
        return signature(types.int64, *args)

@infer_global(dist_reduce)
class DistReduce(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==1
        return signature(args[0], *args)

@infer_global(dist_exscan)
class DistExscan(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==1
        return signature(args[0], *args)

@infer_global(dist_arr_reduce)
class DistArrReduce(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==1
        return signature(types.int32, *args)

@infer_global(time.time)
class DistTime(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==0
        return signature(types.float64, *args)

@infer_global(dist_cumsum)
@infer_global(dist_cumprod)
class DistCumsumprod(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==2
        return signature(types.int32, *args)

@infer_global(irecv)
@infer_global(isend)
class DistIRecv(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==5
        return signature(types.int32, *args)

@infer_global(wait)
class DistWait(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==2
        return signature(types.int32, *args)

import operator
import numpy as np
import numba
from numba import types
from numba.typing.templates import infer_global, AbstractTemplate, infer
from numba.typing import signature
from numba.extending import models, register_model, intrinsic, overload
import hpat
from hpat.str_arr_ext import (string_array_type, num_total_chars, StringArray,
                              pre_alloc_string_array, get_offset_ptr,
                              get_data_ptr, convert_len_arr_to_offset)
from hpat.utils import (debug_prints, empty_like_type, _numba_to_c_type_map,
    unliteral_all)
import time
from llvmlite import ir as lir
import hdist
import llvmlite.binding as ll
ll.add_symbol('c_alltoall', hdist.c_alltoall)
ll.add_symbol('c_gather_scalar', hdist.c_gather_scalar)
ll.add_symbol('c_gatherv', hdist.c_gatherv)
ll.add_symbol('c_bcast', hdist.c_bcast)
ll.add_symbol('c_recv', hdist.hpat_dist_recv)
ll.add_symbol('c_send', hdist.hpat_dist_send)

from enum import Enum

# get size dynamically from C code (mpich 3.2 is 4 bytes but openmpi 1.6 is 8)
import hdist
mpi_req_numba_type = getattr(types, "int"+str(8 * hdist.mpi_req_num_bytes))

MPI_ROOT = 0

class Reduce_Type(Enum):
    Sum = 0
    Prod = 1
    Min = 2
    Max = 3
    Argmin = 4
    Argmax = 5
    Or = 6


def get_type_enum(arr):
    return np.int32(-1)

@overload(get_type_enum)
def get_type_enum_overload(arr):
    dtype = arr.dtype
    if isinstance(dtype, hpat.hiframes.pd_categorical_ext.PDCategoricalDtype):
        dtype = hpat.hiframes.pd_categorical_ext.get_categories_int_type(dtype)

    typ_val = _numba_to_c_type_map[dtype]
    return lambda arr: np.int32(typ_val)

INT_MAX = np.iinfo(np.int32).max

_send = types.ExternalFunction("c_send", types.void(types.voidptr, types.int32, types.int32, types.int32, types.int32))

@numba.njit
def send(val, rank, tag):
    # dummy array for val
    send_arr = np.full(1, val)
    type_enum = get_type_enum(send_arr)
    _send(send_arr.ctypes, 1, type_enum, rank, tag)


_recv = types.ExternalFunction("c_recv", types.void(types.voidptr, types.int32, types.int32, types.int32, types.int32))

@numba.njit
def recv(dtype, rank, tag):
    # dummy array for val
    recv_arr = np.empty(1, dtype)
    type_enum = get_type_enum(recv_arr)
    _recv(recv_arr.ctypes, 1, type_enum, rank, tag)
    return recv_arr[0]


_alltoall = types.ExternalFunction("c_alltoall", types.void(types.voidptr, types.voidptr, types.int32, types.int32))

@numba.njit
def alltoall(send_arr, recv_arr, count):
    # TODO: handle int64 counts
    assert count < INT_MAX
    type_enum = get_type_enum(send_arr)
    _alltoall(send_arr.ctypes, recv_arr.ctypes, np.int32(count), type_enum)

def gather_scalar(data):  # pragma: no cover
    return np.ones(1)

c_gather_scalar = types.ExternalFunction("c_gather_scalar", types.void(types.voidptr, types.voidptr, types.int32))

# TODO: test
@overload(gather_scalar)
def gather_scalar_overload(val):
    assert isinstance(val, (types.Integer, types.Float))
    # TODO: other types like boolean
    typ_val = _numba_to_c_type_map[val]
    func_text = (
    "def gather_scalar_impl(val):\n"
    "  n_pes = hpat.distributed_api.get_size()\n"
    "  rank = hpat.distributed_api.get_rank()\n"
    "  send = np.full(1, val, np.{})\n"
    "  res_size = n_pes if rank == {} else 0\n"
    "  res = np.empty(res_size, np.{})\n"
    "  c_gather_scalar(send.ctypes, res.ctypes, np.int32({}))\n"
    "  return res\n").format(val, MPI_ROOT, val, typ_val)

    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'np': np, 'c_gather_scalar': c_gather_scalar}, loc_vars)
    gather_impl = loc_vars['gather_scalar_impl']
    return gather_impl

# TODO: test
def gatherv(data):  # pragma: no cover
    return data

# sendbuf, sendcount, recvbuf, recv_counts, displs, dtype
c_gatherv = types.ExternalFunction("c_gatherv",
    types.void(types.voidptr, types.int32, types.voidptr, types.voidptr, types.voidptr, types.int32))

@overload(gatherv)
def gatherv_overload(data):
    if isinstance(data, types.Array):
        # TODO: other types like boolean
        typ_val = _numba_to_c_type_map[data.dtype]

        def gatherv_impl(data):
            rank = hpat.distributed_api.get_rank()
            n_loc = len(data)
            recv_counts = gather_scalar(np.int32(n_loc))
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int32)
            if rank == MPI_ROOT:
                displs = hpat.hiframes.join.calc_disp(recv_counts)
            #  print(rank, n_loc, n_total, recv_counts, displs)
            c_gatherv(data.ctypes, np.int32(n_loc), all_data.ctypes, recv_counts.ctypes, displs.ctypes, np.int32(typ_val))
            return all_data

        return gatherv_impl

    if data == string_array_type:
        int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
        char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])

        def gatherv_str_arr_impl(data):
            rank = hpat.distributed_api.get_rank()
            n_loc = len(data)
            n_all_chars = num_total_chars(data)

            # allocate send lens arrays
            send_arr_lens = np.empty(n_loc, np.uint32)  # XXX offset type is uint32
            send_data_ptr = get_data_ptr(data)

            for i in range(n_loc):
                _str = data[i]
                send_arr_lens[i] = len(_str)

            recv_counts = gather_scalar(np.int32(n_loc))
            recv_counts_char = gather_scalar(np.int32(n_all_chars))
            n_total = recv_counts.sum()
            n_total_char = recv_counts_char.sum()


            # displacements
            all_data = StringArray([''])  # dummy arrays on non-root PEs
            displs = np.empty(1, np.int32)
            displs_char = np.empty(1, np.int32)

            if rank == MPI_ROOT:
                all_data = pre_alloc_string_array(n_total, n_total_char)
                displs = hpat.hiframes.join.calc_disp(recv_counts)
                displs_char = hpat.hiframes.join.calc_disp(recv_counts_char)

            #  print(rank, n_loc, n_total, recv_counts, displs)
            offset_ptr = get_offset_ptr(all_data)
            data_ptr = get_data_ptr(all_data)
            c_gatherv(send_arr_lens.ctypes, np.int32(n_loc), offset_ptr, recv_counts.ctypes, displs.ctypes, int32_typ_enum)
            c_gatherv(send_data_ptr, np.int32(n_all_chars), data_ptr, recv_counts_char.ctypes, displs_char.ctypes, char_typ_enum)
            convert_len_arr_to_offset(offset_ptr, n_total)
            return all_data

        return gatherv_str_arr_impl



# TODO: test
# TODO: large BCast

def bcast(data):  # pragma: no cover
    return

@overload(bcast)
def bcast_overload(data):
    if isinstance(data, types.Array):
        def bcast_impl(data):
            typ_enum = get_type_enum(data)
            count = len(data)
            assert count < INT_MAX
            c_bcast(data.ctypes, np.int32(count), typ_enum)
            return
        return bcast_impl

    if data == string_array_type:
        int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
        char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])

        def bcast_str_impl(data):
            rank = hpat.distributed_api.get_rank()
            n_loc = len(data)
            n_all_chars = num_total_chars(data)
            assert n_loc < INT_MAX
            assert n_all_chars < INT_MAX

            offset_ptr = get_offset_ptr(data)
            data_ptr = get_data_ptr(data)

            if rank == MPI_ROOT:
                send_arr_lens = np.empty(n_loc, np.uint32)  # XXX offset type is uint32
                for i in range(n_loc):
                    _str = data[i]
                    send_arr_lens[i] = len(_str)

                c_bcast(send_arr_lens.ctypes, np.int32(n_loc), int32_typ_enum)
            else:
                c_bcast(offset_ptr, np.int32(n_loc), int32_typ_enum)

            c_bcast(data_ptr, np.int32(n_all_chars), char_typ_enum)
            if rank != MPI_ROOT:
                convert_len_arr_to_offset(offset_ptr, n_loc)

        return bcast_str_impl

# sendbuf, sendcount, dtype
c_bcast = types.ExternalFunction("c_bcast",
    types.void(types.voidptr, types.int32, types.int32))

def bcast_scalar(val):  # pragma: no cover
    return val

# TODO: test
@overload(bcast_scalar)
def bcast_scalar_overload(val):
    assert isinstance(val, (types.Integer, types.Float))
    # TODO: other types like boolean
    typ_val = _numba_to_c_type_map[val]
    # TODO: fix np.full and refactor
    func_text = (
    "def bcast_scalar_impl(val):\n"
    "  send = np.full(1, val, np.{})\n"
    "  c_bcast(send.ctypes, np.int32(1), np.int32({}))\n"
    "  return send[0]\n").format(val, typ_val)

    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'np': np, 'c_bcast': c_bcast}, loc_vars)
    bcast_scalar_impl = loc_vars['bcast_scalar_impl']
    return bcast_scalar_impl

# if arr is string array, pre-allocate on non-root the same size as root
def prealloc_str_for_bcast(arr):
    return arr

@overload(prealloc_str_for_bcast)
def prealloc_str_for_bcast_overload(arr):
    if arr == string_array_type:
        def prealloc_impl(arr):
            rank = hpat.distributed_api.get_rank()
            n_loc = bcast_scalar(len(arr))
            n_all_char = bcast_scalar(np.int64(num_total_chars(arr)))
            if rank != MPI_ROOT:
                arr = pre_alloc_string_array(n_loc, n_all_char)
            return arr

        return prealloc_impl

    return lambda arr: arr

# send_data, recv_data, send_counts, recv_counts, send_disp, recv_disp, typ_enum
c_alltoallv = types.ExternalFunction("c_alltoallv", types.void(types.voidptr,
    types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.int32))

# TODO: test
# TODO: big alltoallv
@numba.njit
def alltoallv(send_data, out_data, send_counts, recv_counts, send_disp, recv_disp):  # pragma: no cover
    typ_enum = get_type_enum(send_data)
    typ_enum_o = get_type_enum(out_data)
    assert typ_enum == typ_enum_o

    c_alltoallv(send_data.ctypes, out_data.ctypes, send_counts.ctypes,
              recv_counts.ctypes, send_disp.ctypes, recv_disp.ctypes, typ_enum)
    return

def alltoallv_tup(send_data, out_data, send_counts, recv_counts, send_disp, recv_disp):  # pragma: no cover
    return

@overload(alltoallv_tup)
def alltoallv_tup_overload(send_data, out_data, send_counts, recv_counts, send_disp, recv_disp):

    count = send_data.count
    assert out_data.count == count

    func_text = "def f(send_data, out_data, send_counts, recv_counts, send_disp, recv_disp):\n"
    for i in range(count):
        func_text += "  alltoallv(send_data[{}], out_data[{}], send_counts, recv_counts, send_disp, recv_disp)\n".format(i, i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {'alltoallv': alltoallv}, loc_vars)
    a2a_impl = loc_vars['f']
    return a2a_impl

def get_rank():  # pragma: no cover
    """dummy function for C mpi get_rank"""
    return 0


def barrier():  # pragma: no cover
    return 0


def get_size():  # pragma: no cover
    """dummy function for C mpi get_size"""
    return 0


def get_start(total_size, pes, rank):  # pragma: no cover
    """get end point of range for parfor division"""
    return 0


def get_end(total_size, pes, rank):  # pragma: no cover
    """get end point of range for parfor division"""
    return 0


def get_node_portion(total_size, pes, rank):  # pragma: no cover
    """get portion of size for alloc division"""
    return 0


def dist_reduce(value, op):  # pragma: no cover
    """dummy to implement simple reductions"""
    return value


def dist_arr_reduce(arr):  # pragma: no cover
    """dummy to implement array reductions"""
    return -1


def dist_cumsum(arr):  # pragma: no cover
    """dummy to implement cumsum"""
    return arr


def dist_cumprod(arr):  # pragma: no cover
    """dummy to implement cumprod"""
    return arr


def dist_exscan(value):  # pragma: no cover
    """dummy to implement simple exscan"""
    return value


def dist_setitem(arr, index, val):  # pragma: no cover
    return 0

def allgather(arr, val):  # pragma: no cover
    arr[0] = val

def dist_time():  # pragma: no cover
    return time.time()

def dist_return(A):  # pragma: no cover
    return A

def threaded_return(A):  # pragma: no cover
    return A

def rebalance_array(A):
    return A

def rebalance_array_parallel(A):
    return A


@overload(rebalance_array)
def dist_return_overload(A):
    return dist_return

# TODO: move other funcs to old API?
@infer_global(threaded_return)
@infer_global(dist_return)
class ThreadedRetTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1  # array
        return signature(args[0], *args)


def irecv():  # pragma: no cover
    return 0


def isend():  # pragma: no cover
    return 0


def wait():  # pragma: no cover
    return 0

def waitall():  # pragma: no cover
    return 0

@infer_global(allgather)
class DistAllgather(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2  # array and val
        return signature(types.none, *unliteral_all(args))

@infer_global(rebalance_array_parallel)
class DistRebalanceParallel(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2  # array and count
        return signature(args[0], *unliteral_all(args))


@infer_global(get_rank)
class DistRank(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 0
        return signature(types.int32, *unliteral_all(args))


@infer_global(get_size)
class DistSize(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 0
        return signature(types.int32, *unliteral_all(args))


@infer_global(get_start)
class DistStart(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        return signature(types.int64, *unliteral_all(args))


@infer_global(get_end)
class DistEnd(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        return signature(types.int64, *unliteral_all(args))


@infer_global(get_node_portion)
class DistPortion(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        return signature(types.int64, *unliteral_all(args))


@infer_global(dist_reduce)
class DistReduce(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2  # value and reduce_op
        return signature(args[0], *unliteral_all(args))


@infer_global(dist_exscan)
class DistExscan(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(args[0], *unliteral_all(args))


@infer_global(dist_arr_reduce)
class DistArrReduce(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2  # value and reduce_op
        return signature(types.int32, *unliteral_all(args))


@infer_global(time.time)
class DistTime(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 0
        return signature(types.float64, *unliteral_all(args))


@infer_global(dist_time)
class DistDistTime(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 0
        return signature(types.float64, *unliteral_all(args))


@infer_global(barrier)
class DistBarrier(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 0
        return signature(types.int32, *unliteral_all(args))


@infer_global(dist_cumsum)
@infer_global(dist_cumprod)
class DistCumsumprod(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.int32, *unliteral_all(args))


@infer_global(irecv)
@infer_global(isend)
class DistIRecv(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) in [4, 5]
        return signature(mpi_req_numba_type, *unliteral_all(args))


@infer_global(wait)
class DistWait(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.int32, *unliteral_all(args))

@infer_global(waitall)
class DistWaitAll(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2 and args == (types.int32, req_array_type)
        return signature(types.none, *unliteral_all(args))

# @infer_global(dist_setitem)
# class DistSetitem(AbstractTemplate):
#     def generic(self, args, kws):
#         assert not kws
#         assert len(args)==5
#         return signature(types.int32, *unliteral_all(args))


class ReqArrayType(types.Type):
    def __init__(self):
        super(ReqArrayType, self).__init__(
            name='ReqArrayType()')

req_array_type = ReqArrayType()
register_model(ReqArrayType)(models.OpaqueModel)

def comm_req_alloc():
    return 0

def comm_req_dealloc():
    return 0

@infer_global(comm_req_alloc)
class DistCommReqAlloc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1 and args[0] == types.int32
        return signature(req_array_type, *unliteral_all(args))

@infer_global(comm_req_dealloc)
class DistCommReqDeAlloc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1 and args[0] == req_array_type
        return signature(types.none, *unliteral_all(args))

@infer_global(operator.setitem)
class SetItemReqArray(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [ary, idx, val] = args
        if isinstance(ary, ReqArrayType) and idx == types.intp and val == mpi_req_numba_type:
            return signature(types.none, *unliteral_all(args))

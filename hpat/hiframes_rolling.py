import numpy as np
import pandas as pd
import hpat
import numba
from numba import types
from numba.extending import lower_builtin
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.ir_utils import guard, find_const


def get_rolling_setup_args(func_ir, rhs, get_consts=True):
    """
    Handle Series rolling calls like:
        r = df.column.rolling(3)
    """
    center = False
    kws = dict(rhs.kws)
    if rhs.args:
        window = rhs.args[0]
    elif 'window' in kws:
        window = kws['window']
    else:  # pragma: no cover
        raise ValueError("window argument to rolling() required")
    if get_consts:
        window_const = guard(find_const, func_ir, window)
        window = window_const if window_const is not None else window
    if 'center' in kws:
        center = kws['center']
        if get_consts:
            center_const = guard(find_const, func_ir, center)
            center = center_const if center_const is not None else center
    return window, center


def rolling_fixed(arr, win):  # pragma: no cover
    return arr

def rolling_fixed_parallel(arr, win):  # pragma: no cover
    return arr


@infer_global(rolling_fixed)
@infer_global(rolling_fixed_parallel)
class RollingType(AbstractTemplate):
    def generic(self, args, kws):
        arr = args[0]  # array or series
        # result is always float64 in pandas
        # see _prep_values() in window.py
        return signature(arr.copy(dtype=types.float64), arr, types.intp,
                         types.bool_, types.bool_, args[4])

RollingType.support_literals = True


@lower_builtin(rolling_fixed, types.Array, types.Integer, types.Boolean,
               types.Boolean, types.Const)
def lower_rolling_fixed(context, builder, sig, args):
    func_name = sig.args[-1].value
    if func_name == 'sum':
        func = roll_sum_fixed

    res = context.compile_internal(
        builder, func, signature(sig.return_type, *sig.args[:-1]), args[:-1])
    return impl_ret_borrowed(context, builder, sig.return_type, res)


#### adapted from pandas window.pyx ####

def roll_sum_fixed(in_arr, win, center, parallel):
    sum_x = 0.0
    nobs = 0
    N = len(in_arr)
    output = np.empty(N, dtype=np.float64)
    rank = hpat.distributed_api.get_rank()
    n_pes = hpat.distributed_api.get_size()
    comm_tag = np.int32(22)  # arbitrary

    # TODO: support minp arg end_range etc.
    minp = win
    offset = (win - 1) // 2 if center else 0
    range_endpoint = max(minp, 1) - 1
    # in case window is smaller than array
    range_endpoint = min(range_endpoint, N)

    if parallel:
        # TODO: center
        halo_size = np.int32((win - 1) // 2) if center else np.int32(win-1)
        recv_buff = np.empty(halo_size, in_arr.dtype)
        if center:
            r_recv_buff = np.empty(halo_size, in_arr.dtype)
        # send right
        if rank != n_pes - 1:
            send_req = hpat.distributed_api.isend(in_arr[-halo_size:], halo_size, np.int32(rank+1), comm_tag, True)
        # recv left
        if rank != 0:
            recv_req = hpat.distributed_api.irecv(recv_buff, halo_size, np.int32(rank-1), comm_tag, True)
        # center cases
        # send left
        if center and rank != 0:
            l_send_req = hpat.distributed_api.isend(in_arr[:halo_size], halo_size, np.int32(rank-1), comm_tag, True)
        # recv right
        if center and rank != n_pes - 1:
            r_recv_req = hpat.distributed_api.irecv(r_recv_buff, halo_size, np.int32(rank+1), comm_tag, True)

    for i in range(0, range_endpoint):
        nobs, sum_x = add_sum(in_arr[i], nobs, sum_x)
        if i >= offset:
            output[i - offset] = np.nan

    for i in range(range_endpoint, N):
        val = in_arr[i]
        nobs, sum_x = add_sum(val, nobs, sum_x)

        if i > win - 1:
            prev_x = in_arr[i - win]
            nobs, sum_x = remove_sum(prev_x, nobs, sum_x)

        output[i - offset] = calc_sum(minp, nobs, sum_x)

    for j in range(N - offset, N):
        output[j] = np.nan

    if parallel:
        # wait on send right
        if rank != n_pes - 1:
            hpat.distributed_api.wait(send_req, True)
        # wait on send left
        if center and rank != 0:
            hpat.distributed_api.wait(l_send_req, True)
        # recv right
        if center and rank != n_pes - 1:
            hpat.distributed_api.wait(r_recv_req, True)

            for i in range(0, halo_size):
                nobs, sum_x = add_sum(r_recv_buff[i], nobs, sum_x)

                prev_x = in_arr[N + i - win]
                nobs, sum_x = remove_sum(prev_x, nobs, sum_x)

                output[N + i - offset] = calc_sum(minp, nobs, sum_x)

        # recv left
        if rank != 0:
            hpat.distributed_api.wait(recv_req, True)
            sum_x = 0.0
            nobs = 0
            for i in range(0, halo_size):
                nobs, sum_x = add_sum(recv_buff[i], nobs, sum_x)

            for i in range(0, win - 1):
                nobs, sum_x = add_sum(in_arr[i], nobs, sum_x)

                if i > offset:
                    prev_x = recv_buff[i - offset - 1]
                    nobs, sum_x = remove_sum(prev_x, nobs, sum_x)

                if i >= offset:
                    output[i - offset] = calc_sum(minp, nobs, sum_x)

    return output

@numba.njit
def add_sum(val, nobs, sum_x):
    if not np.isnan(val):
        nobs += 1
        sum_x += val
    return nobs, sum_x

@numba.njit
def remove_sum(val, nobs, sum_x):
    if not np.isnan(val):
        nobs -= 1
        sum_x -= val
    return nobs, sum_x

@numba.njit
def calc_sum(minp, nobs, sum_x):
    return sum_x if nobs >= minp else np.nan

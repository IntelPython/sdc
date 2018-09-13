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

from hpat.distributed_api import Reduce_Type
from hpat.pd_timestamp_ext import integer_to_dt64

supported_rolling_funcs = ('sum', 'mean', 'var', 'std', 'apply')


def get_rolling_setup_args(func_ir, rhs, get_consts=True):
    """
    Handle Series rolling calls like:
        r = df.column.rolling(3)
    """
    center = False
    on = None
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
    if 'on' in kws:
        on = guard(find_const, func_ir, kws['on'])
        if on is None:
            raise ValueError("'on' argument to rolling() should be constant string")
    # convert string offset window statically to nanos
    # TODO: support dynamic conversion
    # TODO: support other offsets types (time delta, etc.)
    if on is not None:
        window = guard(find_const, func_ir, window)
        if not isinstance(window, str):
            raise ValueError("window argument to rolling should be constant"
                             "string in the offset case (variable window)")
        window = pd.tseries.frequencies.to_offset(window).nanos
    return window, center, on


def rolling_fixed(arr, win):  # pragma: no cover
    return arr

def rolling_fixed_parallel(arr, win):  # pragma: no cover
    return arr

def rolling_variable(arr, on_arr, win):  # pragma: no cover
    return arr


@infer_global(rolling_fixed)
@infer_global(rolling_fixed_parallel)
class RollingType(AbstractTemplate):
    def generic(self, args, kws):
        arr = args[0]  # array or series
        # result is always float64 in pandas
        # see _prep_values() in window.py
        f_type = args[4]
        # Const of CPUDispatcher needs to be converted to type since lowerer
        # cannot handle it
        if (isinstance(f_type, types.Const)
                and isinstance(f_type.value,
                numba.targets.registry.CPUDispatcher)):
            f_type = types.functions.Dispatcher(f_type.value)
        return signature(arr.copy(dtype=types.float64), arr, types.intp,
                         types.bool_, types.bool_, f_type)

RollingType.support_literals = True


@infer_global(rolling_variable)
class RollingVarType(AbstractTemplate):
    def generic(self, args, kws):
        arr = args[0]  # array or series
        on_arr = args[1]
        # result is always float64 in pandas
        # see _prep_values() in window.py
        f_type = args[5]
        # Const of CPUDispatcher needs to be converted to type since lowerer
        # cannot handle it
        if (isinstance(f_type, types.Const)
                and isinstance(f_type.value,
                numba.targets.registry.CPUDispatcher)):
            f_type = types.functions.Dispatcher(f_type.value)
        return signature(arr.copy(dtype=types.float64), arr, on_arr, types.intp,
                         types.bool_, types.bool_, f_type)

RollingVarType.support_literals = True


@lower_builtin(rolling_fixed, types.Array, types.Integer, types.Boolean,
               types.Boolean, types.Const)
def lower_rolling_fixed(context, builder, sig, args):
    func_name = sig.args[-1].value
    if func_name == 'sum':
        func = lambda a,w,c,p: roll_fixed_linear_generic(a,w,c,p, init_data_sum, add_sum, remove_sum, calc_sum)
    elif func_name == 'mean':
        func = lambda a,w,c,p: roll_fixed_linear_generic(a,w,c,p, init_data_mean, add_mean, remove_mean, calc_mean)
    elif func_name == 'var':
        func = lambda a,w,c,p: roll_fixed_linear_generic(a,w,c,p, init_data_var, add_var, remove_var, calc_var)
    elif func_name == 'std':
        func = lambda a,w,c,p: roll_fixed_linear_generic(a,w,c,p, init_data_var, add_var, remove_var, calc_std)

    res = context.compile_internal(
        builder, func, signature(sig.return_type, *sig.args[:-1]), args[:-1])
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@lower_builtin(rolling_fixed, types.Array, types.Integer, types.Boolean,
               types.Boolean, types.functions.Dispatcher)
def lower_rolling_fixed_apply(context, builder, sig, args):
    func = lambda a,w,c,p,f: roll_fixed_apply(a,w,c,p,f)
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin(rolling_variable, types.Array, types.Array, types.Integer, types.Boolean,
               types.Boolean, types.Const)
def lower_rolling_variable(context, builder, sig, args):
    func_name = sig.args[-1].value
    if func_name == 'sum':
        func = lambda a,o,w,c,p: roll_var_linear_generic(a,o,w,c,p, init_data_sum, add_sum, remove_sum, calc_sum)
    elif func_name == 'mean':
        func = lambda a,o,w,c,p: roll_var_linear_generic(a,o,w,c,p, init_data_mean, add_mean, remove_mean, calc_mean)
    elif func_name == 'var':
        func = lambda a,o,w,c,p: roll_var_linear_generic(a,o,w,c,p, init_data_var, add_var, remove_var, calc_var)
    elif func_name == 'std':
        func = lambda a,o,w,c,p: roll_var_linear_generic(a,o,w,c,p, init_data_var, add_var, remove_var, calc_std)

    res = context.compile_internal(
        builder, func, signature(sig.return_type, *sig.args[:-1]), args[:-1])
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin(rolling_variable, types.Array, types.Array, types.Integer, types.Boolean,
               types.Boolean, types.functions.Dispatcher)
def lower_rolling_variable_apply(context, builder, sig, args):
    func = lambda a,o,w,c,p,f: roll_variable_apply(a,o,w,c,p,f)
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)

#### adapted from pandas window.pyx ####

comm_border_tag = 22  # arbitrary, TODO: revisit comm tags


@numba.njit
def roll_fixed_linear_generic(in_arr, win, center, parallel, init_data,
                              add_obs, remove_obs, calc_out):  # pragma: no cover
    rank = hpat.distributed_api.get_rank()
    n_pes = hpat.distributed_api.get_size()
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = win
    offset = (win - 1) // 2 if center else 0

    if parallel:
        # halo length is w/2 to handle even w such as w=4
        halo_size = np.int32(win // 2) if center else np.int32(win-1)
        if _is_small_for_parallel(N, halo_size):
            return _handle_small_data(in_arr, win, center, rank, n_pes,
                              init_data, add_obs, remove_obs, calc_out)

        comm_data = _border_icomm(
            in_arr, rank, n_pes, halo_size, in_arr.dtype, center)
        (l_recv_buff, r_recv_buff, l_send_req, r_send_req, l_recv_req,
            r_recv_req) = comm_data

    output, data = roll_fixed_linear_generic_seq(in_arr, win, center,
                            init_data, add_obs, remove_obs, calc_out)

    if parallel:
        _border_send_wait(r_send_req, l_send_req, rank, n_pes, center)

        # recv right
        if center and rank != n_pes - 1:
            hpat.distributed_api.wait(r_recv_req, True)

            for i in range(0, halo_size):
                data = add_obs(r_recv_buff[i], *data)

                prev_x = in_arr[N + i - win]
                data = remove_obs(prev_x, *data)

                output[N + i - offset] = calc_out(minp, *data)

        # recv left
        if rank != 0:
            hpat.distributed_api.wait(l_recv_req, True)
            data = init_data()
            for i in range(0, halo_size):
                data = add_obs(l_recv_buff[i], *data)

            for i in range(0, win - 1):
                data = add_obs(in_arr[i], *data)

                if i > offset:
                    prev_x = l_recv_buff[i - offset - 1]
                    data = remove_obs(prev_x, *data)

                if i >= offset:
                    output[i - offset] = calc_out(minp, *data)

    return output

@numba.njit
def roll_fixed_linear_generic_seq(in_arr, win, center, init_data, add_obs,
                                  remove_obs, calc_out):  # pragma: no cover
    data = init_data()
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = win
    offset = (win - 1) // 2 if center else 0
    output = np.empty(N, dtype=np.float64)
    range_endpoint = max(minp, 1) - 1
    # in case window is smaller than array
    range_endpoint = min(range_endpoint, N)

    for i in range(0, range_endpoint):
        data = add_obs(in_arr[i], *data)
        if i >= offset:
            output[i - offset] = np.nan

    for i in range(range_endpoint, N):
        val = in_arr[i]
        data = add_obs(val, *data)

        if i > win - 1:
            prev_x = in_arr[i - win]
            data = remove_obs(prev_x, *data)

        output[i - offset] = calc_out(minp, *data)

    for j in range(N - offset, N):
        output[j] = np.nan

    return output, data

@numba.njit
def roll_fixed_apply(in_arr, win, center, parallel, kernel_func):  # pragma: no cover
    rank = hpat.distributed_api.get_rank()
    n_pes = hpat.distributed_api.get_size()
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = win
    offset = (win - 1) // 2 if center else 0

    if parallel:
        # halo length is w/2 to handle even w such as w=4
        halo_size = np.int32(win // 2) if center else np.int32(win-1)
        if _is_small_for_parallel(N, halo_size):
            return _handle_small_data_apply(in_arr, win, center, rank, n_pes,
                                      kernel_func)

        comm_data = _border_icomm(
            in_arr, rank, n_pes, halo_size, in_arr.dtype, center)
        (l_recv_buff, r_recv_buff, l_send_req, r_send_req, l_recv_req,
            r_recv_req) = comm_data

    output = roll_fixed_apply_seq(in_arr, win, center, kernel_func)

    if parallel:
        _border_send_wait(r_send_req, l_send_req, rank, n_pes, center)

        # recv right
        if center and rank != n_pes - 1:
            hpat.distributed_api.wait(r_recv_req, True)
            border_data = np.concatenate((in_arr[N-win+1:], r_recv_buff))
            ind = 0
            for i in range(max(N-offset, 0), N):
                output[i] = kernel_func(border_data[ind:ind+win])
                ind += 1

        # recv left
        if rank != 0:
            hpat.distributed_api.wait(l_recv_req, True)
            border_data = np.concatenate((l_recv_buff, in_arr[:win-1]))
            for i in range(0, win - offset - 1):
                output[i] = kernel_func(border_data[i:i+win])

    return output

@numba.njit
def roll_fixed_apply_seq(in_arr, win, center, kernel_func):  # pragma: no cover
    # TODO
    N = len(in_arr)
    output = np.empty(N, dtype=np.float64)
    # minp = win
    offset = (win - 1) // 2 if center else 0
    output = np.empty(N, dtype=np.float64)

    # TODO: handle count and minp
    for i in range(0, min(win - 1, N) - offset):
        output[i] = np.nan
    ind = 0
    for i in range(win - 1 - offset, N - offset):
        output[i] = kernel_func(in_arr[ind:ind+win])
        ind += 1
    for i in range(max(N - offset, 0), N):
        output[i] = np.nan

    return output


# -----------------------------
# variable window

@numba.njit
def roll_var_linear_generic(in_arr, on_arr, win, center, parallel, init_data,
                              add_obs, remove_obs, calc_out):  # pragma: no cover
    #
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = 1
    output = np.empty(N, np.float64)

    # Pandas is right closed by default, TODO: extend to arg
    start, end = _build_indexer(on_arr, N, win, False, True)
    data = init_data()

    # setup (first element)
    for j in range(start[0], end[0]):
        data = add_obs(in_arr[j], *data)

    output[0] = calc_out(minp, *data)

    for i in range(1, N):
        s = start[i]
        e = end[i]

        # calculate deletes
        for j in range(start[i - 1], s):
            data = remove_obs(in_arr[j], *data)

        # calculate adds
        for j in range(end[i - 1], e):
            data = add_obs(in_arr[j], *data)

        output[i] = calc_out(minp, *data)

    return output

@numba.njit
def roll_variable_apply(in_arr, on_arr, win, center, parallel, kernel_func):  # pragma: no cover
    # TODO
    N = len(in_arr)
    minp = 1
    output = np.empty(N, dtype=np.float64)
    start, end = _build_indexer(on_arr, N, win, False, True)

    # TODO: handle count and minp
    for i in range(0, N):
        s = start[i]
        e = end[i]
        if e - s >= minp:
            output[i] = kernel_func(in_arr[s:e])
        else:
            output[i] = np.nan

    return output


@numba.njit
def _build_indexer(on_arr, N, win, left_closed, right_closed):
    index = cast_dt64_arr_to_int(on_arr)
    start = np.empty(N, np.int64)  # XXX pandas inits to -1 but doesn't seem required?
    end = np.empty(N, np.int64)
    start[0] = 0

    # right endpoint is closed
    if right_closed:
        end[0] = 1
    # right endpoint is open
    else:
        end[0] = 0

    # start is start of slice interval (including)
    # end is end of slice interval (not including)
    for i in range(1, N):
        end_bound = index[i]
        start_bound = index[i] - win

        # left endpoint is closed
        if left_closed:
            start_bound -= 1

        # advance the start bound until we are
        # within the constraint
        start[i] = i
        for j in range(start[i - 1], i):
            if index[j] > start_bound:
                start[i] = j
                break

        # end bound is previous end
        # or current index
        if index[end[i - 1]] <= end_bound:
            end[i] = i + 1
        else:
            end[i] = end[i - 1]

        # right endpoint is open
        if not right_closed:
            end[i] -= 1

    return start, end

# -------------------
# sum

@numba.njit
def init_data_sum():  # pragma: no cover
    return 0, 0.0

@numba.njit
def add_sum(val, nobs, sum_x):  # pragma: no cover
    if not np.isnan(val):
        nobs += 1
        sum_x += val
    return nobs, sum_x

@numba.njit
def remove_sum(val, nobs, sum_x):  # pragma: no cover
    if not np.isnan(val):
        nobs -= 1
        sum_x -= val
    return nobs, sum_x

@numba.njit
def calc_sum(minp, nobs, sum_x):  # pragma: no cover
    return sum_x if nobs >= minp else np.nan


# -------------------------------
# mean

@numba.njit
def init_data_mean():  # pragma: no cover
    return 0, 0.0, 0

@numba.njit
def add_mean(val, nobs, sum_x, neg_ct):  # pragma: no cover
    if not np.isnan(val):
        nobs += 1
        sum_x += val
        if val < 0:
            neg_ct += 1
    return nobs, sum_x, neg_ct

@numba.njit
def remove_mean(val, nobs, sum_x, neg_ct):  # pragma: no cover
    if not np.isnan(val):
        nobs -= 1
        sum_x -= val
        if val < 0:
            neg_ct -= 1
    return nobs, sum_x, neg_ct

@numba.njit
def calc_mean(minp, nobs, sum_x, neg_ct):  # pragma: no cover
    if nobs >= minp:
        result = sum_x / nobs
        if neg_ct == 0 and result < 0.0:
            # all positive
            result = 0
        elif neg_ct == nobs and result > 0.0:
            # all negative
            result = 0
    else:
        result = np.nan
    return result


# -------------------
# var

# TODO: combine add/remove similar to pandas?

@numba.njit
def init_data_var():  # pragma: no cover
    return 0, 0.0, 0.0

@numba.njit
def add_var(val, nobs, mean_x, ssqdm_x):  # pragma: no cover
    if not np.isnan(val):
        nobs += 1
        delta = val - mean_x
        mean_x += delta / nobs
        ssqdm_x += ((nobs - 1) * delta ** 2) / nobs
    return nobs, mean_x, ssqdm_x

@numba.njit
def remove_var(val, nobs, mean_x, ssqdm_x):  # pragma: no cover
    if not np.isnan(val):
        nobs -= 1
        if nobs != 0:
            delta = val - mean_x
            mean_x -= delta / nobs
            ssqdm_x -= ((nobs + 1) * delta ** 2) / nobs
        else:
            mean_x = 0.0
            ssqdm_x = 0.0
    return nobs, mean_x, ssqdm_x

@numba.njit
def calc_var(minp, nobs, mean_x, ssqdm_x):  # pragma: no cover
    ddof = 1.0  # TODO: make argument
    result = np.nan
    if nobs >= minp and nobs > ddof:
        # pathological case
        if nobs == 1:
            result = 0.0
        else:
            result = ssqdm_x / (nobs - ddof)
            if result < 0.0:
                result = 0.0

    return result


# --------------------------
# std

@numba.njit
def calc_std(minp, nobs, mean_x, ssqdm_x):  # pragma: no cover
    v = calc_var(minp, nobs, mean_x, ssqdm_x)
    return np.sqrt(v)


# shift -------------
@numba.njit
def shift(in_arr, shift, parallel):  # pragma: no cover
    N = len(in_arr)
    if parallel:
        rank = hpat.distributed_api.get_rank()
        n_pes = hpat.distributed_api.get_size()
        halo_size = np.int32(shift)
        if _is_small_for_parallel(N, halo_size):
            return _handle_small_data_shift(in_arr, shift, rank, n_pes)

        comm_data = _border_icomm(
            in_arr, rank, n_pes, halo_size, in_arr.dtype, False)
        (l_recv_buff, r_recv_buff, l_send_req, r_send_req, l_recv_req,
            r_recv_req) = comm_data

    output = shift_seq(in_arr, shift)

    if parallel:
        _border_send_wait(r_send_req, l_send_req, rank, n_pes, False)

        # recv left
        if rank != 0:
            hpat.distributed_api.wait(l_recv_req, True)

            for i in range(0, halo_size):
                output[i] = l_recv_buff[i]

    return output

@numba.njit
def shift_seq(in_arr, shift):  # pragma: no cover
    N = len(in_arr)
    output = hpat.hiframes_api.alloc_shift(in_arr)
    shift = min(shift, N)
    output[:shift] = np.nan

    for i in range(shift, N):
        output[i] = in_arr[i-shift]

    return output

# pct_change -------------

@numba.njit
def pct_change(in_arr, shift, parallel):  # pragma: no cover
    N = len(in_arr)
    if parallel:
        rank = hpat.distributed_api.get_rank()
        n_pes = hpat.distributed_api.get_size()
        halo_size = np.int32(shift)
        if _is_small_for_parallel(N, halo_size):
            return _handle_small_data_pct_change(in_arr, shift, rank, n_pes)

        comm_data = _border_icomm(
            in_arr, rank, n_pes, halo_size, in_arr.dtype, False)
        (l_recv_buff, r_recv_buff, l_send_req, r_send_req, l_recv_req,
            r_recv_req) = comm_data

    output = pct_change_seq(in_arr, shift)

    if parallel:
        _border_send_wait(r_send_req, l_send_req, rank, n_pes, False)

        # recv left
        if rank != 0:
            hpat.distributed_api.wait(l_recv_req, True)

            for i in range(0, halo_size):
                prev = l_recv_buff[i]
                output[i] = (in_arr[i] - prev) / prev

    return output

@numba.njit
def pct_change_seq(in_arr, shift):  # pragma: no cover
    N = len(in_arr)
    output = hpat.hiframes_api.alloc_shift(in_arr)
    shift = min(shift, N)
    output[:shift] = np.nan

    for i in range(shift, N):
        prev = in_arr[i-shift]
        output[i] = (in_arr[i] - prev) / prev

    return output

# communication calls -----------

@numba.njit
def _border_icomm(in_arr, rank, n_pes, halo_size, dtype, center):  # pragma: no cover
    comm_tag = np.int32(comm_border_tag)
    l_recv_buff = np.empty(halo_size, dtype)
    if center:
        r_recv_buff = np.empty(halo_size, dtype)
    # send right
    if rank != n_pes - 1:
        r_send_req = hpat.distributed_api.isend(in_arr[-halo_size:], halo_size, np.int32(rank+1), comm_tag, True)
    # recv left
    if rank != 0:
        l_recv_req = hpat.distributed_api.irecv(l_recv_buff, halo_size, np.int32(rank-1), comm_tag, True)
    # center cases
    # send left
    if center and rank != 0:
        l_send_req = hpat.distributed_api.isend(in_arr[:halo_size], halo_size, np.int32(rank-1), comm_tag, True)
    # recv right
    if center and rank != n_pes - 1:
        r_recv_req = hpat.distributed_api.irecv(r_recv_buff, halo_size, np.int32(rank+1), comm_tag, True)

    return l_recv_buff, r_recv_buff, l_send_req, r_send_req, l_recv_req, r_recv_req

@numba.njit
def _border_send_wait(r_send_req, l_send_req, rank, n_pes, center):  # pragma: no cover
    # wait on send right
    if rank != n_pes - 1:
        hpat.distributed_api.wait(r_send_req, True)
    # wait on send left
    if center and rank != 0:
        hpat.distributed_api.wait(l_send_req, True)

@numba.njit
def _is_small_for_parallel(N, halo_size):  # pragma: no cover
    # gather data on one processor and compute sequentially if data of any
    # processor is too small for halo size
    # TODO: handle 1D_Var or other cases where data is actually large but
    # highly imbalanced
    # TODO: avoid reduce for obvious cases like no center and large 1D_Block
    # using 2*halo_size+1 to accomodate center cases with data on more than
    # 2 processor
    num_small = hpat.distributed_api.dist_reduce(
        int(N<=2*halo_size+1), np.int32(Reduce_Type.Sum.value))
    return num_small != 0

# TODO: refactor small data functions
@numba.njit
def _handle_small_data(in_arr, win, center, rank, n_pes, init_data, add_obs,
                                                         remove_obs, calc_out):  # pragma: no cover
    all_N = hpat.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = hpat.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out, _ = roll_fixed_linear_generic_seq(all_in_arr, win, center,
                                      init_data, add_obs, remove_obs, calc_out)
    else:
        all_out = np.empty(all_N, np.float64)
    hpat.distributed_api.bcast(all_out)
    start = hpat.distributed_api.get_start(all_N, n_pes, rank)
    end = hpat.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]

@numba.njit
def _handle_small_data_apply(in_arr, win, center, rank, n_pes, kernel_func):  # pragma: no cover
    all_N = hpat.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = hpat.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out = roll_fixed_apply_seq(all_in_arr, win, center,
                                                   kernel_func)
    else:
        all_out = np.empty(all_N, np.float64)
    hpat.distributed_api.bcast(all_out)
    start = hpat.distributed_api.get_start(all_N, n_pes, rank)
    end = hpat.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]

@numba.njit
def _handle_small_data_shift(in_arr, shift, rank, n_pes):  # pragma: no cover
    all_N = hpat.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = hpat.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out = shift_seq(all_in_arr, shift)
    else:
        all_out = np.empty(all_N, np.float64)
    hpat.distributed_api.bcast(all_out)
    start = hpat.distributed_api.get_start(all_N, n_pes, rank)
    end = hpat.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]

@numba.njit
def _handle_small_data_pct_change(in_arr, shift, rank, n_pes):  # pragma: no cover
    all_N = hpat.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = hpat.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out = pct_change_seq(all_in_arr, shift)
    else:
        all_out = np.empty(all_N, np.float64)
    hpat.distributed_api.bcast(all_out)
    start = hpat.distributed_api.get_start(all_N, n_pes, rank)
    end = hpat.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]

def cast_dt64_arr_to_int(arr):  # pragma: no cover
    return arr

@infer_global(cast_dt64_arr_to_int)
class DtArrToIntType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        assert args[0] == types.Array(types.NPDatetime('ns'), 1, 'C')
        return signature(types.Array(types.int64, 1, 'C'), *args)

@lower_builtin(cast_dt64_arr_to_int, types.Array(types.NPDatetime('ns'), 1, 'C'))
def lower_cast_dt64_arr_to_int(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


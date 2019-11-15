# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


import numpy as np
import pandas as pd
import sdc
import numba
from numba import types
from numba.extending import lower_builtin, overload
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.ir_utils import guard, find_const

from sdc.distributed_api import Reduce_Type
from sdc.hiframes.pd_timestamp_ext import integer_to_dt64
from sdc.utils import unliteral_all


supported_rolling_funcs = ('sum', 'mean', 'var', 'std', 'count', 'median',
                           'min', 'max', 'cov', 'corr', 'apply')


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


def rolling_cov(arr, arr2, win):  # pragma: no cover
    return arr


def rolling_corr(arr, arr2, win):  # pragma: no cover
    return arr


@infer_global(rolling_fixed)
@infer_global(rolling_fixed_parallel)
class RollingType(AbstractTemplate):
    def generic(self, args, kws):
        arr = args[0]  # array or series
        # result is always float64 in pandas
        # see _prep_values() in window.py
        f_type = args[4]
        from sdc.hiframes.pd_series_ext import if_series_to_array_type
        ret_typ = if_series_to_array_type(arr).copy(dtype=types.float64)
        return signature(ret_typ, arr, types.intp,
                         types.bool_, types.bool_, f_type)


@infer_global(rolling_variable)
class RollingVarType(AbstractTemplate):
    def generic(self, args, kws):
        arr = args[0]  # array or series
        on_arr = args[1]
        # result is always float64 in pandas
        # see _prep_values() in window.py
        f_type = args[5]
        from sdc.hiframes.pd_series_ext import if_series_to_array_type
        ret_typ = if_series_to_array_type(arr).copy(dtype=types.float64)
        return signature(ret_typ, arr, on_arr, types.intp,
                         types.bool_, types.bool_, f_type)


@infer_global(rolling_cov)
@infer_global(rolling_corr)
class RollingCovType(AbstractTemplate):
    def generic(self, args, kws):
        arr = args[0]  # array or series
        # hiframes_typed pass replaces series input with array after typing
        from sdc.hiframes.pd_series_ext import if_series_to_array_type
        ret_typ = if_series_to_array_type(arr).copy(dtype=types.float64)
        return signature(ret_typ, *unliteral_all(args))


@lower_builtin(rolling_fixed, types.Array, types.Integer, types.Boolean,
               types.Boolean, types.StringLiteral)
def lower_rolling_fixed(context, builder, sig, args):
    func_name = sig.args[-1].literal_value
    if func_name == 'sum':
        def func(a, w, c, p):
            return roll_fixed_linear_generic(a, w, c, p, init_data_sum, add_sum, remove_sum, calc_sum)
    elif func_name == 'mean':
        def func(a, w, c, p):
            return roll_fixed_linear_generic(a, w, c, p, init_data_mean, add_mean, remove_mean, calc_mean)
    elif func_name == 'var':
        def func(a, w, c, p):
            return roll_fixed_linear_generic(a, w, c, p, init_data_var, add_var, remove_var, calc_var)
    elif func_name == 'std':
        def func(a, w, c, p):
            return roll_fixed_linear_generic(a, w, c, p, init_data_var, add_var, remove_var, calc_std)
    elif func_name == 'count':
        def func(a, w, c, p):
            return roll_fixed_linear_generic(a, w, c, p, init_data_count, add_count, remove_count, calc_count)
    elif func_name in ['median', 'min', 'max']:
        # just using 'apply' since we don't have streaming/linear support
        # TODO: implement linear support similar to others
        func_text = "def kernel_func(A):\n"
        func_text += "  if np.isnan(A).sum() != 0: return np.nan\n"
        func_text += "  return np.{}(A)\n".format(func_name)
        loc_vars = {}
        exec(func_text, {'np': np}, loc_vars)
        kernel_func = numba.njit(loc_vars['kernel_func'])

        def func(a, w, c, p):
            return roll_fixed_apply(a, w, c, p, kernel_func)
    else:
        raise ValueError("invalid rolling (fixed) function {}".format(func_name))

    res = context.compile_internal(
        builder, func, signature(sig.return_type, *sig.args[:-1]), args[:-1])
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@lower_builtin(rolling_fixed, types.Array, types.Integer, types.Boolean,
               types.Boolean, types.functions.Dispatcher)
def lower_rolling_fixed_apply(context, builder, sig, args):

    def func(a, w, c, p, f):
        return roll_fixed_apply(a, w, c, p, f)
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin(rolling_variable, types.Array, types.Array, types.Integer, types.Boolean,
               types.Boolean, types.StringLiteral)
def lower_rolling_variable(context, builder, sig, args):
    func_name = sig.args[-1].literal_value
    if func_name == 'sum':
        def func(a, o, w, c, p):
            return roll_var_linear_generic(a, o, w, c, p, init_data_sum, add_sum, remove_sum, calc_sum)
    elif func_name == 'mean':
        def func(a, o, w, c, p):
            return roll_var_linear_generic(a, o, w, c, p, init_data_mean, add_mean, remove_mean, calc_mean)
    elif func_name == 'var':
        def func(a, o, w, c, p):
            return roll_var_linear_generic(a, o, w, c, p, init_data_var, add_var, remove_var, calc_var)
    elif func_name == 'std':
        def func(a, o, w, c, p):
            return roll_var_linear_generic(a, o, w, c, p, init_data_var, add_var, remove_var, calc_std)
    elif func_name == 'count':
        def func(a, o, w, c, p):
            return roll_var_linear_generic(a, o, w, c, p, init_data_count, add_count, remove_count, calc_count_var)
    elif func_name in ['median', 'min', 'max']:
        # TODO: linear support
        func_text = "def kernel_func(A):\n"
        func_text += "  arr  = dropna(A)\n"
        func_text += "  if len(arr) == 0: return np.nan\n"
        func_text += "  return np.{}(arr)\n".format(func_name)
        loc_vars = {}
        exec(func_text, {'np': np, 'dropna': _dropna}, loc_vars)
        kernel_func = numba.njit(loc_vars['kernel_func'])

        def func(a, o, w, c, p,):
            return roll_variable_apply(a, o, w, c, p, kernel_func)
    else:
        raise ValueError("invalid rolling (variable) function {}".format(func_name))

    res = context.compile_internal(
        builder, func, signature(sig.return_type, *sig.args[:-1]), args[:-1])
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin(
    rolling_variable,
    types.Array,
    types.Array,
    types.Integer,
    types.Boolean,
    types.Boolean,
    types.functions.Dispatcher)
def lower_rolling_variable_apply(context, builder, sig, args):

    def func(a, o, w, c, p, f):
        return roll_variable_apply(a, o, w, c, p, f)
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)

# ** adapted from pandas window.pyx ****


comm_border_tag = 22  # arbitrary, TODO: revisit comm tags


@numba.njit
def roll_fixed_linear_generic(in_arr, win, center, parallel, init_data,
                              add_obs, remove_obs, calc_out):  # pragma: no cover
    rank = sdc.distributed_api.get_rank()
    n_pes = sdc.distributed_api.get_size()
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = win
    offset = (win - 1) // 2 if center else 0

    if parallel:
        # halo length is w/2 to handle even w such as w=4
        halo_size = np.int32(win // 2) if center else np.int32(win - 1)
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
            sdc.distributed_api.wait(r_recv_req, True)

            for i in range(0, halo_size):
                data = add_obs(r_recv_buff[i], *data)

                prev_x = in_arr[N + i - win]
                data = remove_obs(prev_x, *data)

                output[N + i - offset] = calc_out(minp, *data)

        # recv left
        if rank != 0:
            sdc.distributed_api.wait(l_recv_req, True)
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
            output[i - offset] = calc_out(minp, *data)

    for i in range(range_endpoint, N):
        val = in_arr[i]
        data = add_obs(val, *data)

        if i > win - 1:
            prev_x = in_arr[i - win]
            data = remove_obs(prev_x, *data)

        output[i - offset] = calc_out(minp, *data)

    border_data = data  # used for parallel case with center=True

    for i in range(N, N + offset):
        if i > win - 1:
            prev_x = in_arr[i - win]
            data = remove_obs(prev_x, *data)

        output[i - offset] = calc_out(minp, *data)

    return output, border_data


@numba.njit
def roll_fixed_apply(in_arr, win, center, parallel, kernel_func):  # pragma: no cover
    rank = sdc.distributed_api.get_rank()
    n_pes = sdc.distributed_api.get_size()
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = win
    offset = (win - 1) // 2 if center else 0

    if parallel:
        # halo length is w/2 to handle even w such as w=4
        halo_size = np.int32(win // 2) if center else np.int32(win - 1)
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
            sdc.distributed_api.wait(r_recv_req, True)
            border_data = np.concatenate((in_arr[N - win + 1:], r_recv_buff))
            ind = 0
            for i in range(max(N - offset, 0), N):
                output[i] = kernel_func(border_data[ind:ind + win])
                ind += 1

        # recv left
        if rank != 0:
            sdc.distributed_api.wait(l_recv_req, True)
            border_data = np.concatenate((l_recv_buff, in_arr[:win - 1]))
            for i in range(0, win - offset - 1):
                output[i] = kernel_func(border_data[i:i + win])

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
        output[i] = kernel_func(in_arr[ind:ind + win])
        ind += 1
    for i in range(max(N - offset, 0), N):
        output[i] = np.nan

    return output


# -----------------------------
# variable window

@numba.njit
def roll_var_linear_generic(in_arr, on_arr_dt, win, center, parallel, init_data,
                            add_obs, remove_obs, calc_out):  # pragma: no cover
    rank = sdc.distributed_api.get_rank()
    n_pes = sdc.distributed_api.get_size()
    on_arr = cast_dt64_arr_to_int(on_arr_dt)
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = 1
    # Pandas is right closed by default, TODO: extend to support arg
    left_closed = False
    right_closed = True

    if parallel:
        if _is_small_for_parallel_variable(on_arr, win):
            return _handle_small_data_variable(in_arr, on_arr, win, rank,
                                               n_pes, init_data, add_obs, remove_obs, calc_out)

        comm_data = _border_icomm_var(
            in_arr, on_arr, rank, n_pes, win, in_arr.dtype)
        (l_recv_buff, l_recv_t_buff, r_send_req, r_send_t_req, l_recv_req,
         l_recv_t_req) = comm_data

    start, end = _build_indexer(on_arr, N, win, left_closed, right_closed)
    output = roll_var_linear_generic_seq(in_arr, on_arr, win, start, end,
                                         init_data, add_obs, remove_obs, calc_out)

    if parallel:
        _border_send_wait(r_send_req, r_send_req, rank, n_pes, False)
        _border_send_wait(r_send_t_req, r_send_t_req, rank, n_pes, False)

        # recv left
        if rank != 0:
            sdc.distributed_api.wait(l_recv_req, True)
            sdc.distributed_api.wait(l_recv_t_req, True)

            # values with start == 0 could potentially have left halo starts
            num_zero_starts = 0
            for i in range(0, N):
                if start[i] != 0:
                    break
                num_zero_starts += 1

            if num_zero_starts == 0:
                return output

            recv_starts = _get_var_recv_starts(on_arr, l_recv_t_buff, num_zero_starts, win)
            data = init_data()
            # setup (first element)
            for j in range(recv_starts[0], len(l_recv_t_buff)):
                data = add_obs(l_recv_buff[j], *data)
            if right_closed:
                data = add_obs(in_arr[0], *data)
            output[0] = calc_out(minp, *data)

            for i in range(1, num_zero_starts):
                s = recv_starts[i]
                e = end[i]

                # calculate deletes (can only happen in left recv buffer)
                for j in range(recv_starts[i - 1], s):
                    data = remove_obs(l_recv_buff[j], *data)

                # calculate adds (can only happen in local data)
                for j in range(end[i - 1], e):
                    data = add_obs(in_arr[j], *data)

                output[i] = calc_out(minp, *data)

    return output


@numba.njit
def _get_var_recv_starts(on_arr, l_recv_t_buff, num_zero_starts, win):
    recv_starts = np.zeros(num_zero_starts, np.int64)
    halo_size = len(l_recv_t_buff)
    index = cast_dt64_arr_to_int(on_arr)
    left_closed = False

    # handle first element
    start_bound = index[0] - win
    # left endpoint is closed
    if left_closed:
        start_bound -= 1
    recv_starts[0] = halo_size
    for j in range(0, halo_size):
        if l_recv_t_buff[j] > start_bound:
            recv_starts[0] = j
            break

    # rest of elements
    for i in range(1, num_zero_starts):
        start_bound = index[i] - win
        # left endpoint is closed
        if left_closed:
            start_bound -= 1
        recv_starts[i] = halo_size
        for j in range(recv_starts[i - 1], halo_size):
            if l_recv_t_buff[j] > start_bound:
                recv_starts[i] = j
                break

    return recv_starts


@numba.njit
def roll_var_linear_generic_seq(in_arr, on_arr, win, start, end, init_data,
                                add_obs, remove_obs, calc_out):  # pragma: no cover
    #
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = 1
    output = np.empty(N, np.float64)

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
def roll_variable_apply(in_arr, on_arr_dt, win, center, parallel, kernel_func):  # pragma: no cover
    rank = sdc.distributed_api.get_rank()
    n_pes = sdc.distributed_api.get_size()
    on_arr = cast_dt64_arr_to_int(on_arr_dt)
    N = len(in_arr)
    # TODO: support minp arg end_range etc.
    minp = 1
    # Pandas is right closed by default, TODO: extend to support arg
    left_closed = False
    right_closed = True

    if parallel:
        if _is_small_for_parallel_variable(on_arr, win):
            return _handle_small_data_variable_apply(in_arr, on_arr, win, rank,
                                                     n_pes, kernel_func)

        comm_data = _border_icomm_var(
            in_arr, on_arr, rank, n_pes, win, in_arr.dtype)
        (l_recv_buff, l_recv_t_buff, r_send_req, r_send_t_req, l_recv_req,
         l_recv_t_req) = comm_data

    start, end = _build_indexer(on_arr, N, win, left_closed, right_closed)
    output = roll_variable_apply_seq(in_arr, on_arr, win, start, end, kernel_func)

    if parallel:
        _border_send_wait(r_send_req, r_send_req, rank, n_pes, False)
        _border_send_wait(r_send_t_req, r_send_t_req, rank, n_pes, False)

        # recv left
        if rank != 0:
            sdc.distributed_api.wait(l_recv_req, True)
            sdc.distributed_api.wait(l_recv_t_req, True)

            # values with start == 0 could potentially have left halo starts
            num_zero_starts = 0
            for i in range(0, N):
                if start[i] != 0:
                    break
                num_zero_starts += 1

            if num_zero_starts == 0:
                return output

            recv_starts = _get_var_recv_starts(on_arr, l_recv_t_buff, num_zero_starts, win)
            for i in range(0, num_zero_starts):
                halo_ind = recv_starts[i]
                sub_arr = np.concatenate(
                    (l_recv_buff[halo_ind:], in_arr[:i + 1]))
                if len(sub_arr) >= minp:
                    output[i] = kernel_func(sub_arr)
                else:
                    output[i] = np.nan

    return output


@numba.njit
def roll_variable_apply_seq(in_arr, on_arr, win, start, end, kernel_func):  # pragma: no cover
    # TODO
    N = len(in_arr)
    minp = 1
    output = np.empty(N, dtype=np.float64)

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

# -------------------
# count


@numba.njit
def init_data_count():  # pragma: no cover
    return (0.0,)


@numba.njit
def add_count(val, count_x):  # pragma: no cover
    if not np.isnan(val):
        count_x += 1.0
    return (count_x,)


@numba.njit
def remove_count(val, count_x):  # pragma: no cover
    if not np.isnan(val):
        count_x -= 1.0
    return (count_x,)

# XXX: pandas uses minp=0 for fixed window count but minp=1 for variable window


@numba.njit
def calc_count(minp, count_x):  # pragma: no cover
    return count_x


@numba.njit
def calc_count_var(minp, count_x):  # pragma: no cover
    return count_x if count_x >= minp else np.nan


# shift -------------

# dummy
def shift():  # pragma: no cover
    return

# using overload since njit bakes in Literal[bool](False) for parallel
@overload(shift)
def shift_overload(in_arr, shift, parallel):
    if not isinstance(parallel, types.Literal):
        return shift_impl


def shift_impl(in_arr, shift, parallel):  # pragma: no cover
    N = len(in_arr)
    if parallel:
        rank = sdc.distributed_api.get_rank()
        n_pes = sdc.distributed_api.get_size()
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
            sdc.distributed_api.wait(l_recv_req, True)

            for i in range(0, halo_size):
                output[i] = l_recv_buff[i]

    return output


@numba.njit
def shift_seq(in_arr, shift):  # pragma: no cover
    N = len(in_arr)
    output = sdc.hiframes.api.alloc_shift(in_arr)
    shift = min(shift, N)
    output[:shift] = np.nan

    for i in range(shift, N):
        output[i] = in_arr[i - shift]

    return output

# pct_change -------------

# dummy


def pct_change():  # pragma: no cover
    return

# using overload since njit bakes in Literal[bool](False) for parallel
@overload(pct_change)
def pct_change_overload(in_arr, shift, parallel):
    if not isinstance(parallel, types.Literal):
        return pct_change_impl


def pct_change_impl(in_arr, shift, parallel):  # pragma: no cover
    N = len(in_arr)
    if parallel:
        rank = sdc.distributed_api.get_rank()
        n_pes = sdc.distributed_api.get_size()
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
            sdc.distributed_api.wait(l_recv_req, True)

            for i in range(0, halo_size):
                prev = l_recv_buff[i]
                output[i] = (in_arr[i] - prev) / prev

    return output


@numba.njit
def pct_change_seq(in_arr, shift):  # pragma: no cover
    N = len(in_arr)
    output = sdc.hiframes.api.alloc_shift(in_arr)
    shift = min(shift, N)
    output[:shift] = np.nan

    for i in range(shift, N):
        prev = in_arr[i - shift]
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
        r_send_req = sdc.distributed_api.isend(in_arr[-halo_size:], halo_size, np.int32(rank + 1), comm_tag, True)
    # recv left
    if rank != 0:
        l_recv_req = sdc.distributed_api.irecv(l_recv_buff, halo_size, np.int32(rank - 1), comm_tag, True)
    # center cases
    # send left
    if center and rank != 0:
        l_send_req = sdc.distributed_api.isend(in_arr[:halo_size], halo_size, np.int32(rank - 1), comm_tag, True)
    # recv right
    if center and rank != n_pes - 1:
        r_recv_req = sdc.distributed_api.irecv(r_recv_buff, halo_size, np.int32(rank + 1), comm_tag, True)

    return l_recv_buff, r_recv_buff, l_send_req, r_send_req, l_recv_req, r_recv_req


@numba.njit
def _border_icomm_var(in_arr, on_arr, rank, n_pes, win_size, dtype):  # pragma: no cover
    comm_tag = np.int32(comm_border_tag)
    # find halo size from time array
    N = len(on_arr)
    halo_size = N
    end = on_arr[-1]
    for j in range(-2, -N, -1):
        t = on_arr[j]
        if end - t >= win_size:
            halo_size = -j
            break

    # send right
    if rank != n_pes - 1:
        sdc.distributed_api.send(halo_size, np.int32(rank + 1), comm_tag)
        r_send_req = sdc.distributed_api.isend(
            in_arr[-halo_size:], np.int32(halo_size), np.int32(rank + 1), comm_tag, True)
        r_send_t_req = sdc.distributed_api.isend(
            on_arr[-halo_size:], np.int32(halo_size), np.int32(rank + 1), comm_tag, True)
    # recv left
    if rank != 0:
        halo_size = sdc.distributed_api.recv(np.int64, np.int32(rank - 1), comm_tag)
        l_recv_buff = np.empty(halo_size, dtype)
        l_recv_req = sdc.distributed_api.irecv(l_recv_buff, np.int32(halo_size), np.int32(rank - 1), comm_tag, True)
        l_recv_t_buff = np.empty(halo_size, np.int64)
        l_recv_t_req = sdc.distributed_api.irecv(
            l_recv_t_buff, np.int32(halo_size), np.int32(
                rank - 1), comm_tag, True)

    return l_recv_buff, l_recv_t_buff, r_send_req, r_send_t_req, l_recv_req, l_recv_t_req


@numba.njit
def _border_send_wait(r_send_req, l_send_req, rank, n_pes, center):  # pragma: no cover
    # wait on send right
    if rank != n_pes - 1:
        sdc.distributed_api.wait(r_send_req, True)
    # wait on send left
    if center and rank != 0:
        sdc.distributed_api.wait(l_send_req, True)


@numba.njit
def _is_small_for_parallel(N, halo_size):  # pragma: no cover
    # gather data on one processor and compute sequentially if data of any
    # processor is too small for halo size
    # TODO: handle 1D_Var or other cases where data is actually large but
    # highly imbalanced
    # TODO: avoid reduce for obvious cases like no center and large 1D_Block
    # using 2*halo_size+1 to accomodate center cases with data on more than
    # 2 processor
    num_small = sdc.distributed_api.dist_reduce(
        int(N <= 2 * halo_size + 1), np.int32(Reduce_Type.Sum.value))
    return num_small != 0

# TODO: refactor small data functions
@numba.njit
def _handle_small_data(in_arr, win, center, rank, n_pes, init_data, add_obs,
                       remove_obs, calc_out):  # pragma: no cover
    all_N = sdc.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = sdc.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out, _ = roll_fixed_linear_generic_seq(all_in_arr, win, center,
                                                   init_data, add_obs, remove_obs, calc_out)
    else:
        all_out = np.empty(all_N, np.float64)
    sdc.distributed_api.bcast(all_out)
    start = sdc.distributed_api.get_start(all_N, n_pes, rank)
    end = sdc.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@numba.njit
def _handle_small_data_apply(in_arr, win, center, rank, n_pes, kernel_func):  # pragma: no cover
    all_N = sdc.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = sdc.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out = roll_fixed_apply_seq(all_in_arr, win, center,
                                       kernel_func)
    else:
        all_out = np.empty(all_N, np.float64)
    sdc.distributed_api.bcast(all_out)
    start = sdc.distributed_api.get_start(all_N, n_pes, rank)
    end = sdc.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@numba.njit
def _handle_small_data_shift(in_arr, shift, rank, n_pes):  # pragma: no cover
    all_N = sdc.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = sdc.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out = shift_seq(all_in_arr, shift)
    else:
        all_out = np.empty(all_N, np.float64)
    sdc.distributed_api.bcast(all_out)
    start = sdc.distributed_api.get_start(all_N, n_pes, rank)
    end = sdc.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@numba.njit
def _handle_small_data_pct_change(in_arr, shift, rank, n_pes):  # pragma: no cover
    all_N = sdc.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = sdc.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out = pct_change_seq(all_in_arr, shift)
    else:
        all_out = np.empty(all_N, np.float64)
    sdc.distributed_api.bcast(all_out)
    start = sdc.distributed_api.get_start(all_N, n_pes, rank)
    end = sdc.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


def cast_dt64_arr_to_int(arr):  # pragma: no cover
    return arr


@infer_global(cast_dt64_arr_to_int)
class DtArrToIntType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        assert (args[0] == types.Array(types.NPDatetime('ns'), 1, 'C')
                or args[0] == types.Array(types.int64, 1, 'C'))
        return signature(types.Array(types.int64, 1, 'C'), *args)


@lower_builtin(cast_dt64_arr_to_int, types.Array(types.NPDatetime('ns'), 1, 'C'))
@lower_builtin(cast_dt64_arr_to_int, types.Array(types.int64, 1, 'C'))
def lower_cast_dt64_arr_to_int(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# ----------------------------------------
# variable window comm routines


@numba.njit
def _is_small_for_parallel_variable(on_arr, win_size):  # pragma: no cover
    # assume small if current processor's whole range is smaller than win_size
    if len(on_arr) < 2:
        return True
    start = on_arr[0]
    end = on_arr[-1]
    pe_range = end - start
    num_small = sdc.distributed_api.dist_reduce(
        int(pe_range <= win_size), np.int32(Reduce_Type.Sum.value))
    return num_small != 0


@numba.njit
def _handle_small_data_variable(in_arr, on_arr, win, rank, n_pes, init_data,
                                add_obs, remove_obs, calc_out):  # pragma: no cover
    all_N = sdc.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = sdc.distributed_api.gatherv(in_arr)
    all_on_arr = sdc.distributed_api.gatherv(on_arr)
    if rank == 0:
        start, end = _build_indexer(all_on_arr, all_N, win, False, True)
        all_out = roll_var_linear_generic_seq(all_in_arr, all_on_arr, win,
                                              start, end, init_data, add_obs, remove_obs, calc_out)
    else:
        all_out = np.empty(all_N, np.float64)
    sdc.distributed_api.bcast(all_out)
    start = sdc.distributed_api.get_start(all_N, n_pes, rank)
    end = sdc.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@numba.njit
def _handle_small_data_variable_apply(in_arr, on_arr, win, rank, n_pes,
                                      kernel_func):  # pragma: no cover
    all_N = sdc.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value))
    all_in_arr = sdc.distributed_api.gatherv(in_arr)
    all_on_arr = sdc.distributed_api.gatherv(on_arr)
    if rank == 0:
        start, end = _build_indexer(all_on_arr, all_N, win, False, True)
        all_out = roll_variable_apply_seq(all_in_arr, all_on_arr, win,
                                          start, end, kernel_func)
    else:
        all_out = np.empty(all_N, np.float64)
    sdc.distributed_api.bcast(all_out)
    start = sdc.distributed_api.get_start(all_N, n_pes, rank)
    end = sdc.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@numba.njit
def _dropna(arr):  # pragma: no cover
    old_len = len(arr)
    new_len = old_len - np.isnan(arr).sum()
    A = np.empty(new_len, arr.dtype)
    curr_ind = 0
    for i in range(old_len):
        val = arr[i]
        if not np.isnan(val):
            A[curr_ind] = val
            curr_ind += 1

    return A

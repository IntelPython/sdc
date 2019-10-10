from collections import defaultdict
import numpy as np
import re

import numba
from numba import types
from numba.extending import overload
from numba.typing.templates import infer_global, AbstractTemplate, signature

import hpat
from hpat.str_ext import string_type, unicode_to_std_str, std_str_to_unicode
from hpat.str_arr_ext import (string_array_type, StringArrayType,
                              is_str_arr_typ, pre_alloc_string_array, get_utf8_size)


# float columns can have regular np.nan
def _column_filter_impl(B, ind):  # pragma: no cover
    dtype = hpat.hiframes.api.shift_dtype(B.dtype)
    A = np.empty(len(B), dtype)
    for i in numba.parfor.internal_prange(len(A)):
        if ind[i]:
            A[i] = B[i]
        else:
            hpat.hiframes.join.setitem_arr_nan(A, i)
    return hpat.hiframes.api.init_series(A)


def _column_count_impl(A):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    for i in numba.parfor.internal_prange(len(A)):
        if not hpat.hiframes.api.isna(A, i):
            count += 1

    res = count
    return res


def _column_fillna_impl(A, B, fill):  # pragma: no cover
    for i in numba.parfor.internal_prange(len(A)):
        s = B[i]
        if hpat.hiframes.api.isna(B, i):
            s = fill
        A[i] = s


def _series_fillna_str_alloc_impl(B, fill, name):  # pragma: no cover
    n = len(B)
    num_chars = 0
    # get total chars in new array
    for i in numba.parfor.internal_prange(n):
        s = B[i]
        if hpat.hiframes.api.isna(B, i):
            num_chars += len(fill)
        else:
            num_chars += len(s)
    A = hpat.str_arr_ext.pre_alloc_string_array(n, num_chars)
    hpat.hiframes.api.fillna(A, B, fill)
    return hpat.hiframes.api.init_series(A, None, name)


def _series_dropna_float_impl(S, name):  # pragma: no cover
    old_len = len(S)
    new_len = old_len - hpat.hiframes.api.init_series(S).isna().sum()
    A = np.empty(new_len, S.dtype)
    curr_ind = 0
    for i in numba.parfor.internal_prange(old_len):
        val = S[i]
        if not np.isnan(val):
            A[curr_ind] = val
            curr_ind += 1

    return hpat.hiframes.api.init_series(A, None, name)


# using njit since 1D_var is broken for alloc when there is calculation of len
@numba.njit(no_cpython_wrapper=True)
def _series_dropna_str_alloc_impl_inner(B):  # pragma: no cover
    # TODO: test
    # TODO: generalize
    old_len = len(B)
    na_count = 0
    for i in range(len(B)):
        if hpat.str_arr_ext.str_arr_is_na(B, i):
            na_count += 1
    # TODO: more efficient null counting
    new_len = old_len - na_count
    num_chars = hpat.str_arr_ext.num_total_chars(B)
    A = hpat.str_arr_ext.pre_alloc_string_array(new_len, num_chars)
    hpat.str_arr_ext.copy_non_null_offsets(A, B)
    hpat.str_arr_ext.copy_data(A, B)
    return A


def _series_dropna_str_alloc_impl(B, name):  # pragma: no cover
    A = hpat.hiframes.series_kernels._series_dropna_str_alloc_impl_inner(B)
    return hpat.hiframes.api.init_series(A, None, name)


# return the nan value for the type (handle dt64)
def _get_nan(val):
    return np.nan


@overload(_get_nan)
def _get_nan_overload(val):
    if isinstance(val, (types.NPDatetime, types.NPTimedelta)):
        nat = val('NaT')
        return lambda val: nat
    # TODO: other types
    return lambda val: np.nan


def _get_type_max_value(dtype):
    return 0


@overload(_get_type_max_value)
def _get_type_max_value_overload(dtype):
    if isinstance(dtype.dtype, (types.NPDatetime, types.NPTimedelta)):
        return lambda dtype: hpat.hiframes.pd_timestamp_ext.integer_to_dt64(
            numba.targets.builtins.get_type_max_value(numba.types.int64))
    return lambda dtype: numba.targets.builtins.get_type_max_value(dtype)


@numba.njit
def _sum_handle_nan(s, count):  # pragma: no cover
    if not count:
        s = hpat.hiframes.series_kernels._get_nan(s)
    return s


def _column_sum_impl_basic(A):  # pragma: no cover
    numba.parfor.init_prange()
    # TODO: fix output type
    s = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += val

    res = s
    return res


def _column_sum_impl_count(A):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += val
            count += 1

    res = hpat.hiframes.series_kernels._sum_handle_nan(s, count)
    return res


def _column_prod_impl_basic(A):  # pragma: no cover
    numba.parfor.init_prange()
    # TODO: fix output type
    s = 1
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s *= val

    res = s
    return res


@numba.njit
def _mean_handle_nan(s, count):  # pragma: no cover
    if not count:
        s = np.nan
    else:
        s = s / count
    return s


def _column_mean_impl(A):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += val
            count += 1

    res = hpat.hiframes.series_kernels._mean_handle_nan(s, count)
    return res


@numba.njit
def _var_handle_nan(s, count):  # pragma: no cover
    if count <= 1:
        s = np.nan
    else:
        s = s / (count - 1)
    return s


def _column_var_impl(A):  # pragma: no cover
    numba.parfor.init_prange()
    count_m = 0
    m = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            m += val
            count_m += 1

    numba.parfor.init_prange()
    m = hpat.hiframes.series_kernels._mean_handle_nan(m, count_m)
    s = 0
    count = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += (val - m)**2
            count += 1

    res = hpat.hiframes.series_kernels._var_handle_nan(s, count)
    return res


def _column_std_impl(A):  # pragma: no cover
    var = hpat.hiframes.api.var(A)
    return var**0.5


def _column_min_impl(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = hpat.hiframes.series_kernels._get_type_max_value(in_arr.dtype)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = in_arr[i]
        if not hpat.hiframes.api.isna(in_arr, i):
            s = min(s, val)
            count += 1
    res = hpat.hiframes.series_kernels._sum_handle_nan(s, count)
    return res


def _column_min_impl_no_isnan(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    s = numba.targets.builtins.get_type_max_value(numba.types.int64)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = hpat.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
        s = min(s, val)
    return hpat.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)

# TODO: fix for dt64


def _column_max_impl(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    count = 0
    s = numba.targets.builtins.get_type_min_value(in_arr.dtype)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = in_arr[i]
        if not np.isnan(val):
            s = max(s, val)
            count += 1
    res = hpat.hiframes.series_kernels._sum_handle_nan(s, count)
    return res


def _column_max_impl_no_isnan(in_arr):  # pragma: no cover
    numba.parfor.init_prange()
    s = numba.targets.builtins.get_type_min_value(numba.types.int64)
    for i in numba.parfor.internal_prange(len(in_arr)):
        val = in_arr[i]
        s = max(s, hpat.hiframes.pd_timestamp_ext.dt64_to_integer(val))
    return hpat.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)


def _column_sub_impl_datetime_series_timestamp(in_arr, ts):  # pragma: no cover
    numba.parfor.init_prange()
    n = len(in_arr)
    S = numba.unsafe.ndarray.empty_inferred((n,))
    tsint = hpat.hiframes.pd_timestamp_ext.convert_timestamp_to_datetime64(ts)
    for i in numba.parfor.internal_prange(n):
        S[i] = hpat.hiframes.pd_timestamp_ext.integer_to_timedelta64(
            hpat.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i]) - tsint)
    return hpat.hiframes.api.init_series(S)


def _column_sub_impl_datetimeindex_timestamp(in_arr, ts):  # pragma: no cover
    numba.parfor.init_prange()
    n = len(in_arr)
    S = numba.unsafe.ndarray.empty_inferred((n,))
    tsint = hpat.hiframes.pd_timestamp_ext.convert_timestamp_to_datetime64(ts)
    for i in numba.parfor.internal_prange(n):
        S[i] = hpat.hiframes.pd_timestamp_ext.integer_to_timedelta64(
            hpat.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i]) - tsint)
    return hpat.hiframes.api.init_timedelta_index(S)


def _column_describe_impl(S):  # pragma: no cover
    a_count = np.float64(S.count())
    a_min = S.min()
    a_max = S.max()
    a_mean = S.mean()
    a_std = S.std()
    q25 = S.quantile(.25)
    q50 = S.quantile(.5)
    q75 = S.quantile(.75)
    # TODO: pandas returns dataframe, maybe return namedtuple instread of
    # string?
    # TODO: fix string formatting to match python/pandas
    res = "count    " + str(a_count) + "\n"\
        "mean     " + str(a_mean) + "\n"\
        "std      " + str(a_std) + "\n"\
        "min      " + str(a_min) + "\n"\
        "25%      " + str(q25) + "\n"\
        "50%      " + str(q50) + "\n"\
        "75%      " + str(q75) + "\n"\
        "max      " + str(a_max) + "\n"
    return res


def _column_fillna_alloc_impl(S, val, name):  # pragma: no cover
    # TODO: handle string, etc.
    B = np.empty(len(S), S.dtype)
    hpat.hiframes.api.fillna(B, S, val)
    return hpat.hiframes.api.init_series(B, None, name)


def _str_contains_regex_impl(str_arr, pat):  # pragma: no cover
    e = hpat.str_ext.compile_regex(pat)
    return hpat.hiframes.api.str_contains_regex(str_arr, e)


def _str_contains_noregex_impl(str_arr, pat):  # pragma: no cover
    return hpat.hiframes.api.str_contains_noregex(str_arr, pat)


# TODO: use online algorithm, e.g. StatFunctions.scala
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def _column_cov_impl(S1, S2):  # pragma: no cover
    # TODO: check lens
    ma = S1.mean()
    mb = S2.mean()
    # TODO: check aligned nans, (S1.notna() != S2.notna()).any()
    return ((S1 - ma) * (S2 - mb)).sum() / (S1.count() - 1.0)


def _column_corr_impl(S1, S2):  # pragma: no cover
    n = S1.count()
    # TODO: check lens
    ma = S1.sum()
    mb = S2.sum()
    # TODO: check aligned nans, (S1.notna() != S2.notna()).any()
    a = n * ((S1 * S2).sum()) - ma * mb
    b1 = n * (S1**2).sum() - ma**2
    b2 = n * (S2**2).sum() - mb**2
    # TODO: np.clip
    # TODO: np.true_divide?
    return a / np.sqrt(b1 * b2)


def _series_append_single_impl(arr, other):
    return hpat.hiframes.api.init_series(
        hpat.hiframes.api.concat((arr, other)))


def _series_append_tuple_impl(arr, other):
    tup_other = hpat.hiframes.api.to_const_tuple(other)
    tup_other = hpat.hiframes.api.series_tup_to_arr_tup(tup_other)
    arrs = (arr,) + tup_other
    c_arrs = hpat.hiframes.api.to_const_tuple(arrs)
    return hpat.hiframes.api.init_series(
        hpat.hiframes.api.concat(c_arrs))


def _series_isna_impl(arr):
    numba.parfor.init_prange()
    n = len(arr)
    out_arr = np.empty(n, np.bool_)
    for i in numba.parfor.internal_prange(n):
        out_arr[i] = hpat.hiframes.api.isna(arr, i)
    return hpat.hiframes.api.init_series(out_arr)


def _series_astype_str_impl(arr):
    n = len(arr)
    num_chars = 0
    # get total chars in new array
    for i in numba.parfor.internal_prange(n):
        s = arr[i]
        num_chars += len(str(s))  # TODO: check NA

    A = hpat.str_arr_ext.pre_alloc_string_array(n, num_chars)
    for i in numba.parfor.internal_prange(n):
        s = arr[i]
        A[i] = str(s)  # TODO: check NA
    return hpat.hiframes.api.init_series(A)


# def _str_replace_regex_impl(str_arr, pat, val):
#     numba.parfor.init_prange()
#     e = hpat.str_ext.compile_regex(unicode_to_std_str(pat))
#     val = unicode_to_std_str(val)
#     n = len(str_arr)
#     n_total_chars = 0
#     str_list = hpat.str_ext.alloc_str_list(n)
#     for i in numba.parfor.internal_prange(n):
#         # TODO: support unicode
#         in_str = unicode_to_std_str(str_arr[i])
#         out_str = std_str_to_unicode(
#             hpat.str_ext.str_replace_regex(in_str, e, val))
#         str_list[i] = out_str
#         n_total_chars += len(out_str)
#     numba.parfor.init_prange()
#     out_arr = pre_alloc_string_array(n, n_total_chars)
#     for i in numba.parfor.internal_prange(n):
#         _str = str_list[i]
#         out_arr[i] = _str
#     return hpat.hiframes.api.init_series(out_arr)


def _str_replace_regex_impl(str_arr, pat, val):
    numba.parfor.init_prange()
    e = re.compile(pat)
    n = len(str_arr)
    n_total_chars = 0
    str_list = hpat.str_ext.alloc_str_list(n)
    for i in numba.parfor.internal_prange(n):
        out_str = e.sub(val, str_arr[i])
        str_list[i] = out_str
        n_total_chars += get_utf8_size(out_str)
    numba.parfor.init_prange()
    out_arr = pre_alloc_string_array(n, n_total_chars)
    for i in numba.parfor.internal_prange(n):
        _str = str_list[i]
        out_arr[i] = _str
    return hpat.hiframes.api.init_series(out_arr)


# TODO: refactor regex and noregex
# implementation using std::string
# def _str_replace_noregex_impl(str_arr, pat, val):
#     numba.parfor.init_prange()
#     e = unicode_to_std_str(pat)
#     val = unicode_to_std_str(val)
#     n = len(str_arr)
#     n_total_chars = 0
#     str_list = hpat.str_ext.alloc_str_list(n)
#     for i in numba.parfor.internal_prange(n):
#         # TODO: support unicode
#         in_str = unicode_to_std_str(str_arr[i])
#         out_str = std_str_to_unicode(
#             hpat.str_ext.str_replace_noregex(in_str, e, val))
#         str_list[i] = out_str
#         n_total_chars += len(out_str)
#     numba.parfor.init_prange()
#     out_arr = pre_alloc_string_array(n, n_total_chars)
#     for i in numba.parfor.internal_prange(n):
#         _str = str_list[i]
#         out_arr[i] = _str
#     return hpat.hiframes.api.init_series(out_arr)


def _str_replace_noregex_impl(str_arr, pat, val):
    numba.parfor.init_prange()
    n = len(str_arr)
    n_total_chars = 0
    str_list = hpat.str_ext.alloc_str_list(n)
    for i in numba.parfor.internal_prange(n):
        out_str = str_arr[i].replace(pat, val)
        str_list[i] = out_str
        n_total_chars += get_utf8_size(out_str)
    numba.parfor.init_prange()
    out_arr = pre_alloc_string_array(n, n_total_chars)
    for i in numba.parfor.internal_prange(n):
        _str = str_list[i]
        out_arr[i] = _str
    return hpat.hiframes.api.init_series(out_arr)


@numba.njit
def lt_f(a, b):
    return a < b


@numba.njit
def gt_f(a, b):
    return a > b


series_replace_funcs = {
    'sum': _column_sum_impl_basic,
    'prod': _column_prod_impl_basic,
    'count': _column_count_impl,
    'mean': _column_mean_impl,
    'max': defaultdict(lambda: _column_max_impl, [(types.NPDatetime('ns'), _column_max_impl_no_isnan)]),
    # 'min': defaultdict(lambda: _column_min_impl, [(types.NPDatetime('ns'), _column_min_impl_no_isnan)]),
    'var': _column_var_impl,
    'std': _column_std_impl,
    'nunique': lambda A: hpat.hiframes.api.nunique(A),
    'unique': lambda A: hpat.hiframes.api.unique(A),
    'describe': _column_describe_impl,
    'fillna_alloc': _column_fillna_alloc_impl,
    'fillna_str_alloc': _series_fillna_str_alloc_impl,
    'dropna_float': _series_dropna_float_impl,
    'dropna_str_alloc': _series_dropna_str_alloc_impl,
    'shift': lambda A, shift: hpat.hiframes.api.init_series(hpat.hiframes.rolling.shift(A, shift, False)),
    'shift_default': lambda A: hpat.hiframes.api.init_series(hpat.hiframes.rolling.shift(A, 1, False)),
    'pct_change': lambda A, shift: hpat.hiframes.api.init_series(hpat.hiframes.rolling.pct_change(A, shift, False)),
    'pct_change_default': lambda A: hpat.hiframes.api.init_series(hpat.hiframes.rolling.pct_change(A, 1, False)),
    'str_contains_regex': _str_contains_regex_impl,
    'str_contains_noregex': _str_contains_noregex_impl,
    # 'abs': lambda A: hpat.hiframes.api.init_series(np.abs(A)),  # TODO: timedelta
    'cov': _column_cov_impl,
    'corr': _column_corr_impl,
    'append_single': _series_append_single_impl,
    'append_tuple': _series_append_tuple_impl,
    'isna': _series_isna_impl,
    # isnull is just alias of isna
    'isnull': _series_isna_impl,
    'astype_str': _series_astype_str_impl,
    'nlargest': lambda A, k, name: hpat.hiframes.api.init_series(hpat.hiframes.api.nlargest(A, k, True, gt_f), None, name),
    'nlargest_default': lambda A, name: hpat.hiframes.api.init_series(hpat.hiframes.api.nlargest(A, 5, True, gt_f), None, name),
    'nsmallest': lambda A, k, name: hpat.hiframes.api.init_series(hpat.hiframes.api.nlargest(A, k, False, lt_f), None, name),
    'nsmallest_default': lambda A, name: hpat.hiframes.api.init_series(hpat.hiframes.api.nlargest(A, 5, False, lt_f), None, name),
    'head': lambda A, I, k, name: hpat.hiframes.api.init_series(A[:k], None, name),
    'head_index': lambda A, I, k, name: hpat.hiframes.api.init_series(A[:k], I[:k], name),
    'median': lambda A: hpat.hiframes.api.median(A),
    # TODO: handle NAs in argmin/argmax
    'idxmin': lambda A: A.argmin(),
    'idxmax': lambda A: A.argmax(),
}

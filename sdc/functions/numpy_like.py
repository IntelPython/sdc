# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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
from sdc.hiframes.api import isna

"""

| This file contains SDC numpy modified functions, rewritten according to concurrency

"""

import numba
import numpy
import sys
import pandas
import numpy as np

from numba import types, jit, prange, numpy_support, literally
from numba.errors import TypingError
from numba.targets.arraymath import get_isnan

import sdc
from sdc.utilities.sdc_typing_utils import TypeChecker
from sdc.utilities.utils import (sdc_overload, sdc_register_jitable,
                                 min_dtype_int_val, max_dtype_int_val, min_dtype_float_val,
                                 max_dtype_float_val)
from sdc.str_arr_ext import (StringArrayType, pre_alloc_string_array, get_utf8_size,
                             string_array_type, create_str_arr_from_list, str_arr_set_na_by_mask)
from sdc.utilities.utils import sdc_overload, sdc_register_jitable
from sdc.utilities.prange_utils import parallel_chunks


def astype(self, dtype):
    pass


def astype_no_inline(self, dtype):
    pass


def argmin(self):
    pass


def argmax(self):
    pass


def nanargmin(self):
    pass


def nanargmax(self):
    pass


def fillna(self, inplace=False, value=None):
    pass


def copy(self):
    pass


def isnan(self):
    pass


def notnan(self):
    pass


def sum(self):
    pass


def nansum(self):
    pass


@sdc_overload(astype, inline='always')
@sdc_overload(astype_no_inline)
def sdc_astype_overload(self, dtype):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of numpy.astype.

    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k astype

    """

    ty_checker = TypeChecker("numpy-like 'astype'")
    if not isinstance(self, (types.Array, StringArrayType)):
        return None

    if not isinstance(dtype, (types.functions.NumberClass, types.Function, types.Literal)):
        def impl(self, dtype):
            return astype(self, literally(dtype))

        return impl

    if not isinstance(dtype, (types.StringLiteral, types.UnicodeType, types.Function, types.functions.NumberClass)):
        ty_checker.raise_exc(dtype, 'string or type', 'dtype')

    if (
        (isinstance(dtype, types.Function) and dtype.typing_key == str) or
        (isinstance(dtype, types.StringLiteral) and dtype.literal_value == 'str')
    ):
        def sdc_astype_number_to_string_impl(self, dtype):
            num_bytes = 0
            arr_len = len(self)

            # Get total bytes for new array
            for i in prange(arr_len):
                item = self[i]
                num_bytes += get_utf8_size(str(item))

            data = pre_alloc_string_array(arr_len, num_bytes)

            for i in range(arr_len):
                item = self[i]
                data[i] = str(item)  # TODO: check NA

            return data

        return sdc_astype_number_to_string_impl

    if (isinstance(self, types.Array) and isinstance(dtype, (types.StringLiteral, types.functions.NumberClass))):
        def sdc_astype_number_impl(self, dtype):
            arr = numpy.empty(len(self), dtype=numpy.dtype(dtype))
            for i in numba.prange(len(self)):
                arr[i] = self[i]

            return arr

        return sdc_astype_number_impl


def sdc_nanarg_overload(reduce_op):
    def nanarg_impl(self):
        """
        Intel Scalable Dataframe Compiler Developer Guide
        *************************************************
        Parallel replacement of numpy.nanargmin/numpy.nanargmax.

        .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k nanargmin
        Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k nanargmax

        """

        ty_checker = TypeChecker("numpy-like 'nanargmin'/'nanargmax'")
        dtype = self.dtype
        isnan = get_isnan(dtype)
        max_int64 = max_dtype_int_val(numpy_support.from_dtype(numpy.int64))
        if isinstance(dtype, types.Integer):
            initial_result = {
                min: max_dtype_int_val(dtype),
                max: min_dtype_int_val(dtype),
            }[reduce_op]

        if isinstance(dtype, types.Float):
            initial_result = {
                min: max_dtype_float_val(dtype),
                max: min_dtype_float_val(dtype),
            }[reduce_op]

        if not isinstance(self, types.Array):
            return None

        if isinstance(dtype, types.Number):
            def sdc_nanargmin_impl(self):
                chunks = parallel_chunks(len(self))
                arr_res = numpy.empty(shape=len(chunks), dtype=dtype)
                arr_pos = numpy.empty(shape=len(chunks), dtype=numpy.int64)
                for i in prange(len(chunks)):
                    chunk = chunks[i]
                    res = initial_result
                    pos = max_int64
                    for j in range(chunk.start, chunk.stop):
                        if reduce_op(res, self[j]) != self[j]:
                            continue
                        if isnan(self[j]):
                            continue
                        if res == self[j]:
                            pos = min(pos, j)
                        else:
                            pos = j
                            res = self[j]
                    arr_res[i] = res
                    arr_pos[i] = pos

                general_res = initial_result
                general_pos = max_int64
                for i in range(len(chunks)):
                    if reduce_op(general_res, arr_res[i]) != arr_res[i]:
                        continue
                    if general_res == arr_res[i]:
                        general_pos = min(general_pos, arr_pos[i])
                    else:
                        general_pos = arr_pos[i]
                        general_res = arr_res[i]

                return general_pos

            return sdc_nanargmin_impl

        ty_checker.raise_exc(dtype, 'number', 'self.dtype')
    return nanarg_impl


sdc_overload(nanargmin)(sdc_nanarg_overload(min))
sdc_overload(nanargmax)(sdc_nanarg_overload(max))


def sdc_arg_overload(reduce_op):
    def arg_impl(self):
        """
        Intel Scalable Dataframe Compiler Developer Guide
        *************************************************
        Parallel replacement of numpy.argmin/numpy.argmax.

        .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k argmin
        Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k argmax

        """

        ty_checker = TypeChecker("numpy-like 'argmin'/'argmax'")
        dtype = self.dtype
        isnan = get_isnan(dtype)
        max_int64 = max_dtype_int_val(numpy_support.from_dtype(numpy.int64))
        if isinstance(dtype, types.Integer):
            initial_result = {
                min: max_dtype_int_val(dtype),
                max: min_dtype_int_val(dtype),
            }[reduce_op]

        if isinstance(dtype, types.Float):
            initial_result = {
                min: max_dtype_float_val(dtype),
                max: min_dtype_float_val(dtype),
            }[reduce_op]

        if not isinstance(self, types.Array):
            return None

        if isinstance(dtype, types.Number):
            def sdc_argmin_impl(self):
                chunks = parallel_chunks(len(self))
                arr_res = numpy.empty(shape=len(chunks), dtype=dtype)
                arr_pos = numpy.empty(shape=len(chunks), dtype=numpy.int64)
                for i in prange(len(chunks)):
                    chunk = chunks[i]
                    res = initial_result
                    pos = max_int64
                    for j in range(chunk.start, chunk.stop):
                        if not isnan(self[j]):
                            if reduce_op(res, self[j]) != self[j]:
                                continue
                            if res == self[j]:
                                pos = min(pos, j)
                            else:
                                pos = j
                                res = self[j]
                        else:
                            if numpy.isnan(res):
                                pos = min(pos, j)
                            else:
                                pos = j
                            res = self[j]

                    arr_res[i] = res
                    arr_pos[i] = pos
                general_res = initial_result
                general_pos = max_int64
                for i in range(len(chunks)):
                    if not isnan(arr_res[i]):
                        if reduce_op(general_res, arr_res[i]) != arr_res[i]:
                            continue
                        if general_res == arr_res[i]:
                            general_pos = min(general_pos, arr_pos[i])
                        else:
                            general_pos = arr_pos[i]
                            general_res = arr_res[i]
                    else:
                        if numpy.isnan(general_res):
                            general_pos = min(general_pos, arr_pos[i])
                        else:
                            general_pos = arr_pos[i]
                        general_res = arr_res[i]
                return general_pos

            return sdc_argmin_impl

        ty_checker.raise_exc(dtype, 'number', 'self.dtype')
    return arg_impl


sdc_overload(argmin)(sdc_arg_overload(min))
sdc_overload(argmax)(sdc_arg_overload(max))


@sdc_overload(copy)
def sdc_copy_overload(self):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of numpy.copy.

    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k copy
    """

    if not isinstance(self, (types.Array, StringArrayType)):
        return None

    dtype = self.dtype
    if isinstance(dtype, (types.Number, types.Boolean, bool)):
        def sdc_copy_number_impl(self):
            length = len(self)
            res = numpy.empty(length, dtype=dtype)
            for i in prange(length):
                res[i] = self[i]

            return res

        return sdc_copy_number_impl

    if isinstance(dtype, (types.npytypes.UnicodeCharSeq, types.UnicodeType, types.StringLiteral)):
        def sdc_copy_string_impl(self):
            return self.copy()

        return sdc_copy_string_impl


@sdc_overload(notnan)
def sdc_notnan_overload(self):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of numpy.notnan.
    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k notnan
    """

    if not isinstance(self, types.Array):
        return None

    dtype = self.dtype
    isnan = get_isnan(dtype)
    if isinstance(dtype, (types.Integer, types.Boolean, bool)):
        def sdc_notnan_int_impl(self):
            length = len(self)
            res = numpy.ones(shape=length, dtype=numpy.bool_)

            return res

        return sdc_notnan_int_impl

    if isinstance(dtype, types.Float):
        def sdc_notnan_float_impl(self):
            length = len(self)
            res = numpy.empty(shape=length, dtype=numpy.bool_)
            for i in prange(length):
                res[i] = not isnan(self[i])

            return res

        return sdc_notnan_float_impl

    ty_checker.raise_exc(dtype, 'int or float', 'self.dtype')


@sdc_overload(isnan)
def sdc_isnan_overload(self):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of numpy.isnan.
    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k isnan
    """

    if not isinstance(self, types.Array):
        return None

    dtype = self.dtype
    isnan = get_isnan(dtype)
    if isinstance(dtype, (types.Integer, types.Boolean, bool)):
        def sdc_isnan_int_impl(self):
            length = len(self)
            res = numpy.zeros(shape=length, dtype=numpy.bool_)

            return res

        return sdc_isnan_int_impl

    if isinstance(dtype, types.Float):
        def sdc_isnan_float_impl(self):
            length = len(self)
            res = numpy.empty(shape=length, dtype=numpy.bool_)
            for i in prange(length):
                res[i] = isnan(self[i])

            return res

        return sdc_isnan_float_impl

    ty_checker.raise_exc(dtype, 'int or float', 'self.dtype')


def gen_sum_bool_impl():
    """Generate sum bool implementation."""
    def _sum_bool_impl(self):
        length = len(self)
        result = 0
        for i in prange(length):
            result += self[i]

        return result

    return _sum_bool_impl


@sdc_overload(sum)
def sdc_sum_overload(self):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of numpy.sum.
    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k sum
    """

    dtype = self.dtype
    isnan = get_isnan(dtype)
    if not isinstance(self, types.Array):
        return None

    if isinstance(dtype, types.Number):
        def sdc_sum_number_impl(self):
            length = len(self)
            result = 0
            for i in prange(length):
                if not isnan(self[i]):
                    result += self[i]
                else:
                    return numpy.nan

            return result

        return sdc_sum_number_impl

    if isinstance(dtype, (types.Boolean, bool)):
        return gen_sum_bool_impl()


@sdc_overload(nansum)
def sdc_nansum_overload(self):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of numpy.nansum.
    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k nansum
    """

    dtype = self.dtype
    isnan = get_isnan(dtype)
    if not isinstance(self, types.Array):
        return None

    if isinstance(dtype, types.Number):
        def sdc_nansum_number_impl(self):
            length = len(self)
            result = 0
            for i in prange(length):
                if not numpy.isnan(self[i]):
                    result += self[i]

            return result

        return sdc_nansum_number_impl

    if isinstance(dtype, (types.Boolean, bool)):
        return gen_sum_bool_impl()


@sdc_overload(fillna)
def sdc_fillna_overload(self, inplace=False, value=None):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of fillna.
    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k fillna
    """
    if not isinstance(self, (types.Array, StringArrayType)):
        return None

    dtype = self.dtype
    isnan = get_isnan(dtype)
    if (
        (isinstance(inplace, types.Literal) and inplace.literal_value == True) or  # noqa
        (isinstance(inplace, bool) and inplace == True)  # noqa
    ):
        if isinstance(dtype, (types.Integer, types.Boolean)):
            def sdc_fillna_inplace_int_impl(self, inplace=False, value=None):
                return None

            return sdc_fillna_inplace_int_impl

        def sdc_fillna_inplace_float_impl(self, inplace=False, value=None):
            length = len(self)
            for i in prange(length):
                if isnan(self[i]):
                    self[i] = value
            return None

        return sdc_fillna_inplace_float_impl

    else:
        if isinstance(self.dtype, types.UnicodeType):
            def sdc_fillna_str_impl(self, inplace=False, value=None):
                n = len(self)
                num_chars = 0
                # get total chars in new array
                for i in prange(n):
                    s = self[i]
                    if sdc.hiframes.api.isna(self, i):
                        num_chars += len(value)
                    else:
                        num_chars += len(s)

                filled_data = pre_alloc_string_array(n, num_chars)
                for i in prange(n):
                    if sdc.hiframes.api.isna(self, i):
                        filled_data[i] = value
                    else:
                        filled_data[i] = self[i]
                return filled_data

            return sdc_fillna_str_impl

        if isinstance(dtype, (types.Integer, types.Boolean)):
            def sdc_fillna_int_impl(self, inplace=False, value=None):
                return copy(self)

            return sdc_fillna_int_impl

        def sdc_fillna_impl(self, inplace=False, value=None):
            length = len(self)
            filled_data = numpy.empty(length, dtype=dtype)
            for i in prange(length):
                if isnan(self[i]):
                    filled_data[i] = value
                else:
                    filled_data[i] = self[i]
            return filled_data

        return sdc_fillna_impl


def nanmin(a):
    pass


def nanmax(a):
    pass


def nan_min_max_overload_factory(reduce_op):
    def ov_impl(a):
        if not isinstance(a, types.Array):
            return

        if isinstance(a.dtype, (types.Float, types.Complex)):
            isnan = get_isnan(a.dtype)
            initial_result = {
                min: numpy.inf,
                max: -numpy.inf,
            }[reduce_op]

            def impl(a):
                result = initial_result
                nan_count = 0
                length = len(a)
                for i in prange(length):
                    v = a[i]
                    if not isnan(v):
                        result = reduce_op(result, v)
                    else:
                        nan_count += 1

                if nan_count == length:
                    return numpy.nan

                return result
            return impl
        else:
            def impl(a):
                result = a[0]
                for i in prange(len(a) - 1):
                    result = reduce_op(result, a[i + 1])
                return result
            return impl

    return ov_impl


sdc_overload(nanmin)(nan_min_max_overload_factory(min))
sdc_overload(nanmax)(nan_min_max_overload_factory(max))


def nanprod(a):
    pass


@sdc_overload(nanprod)
def np_nanprod(a):
    """
    Reimplemented with parfor from numba.targets.arraymath.
    """
    if not isinstance(a, types.Array):
        return
    if isinstance(a.dtype, types.Integer):
        retty = types.intp
    else:
        retty = a.dtype
    one = retty(1)
    isnan = get_isnan(a.dtype)

    def nanprod_impl(a):
        c = one
        for i in prange(len(a)):
            v = a[i]
            if not isnan(v):
                c *= v
        return c

    return nanprod_impl


def dropna(arr, idx, name):
    pass


@sdc_overload(dropna)
def dropna_overload(arr, idx, name):
    dtype = arr.dtype
    dtype_idx = idx.dtype
    isnan = get_isnan(dtype)

    def dropna_impl(arr, idx, name):
        chunks = parallel_chunks(len(arr))
        arr_len = numpy.empty(len(chunks), dtype=numpy.int64)
        length = 0

        for i in prange(len(chunks)):
            chunk = chunks[i]
            res = 0
            for j in range(chunk.start, chunk.stop):
                if not isnan(arr[j]):
                    res += 1
            length += res
            arr_len[i] = res

        result_data = numpy.empty(shape=length, dtype=dtype)
        result_index = numpy.empty(shape=length, dtype=dtype_idx)
        for i in prange(len(chunks)):
            chunk = chunks[i]
            new_start = int(sum(arr_len[0:i]))
            new_stop = new_start + arr_len[i]
            current_pos = new_start

            for j in range(chunk.start, chunk.stop):
                if not isnan(arr[j]):
                    result_data[current_pos] = arr[j]
                    result_index[current_pos] = idx[j]
                    current_pos += 1

        return pandas.Series(result_data, result_index, name)

    return dropna_impl


def nanmean(a):
    pass


@sdc_overload(nanmean)
def np_nanmean(a):
    if not isinstance(a, types.Array):
        return
    isnan = get_isnan(a.dtype)

    def nanmean_impl(a):
        c = 0.0
        count = 0
        for i in prange(len(a)):
            v = a[i]
            if not isnan(v):
                c += v
                count += 1
        # np.divide() doesn't raise ZeroDivisionError
        return np.divide(c, count)

    return nanmean_impl


def corr(self, other, method='pearson', min_periods=None):
    pass


@sdc_overload(corr)
def corr_overload(self, other, method='pearson', min_periods=None):
    def corr_impl(self, other, method='pearson', min_periods=None):
        if method not in ('pearson', ''):
            raise ValueError("Method corr(). Unsupported parameter. Given method != 'pearson'")

        if min_periods is None or min_periods < 1:
            min_periods = 1

        min_len = min(len(self._data), len(other._data))

        if min_len == 0:
            return numpy.nan

        sum_y = 0.
        sum_x = 0.
        sum_xy = 0.
        sum_xx = 0.
        sum_yy = 0.
        total_count = 0
        for i in prange(min_len):
            x = self._data[i]
            y = other._data[i]
            if not (numpy.isnan(x) or numpy.isnan(y)):
                sum_x += x
                sum_y += y
                sum_xy += x * y
                sum_xx += x * x
                sum_yy += y * y
                total_count += 1

        if total_count < min_periods:
            return numpy.nan

        cov_xy = (sum_xy - sum_x * sum_y / total_count)
        var_x = (sum_xx - sum_x * sum_x / total_count)
        var_y = (sum_yy - sum_y * sum_y / total_count)
        corr_xy = cov_xy / numpy.sqrt(var_x * var_y)

        return corr_xy

    return corr_impl


def nanvar(a):
    pass


@sdc_overload(nanvar)
def np_nanvar(a):
    if not isinstance(a, types.Array):
        return
    isnan = get_isnan(a.dtype)

    def nanvar_impl(a):
        # Compute the mean
        m = nanmean(a)

        # Compute the sum of square diffs
        ssd = 0.0
        count = 0
        for i in prange(len(a)):
            v = a[i]
            if not isnan(v):
                val = (v.item() - m)
                ssd += np.real(val * np.conj(val))
                count += 1
        # np.divide() doesn't raise ZeroDivisionError
        return np.divide(ssd, count)

    return nanvar_impl


def cumsum(a):
    pass


def nancumsum(a):
    pass


@sdc_overload(cumsum)
def np_cumsum(arr):
    if not isinstance(arr, types.Array):
        return

    retty = arr.dtype
    zero = retty(0)

    def cumsum_impl(arr):
        chunks = parallel_chunks(len(arr))
        partial_sum = numpy.zeros(len(chunks), dtype=retty)
        result = numpy.empty_like(arr)

        for i in prange(len(chunks)):
            chunk = chunks[i]
            partial = zero
            for j in range(chunk.start, chunk.stop):
                result[j] = partial + arr[j]
                partial = result[j]
            partial_sum[i] = partial

        for i in prange(len(chunks)):
            prefix = sum(partial_sum[0:i])
            chunk = chunks[i]
            for j in range(chunk.start, chunk.stop):
                result[j] += prefix

        return result

    return cumsum_impl


@sdc_overload(nancumsum)
def np_nancumsum(arr, like_pandas=False):
    if not isinstance(arr, types.Array):
        return

    if isinstance(arr.dtype, (types.Boolean, types.Integer)):
        # dtype cannot possibly contain NaN
        return lambda arr, like_pandas=False: cumsum(arr)
    else:
        retty = arr.dtype
        is_nan = get_isnan(retty)
        zero = retty(0)

        def nancumsum_impl(arr, like_pandas=False):
            chunks = parallel_chunks(len(arr))
            partial_sum = numpy.zeros(len(chunks), dtype=retty)
            result = numpy.empty_like(arr)

            for i in prange(len(chunks)):
                chunk = chunks[i]
                partial = zero
                for j in range(chunk.start, chunk.stop):
                    if like_pandas:
                        result[j] = partial + arr[j]
                        if ~is_nan(arr[j]):
                            partial = result[j]
                    else:
                        if ~is_nan(arr[j]):
                            partial += arr[j]
                        result[j] = partial
                partial_sum[i] = partial

            for i in prange(len(chunks)):
                prefix = sum(partial_sum[0:i])
                chunk = chunks[i]
                for j in range(chunk.start, chunk.stop):
                    result[j] += prefix

            return result

        return nancumsum_impl


def getitem_by_mask(arr, idx):
    pass


@sdc_overload(getitem_by_mask)
def getitem_by_mask_overload(arr, idx):
    """
    Creates a new array from arr by selecting elements indicated by Boolean mask idx.

    Parameters
    -----------
    arr: :obj:`Array` or :obj:`Range`
        Input array or range
    idx: :obj:`Array` of dtype :class:`bool`
        Boolean mask

    Returns
    -------
    :obj:`Array` of the same dtype as arr
        Array with only elements indicated by mask left

    """

    is_range = isinstance(arr, types.RangeType) and isinstance(arr.dtype, types.Integer)
    is_str_arr = arr == string_array_type
    if not (isinstance(arr, types.Array) or is_str_arr or is_range):
        return

    res_dtype = arr.dtype
    is_str_arr = arr == string_array_type
    def getitem_by_mask_impl(arr, idx):
        chunks = parallel_chunks(len(arr))
        arr_len = numpy.empty(len(chunks), dtype=numpy.int64)
        length = 0

        for i in prange(len(chunks)):
            chunk = chunks[i]
            res = 0
            for j in range(chunk.start, chunk.stop):
                if idx[j]:
                    res += 1
            length += res
            arr_len[i] = res

        if is_str_arr == True:  # noqa
            result_data = [''] * length
            result_nan_mask = numpy.empty(shape=length, dtype=types.bool_)
        else:
            result_data = numpy.empty(shape=length, dtype=res_dtype)
        for i in prange(len(chunks)):
            chunk = chunks[i]
            new_start = int(sum(arr_len[0:i]))
            current_pos = new_start

            for j in range(chunk.start, chunk.stop):
                if idx[j]:
                    if is_range == True:  # noqa
                        value = arr.start + arr.step * j
                    else:
                        value = arr[j]
                    result_data[current_pos] = value
                    if is_str_arr == True:  # noqa
                        result_nan_mask[current_pos] = isna(arr, j)
                    current_pos += 1

        if is_str_arr == True:  # noqa
            result_data_as_str_arr = create_str_arr_from_list(result_data)
            str_arr_set_na_by_mask(result_data_as_str_arr, result_nan_mask)
            return result_data_as_str_arr
        else:
            return result_data

    return getitem_by_mask_impl


def skew(a):
    pass


def nanskew(a):
    pass


@sdc_overload(skew)
def np_skew(arr):
    if not isinstance(arr, types.Array):
        return

    def skew_impl(arr):
        len_val = len(arr)
        n = 0
        _sum = 0.0
        square_sum = 0.0
        cube_sum = 0.0

        for idx in numba.prange(len_val):
            if not numpy.isnan(arr[idx]):
                n += 1
                _sum += arr[idx]
                square_sum += arr[idx] ** 2
                cube_sum += arr[idx] ** 3

        if n == 0 or n < len_val:
            return numpy.nan

        m2 = (square_sum - _sum * _sum / n) / n
        m3 = (cube_sum - 3. * _sum * square_sum / n + 2. * _sum * _sum * _sum / n / n) / n
        res = numpy.nan if m2 == 0 else m3 / m2 ** 1.5

        if (n > 2) & (m2 > 0):
            res = numpy.sqrt((n - 1.) * n) / (n - 2.) * m3 / m2 ** 1.5

        return res

    return skew_impl


@sdc_overload(nanskew)
def np_nanskew(arr):
    if not isinstance(arr, types.Array):
        return

    def nanskew_impl(arr):
        len_val = len(arr)
        n = 0
        _sum = 0.0
        square_sum = 0.0
        cube_sum = 0.0

        for idx in numba.prange(len_val):
            if not numpy.isnan(arr[idx]):
                n += 1
                _sum += arr[idx]
                square_sum += arr[idx] ** 2
                cube_sum += arr[idx] ** 3

        if n == 0:
            return numpy.nan

        m2 = (square_sum - _sum * _sum / n) / n
        m3 = (cube_sum - 3. * _sum * square_sum / n + 2. * _sum * _sum * _sum / n / n) / n
        res = numpy.nan if m2 == 0 else m3 / m2 ** 1.5

        if (n > 2) & (m2 > 0):
            res = numpy.sqrt((n - 1.) * n) / (n - 2.) * m3 / m2 ** 1.5

        return res

    return nanskew_impl

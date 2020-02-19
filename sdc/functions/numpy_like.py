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

"""

| This file contains SDC numpy modified functions, rewritten according to concurrency

"""

import numba
import numpy
import pandas
import numpy as np

from numba import types, jit, prange, numpy_support, literally
from numba.errors import TypingError
from numba.targets.arraymath import get_isnan

import sdc
from sdc.utilities.sdc_typing_utils import TypeChecker
from sdc.str_arr_ext import (StringArrayType, pre_alloc_string_array, get_utf8_size, str_arr_is_na)
from sdc.utilities.utils import sdc_overload, sdc_register_jitable
from sdc.utilities.prange_utils import parallel_chunks


def astype(self, dtype):
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


@sdc_overload(astype)
def sdc_astype_overload(self, dtype):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of numpy.astype.

    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k astype

    """

    ty_checker = TypeChecker("numpy-like 'astype'")
    if not isinstance(self, types.Array):
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

    ty_checker.raise_exc(self.dtype, 'str or type', 'self.dtype')


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

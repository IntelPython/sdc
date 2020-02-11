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

from numba import types, jit, prange, numpy_support, literally
from numba.errors import TypingError
from numba.targets.arraymath import get_isnan

import sdc
from sdc.utilities.sdc_typing_utils import TypeChecker
from sdc.str_arr_ext import (StringArrayType, pre_alloc_string_array, get_utf8_size, str_arr_is_na)
from sdc.utilities.utils import sdc_overload, sdc_register_jitable


def astype(self, dtype):
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


@sdc_overload(notnan)
def sdc_isnan_overload(self):
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
    if isinstance(dtype, types.Integer):
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
    if isinstance(dtype, types.Integer):
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
            if self[i]:
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

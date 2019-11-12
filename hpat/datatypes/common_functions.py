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

"""

| This file contains internal common functions used in SDC implementation across different files

"""

import numpy

from numba import types, njit, prange
from numba.extending import overload
from numba import numpy_support

import hpat
from hpat.str_arr_ext import (string_array_type, num_total_chars, append_string_array_to)


def has_literal_value(var, value):
    '''Used during typing to check that variable var is a Numba literal value equal to value'''

    if not isinstance(var, types.Literal):
        return False

    if value is None or isinstance(value, type(bool)):
        return var.literal_value is value
    else:
        return var.literal_value == value


def has_python_value(var, value):
    '''Used during typing to check that variable var was resolved as Python type and has specific value'''

    if not isinstance(var, type(value)):
        return False

    if value is None or isinstance(value, type(bool)):
        return var is value
    else:
        return var == value


def hpat_arrays_append(A, B):
    pass


@overload(hpat_arrays_append)
def hpat_arrays_append_overload(A, B):
    '''Function for appending underlying arrays (A and B) or list/tuple of arrays B to an array A'''

    if isinstance(A, types.Array):
        if isinstance(B, types.Array):
            def _append_single_numeric_impl(A, B):
                return numpy.concatenate((A, B,))

            return _append_single_numeric_impl
        elif isinstance(B, (types.UniTuple, types.List)):
            # TODO: this heavily relies on B being a homogeneous tuple/list - find a better way
            # to resolve common dtype of heterogeneous sequence of arrays
            np_dtypes = [numpy_support.as_dtype(A.dtype), numpy_support.as_dtype(B.dtype.dtype)]
            np_common_dtype = numpy.find_common_type([], np_dtypes)
            numba_common_dtype = numpy_support.from_dtype(np_common_dtype)

            # TODO: refactor to use numpy.concatenate when Numba supports building a tuple at runtime
            def _append_list_numeric_impl(A, B):

                total_length = len(A) + numpy.array([len(arr) for arr in B]).sum()
                new_data = numpy.empty(total_length, numba_common_dtype)

                stop = len(A)
                new_data[:stop] = A
                for arr in B:
                    start = stop
                    stop = start + len(arr)
                    new_data[start:stop] = arr
                return new_data

            return _append_list_numeric_impl

    elif A == string_array_type:
        if B == string_array_type:
            def _append_single_string_array_impl(A, B):
                total_size = len(A) + len(B)
                total_chars = num_total_chars(A) + num_total_chars(B)
                new_data = hpat.str_arr_ext.pre_alloc_string_array(total_size, total_chars)

                pos = 0
                pos += append_string_array_to(new_data, pos, A)
                pos += append_string_array_to(new_data, pos, B)

                return new_data

            return _append_single_string_array_impl
        elif (isinstance(B, (types.UniTuple, types.List)) and B.dtype == string_array_type):
            def _append_list_string_array_impl(A, B):
                array_list = [A] + list(B)
                total_size = numpy.array([len(arr) for arr in array_list]).sum()
                total_chars = numpy.array([num_total_chars(arr) for arr in array_list]).sum()

                new_data = hpat.str_arr_ext.pre_alloc_string_array(total_size, total_chars)

                pos = 0
                pos += append_string_array_to(new_data, pos, A)
                for arr in B:
                    pos += append_string_array_to(new_data, pos, arr)

                return new_data

            return _append_list_string_array_impl


@njit
def _compute_map_chunks(l, n):
    assert n > 0
    a = len(l) // n
    b = a + 1
    c = len(l) % n
    return [l[i * b: i * b + b] if i < c else l[c * b + (i - c) * a: c * b + (i - c) * a + a] for i in range(n)]


@njit(parallel=True)
def map_reduce(arg, init_val, map_func, reduce_func):
    res = init_val
    for i in prange(len(arg)):
        val = map_func(arg[i])
        res = reduce_func(res, val)
    return res


@njit(parallel=True)
def map_reduce_chunked(arg, init_val, map_func, reduce_func):
    res = init_val
    # TODO: proper cores/nodes count
    chunks_count = 4
    if 1 == chunks_count:
        return map_func(arg)
    else:
        c = _compute_map_chunks(arg, chunks_count)
        for i in range(len(c)):
            val = map_func(c[i])
            res = reduce_func(res, val)
        return res

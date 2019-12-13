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
import pandas

import numba
from numba import types
from numba.errors import TypingError
from numba.extending import overload
from numba import numpy_support

import sdc
from sdc.str_arr_ext import (string_array_type, num_total_chars, append_string_array_to,
                             str_arr_is_na, pre_alloc_string_array, str_arr_set_na)


class TypeChecker:
    """
        Validate object type and raise TypingError if the type is invalid, e.g.:
            Method nsmallest(). The object n
             given: bool
             expected: int
    """
    msg_template = '{} The object {}\n given: {}\n expected: {}'

    def __init__(self, func_name):
        """
        Parameters
        ----------
        func_name: :obj:`str`
            name of the function where types checking
        """
        self.func_name = func_name

    def raise_exc(self, data, expected_types, name=''):
        """
        Raise exception with unified message
        Parameters
        ----------
        data: :obj:`any`
            real type of the data
        expected_types: :obj:`str`
            expected types inserting directly to the exception
        name: :obj:`str`
            name of the parameter
        """
        msg = self.msg_template.format(self.func_name, name, data, expected_types)
        raise TypingError(msg)

    def check(self, data, accepted_type, name=''):
        """
        Check data type belongs to specified type
        Parameters
        ----------
        data: :obj:`any`
            real type of the data
        accepted_type: :obj:`type`
            accepted type
        name: :obj:`str`
            name of the parameter
        """
        if not isinstance(data, accepted_type):
            self.raise_exc(data, accepted_type.__name__, name=name)


def has_literal_value(var, value):
    """Used during typing to check that variable var is a Numba literal value equal to value"""

    if not isinstance(var, types.Literal):
        return False

    if value is None or isinstance(value, type(bool)):
        return var.literal_value is value
    else:
        return var.literal_value == value


def has_python_value(var, value):
    """Used during typing to check that variable var was resolved as Python type and has specific value"""

    if not isinstance(var, type(value)):
        return False

    if value is None or isinstance(value, type(bool)):
        return var is value
    else:
        return var == value


def check_is_numeric_array(type_var):
    """Used during typing to check that type_var is a numeric numpy arrays"""
    return isinstance(type_var, types.Array) and isinstance(type_var.dtype, types.Number)


def check_index_is_numeric(ty_series):
    """Used during typing to check that series has numeric index"""
    return check_is_numeric_array(ty_series.index)


def check_types_comparable(ty_left, ty_right):
    """Used during typing to check that underlying arrays of specified types can be compared"""
    return ((ty_left == string_array_type and ty_right == string_array_type)
            or (check_is_numeric_array(ty_left) and check_is_numeric_array(ty_right)))


def hpat_arrays_append(A, B):
    pass


@overload(hpat_arrays_append)
def hpat_arrays_append_overload(A, B):
    """Function for appending underlying arrays (A and B) or list/tuple of arrays B to an array A"""

    if isinstance(A, types.Array):
        if isinstance(B, types.Array):
            def _append_single_numeric_impl(A, B):
                return numpy.concatenate((A, B,))

            return _append_single_numeric_impl
        elif isinstance(B, (types.UniTuple, types.List)):
            # TODO: this heavily relies on B being a homogeneous tuple/list - find a better way
            # to resolve common dtype of heterogeneous sequence of arrays
            numba_common_dtype = find_common_dtype_from_numpy_dtypes([A.dtype, B.dtype.dtype], [])

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
                new_data = sdc.str_arr_ext.pre_alloc_string_array(total_size, total_chars)

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

                new_data = sdc.str_arr_ext.pre_alloc_string_array(total_size, total_chars)

                pos = 0
                pos += append_string_array_to(new_data, pos, A)
                for arr in B:
                    pos += append_string_array_to(new_data, pos, arr)

                return new_data

            return _append_list_string_array_impl


@numba.njit
def _hpat_ensure_array_capacity(new_size, arr):
    """ Function ensuring that the size of numpy array is at least as specified
        Returns newly allocated array of bigger size with copied elements if existing size is less than requested
    """

    k = len(arr)
    if k >= new_size:
        return arr

    n = k
    while n < new_size:
        n = 2 * n
    res = numpy.empty(n, arr.dtype)
    res[:k] = arr[:k]
    return res


def find_common_dtype_from_numpy_dtypes(array_types, scalar_types):
    """Used to find common numba dtype for a sequences of numba dtypes each representing some numpy dtype"""
    np_array_dtypes = [numpy_support.as_dtype(dtype) for dtype in array_types]
    np_scalar_dtypes = [numpy_support.as_dtype(dtype) for dtype in scalar_types]
    np_common_dtype = numpy.find_common_type(np_array_dtypes, np_scalar_dtypes)
    numba_common_dtype = numpy_support.from_dtype(np_common_dtype)

    return numba_common_dtype


def sdc_join_series_indexes(left, right):
    pass


@overload(sdc_join_series_indexes)
def sdc_join_series_indexes_overload(left, right):
    """Function for joining arrays left and right in a way similar to pandas.join 'outer' algorithm"""

    # TODO: eliminate code duplication by merging implementations for numeric and StringArray
    # requires equivalents of numpy.arsort and _hpat_ensure_array_capacity for StringArrays
    if (isinstance(left, types.Array) and isinstance(right, types.Array)):

        numba_common_dtype = find_common_dtype_from_numpy_dtypes([left.dtype, right.dtype], [])
        if isinstance(numba_common_dtype, types.Number):

            def sdc_join_series_indexes_impl(left, right):

                # allocate result arrays
                lsize = len(left)
                rsize = len(right)
                est_total_size = int(1.1 * (lsize + rsize))

                lidx = numpy.empty(est_total_size, numpy.int64)
                ridx = numpy.empty(est_total_size, numpy.int64)
                joined = numpy.empty(est_total_size, numba_common_dtype)

                # sort arrays saving the old positions
                sorted_left = numpy.argsort(left, kind='mergesort')
                sorted_right = numpy.argsort(right, kind='mergesort')

                i, j, k = 0, 0, 0
                while (i < lsize and j < rsize):
                    joined = _hpat_ensure_array_capacity(k + 1, joined)
                    lidx = _hpat_ensure_array_capacity(k + 1, lidx)
                    ridx = _hpat_ensure_array_capacity(k + 1, ridx)

                    left_index = left[sorted_left[i]]
                    right_index = right[sorted_right[j]]

                    if (left_index < right_index):
                        joined[k] = left_index
                        lidx[k] = sorted_left[i]
                        ridx[k] = -1
                        i += 1
                        k += 1
                    elif (left_index > right_index):
                        joined[k] = right_index
                        lidx[k] = -1
                        ridx[k] = sorted_right[j]
                        j += 1
                        k += 1
                    else:
                        # find ends of sequences of equal index values in left and right
                        ni, nj = i, j
                        while (ni < lsize and left[sorted_left[ni]] == left_index):
                            ni += 1
                        while (nj < rsize and right[sorted_right[nj]] == right_index):
                            nj += 1

                        # join the blocks found into results
                        for s in numpy.arange(i, ni, 1):
                            block_size = nj - j
                            to_joined = numpy.repeat(left_index, block_size)
                            to_lidx = numpy.repeat(sorted_left[s], block_size)
                            to_ridx = numpy.array([sorted_right[k] for k in numpy.arange(j, nj, 1)], numpy.int64)

                            joined = _hpat_ensure_array_capacity(k + block_size, joined)
                            lidx = _hpat_ensure_array_capacity(k + block_size, lidx)
                            ridx = _hpat_ensure_array_capacity(k + block_size, ridx)

                            joined[k:k + block_size] = to_joined
                            lidx[k:k + block_size] = to_lidx
                            ridx[k:k + block_size] = to_ridx
                            k += block_size
                        i = ni
                        j = nj

                # fill the end of joined with remaining part of left or right
                if i < lsize:
                    block_size = lsize - i
                    joined = _hpat_ensure_array_capacity(k + block_size, joined)
                    lidx = _hpat_ensure_array_capacity(k + block_size, lidx)
                    ridx = _hpat_ensure_array_capacity(k + block_size, ridx)
                    ridx[k: k + block_size] = numpy.repeat(-1, block_size)
                    while i < lsize:
                        joined[k] = left[sorted_left[i]]
                        lidx[k] = sorted_left[i]
                        i += 1
                        k += 1

                elif j < rsize:
                    block_size = rsize - j
                    joined = _hpat_ensure_array_capacity(k + block_size, joined)
                    lidx = _hpat_ensure_array_capacity(k + block_size, lidx)
                    ridx = _hpat_ensure_array_capacity(k + block_size, ridx)
                    lidx[k: k + block_size] = numpy.repeat(-1, block_size)
                    while j < rsize:
                        joined[k] = right[sorted_right[j]]
                        ridx[k] = sorted_right[j]
                        j += 1
                        k += 1

                return joined[:k], lidx[:k], ridx[:k]

            return sdc_join_series_indexes_impl

        else:
            # TODO: support joining indexes with common dtype=object - requires Numba
            # support of such numpy arrays in nopython mode, for now just return None
            return None

    elif (left == string_array_type and right == string_array_type):

        def sdc_join_series_indexes_impl(left, right):

            # allocate result arrays
            lsize = len(left)
            rsize = len(right)
            est_total_size = int(1.1 * (lsize + rsize))

            lidx = numpy.empty(est_total_size, numpy.int64)
            ridx = numpy.empty(est_total_size, numpy.int64)

            # use Series.sort_values since argsort for StringArrays not implemented
            original_left_series = pandas.Series(left)
            original_right_series = pandas.Series(right)

            # sort arrays saving the old positions
            left_series = original_left_series.sort_values(kind='mergesort')
            right_series = original_right_series.sort_values(kind='mergesort')
            sorted_left = left_series._index
            sorted_right = right_series._index

            i, j, k = 0, 0, 0
            while (i < lsize and j < rsize):
                lidx = _hpat_ensure_array_capacity(k + 1, lidx)
                ridx = _hpat_ensure_array_capacity(k + 1, ridx)

                left_index = left[sorted_left[i]]
                right_index = right[sorted_right[j]]

                if (left_index < right_index):
                    lidx[k] = sorted_left[i]
                    ridx[k] = -1
                    i += 1
                    k += 1
                elif (left_index > right_index):
                    lidx[k] = -1
                    ridx[k] = sorted_right[j]
                    j += 1
                    k += 1
                else:
                    # find ends of sequences of equal index values in left and right
                    ni, nj = i, j
                    while (ni < lsize and left[sorted_left[ni]] == left_index):
                        ni += 1
                    while (nj < rsize and right[sorted_right[nj]] == right_index):
                        nj += 1

                    # join the blocks found into results
                    for s in numpy.arange(i, ni, 1):
                        block_size = nj - j
                        to_lidx = numpy.repeat(sorted_left[s], block_size)
                        to_ridx = numpy.array([sorted_right[k] for k in numpy.arange(j, nj, 1)], numpy.int64)

                        lidx = _hpat_ensure_array_capacity(k + block_size, lidx)
                        ridx = _hpat_ensure_array_capacity(k + block_size, ridx)

                        lidx[k:k + block_size] = to_lidx
                        ridx[k:k + block_size] = to_ridx
                        k += block_size
                    i = ni
                    j = nj

            # fill the end of joined with remaining part of left or right
            if i < lsize:
                block_size = lsize - i
                lidx = _hpat_ensure_array_capacity(k + block_size, lidx)
                ridx = _hpat_ensure_array_capacity(k + block_size, ridx)
                ridx[k: k + block_size] = numpy.repeat(-1, block_size)
                while i < lsize:
                    lidx[k] = sorted_left[i]
                    i += 1
                    k += 1

            elif j < rsize:
                block_size = rsize - j
                lidx = _hpat_ensure_array_capacity(k + block_size, lidx)
                ridx = _hpat_ensure_array_capacity(k + block_size, ridx)
                lidx[k: k + block_size] = numpy.repeat(-1, block_size)
                while j < rsize:
                    ridx[k] = sorted_right[j]
                    j += 1
                    k += 1

            # count total number of characters and allocate joined array
            total_joined_size = k
            num_chars_in_joined = 0
            for i in numpy.arange(total_joined_size):
                if lidx[i] != -1:
                    num_chars_in_joined += len(left[lidx[i]])
                elif ridx[i] != -1:
                    num_chars_in_joined += len(right[ridx[i]])

            joined = pre_alloc_string_array(total_joined_size, num_chars_in_joined)

            # iterate over joined and fill it with indexes using lidx and ridx indexers
            for i in numpy.arange(total_joined_size):
                if lidx[i] != -1:
                    joined[i] = left[lidx[i]]
                    if (str_arr_is_na(left, lidx[i])):
                        str_arr_set_na(joined, i)
                elif ridx[i] != -1:
                    joined[i] = right[ridx[i]]
                    if (str_arr_is_na(right, ridx[i])):
                        str_arr_set_na(joined, i)
                else:
                    str_arr_set_na(joined, i)

            return joined, lidx, ridx

        return sdc_join_series_indexes_impl

    return None


def sdc_check_indexes_equal(left, right):
    pass


@overload(sdc_check_indexes_equal)
def sdc_check_indexes_equal_overload(A, B):
    """Function for checking arrays A and B of the same type are equal"""

    if isinstance(A, types.Array):
        def sdc_check_indexes_equal_numeric_impl(A, B):
            return numpy.array_equal(A, B)
        return sdc_check_indexes_equal_numeric_impl

    elif A == string_array_type:
        def sdc_check_indexes_equal_string_impl(A, B):
            # TODO: replace with StringArrays comparison
            is_index_equal = (len(A) == len(B)
                              and num_total_chars(A) == num_total_chars(B))
            for i in numpy.arange(len(A)):
                if (A[i] != B[i]
                        or str_arr_is_na(A, i) is not str_arr_is_na(B, i)):
                    is_index_equal = False

            return is_index_equal

        return sdc_check_indexes_equal_string_impl

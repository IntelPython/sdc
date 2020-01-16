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

| This file contains function templates used by the auto-generation script during build

"""

# below imports are copied into the auto-generated source file as-is
# for the auto-generation script to work ensure they are not mixed up with code
import numba
import numpy
import operator
import pandas

from numba.errors import TypingError
from numba import types

from sdc.datatypes.common_functions import TypeChecker
from sdc.datatypes.common_functions import (check_index_is_numeric, find_common_dtype_from_numpy_dtypes,
                                            sdc_join_series_indexes, sdc_check_indexes_equal, check_types_comparable)
from sdc.hiframes.pd_series_type import SeriesType
from sdc.str_arr_ext import (string_array_type, str_arr_is_na)
from sdc.utils import sdc_overload


def sdc_pandas_series_operator_binop(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.binop` implementation

    Note: Currently implemented for numeric Series only.
        Differs from Pandas in returning Series with fixed dtype :obj:`float64`

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op1*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op2*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_binop*

    Parameters
    ----------
    series: :obj:`pandas.Series`
        Input series
    other: :obj:`pandas.Series` or :obj:`scalar`
        Series or scalar value to be used as a second argument of binary operation

    Returns
    -------
    :obj:`pandas.Series`
        The result of the operation
    """

    _func_name = 'Operator binop().'

    ty_checker = TypeChecker('Operator binop().')
    if not isinstance(self, SeriesType):
        return None

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or scalar', 'other')

    if isinstance(other, SeriesType):
        none_or_numeric_indexes = ((isinstance(self.index, types.NoneType) or check_index_is_numeric(self))
                                   and (isinstance(other.index, types.NoneType) or check_index_is_numeric(other)))
        series_data_comparable = check_types_comparable(self.data, other.data)
        series_indexes_comparable = check_types_comparable(self.index, other.index) or none_or_numeric_indexes

    if isinstance(other, SeriesType) and not series_data_comparable:
        raise TypingError('{} Not supported for series with not-comparable data. \
        Given: self.data={}, other.data={}'.format(_func_name, self.data, other.data))

    if isinstance(other, SeriesType) and not series_indexes_comparable:
        raise TypingError('{} Not implemented for series with not-comparable indexes. \
        Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    # specializations for numeric series - TODO: support arithmetic operation on StringArrays
    if (isinstance(other, types.Number)):
        def _series_operator_binop_scalar_impl(self, other):
            result_data = self._data.astype(numpy.float64) + numpy.float64(other)
            return pandas.Series(result_data, index=self._index, name=self._name)

        return _series_operator_binop_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_binop_none_indexes_impl(self, other):

                if (len(self._data) == len(other._data)):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data + other._data.astype(numpy.float64)
                    return pandas.Series(result_data)
                else:
                    left_size, right_size = len(self._data), len(other._data)
                    min_data_size = min(left_size, right_size)
                    max_data_size = max(left_size, right_size)
                    result_data = numpy.empty(max_data_size, dtype=numpy.float64)
                    if (left_size == min_data_size):
                        result_data[:min_data_size] = self._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = result_data + other._data.astype(numpy.float64)
                    else:
                        result_data[:min_data_size] = other._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = self._data.astype(numpy.float64) + result_data

                    return pandas.Series(result_data, self._index)

            return _series_operator_binop_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_binop_common_impl(self, other):
                left_index, right_index = self.index, other.index

                # check if indexes are equal and series don't have to be aligned
                if sdc_check_indexes_equal(left_index, right_index):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data + other._data.astype(numpy.float64)

                    if none_or_numeric_indexes == True:  # noqa
                        result_index = left_index.astype(numba_index_common_dtype)
                    else:
                        result_index = self._index

                    return pandas.Series(result_data, index=result_index)

                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)

                joined_index_range = numpy.arange(len(joined_index))
                left_values = numpy.asarray(
                    [self._data[left_indexer[i]] for i in joined_index_range],
                    numpy.float64
                )
                left_values[left_indexer == -1] = numpy.nan

                right_values = numpy.asarray(
                    [other._data[right_indexer[i]] for i in joined_index_range],
                    numpy.float64
                )
                right_values[right_indexer == -1] = numpy.nan

                result_data = left_values + right_values
                return pandas.Series(result_data, joined_index)

            return _series_operator_binop_common_impl

    return None


def sdc_pandas_series_operator_comp_binop(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.comp_binop` implementation

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op7*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_comp_binop*

    Parameters
    ----------
    series: :obj:`pandas.Series`
        Input series
    other: :obj:`pandas.Series` or :obj:`scalar`
        Series or scalar value to be used as a second argument of binary operation

    Returns
    -------
    :obj:`pandas.Series`
        The result of the operation
    """

    _func_name = 'Operator comp_binop().'

    ty_checker = TypeChecker('Operator comp_binop().')
    if not isinstance(self, SeriesType):
        return None

    if not isinstance(other, (SeriesType, types.Number, types.UnicodeType)):
        ty_checker.raise_exc(other, 'pandas.series or scalar', 'other')

    if isinstance(other, SeriesType):
        none_or_numeric_indexes = ((isinstance(self.index, types.NoneType) or check_index_is_numeric(self))
                                   and (isinstance(other.index, types.NoneType) or check_index_is_numeric(other)))
        series_data_comparable = check_types_comparable(self.data, other.data)
        series_indexes_comparable = check_types_comparable(self.index, other.index) or none_or_numeric_indexes

    if isinstance(other, SeriesType) and not series_data_comparable:
        raise TypingError('{} Not supported for series with not-comparable data. \
        Given: self.data={}, other.data={}'.format(_func_name, self.data, other.data))

    if isinstance(other, SeriesType) and not series_indexes_comparable:
        raise TypingError('{} Not implemented for series with not-comparable indexes. \
        Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    if (isinstance(other, (types.Number, types.UnicodeType))):
        def _series_operator_comp_binop_scalar_impl(self, other):
            return pandas.Series(self._data < other, index=self._index, name=self._name)

        return _series_operator_comp_binop_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_comp_binop_none_indexes_impl(self, other):
                left_size, right_size = len(self._data), len(other._data)
                if (left_size == right_size):
                    return pandas.Series(self._data < other._data)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_comp_binop_none_indexes_impl
        else:

            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_comp_binop_common_impl(self, other):
                left_index, right_index = self.index, other.index

                if sdc_check_indexes_equal(left_index, right_index):
                    if none_or_numeric_indexes == True:  # noqa
                        new_index = left_index.astype(numba_index_common_dtype)
                    else:
                        new_index = self._index
                    return pandas.Series(self._data < other._data,
                                         new_index)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_comp_binop_common_impl

    return None


def sdc_str_arr_operator_comp_binop(self, other):

    self_is_str_arr = self == string_array_type
    other_is_str_arr = other == string_array_type
    operands_are_arrays = self_is_str_arr and other_is_str_arr

    if not (operands_are_arrays
            or (self_is_str_arr and isinstance(other, types.UnicodeType))
            or (isinstance(self, types.UnicodeType) and other_is_str_arr)):
        return None

    if operands_are_arrays:
        def _sdc_str_arr_operator_comp_binop_impl(self, other):
            if len(self) != len(other):
                raise ValueError("Mismatch of String Arrays sizes in operator.comp_binop")
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] < other[i]
                               and not (str_arr_is_na(self, i) or str_arr_is_na(other, i)))
            return out_list

    elif self_is_str_arr:
        def _sdc_str_arr_operator_comp_binop_impl(self, other):
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] < other and not (str_arr_is_na(self, i)))
            return out_list

    elif other_is_str_arr:
        def _sdc_str_arr_operator_comp_binop_impl(self, other):
            n = len(other)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self < other[i] and not (str_arr_is_na(other, i)))
            return out_list
    else:
        return None

    return _sdc_str_arr_operator_comp_binop_impl

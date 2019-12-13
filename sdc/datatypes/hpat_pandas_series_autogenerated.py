# -*- coding: utf-8 -*-
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

| This file contains SDC Pandas Series methods auto-generated with autogen_sources.py

"""

import numba
import numpy
import operator
import pandas

from numba.errors import TypingError
from numba.extending import overload, overload_method, overload_attribute
from numba import types

import sdc
from sdc.datatypes.common_functions import TypeChecker
from sdc.datatypes.common_functions import (check_index_is_numeric, find_common_dtype_from_numpy_dtypes,
                                            sdc_join_series_indexes, sdc_check_indexes_equal, check_types_comparable)
from sdc.hiframes.pd_series_ext import SeriesType
from sdc.str_arr_ext import (string_array_type, num_total_chars, str_arr_is_na)


@overload(operator.add)
def sdc_pandas_series_operator_add(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.add` implementation

    Note: Currently implemented for numeric Series only.
        Differs from Pandas in returning Series with fixed dtype :obj:`float64`

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op1*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op2*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_add*

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

    _func_name = 'Operator add().'

    ty_checker = TypeChecker('Operator add().')
    ty_checker.check(self, SeriesType)

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
        def _series_operator_add_scalar_impl(self, other):
            result_data = self._data.astype(numpy.float64) + numpy.float64(other)
            return pandas.Series(result_data, index=self._index, name=self._name)

        return _series_operator_add_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_add_none_indexes_impl(self, other):

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

            return _series_operator_add_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_add_common_impl(self, other):
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

            return _series_operator_add_common_impl

    return None


@overload(operator.sub)
def sdc_pandas_series_operator_sub(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.sub` implementation

    Note: Currently implemented for numeric Series only.
        Differs from Pandas in returning Series with fixed dtype :obj:`float64`

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op1*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op2*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_sub*

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

    _func_name = 'Operator sub().'

    ty_checker = TypeChecker('Operator sub().')
    ty_checker.check(self, SeriesType)

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
        def _series_operator_sub_scalar_impl(self, other):
            result_data = self._data.astype(numpy.float64) - numpy.float64(other)
            return pandas.Series(result_data, index=self._index, name=self._name)

        return _series_operator_sub_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_sub_none_indexes_impl(self, other):

                if (len(self._data) == len(other._data)):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data - other._data.astype(numpy.float64)
                    return pandas.Series(result_data)
                else:
                    left_size, right_size = len(self._data), len(other._data)
                    min_data_size = min(left_size, right_size)
                    max_data_size = max(left_size, right_size)
                    result_data = numpy.empty(max_data_size, dtype=numpy.float64)
                    if (left_size == min_data_size):
                        result_data[:min_data_size] = self._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = result_data - other._data.astype(numpy.float64)
                    else:
                        result_data[:min_data_size] = other._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = self._data.astype(numpy.float64) - result_data

                    return pandas.Series(result_data, self._index)

            return _series_operator_sub_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_sub_common_impl(self, other):
                left_index, right_index = self.index, other.index

                # check if indexes are equal and series don't have to be aligned
                if sdc_check_indexes_equal(left_index, right_index):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data - other._data.astype(numpy.float64)

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

                result_data = left_values - right_values
                return pandas.Series(result_data, joined_index)

            return _series_operator_sub_common_impl

    return None


@overload(operator.mul)
def sdc_pandas_series_operator_mul(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.mul` implementation

    Note: Currently implemented for numeric Series only.
        Differs from Pandas in returning Series with fixed dtype :obj:`float64`

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op1*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op2*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_mul*

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

    _func_name = 'Operator mul().'

    ty_checker = TypeChecker('Operator mul().')
    ty_checker.check(self, SeriesType)

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
        def _series_operator_mul_scalar_impl(self, other):
            result_data = self._data.astype(numpy.float64) * numpy.float64(other)
            return pandas.Series(result_data, index=self._index, name=self._name)

        return _series_operator_mul_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_mul_none_indexes_impl(self, other):

                if (len(self._data) == len(other._data)):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data * other._data.astype(numpy.float64)
                    return pandas.Series(result_data)
                else:
                    left_size, right_size = len(self._data), len(other._data)
                    min_data_size = min(left_size, right_size)
                    max_data_size = max(left_size, right_size)
                    result_data = numpy.empty(max_data_size, dtype=numpy.float64)
                    if (left_size == min_data_size):
                        result_data[:min_data_size] = self._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = result_data * other._data.astype(numpy.float64)
                    else:
                        result_data[:min_data_size] = other._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = self._data.astype(numpy.float64) * result_data

                    return pandas.Series(result_data, self._index)

            return _series_operator_mul_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_mul_common_impl(self, other):
                left_index, right_index = self.index, other.index

                # check if indexes are equal and series don't have to be aligned
                if sdc_check_indexes_equal(left_index, right_index):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data * other._data.astype(numpy.float64)

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

                result_data = left_values * right_values
                return pandas.Series(result_data, joined_index)

            return _series_operator_mul_common_impl

    return None


@overload(operator.truediv)
def sdc_pandas_series_operator_truediv(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.truediv` implementation

    Note: Currently implemented for numeric Series only.
        Differs from Pandas in returning Series with fixed dtype :obj:`float64`

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op1*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op2*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_truediv*

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

    _func_name = 'Operator truediv().'

    ty_checker = TypeChecker('Operator truediv().')
    ty_checker.check(self, SeriesType)

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
        def _series_operator_truediv_scalar_impl(self, other):
            result_data = self._data.astype(numpy.float64) / numpy.float64(other)
            return pandas.Series(result_data, index=self._index, name=self._name)

        return _series_operator_truediv_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_truediv_none_indexes_impl(self, other):

                if (len(self._data) == len(other._data)):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data / other._data.astype(numpy.float64)
                    return pandas.Series(result_data)
                else:
                    left_size, right_size = len(self._data), len(other._data)
                    min_data_size = min(left_size, right_size)
                    max_data_size = max(left_size, right_size)
                    result_data = numpy.empty(max_data_size, dtype=numpy.float64)
                    if (left_size == min_data_size):
                        result_data[:min_data_size] = self._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = result_data / other._data.astype(numpy.float64)
                    else:
                        result_data[:min_data_size] = other._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = self._data.astype(numpy.float64) / result_data

                    return pandas.Series(result_data, self._index)

            return _series_operator_truediv_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_truediv_common_impl(self, other):
                left_index, right_index = self.index, other.index

                # check if indexes are equal and series don't have to be aligned
                if sdc_check_indexes_equal(left_index, right_index):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data / other._data.astype(numpy.float64)

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

                result_data = left_values / right_values
                return pandas.Series(result_data, joined_index)

            return _series_operator_truediv_common_impl

    return None


@overload(operator.floordiv)
def sdc_pandas_series_operator_floordiv(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.floordiv` implementation

    Note: Currently implemented for numeric Series only.
        Differs from Pandas in returning Series with fixed dtype :obj:`float64`

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op1*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op2*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_floordiv*

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

    _func_name = 'Operator floordiv().'

    ty_checker = TypeChecker('Operator floordiv().')
    ty_checker.check(self, SeriesType)

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
        def _series_operator_floordiv_scalar_impl(self, other):
            result_data = self._data.astype(numpy.float64) // numpy.float64(other)
            return pandas.Series(result_data, index=self._index, name=self._name)

        return _series_operator_floordiv_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_floordiv_none_indexes_impl(self, other):

                if (len(self._data) == len(other._data)):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data // other._data.astype(numpy.float64)
                    return pandas.Series(result_data)
                else:
                    left_size, right_size = len(self._data), len(other._data)
                    min_data_size = min(left_size, right_size)
                    max_data_size = max(left_size, right_size)
                    result_data = numpy.empty(max_data_size, dtype=numpy.float64)
                    if (left_size == min_data_size):
                        result_data[:min_data_size] = self._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = result_data // other._data.astype(numpy.float64)
                    else:
                        result_data[:min_data_size] = other._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = self._data.astype(numpy.float64) // result_data

                    return pandas.Series(result_data, self._index)

            return _series_operator_floordiv_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_floordiv_common_impl(self, other):
                left_index, right_index = self.index, other.index

                # check if indexes are equal and series don't have to be aligned
                if sdc_check_indexes_equal(left_index, right_index):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data // other._data.astype(numpy.float64)

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

                result_data = left_values // right_values
                return pandas.Series(result_data, joined_index)

            return _series_operator_floordiv_common_impl

    return None


@overload(operator.mod)
def sdc_pandas_series_operator_mod(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.mod` implementation

    Note: Currently implemented for numeric Series only.
        Differs from Pandas in returning Series with fixed dtype :obj:`float64`

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op1*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op2*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_mod*

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

    _func_name = 'Operator mod().'

    ty_checker = TypeChecker('Operator mod().')
    ty_checker.check(self, SeriesType)

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
        def _series_operator_mod_scalar_impl(self, other):
            result_data = self._data.astype(numpy.float64) % numpy.float64(other)
            return pandas.Series(result_data, index=self._index, name=self._name)

        return _series_operator_mod_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_mod_none_indexes_impl(self, other):

                if (len(self._data) == len(other._data)):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data % other._data.astype(numpy.float64)
                    return pandas.Series(result_data)
                else:
                    left_size, right_size = len(self._data), len(other._data)
                    min_data_size = min(left_size, right_size)
                    max_data_size = max(left_size, right_size)
                    result_data = numpy.empty(max_data_size, dtype=numpy.float64)
                    if (left_size == min_data_size):
                        result_data[:min_data_size] = self._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = result_data % other._data.astype(numpy.float64)
                    else:
                        result_data[:min_data_size] = other._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = self._data.astype(numpy.float64) % result_data

                    return pandas.Series(result_data, self._index)

            return _series_operator_mod_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_mod_common_impl(self, other):
                left_index, right_index = self.index, other.index

                # check if indexes are equal and series don't have to be aligned
                if sdc_check_indexes_equal(left_index, right_index):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data % other._data.astype(numpy.float64)

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

                result_data = left_values % right_values
                return pandas.Series(result_data, joined_index)

            return _series_operator_mod_common_impl

    return None


@overload(operator.pow)
def sdc_pandas_series_operator_pow(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.pow` implementation

    Note: Currently implemented for numeric Series only.
        Differs from Pandas in returning Series with fixed dtype :obj:`float64`

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op1*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op2*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_pow*

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

    _func_name = 'Operator pow().'

    ty_checker = TypeChecker('Operator pow().')
    ty_checker.check(self, SeriesType)

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
        def _series_operator_pow_scalar_impl(self, other):
            result_data = self._data.astype(numpy.float64) ** numpy.float64(other)
            return pandas.Series(result_data, index=self._index, name=self._name)

        return _series_operator_pow_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_pow_none_indexes_impl(self, other):

                if (len(self._data) == len(other._data)):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data ** other._data.astype(numpy.float64)
                    return pandas.Series(result_data)
                else:
                    left_size, right_size = len(self._data), len(other._data)
                    min_data_size = min(left_size, right_size)
                    max_data_size = max(left_size, right_size)
                    result_data = numpy.empty(max_data_size, dtype=numpy.float64)
                    if (left_size == min_data_size):
                        result_data[:min_data_size] = self._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = result_data ** other._data.astype(numpy.float64)
                    else:
                        result_data[:min_data_size] = other._data
                        result_data[min_data_size:] = numpy.nan
                        result_data = self._data.astype(numpy.float64) ** result_data

                    return pandas.Series(result_data, self._index)

            return _series_operator_pow_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_pow_common_impl(self, other):
                left_index, right_index = self.index, other.index

                # check if indexes are equal and series don't have to be aligned
                if sdc_check_indexes_equal(left_index, right_index):
                    result_data = self._data.astype(numpy.float64)
                    result_data = result_data ** other._data.astype(numpy.float64)

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

                result_data = left_values ** right_values
                return pandas.Series(result_data, joined_index)

            return _series_operator_pow_common_impl

    return None


@overload(operator.lt)
def sdc_pandas_series_operator_lt(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.lt` implementation

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op7*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_lt*

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

    _func_name = 'Operator lt().'

    ty_checker = TypeChecker('Operator lt().')
    ty_checker.check(self, SeriesType)

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

    # specializations for numeric series
    if (isinstance(other, types.Number)):
        def _series_operator_lt_scalar_impl(self, other):
            return pandas.Series(self._data < other, index=self._index, name=self._name)

        return _series_operator_lt_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_lt_none_indexes_impl(self, other):
                left_size, right_size = len(self._data), len(other._data)
                if (left_size == right_size):
                    return pandas.Series(self._data < other._data)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_lt_none_indexes_impl
        else:

            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_lt_common_impl(self, other):
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

            return _series_operator_lt_common_impl

    return None


@overload(operator.gt)
def sdc_pandas_series_operator_gt(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.gt` implementation

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op7*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_gt*

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

    _func_name = 'Operator gt().'

    ty_checker = TypeChecker('Operator gt().')
    ty_checker.check(self, SeriesType)

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

    # specializations for numeric series
    if (isinstance(other, types.Number)):
        def _series_operator_gt_scalar_impl(self, other):
            return pandas.Series(self._data > other, index=self._index, name=self._name)

        return _series_operator_gt_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_gt_none_indexes_impl(self, other):
                left_size, right_size = len(self._data), len(other._data)
                if (left_size == right_size):
                    return pandas.Series(self._data > other._data)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_gt_none_indexes_impl
        else:

            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_gt_common_impl(self, other):
                left_index, right_index = self.index, other.index

                if sdc_check_indexes_equal(left_index, right_index):
                    if none_or_numeric_indexes == True:  # noqa
                        new_index = left_index.astype(numba_index_common_dtype)
                    else:
                        new_index = self._index
                    return pandas.Series(self._data > other._data,
                                         new_index)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_gt_common_impl

    return None


@overload(operator.le)
def sdc_pandas_series_operator_le(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.le` implementation

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op7*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_le*

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

    _func_name = 'Operator le().'

    ty_checker = TypeChecker('Operator le().')
    ty_checker.check(self, SeriesType)

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

    # specializations for numeric series
    if (isinstance(other, types.Number)):
        def _series_operator_le_scalar_impl(self, other):
            return pandas.Series(self._data <= other, index=self._index, name=self._name)

        return _series_operator_le_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_le_none_indexes_impl(self, other):
                left_size, right_size = len(self._data), len(other._data)
                if (left_size == right_size):
                    return pandas.Series(self._data <= other._data)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_le_none_indexes_impl
        else:

            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_le_common_impl(self, other):
                left_index, right_index = self.index, other.index

                if sdc_check_indexes_equal(left_index, right_index):
                    if none_or_numeric_indexes == True:  # noqa
                        new_index = left_index.astype(numba_index_common_dtype)
                    else:
                        new_index = self._index
                    return pandas.Series(self._data <= other._data,
                                         new_index)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_le_common_impl

    return None


@overload(operator.ge)
def sdc_pandas_series_operator_ge(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.ge` implementation

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op7*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_ge*

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

    _func_name = 'Operator ge().'

    ty_checker = TypeChecker('Operator ge().')
    ty_checker.check(self, SeriesType)

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

    # specializations for numeric series
    if (isinstance(other, types.Number)):
        def _series_operator_ge_scalar_impl(self, other):
            return pandas.Series(self._data >= other, index=self._index, name=self._name)

        return _series_operator_ge_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_ge_none_indexes_impl(self, other):
                left_size, right_size = len(self._data), len(other._data)
                if (left_size == right_size):
                    return pandas.Series(self._data >= other._data)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_ge_none_indexes_impl
        else:

            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_ge_common_impl(self, other):
                left_index, right_index = self.index, other.index

                if sdc_check_indexes_equal(left_index, right_index):
                    if none_or_numeric_indexes == True:  # noqa
                        new_index = left_index.astype(numba_index_common_dtype)
                    else:
                        new_index = self._index
                    return pandas.Series(self._data >= other._data,
                                         new_index)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_ge_common_impl

    return None


@overload(operator.ne)
def sdc_pandas_series_operator_ne(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.ne` implementation

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op7*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_ne*

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

    _func_name = 'Operator ne().'

    ty_checker = TypeChecker('Operator ne().')
    ty_checker.check(self, SeriesType)

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

    # specializations for numeric series
    if (isinstance(other, types.Number)):
        def _series_operator_ne_scalar_impl(self, other):
            return pandas.Series(self._data != other, index=self._index, name=self._name)

        return _series_operator_ne_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_ne_none_indexes_impl(self, other):
                left_size, right_size = len(self._data), len(other._data)
                if (left_size == right_size):
                    return pandas.Series(self._data != other._data)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_ne_none_indexes_impl
        else:

            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_ne_common_impl(self, other):
                left_index, right_index = self.index, other.index

                if sdc_check_indexes_equal(left_index, right_index):
                    if none_or_numeric_indexes == True:  # noqa
                        new_index = left_index.astype(numba_index_common_dtype)
                    else:
                        new_index = self._index
                    return pandas.Series(self._data != other._data,
                                         new_index)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_ne_common_impl

    return None


@overload(operator.eq)
def sdc_pandas_series_operator_eq(self, other):
    """
    Pandas Series operator :attr:`pandas.Series.eq` implementation

    .. only:: developer

    **Test**: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op7*
              python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_operator_eq*

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

    _func_name = 'Operator eq().'

    ty_checker = TypeChecker('Operator eq().')
    ty_checker.check(self, SeriesType)

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

    # specializations for numeric series
    if (isinstance(other, types.Number)):
        def _series_operator_eq_scalar_impl(self, other):
            return pandas.Series(self._data == other, index=self._index, name=self._name)

        return _series_operator_eq_scalar_impl

    elif (isinstance(other, SeriesType)):

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_operator_eq_none_indexes_impl(self, other):
                left_size, right_size = len(self._data), len(other._data)
                if (left_size == right_size):
                    return pandas.Series(self._data == other._data)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_eq_none_indexes_impl
        else:

            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_operator_eq_common_impl(self, other):
                left_index, right_index = self.index, other.index

                if sdc_check_indexes_equal(left_index, right_index):
                    if none_or_numeric_indexes == True:  # noqa
                        new_index = left_index.astype(numba_index_common_dtype)
                    else:
                        new_index = self._index
                    return pandas.Series(self._data == other._data,
                                         new_index)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_operator_eq_common_impl

    return None

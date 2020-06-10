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

| This file contains function templates used by the auto-generation script

"""

# below imports are copied into the auto-generated source file as-is
# for the auto-generation script to work ensure they are not mixed up with code
import numba
import numpy
import operator
import pandas

from numba.core.errors import TypingError
from numba import types

from sdc.utilities.sdc_typing_utils import (TypeChecker, check_index_is_numeric, check_types_comparable,
                                            find_common_dtype_from_numpy_dtypes)
from sdc.datatypes.common_functions import (sdc_join_series_indexes, sdc_check_indexes_equal)
from sdc.hiframes.pd_series_type import SeriesType
from sdc.str_arr_ext import (string_array_type, str_arr_is_na)
from sdc.utilities.utils import sdc_overload, sdc_overload_method
from sdc.functions import numpy_like


def sdc_pandas_series_binop(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.binop

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_binop.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_binop

    .. command-output:: python ./series/series_binop.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.binop` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method binop().'
    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, (SeriesType, types.Number)):
        ty_checker.raise_exc(self, 'pandas.series or scalar', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or scalar', 'other')

    operands_are_series = self_is_series and other_is_series
    if operands_are_series:
        none_or_numeric_indexes = ((isinstance(self.index, types.NoneType) or check_index_is_numeric(self))
                                   and (isinstance(other.index, types.NoneType) or check_index_is_numeric(other)))
        series_indexes_comparable = check_types_comparable(self.index, other.index) or none_or_numeric_indexes
        if not series_indexes_comparable:
            raise TypingError('{} Not implemented for series with not-comparable indexes. \
            Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    series_data_comparable = check_types_comparable(self, other)
    if not series_data_comparable:
        raise TypingError('{} Not supported for not-comparable operands. \
        Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')
    # specializations for numeric series only
    if not operands_are_series:
        def _series_binop_scalar_impl(self, other, level=None, fill_value=None, axis=0):
            if self_is_series == True:  # noqa
                numpy_like.fillna(self._data, inplace=True, value=fill_value)
                result_data = numpy.empty(len(self._data), dtype=numpy.float64)
                result_data[:] = self._data + numpy.float64(other)
                return pandas.Series(result_data, index=self._index, name=self._name)
            else:
                numpy_like.fillna(other._data, inplace=True, value=fill_value)
                result_data = numpy.empty(len(other._data), dtype=numpy.float64)
                result_data[:] = numpy.float64(self) + other._data
                return pandas.Series(result_data, index=other._index, name=other._name)

        return _series_binop_scalar_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_binop_none_indexes_impl(self, other, level=None, fill_value=None, axis=0):
                numpy_like.fillna(self._data, inplace=True, value=fill_value)
                numpy_like.fillna(other._data, inplace=True, value=fill_value)

                if (len(self._data) == len(other._data)):
                    result_data = numpy_like.astype(self._data, numpy.float64)
                    result_data = result_data + other._data
                    return pandas.Series(result_data)
                else:
                    left_size, right_size = len(self._data), len(other._data)
                    min_data_size = min(left_size, right_size)
                    max_data_size = max(left_size, right_size)
                    result_data = numpy.empty(max_data_size, dtype=numpy.float64)
                    _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                    if (left_size == min_data_size):
                        result_data[:min_data_size] = self._data
                        for i in range(min_data_size, len(result_data)):
                            result_data[i] = _fill_value
                        result_data = result_data + other._data
                    else:
                        result_data[:min_data_size] = other._data
                        for i in range(min_data_size, len(result_data)):
                            result_data[i] = _fill_value
                        result_data = self._data + result_data

                    return pandas.Series(result_data)

            return _series_binop_none_indexes_impl
        else:
            # for numeric indexes find common dtype to be used when creating joined index
            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_binop_common_impl(self, other, level=None, fill_value=None, axis=0):
                left_index, right_index = self.index, other.index
                numpy_like.fillna(self._data, inplace=True, value=fill_value)
                numpy_like.fillna(other._data, inplace=True, value=fill_value)
                # check if indexes are equal and series don't have to be aligned
                if sdc_check_indexes_equal(left_index, right_index):
                    result_data = numpy.empty(len(self._data), dtype=numpy.float64)
                    result_data[:] = self._data + other._data

                    if none_or_numeric_indexes == True:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = self._index

                    return pandas.Series(result_data, index=result_index)

                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                left_values = numpy.empty(result_size, dtype=numpy.float64)
                right_values = numpy.empty(result_size, dtype=numpy.float64)
                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in range(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_values[i] = self._data[left_pos] if left_pos != -1 else _fill_value
                    right_values[i] = other._data[right_pos] if right_pos != -1 else _fill_value
                result_data = left_values + right_values
                return pandas.Series(result_data, joined_index)

            return _series_binop_common_impl

    return None


def sdc_pandas_series_comp_binop(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.comp_binop

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_comp_binop.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_comp_binop

    .. command-output:: python ./series/series_comp_binop.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.comp_binop` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method comp_binop().'

    ty_checker = TypeChecker(_func_name)
    ty_checker.check(self, SeriesType)

    if not (isinstance(level, types.Omitted) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not (isinstance(axis, types.Omitted) or axis == 0):
        ty_checker.raise_exc(axis, 'int', 'axis')

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    if not isinstance(self, (SeriesType, types.Number, types.UnicodeType)):
        ty_checker.raise_exc(self, 'pandas.series or scalar', 'self')

    if not isinstance(other, (SeriesType, types.Number, types.UnicodeType)):
        ty_checker.raise_exc(other, 'pandas.series or scalar', 'other')

    operands_are_series = self_is_series and other_is_series
    if operands_are_series:
        none_or_numeric_indexes = ((isinstance(self.index, types.NoneType) or check_index_is_numeric(self))
                                   and (isinstance(other.index, types.NoneType) or check_index_is_numeric(other)))
        series_indexes_comparable = check_types_comparable(self.index, other.index) or none_or_numeric_indexes
        if not series_indexes_comparable:
            raise TypingError('{} Not implemented for series with not-comparable indexes. \
            Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    series_data_comparable = check_types_comparable(self, other)
    if not series_data_comparable:
        raise TypingError('{} Not supported for not-comparable operands. \
        Given: self={}, other={}'.format(_func_name, self, other))

    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None
    if not operands_are_series:
        def _series_comp_binop_scalar_impl(self, other, level=None, fill_value=None, axis=0):
            if self_is_series == True:  # noqa
                numpy_like.fillna(self._data, inplace=True, value=fill_value)
                return pandas.Series(self._data < other, index=self._index, name=self._name)
            else:
                numpy_like.fillna(other._data, inplace=True, value=fill_value)
                return pandas.Series(self < other._data, index=other._index, name=other._name)

        return _series_comp_binop_scalar_impl

    else:

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_comp_binop_none_indexes_impl(self, other, level=None, fill_value=None, axis=0):
                numpy_like.fillna(self._data, inplace=True, value=fill_value)
                numpy_like.fillna(other._data, inplace=True, value=fill_value)
                left_size, right_size = len(self._data), len(other._data)
                if (left_size == right_size):
                    return pandas.Series(self._data < other._data)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_comp_binop_none_indexes_impl
        else:

            if none_or_numeric_indexes:
                ty_left_index_dtype = types.int64 if isinstance(self.index, types.NoneType) else self.index.dtype
                ty_right_index_dtype = types.int64 if isinstance(other.index, types.NoneType) else other.index.dtype
                numba_index_common_dtype = find_common_dtype_from_numpy_dtypes(
                    [ty_left_index_dtype, ty_right_index_dtype], [])

            def _series_comp_binop_common_impl(self, other, level=None, fill_value=None, axis=0):
                numpy_like.fillna(self._data, inplace=True, value=fill_value)
                numpy_like.fillna(other._data, inplace=True, value=fill_value)
                left_index, right_index = self.index, other.index

                if sdc_check_indexes_equal(left_index, right_index):
                    if none_or_numeric_indexes == True:  # noqa
                        new_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        new_index = self._index
                    return pandas.Series(self._data < other._data,
                                         new_index)
                else:
                    raise ValueError("Can only compare identically-labeled Series objects")

            return _series_comp_binop_common_impl

    return None


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
    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, (SeriesType, types.Number)):
        ty_checker.raise_exc(self, 'pandas.series or scalar', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or scalar', 'other')

    operands_are_series = self_is_series and other_is_series
    if operands_are_series:
        none_or_numeric_indexes = ((isinstance(self.index, types.NoneType) or check_index_is_numeric(self))
                                   and (isinstance(other.index, types.NoneType) or check_index_is_numeric(other)))
        series_indexes_comparable = check_types_comparable(self.index, other.index) or none_or_numeric_indexes
        if not series_indexes_comparable:
            raise TypingError('{} Not implemented for series with not-comparable indexes. \
            Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    series_data_comparable = check_types_comparable(self, other)
    if not series_data_comparable:
        raise TypingError('{} Not supported for not-comparable operands. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def sdc_pandas_series_operator_binop_impl(self, other):
        return self.binop(other)

    return sdc_pandas_series_operator_binop_impl


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
    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    if not isinstance(self, (SeriesType, types.Number, types.UnicodeType)):
        ty_checker.raise_exc(self, 'pandas.series or scalar', 'self')

    if not isinstance(other, (SeriesType, types.Number, types.UnicodeType)):
        ty_checker.raise_exc(other, 'pandas.series or scalar', 'other')

    operands_are_series = self_is_series and other_is_series
    if operands_are_series:
        none_or_numeric_indexes = ((isinstance(self.index, types.NoneType) or check_index_is_numeric(self))
                                   and (isinstance(other.index, types.NoneType) or check_index_is_numeric(other)))
        series_indexes_comparable = check_types_comparable(self.index, other.index) or none_or_numeric_indexes
        if not series_indexes_comparable:
            raise TypingError('{} Not implemented for series with not-comparable indexes. \
            Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    series_data_comparable = check_types_comparable(self, other)
    if not series_data_comparable:
        raise TypingError('{} Not supported for not-comparable operands. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def sdc_pandas_series_operator_comp_binop_impl(self, other):
        return self.comp_binop(other)

    return sdc_pandas_series_operator_comp_binop_impl


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

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

| This file contains overloads for various extension types auto-generated with autogen_sources.py

"""

import numba
import numpy
import operator
import pandas

from numba.core.errors import TypingError
from numba import types

from sdc.utilities.sdc_typing_utils import (TypeChecker, check_index_is_numeric, check_types_comparable,
                                            find_common_dtype_from_numpy_dtypes, find_index_common_dtype)
from sdc.datatypes.common_functions import (sdc_join_series_indexes, )
from sdc.hiframes.api import isna
from sdc.hiframes.pd_series_type import SeriesType
from sdc.str_arr_ext import (string_array_type, str_arr_is_na)
from sdc.utilities.utils import sdc_overload, sdc_overload_method
from sdc.functions import numpy_like
from sdc.datatypes.range_index_type import RangeIndexType


def sdc_add(self, other, fill_value=None):
    pass


@sdc_overload(sdc_add)
def sdc_add_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_add_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self + _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_add_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def sdc_add_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                max_data_size = max(left_size, right_size)
                result_data = numpy.empty(max_data_size, dtype=numpy.float64)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in numba.prange(max_data_size):
                    left_nan = (i >= left_size or numpy.isnan(self._data[i]))
                    right_nan = (i >= right_size or numpy.isnan(other._data[i]))
                    _left = _fill_value if left_nan else self._data[i]
                    _right = _fill_value if right_nan else other._data[i]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left + _right

                return pandas.Series(result_data)

            return sdc_add_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def sdc_add_impl(self, other, fill_value=None):

                # check if indexes are equal and series don't have to be aligned
                left_index, right_index = self.index, other.index
                if (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):

                    _left = pandas.Series(self._data)
                    _right = pandas.Series(other._data)
                    partial_res = _left.add(_right, fill_value=fill_value)

                    if index_dtypes_match == False:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                    return pandas.Series(partial_res._data, index=result_index)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                result_data = numpy.empty(result_size, dtype=numpy.float64)
                for i in numba.prange(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                    right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                    _left = _fill_value if left_nan else self._data[left_pos]
                    _right = _fill_value if right_nan else other._data[right_pos]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left + _right

                return pandas.Series(result_data, index=joined_index)

            return sdc_add_impl


@sdc_overload_method(SeriesType, 'add')
def sdc_pandas_series_add(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.add

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_add.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_add

    .. command-output:: python ./series/series_add.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.add` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method add().'

    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

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
        raise TypingError('{} Not supported for not-comparable operands. '
                          'Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    # specialization for numeric series only
    def series_add_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_add(self, other, fill_value)

    return series_add_wrapper


def sdc_div(self, other, fill_value=None):
    pass


@sdc_overload(sdc_div)
def sdc_div_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_div_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self / _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_div_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def sdc_div_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                max_data_size = max(left_size, right_size)
                result_data = numpy.empty(max_data_size, dtype=numpy.float64)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in numba.prange(max_data_size):
                    left_nan = (i >= left_size or numpy.isnan(self._data[i]))
                    right_nan = (i >= right_size or numpy.isnan(other._data[i]))
                    _left = _fill_value if left_nan else self._data[i]
                    _right = _fill_value if right_nan else other._data[i]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left / _right

                return pandas.Series(result_data)

            return sdc_div_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def sdc_div_impl(self, other, fill_value=None):

                # check if indexes are equal and series don't have to be aligned
                left_index, right_index = self.index, other.index
                if (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):

                    _left = pandas.Series(self._data)
                    _right = pandas.Series(other._data)
                    partial_res = _left.div(_right, fill_value=fill_value)

                    if index_dtypes_match == False:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                    return pandas.Series(partial_res._data, index=result_index)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                result_data = numpy.empty(result_size, dtype=numpy.float64)
                for i in numba.prange(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                    right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                    _left = _fill_value if left_nan else self._data[left_pos]
                    _right = _fill_value if right_nan else other._data[right_pos]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left / _right

                return pandas.Series(result_data, index=joined_index)

            return sdc_div_impl


@sdc_overload_method(SeriesType, 'div')
def sdc_pandas_series_div(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.div

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_div.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_div

    .. command-output:: python ./series/series_div.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.div` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method div().'

    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

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
        raise TypingError('{} Not supported for not-comparable operands. '
                          'Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    # specialization for numeric series only
    def series_div_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_div(self, other, fill_value)

    return series_div_wrapper


def sdc_sub(self, other, fill_value=None):
    pass


@sdc_overload(sdc_sub)
def sdc_sub_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_sub_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self - _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_sub_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def sdc_sub_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                max_data_size = max(left_size, right_size)
                result_data = numpy.empty(max_data_size, dtype=numpy.float64)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in numba.prange(max_data_size):
                    left_nan = (i >= left_size or numpy.isnan(self._data[i]))
                    right_nan = (i >= right_size or numpy.isnan(other._data[i]))
                    _left = _fill_value if left_nan else self._data[i]
                    _right = _fill_value if right_nan else other._data[i]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left - _right

                return pandas.Series(result_data)

            return sdc_sub_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def sdc_sub_impl(self, other, fill_value=None):

                # check if indexes are equal and series don't have to be aligned
                left_index, right_index = self.index, other.index
                if (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):

                    _left = pandas.Series(self._data)
                    _right = pandas.Series(other._data)
                    partial_res = _left.sub(_right, fill_value=fill_value)

                    if index_dtypes_match == False:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                    return pandas.Series(partial_res._data, index=result_index)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                result_data = numpy.empty(result_size, dtype=numpy.float64)
                for i in numba.prange(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                    right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                    _left = _fill_value if left_nan else self._data[left_pos]
                    _right = _fill_value if right_nan else other._data[right_pos]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left - _right

                return pandas.Series(result_data, index=joined_index)

            return sdc_sub_impl


@sdc_overload_method(SeriesType, 'sub')
def sdc_pandas_series_sub(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.sub

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_sub.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_sub

    .. command-output:: python ./series/series_sub.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.sub` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method sub().'

    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

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
        raise TypingError('{} Not supported for not-comparable operands. '
                          'Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    # specialization for numeric series only
    def series_sub_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_sub(self, other, fill_value)

    return series_sub_wrapper


def sdc_mul(self, other, fill_value=None):
    pass


@sdc_overload(sdc_mul)
def sdc_mul_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_mul_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self * _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_mul_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def sdc_mul_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                max_data_size = max(left_size, right_size)
                result_data = numpy.empty(max_data_size, dtype=numpy.float64)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in numba.prange(max_data_size):
                    left_nan = (i >= left_size or numpy.isnan(self._data[i]))
                    right_nan = (i >= right_size or numpy.isnan(other._data[i]))
                    _left = _fill_value if left_nan else self._data[i]
                    _right = _fill_value if right_nan else other._data[i]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left * _right

                return pandas.Series(result_data)

            return sdc_mul_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def sdc_mul_impl(self, other, fill_value=None):

                # check if indexes are equal and series don't have to be aligned
                left_index, right_index = self.index, other.index
                if (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):

                    _left = pandas.Series(self._data)
                    _right = pandas.Series(other._data)
                    partial_res = _left.mul(_right, fill_value=fill_value)

                    if index_dtypes_match == False:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                    return pandas.Series(partial_res._data, index=result_index)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                result_data = numpy.empty(result_size, dtype=numpy.float64)
                for i in numba.prange(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                    right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                    _left = _fill_value if left_nan else self._data[left_pos]
                    _right = _fill_value if right_nan else other._data[right_pos]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left * _right

                return pandas.Series(result_data, index=joined_index)

            return sdc_mul_impl


@sdc_overload_method(SeriesType, 'mul')
def sdc_pandas_series_mul(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.mul

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_mul.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_mul

    .. command-output:: python ./series/series_mul.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.mul` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method mul().'

    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

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
        raise TypingError('{} Not supported for not-comparable operands. '
                          'Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    # specialization for numeric series only
    def series_mul_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_mul(self, other, fill_value)

    return series_mul_wrapper


def sdc_truediv(self, other, fill_value=None):
    pass


@sdc_overload(sdc_truediv)
def sdc_truediv_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_truediv_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self / _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_truediv_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def sdc_truediv_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                max_data_size = max(left_size, right_size)
                result_data = numpy.empty(max_data_size, dtype=numpy.float64)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in numba.prange(max_data_size):
                    left_nan = (i >= left_size or numpy.isnan(self._data[i]))
                    right_nan = (i >= right_size or numpy.isnan(other._data[i]))
                    _left = _fill_value if left_nan else self._data[i]
                    _right = _fill_value if right_nan else other._data[i]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left / _right

                return pandas.Series(result_data)

            return sdc_truediv_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def sdc_truediv_impl(self, other, fill_value=None):

                # check if indexes are equal and series don't have to be aligned
                left_index, right_index = self.index, other.index
                if (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):

                    _left = pandas.Series(self._data)
                    _right = pandas.Series(other._data)
                    partial_res = _left.truediv(_right, fill_value=fill_value)

                    if index_dtypes_match == False:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                    return pandas.Series(partial_res._data, index=result_index)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                result_data = numpy.empty(result_size, dtype=numpy.float64)
                for i in numba.prange(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                    right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                    _left = _fill_value if left_nan else self._data[left_pos]
                    _right = _fill_value if right_nan else other._data[right_pos]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left / _right

                return pandas.Series(result_data, index=joined_index)

            return sdc_truediv_impl


@sdc_overload_method(SeriesType, 'truediv')
def sdc_pandas_series_truediv(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.truediv

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_truediv.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_truediv

    .. command-output:: python ./series/series_truediv.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.truediv` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method truediv().'

    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

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
        raise TypingError('{} Not supported for not-comparable operands. '
                          'Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    # specialization for numeric series only
    def series_truediv_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_truediv(self, other, fill_value)

    return series_truediv_wrapper


def sdc_floordiv(self, other, fill_value=None):
    pass


@sdc_overload(sdc_floordiv)
def sdc_floordiv_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_floordiv_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self // _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_floordiv_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def sdc_floordiv_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                max_data_size = max(left_size, right_size)
                result_data = numpy.empty(max_data_size, dtype=numpy.float64)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in numba.prange(max_data_size):
                    left_nan = (i >= left_size or numpy.isnan(self._data[i]))
                    right_nan = (i >= right_size or numpy.isnan(other._data[i]))
                    _left = _fill_value if left_nan else self._data[i]
                    _right = _fill_value if right_nan else other._data[i]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left // _right

                return pandas.Series(result_data)

            return sdc_floordiv_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def sdc_floordiv_impl(self, other, fill_value=None):

                # check if indexes are equal and series don't have to be aligned
                left_index, right_index = self.index, other.index
                if (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):

                    _left = pandas.Series(self._data)
                    _right = pandas.Series(other._data)
                    partial_res = _left.floordiv(_right, fill_value=fill_value)

                    if index_dtypes_match == False:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                    return pandas.Series(partial_res._data, index=result_index)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                result_data = numpy.empty(result_size, dtype=numpy.float64)
                for i in numba.prange(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                    right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                    _left = _fill_value if left_nan else self._data[left_pos]
                    _right = _fill_value if right_nan else other._data[right_pos]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left // _right

                return pandas.Series(result_data, index=joined_index)

            return sdc_floordiv_impl


@sdc_overload_method(SeriesType, 'floordiv')
def sdc_pandas_series_floordiv(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.floordiv

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_floordiv.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_floordiv

    .. command-output:: python ./series/series_floordiv.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.floordiv` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method floordiv().'

    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

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
        raise TypingError('{} Not supported for not-comparable operands. '
                          'Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    # specialization for numeric series only
    def series_floordiv_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_floordiv(self, other, fill_value)

    return series_floordiv_wrapper


def sdc_mod(self, other, fill_value=None):
    pass


@sdc_overload(sdc_mod)
def sdc_mod_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_mod_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self % _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_mod_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def sdc_mod_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                max_data_size = max(left_size, right_size)
                result_data = numpy.empty(max_data_size, dtype=numpy.float64)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in numba.prange(max_data_size):
                    left_nan = (i >= left_size or numpy.isnan(self._data[i]))
                    right_nan = (i >= right_size or numpy.isnan(other._data[i]))
                    _left = _fill_value if left_nan else self._data[i]
                    _right = _fill_value if right_nan else other._data[i]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left % _right

                return pandas.Series(result_data)

            return sdc_mod_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def sdc_mod_impl(self, other, fill_value=None):

                # check if indexes are equal and series don't have to be aligned
                left_index, right_index = self.index, other.index
                if (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):

                    _left = pandas.Series(self._data)
                    _right = pandas.Series(other._data)
                    partial_res = _left.mod(_right, fill_value=fill_value)

                    if index_dtypes_match == False:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                    return pandas.Series(partial_res._data, index=result_index)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                result_data = numpy.empty(result_size, dtype=numpy.float64)
                for i in numba.prange(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                    right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                    _left = _fill_value if left_nan else self._data[left_pos]
                    _right = _fill_value if right_nan else other._data[right_pos]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left % _right

                return pandas.Series(result_data, index=joined_index)

            return sdc_mod_impl


@sdc_overload_method(SeriesType, 'mod')
def sdc_pandas_series_mod(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.mod

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_mod.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_mod

    .. command-output:: python ./series/series_mod.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.mod` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method mod().'

    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

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
        raise TypingError('{} Not supported for not-comparable operands. '
                          'Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    # specialization for numeric series only
    def series_mod_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_mod(self, other, fill_value)

    return series_mod_wrapper


def sdc_pow(self, other, fill_value=None):
    pass


@sdc_overload(sdc_pow)
def sdc_pow_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_pow_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self ** _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_pow_impl

    else:   # both operands are numeric series
        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def sdc_pow_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                max_data_size = max(left_size, right_size)
                result_data = numpy.empty(max_data_size, dtype=numpy.float64)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                for i in numba.prange(max_data_size):
                    left_nan = (i >= left_size or numpy.isnan(self._data[i]))
                    right_nan = (i >= right_size or numpy.isnan(other._data[i]))
                    _left = _fill_value if left_nan else self._data[i]
                    _right = _fill_value if right_nan else other._data[i]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left ** _right

                return pandas.Series(result_data)

            return sdc_pow_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def sdc_pow_impl(self, other, fill_value=None):

                # check if indexes are equal and series don't have to be aligned
                left_index, right_index = self.index, other.index
                if (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):

                    _left = pandas.Series(self._data)
                    _right = pandas.Series(other._data)
                    partial_res = _left.pow(_right, fill_value=fill_value)

                    if index_dtypes_match == False:  # noqa
                        result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                    else:
                        result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                    return pandas.Series(partial_res._data, index=result_index)

                _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
                # TODO: replace below with core join(how='outer', return_indexers=True) when implemented
                joined_index, left_indexer, right_indexer = sdc_join_series_indexes(left_index, right_index)
                result_size = len(joined_index)
                result_data = numpy.empty(result_size, dtype=numpy.float64)
                for i in numba.prange(result_size):
                    left_pos, right_pos = left_indexer[i], right_indexer[i]
                    left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                    right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                    _left = _fill_value if left_nan else self._data[left_pos]
                    _right = _fill_value if right_nan else other._data[right_pos]
                    result_data[i] = numpy.nan if (left_nan and right_nan) else _left ** _right

                return pandas.Series(result_data, index=joined_index)

            return sdc_pow_impl


@sdc_overload_method(SeriesType, 'pow')
def sdc_pandas_series_pow(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.pow

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_pow.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_pow

    .. command-output:: python ./series/series_pow.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.pow` implementation.

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method pow().'

    ty_checker = TypeChecker(_func_name)
    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    # this overload is not for string series
    self_is_string_series = self_is_series and isinstance(self.dtype, types.UnicodeType)
    other_is_string_series = other_is_series and isinstance(other.dtype, types.UnicodeType)
    if self_is_string_series or other_is_string_series:
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

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
        raise TypingError('{} Not supported for not-comparable operands. '
                          'Given: self={}, other={}'.format(_func_name, self, other))

    if not isinstance(level, types.Omitted) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        ty_checker.raise_exc(fill_value, 'number', 'fill_value')

    if not isinstance(axis, types.Omitted) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    # specialization for numeric series only
    def series_pow_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_pow(self, other, fill_value)

    return series_pow_wrapper


def sdc_lt(self, other, fill_value=None):
    pass


@sdc_overload(sdc_lt)
def sdc_lt_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    if not operands_are_series:
        def _series_lt_scalar_impl(self, other, fill_value=None):

            _self = numpy_like.fillna(self._data, inplace=False, value=fill_value)
            return pandas.Series(_self < other, index=self._index, name=self._name)

        return _series_lt_scalar_impl

    else:

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_lt_none_indexes_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                if left_size != right_size:
                    raise ValueError("Can only compare identically-labeled Series objects")

                if fill_value_is_none == True:  # noqa
                    result_data = self._data < other._data
                else:
                    result_data = numpy.empty(left_size, dtype=types.bool_)
                    for i in numba.prange(left_size):
                        left_nan = isna(self._data, i)
                        right_nan = isna(other._data, i)
                        _left = fill_value if left_nan else self._data[i]
                        _right = fill_value if right_nan else other._data[i]
                        result_data[i] = False if (left_nan and right_nan) else _left < _right

                return pandas.Series(result_data)

            return _series_lt_none_indexes_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def _series_lt_common_impl(self, other, fill_value=None):

                left_index, right_index = self.index, other.index
                if not (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):
                    raise ValueError("Can only compare identically-labeled Series objects")

                _left, _right = pandas.Series(self._data), pandas.Series(other._data)
                partial_res = _left.lt(_right, fill_value=fill_value)

                if index_dtypes_match == False:  # noqa
                    result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                else:
                    result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                return pandas.Series(partial_res._data, result_index)

            return _series_lt_common_impl

    return None


@sdc_overload_method(SeriesType, 'lt')
def sdc_pandas_series_lt(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.lt

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_lt.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_lt

    .. command-output:: python ./series/series_lt.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.lt` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method lt().'

    ty_checker = TypeChecker(_func_name)
    ty_checker.check(self, SeriesType)

    if not (isinstance(level, types.Omitted) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not (isinstance(fill_value, (types.Omitted, types.Number, types.UnicodeType, types.NoneType))
            or fill_value is None):
        ty_checker.raise_exc(fill_value, 'scalar', 'fill_value')

    if not (isinstance(axis, types.Omitted) or axis == 0):
        ty_checker.raise_exc(axis, 'int', 'axis')

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

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

    # specializations for both numeric and string series
    def series_lt_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_lt(self, other, fill_value)

    return series_lt_wrapper


def sdc_gt(self, other, fill_value=None):
    pass


@sdc_overload(sdc_gt)
def sdc_gt_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    if not operands_are_series:
        def _series_gt_scalar_impl(self, other, fill_value=None):

            _self = numpy_like.fillna(self._data, inplace=False, value=fill_value)
            return pandas.Series(_self > other, index=self._index, name=self._name)

        return _series_gt_scalar_impl

    else:

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_gt_none_indexes_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                if left_size != right_size:
                    raise ValueError("Can only compare identically-labeled Series objects")

                if fill_value_is_none == True:  # noqa
                    result_data = self._data > other._data
                else:
                    result_data = numpy.empty(left_size, dtype=types.bool_)
                    for i in numba.prange(left_size):
                        left_nan = isna(self._data, i)
                        right_nan = isna(other._data, i)
                        _left = fill_value if left_nan else self._data[i]
                        _right = fill_value if right_nan else other._data[i]
                        result_data[i] = False if (left_nan and right_nan) else _left > _right

                return pandas.Series(result_data)

            return _series_gt_none_indexes_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def _series_gt_common_impl(self, other, fill_value=None):

                left_index, right_index = self.index, other.index
                if not (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):
                    raise ValueError("Can only compare identically-labeled Series objects")

                _left, _right = pandas.Series(self._data), pandas.Series(other._data)
                partial_res = _left.gt(_right, fill_value=fill_value)

                if index_dtypes_match == False:  # noqa
                    result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                else:
                    result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                return pandas.Series(partial_res._data, result_index)

            return _series_gt_common_impl

    return None


@sdc_overload_method(SeriesType, 'gt')
def sdc_pandas_series_gt(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.gt

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_gt.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_gt

    .. command-output:: python ./series/series_gt.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.gt` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method gt().'

    ty_checker = TypeChecker(_func_name)
    ty_checker.check(self, SeriesType)

    if not (isinstance(level, types.Omitted) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not (isinstance(fill_value, (types.Omitted, types.Number, types.UnicodeType, types.NoneType))
            or fill_value is None):
        ty_checker.raise_exc(fill_value, 'scalar', 'fill_value')

    if not (isinstance(axis, types.Omitted) or axis == 0):
        ty_checker.raise_exc(axis, 'int', 'axis')

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

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

    # specializations for both numeric and string series
    def series_gt_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_gt(self, other, fill_value)

    return series_gt_wrapper


def sdc_le(self, other, fill_value=None):
    pass


@sdc_overload(sdc_le)
def sdc_le_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    if not operands_are_series:
        def _series_le_scalar_impl(self, other, fill_value=None):

            _self = numpy_like.fillna(self._data, inplace=False, value=fill_value)
            return pandas.Series(_self <= other, index=self._index, name=self._name)

        return _series_le_scalar_impl

    else:

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_le_none_indexes_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                if left_size != right_size:
                    raise ValueError("Can only compare identically-labeled Series objects")

                if fill_value_is_none == True:  # noqa
                    result_data = self._data <= other._data
                else:
                    result_data = numpy.empty(left_size, dtype=types.bool_)
                    for i in numba.prange(left_size):
                        left_nan = isna(self._data, i)
                        right_nan = isna(other._data, i)
                        _left = fill_value if left_nan else self._data[i]
                        _right = fill_value if right_nan else other._data[i]
                        result_data[i] = False if (left_nan and right_nan) else _left <= _right

                return pandas.Series(result_data)

            return _series_le_none_indexes_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def _series_le_common_impl(self, other, fill_value=None):

                left_index, right_index = self.index, other.index
                if not (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):
                    raise ValueError("Can only compare identically-labeled Series objects")

                _left, _right = pandas.Series(self._data), pandas.Series(other._data)
                partial_res = _left.le(_right, fill_value=fill_value)

                if index_dtypes_match == False:  # noqa
                    result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                else:
                    result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                return pandas.Series(partial_res._data, result_index)

            return _series_le_common_impl

    return None


@sdc_overload_method(SeriesType, 'le')
def sdc_pandas_series_le(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.le

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_le.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_le

    .. command-output:: python ./series/series_le.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.le` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method le().'

    ty_checker = TypeChecker(_func_name)
    ty_checker.check(self, SeriesType)

    if not (isinstance(level, types.Omitted) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not (isinstance(fill_value, (types.Omitted, types.Number, types.UnicodeType, types.NoneType))
            or fill_value is None):
        ty_checker.raise_exc(fill_value, 'scalar', 'fill_value')

    if not (isinstance(axis, types.Omitted) or axis == 0):
        ty_checker.raise_exc(axis, 'int', 'axis')

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

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

    # specializations for both numeric and string series
    def series_le_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_le(self, other, fill_value)

    return series_le_wrapper


def sdc_ge(self, other, fill_value=None):
    pass


@sdc_overload(sdc_ge)
def sdc_ge_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    if not operands_are_series:
        def _series_ge_scalar_impl(self, other, fill_value=None):

            _self = numpy_like.fillna(self._data, inplace=False, value=fill_value)
            return pandas.Series(_self >= other, index=self._index, name=self._name)

        return _series_ge_scalar_impl

    else:

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_ge_none_indexes_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                if left_size != right_size:
                    raise ValueError("Can only compare identically-labeled Series objects")

                if fill_value_is_none == True:  # noqa
                    result_data = self._data >= other._data
                else:
                    result_data = numpy.empty(left_size, dtype=types.bool_)
                    for i in numba.prange(left_size):
                        left_nan = isna(self._data, i)
                        right_nan = isna(other._data, i)
                        _left = fill_value if left_nan else self._data[i]
                        _right = fill_value if right_nan else other._data[i]
                        result_data[i] = False if (left_nan and right_nan) else _left >= _right

                return pandas.Series(result_data)

            return _series_ge_none_indexes_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def _series_ge_common_impl(self, other, fill_value=None):

                left_index, right_index = self.index, other.index
                if not (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):
                    raise ValueError("Can only compare identically-labeled Series objects")

                _left, _right = pandas.Series(self._data), pandas.Series(other._data)
                partial_res = _left.ge(_right, fill_value=fill_value)

                if index_dtypes_match == False:  # noqa
                    result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                else:
                    result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                return pandas.Series(partial_res._data, result_index)

            return _series_ge_common_impl

    return None


@sdc_overload_method(SeriesType, 'ge')
def sdc_pandas_series_ge(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.ge

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_ge.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_ge

    .. command-output:: python ./series/series_ge.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.ge` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method ge().'

    ty_checker = TypeChecker(_func_name)
    ty_checker.check(self, SeriesType)

    if not (isinstance(level, types.Omitted) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not (isinstance(fill_value, (types.Omitted, types.Number, types.UnicodeType, types.NoneType))
            or fill_value is None):
        ty_checker.raise_exc(fill_value, 'scalar', 'fill_value')

    if not (isinstance(axis, types.Omitted) or axis == 0):
        ty_checker.raise_exc(axis, 'int', 'axis')

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

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

    # specializations for both numeric and string series
    def series_ge_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_ge(self, other, fill_value)

    return series_ge_wrapper


def sdc_ne(self, other, fill_value=None):
    pass


@sdc_overload(sdc_ne)
def sdc_ne_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    if not operands_are_series:
        def _series_ne_scalar_impl(self, other, fill_value=None):

            _self = numpy_like.fillna(self._data, inplace=False, value=fill_value)
            return pandas.Series(_self != other, index=self._index, name=self._name)

        return _series_ne_scalar_impl

    else:

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_ne_none_indexes_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                if left_size != right_size:
                    raise ValueError("Can only compare identically-labeled Series objects")

                if fill_value_is_none == True:  # noqa
                    result_data = self._data != other._data
                else:
                    result_data = numpy.empty(left_size, dtype=types.bool_)
                    for i in numba.prange(left_size):
                        left_nan = isna(self._data, i)
                        right_nan = isna(other._data, i)
                        _left = fill_value if left_nan else self._data[i]
                        _right = fill_value if right_nan else other._data[i]
                        result_data[i] = False if (left_nan and right_nan) else _left != _right

                return pandas.Series(result_data)

            return _series_ne_none_indexes_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def _series_ne_common_impl(self, other, fill_value=None):

                left_index, right_index = self.index, other.index
                if not (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):
                    raise ValueError("Can only compare identically-labeled Series objects")

                _left, _right = pandas.Series(self._data), pandas.Series(other._data)
                partial_res = _left.ne(_right, fill_value=fill_value)

                if index_dtypes_match == False:  # noqa
                    result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                else:
                    result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                return pandas.Series(partial_res._data, result_index)

            return _series_ne_common_impl

    return None


@sdc_overload_method(SeriesType, 'ne')
def sdc_pandas_series_ne(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.ne

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_ne.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_ne

    .. command-output:: python ./series/series_ne.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.ne` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method ne().'

    ty_checker = TypeChecker(_func_name)
    ty_checker.check(self, SeriesType)

    if not (isinstance(level, types.Omitted) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not (isinstance(fill_value, (types.Omitted, types.Number, types.UnicodeType, types.NoneType))
            or fill_value is None):
        ty_checker.raise_exc(fill_value, 'scalar', 'fill_value')

    if not (isinstance(axis, types.Omitted) or axis == 0):
        ty_checker.raise_exc(axis, 'int', 'axis')

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

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

    # specializations for both numeric and string series
    def series_ne_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_ne(self, other, fill_value)

    return series_ne_wrapper


def sdc_eq(self, other, fill_value=None):
    pass


@sdc_overload(sdc_eq)
def sdc_eq_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    if not operands_are_series:
        def _series_eq_scalar_impl(self, other, fill_value=None):

            _self = numpy_like.fillna(self._data, inplace=False, value=fill_value)
            return pandas.Series(_self == other, index=self._index, name=self._name)

        return _series_eq_scalar_impl

    else:

        # optimization for series with default indexes, that can be aligned differently
        if (isinstance(self.index, types.NoneType) and isinstance(other.index, types.NoneType)):
            def _series_eq_none_indexes_impl(self, other, fill_value=None):

                left_size, right_size = len(self._data), len(other._data)
                if left_size != right_size:
                    raise ValueError("Can only compare identically-labeled Series objects")

                if fill_value_is_none == True:  # noqa
                    result_data = self._data == other._data
                else:
                    result_data = numpy.empty(left_size, dtype=types.bool_)
                    for i in numba.prange(left_size):
                        left_nan = isna(self._data, i)
                        right_nan = isna(other._data, i)
                        _left = fill_value if left_nan else self._data[i]
                        _right = fill_value if right_nan else other._data[i]
                        result_data[i] = False if (left_nan and right_nan) else _left == _right

                return pandas.Series(result_data)

            return _series_eq_none_indexes_impl
        else:
            left_index_is_range = isinstance(self.index, (RangeIndexType, types.NoneType))
            index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(self, other)

            def _series_eq_common_impl(self, other, fill_value=None):

                left_index, right_index = self.index, other.index
                if not (left_index is right_index
                        or numpy_like.array_equal(left_index, right_index)):
                    raise ValueError("Can only compare identically-labeled Series objects")

                _left, _right = pandas.Series(self._data), pandas.Series(other._data)
                partial_res = _left.eq(_right, fill_value=fill_value)

                if index_dtypes_match == False:  # noqa
                    result_index = numpy_like.astype(left_index, numba_index_common_dtype)
                else:
                    result_index = left_index.values if left_index_is_range == True else left_index  # noqa

                return pandas.Series(partial_res._data, result_index)

            return _series_eq_common_impl

    return None


@sdc_overload_method(SeriesType, 'eq')
def sdc_pandas_series_eq(self, other, level=None, fill_value=None, axis=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.eq

    Limitations
    -----------
    Parameters ``level`` and ``axis`` are currently unsupported by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_eq.py
       :language: python
       :lines: 27-
       :caption:
       :name: ex_series_eq

    .. command-output:: python ./series/series_eq.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.eq` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method eq().'

    ty_checker = TypeChecker(_func_name)
    ty_checker.check(self, SeriesType)

    if not (isinstance(level, types.Omitted) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not (isinstance(fill_value, (types.Omitted, types.Number, types.UnicodeType, types.NoneType))
            or fill_value is None):
        ty_checker.raise_exc(fill_value, 'scalar', 'fill_value')

    if not (isinstance(axis, types.Omitted) or axis == 0):
        ty_checker.raise_exc(axis, 'int', 'axis')

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    if not (self_is_series or other_is_series):
        return None

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

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

    # specializations for both numeric and string series
    def series_eq_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_eq(self, other, fill_value)

    return series_eq_wrapper


@sdc_overload(operator.add)
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

    _func_name = 'Method comp_add().'
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

    def series_operator_add_wrapper(self, other):
        return sdc_add(self, other)

    return series_operator_add_wrapper


@sdc_overload(operator.sub)
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

    _func_name = 'Method comp_sub().'
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

    def series_operator_sub_wrapper(self, other):
        return sdc_sub(self, other)

    return series_operator_sub_wrapper


@sdc_overload(operator.mul)
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

    _func_name = 'Method comp_mul().'
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

    def series_operator_mul_wrapper(self, other):
        return sdc_mul(self, other)

    return series_operator_mul_wrapper


@sdc_overload(operator.truediv)
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

    _func_name = 'Method comp_truediv().'
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

    def series_operator_truediv_wrapper(self, other):
        return sdc_truediv(self, other)

    return series_operator_truediv_wrapper


@sdc_overload(operator.floordiv)
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

    _func_name = 'Method comp_floordiv().'
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

    def series_operator_floordiv_wrapper(self, other):
        return sdc_floordiv(self, other)

    return series_operator_floordiv_wrapper


@sdc_overload(operator.mod)
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

    _func_name = 'Method comp_mod().'
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

    def series_operator_mod_wrapper(self, other):
        return sdc_mod(self, other)

    return series_operator_mod_wrapper


@sdc_overload(operator.pow)
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

    _func_name = 'Method comp_pow().'
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

    def series_operator_pow_wrapper(self, other):
        return sdc_pow(self, other)

    return series_operator_pow_wrapper


@sdc_overload(operator.lt)
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

    def series_operator_lt_wrapper(self, other):
        return sdc_lt(self, other)

    return series_operator_lt_wrapper


@sdc_overload(operator.gt)
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

    def series_operator_gt_wrapper(self, other):
        return sdc_gt(self, other)

    return series_operator_gt_wrapper


@sdc_overload(operator.le)
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

    def series_operator_le_wrapper(self, other):
        return sdc_le(self, other)

    return series_operator_le_wrapper


@sdc_overload(operator.ge)
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

    def series_operator_ge_wrapper(self, other):
        return sdc_ge(self, other)

    return series_operator_ge_wrapper


@sdc_overload(operator.ne)
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

    def series_operator_ne_wrapper(self, other):
        return sdc_ne(self, other)

    return series_operator_ne_wrapper


@sdc_overload(operator.eq)
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

    def series_operator_eq_wrapper(self, other):
        return sdc_eq(self, other)

    return series_operator_eq_wrapper


@sdc_overload(operator.lt)
def sdc_str_arr_operator_lt(self, other):

    self_is_str_arr = self == string_array_type
    other_is_str_arr = other == string_array_type
    operands_are_arrays = self_is_str_arr and other_is_str_arr

    if not (operands_are_arrays
            or (self_is_str_arr and isinstance(other, types.UnicodeType))
            or (isinstance(self, types.UnicodeType) and other_is_str_arr)):
        return None

    if operands_are_arrays:
        def _sdc_str_arr_operator_lt_impl(self, other):
            if len(self) != len(other):
                raise ValueError("Mismatch of String Arrays sizes in operator.lt")
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] < other[i]
                               and not (str_arr_is_na(self, i) or str_arr_is_na(other, i)))
            return out_list

    elif self_is_str_arr:
        def _sdc_str_arr_operator_lt_impl(self, other):
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] < other and not (str_arr_is_na(self, i)))
            return out_list

    elif other_is_str_arr:
        def _sdc_str_arr_operator_lt_impl(self, other):
            n = len(other)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self < other[i] and not (str_arr_is_na(other, i)))
            return out_list
    else:
        return None

    return _sdc_str_arr_operator_lt_impl


@sdc_overload(operator.gt)
def sdc_str_arr_operator_gt(self, other):

    self_is_str_arr = self == string_array_type
    other_is_str_arr = other == string_array_type
    operands_are_arrays = self_is_str_arr and other_is_str_arr

    if not (operands_are_arrays
            or (self_is_str_arr and isinstance(other, types.UnicodeType))
            or (isinstance(self, types.UnicodeType) and other_is_str_arr)):
        return None

    if operands_are_arrays:
        def _sdc_str_arr_operator_gt_impl(self, other):
            if len(self) != len(other):
                raise ValueError("Mismatch of String Arrays sizes in operator.gt")
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] > other[i]
                               and not (str_arr_is_na(self, i) or str_arr_is_na(other, i)))
            return out_list

    elif self_is_str_arr:
        def _sdc_str_arr_operator_gt_impl(self, other):
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] > other and not (str_arr_is_na(self, i)))
            return out_list

    elif other_is_str_arr:
        def _sdc_str_arr_operator_gt_impl(self, other):
            n = len(other)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self > other[i] and not (str_arr_is_na(other, i)))
            return out_list
    else:
        return None

    return _sdc_str_arr_operator_gt_impl


@sdc_overload(operator.le)
def sdc_str_arr_operator_le(self, other):

    self_is_str_arr = self == string_array_type
    other_is_str_arr = other == string_array_type
    operands_are_arrays = self_is_str_arr and other_is_str_arr

    if not (operands_are_arrays
            or (self_is_str_arr and isinstance(other, types.UnicodeType))
            or (isinstance(self, types.UnicodeType) and other_is_str_arr)):
        return None

    if operands_are_arrays:
        def _sdc_str_arr_operator_le_impl(self, other):
            if len(self) != len(other):
                raise ValueError("Mismatch of String Arrays sizes in operator.le")
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] <= other[i]
                               and not (str_arr_is_na(self, i) or str_arr_is_na(other, i)))
            return out_list

    elif self_is_str_arr:
        def _sdc_str_arr_operator_le_impl(self, other):
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] <= other and not (str_arr_is_na(self, i)))
            return out_list

    elif other_is_str_arr:
        def _sdc_str_arr_operator_le_impl(self, other):
            n = len(other)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self <= other[i] and not (str_arr_is_na(other, i)))
            return out_list
    else:
        return None

    return _sdc_str_arr_operator_le_impl


@sdc_overload(operator.ge)
def sdc_str_arr_operator_ge(self, other):

    self_is_str_arr = self == string_array_type
    other_is_str_arr = other == string_array_type
    operands_are_arrays = self_is_str_arr and other_is_str_arr

    if not (operands_are_arrays
            or (self_is_str_arr and isinstance(other, types.UnicodeType))
            or (isinstance(self, types.UnicodeType) and other_is_str_arr)):
        return None

    if operands_are_arrays:
        def _sdc_str_arr_operator_ge_impl(self, other):
            if len(self) != len(other):
                raise ValueError("Mismatch of String Arrays sizes in operator.ge")
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] >= other[i]
                               and not (str_arr_is_na(self, i) or str_arr_is_na(other, i)))
            return out_list

    elif self_is_str_arr:
        def _sdc_str_arr_operator_ge_impl(self, other):
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] >= other and not (str_arr_is_na(self, i)))
            return out_list

    elif other_is_str_arr:
        def _sdc_str_arr_operator_ge_impl(self, other):
            n = len(other)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self >= other[i] and not (str_arr_is_na(other, i)))
            return out_list
    else:
        return None

    return _sdc_str_arr_operator_ge_impl


@sdc_overload(operator.ne)
def sdc_str_arr_operator_ne(self, other):

    self_is_str_arr = self == string_array_type
    other_is_str_arr = other == string_array_type
    operands_are_arrays = self_is_str_arr and other_is_str_arr

    if not (operands_are_arrays
            or (self_is_str_arr and isinstance(other, types.UnicodeType))
            or (isinstance(self, types.UnicodeType) and other_is_str_arr)):
        return None

    if operands_are_arrays:
        def _sdc_str_arr_operator_ne_impl(self, other):
            if len(self) != len(other):
                raise ValueError("Mismatch of String Arrays sizes in operator.ne")
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] != other[i]
                               or (str_arr_is_na(self, i) or str_arr_is_na(other, i)))
            return out_list

    elif self_is_str_arr:
        def _sdc_str_arr_operator_ne_impl(self, other):
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] != other or (str_arr_is_na(self, i)))
            return out_list

    elif other_is_str_arr:
        def _sdc_str_arr_operator_ne_impl(self, other):
            n = len(other)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self != other[i] or (str_arr_is_na(other, i)))
            return out_list
    else:
        return None

    return _sdc_str_arr_operator_ne_impl


@sdc_overload(operator.eq)
def sdc_str_arr_operator_eq(self, other):

    self_is_str_arr = self == string_array_type
    other_is_str_arr = other == string_array_type
    operands_are_arrays = self_is_str_arr and other_is_str_arr

    if not (operands_are_arrays
            or (self_is_str_arr and isinstance(other, types.UnicodeType))
            or (isinstance(self, types.UnicodeType) and other_is_str_arr)):
        return None

    if operands_are_arrays:
        def _sdc_str_arr_operator_eq_impl(self, other):
            if len(self) != len(other):
                raise ValueError("Mismatch of String Arrays sizes in operator.eq")
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] == other[i]
                               and not (str_arr_is_na(self, i) or str_arr_is_na(other, i)))
            return out_list

    elif self_is_str_arr:
        def _sdc_str_arr_operator_eq_impl(self, other):
            n = len(self)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self[i] == other and not (str_arr_is_na(self, i)))
            return out_list

    elif other_is_str_arr:
        def _sdc_str_arr_operator_eq_impl(self, other):
            n = len(other)
            out_list = [False] * n
            for i in numba.prange(n):
                out_list[i] = (self == other[i] and not (str_arr_is_na(other, i)))
            return out_list
    else:
        return None

    return _sdc_str_arr_operator_eq_impl

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

from sdc.utilities.sdc_typing_utils import (TypeChecker, check_types_comparable, sdc_old_index_types,)
from sdc.hiframes.api import isna
from sdc.hiframes.pd_series_type import SeriesType
from sdc.str_arr_ext import (string_array_type, str_arr_is_na)
from sdc.utilities.utils import sdc_overload, sdc_overload_method
from sdc.functions import numpy_like
from sdc.extensions.indexes.indexes_generic import sdc_indexes_join_outer, sdc_fix_indexes_join, sdc_unify_index_types


def sdc_binop(self, other, fill_value=None):
    pass


def sdc_binop_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    # specializations for numeric series only
    if not operands_are_series:
        def sdc_binop_impl(self, other, fill_value=None):

            series = self if self_is_series == True else other  # noqa
            result_data = numpy.empty(len(series._data), dtype=numpy.float64)
            series_data = numpy_like.fillna(series._data, inplace=False, value=fill_value)
            if self_is_series == True:  # noqa
                _self, _other = series_data, numpy.float64(other)
            else:
                _self, _other = numpy.float64(self), series_data

            result_data[:] = _self + _other
            return pandas.Series(result_data, index=series._index, name=series._name)

        return sdc_binop_impl

    else:   # both operands are numeric series

        use_index_methods = not (isinstance(self.index, sdc_old_index_types)
                                 or isinstance(other.index, sdc_old_index_types))
        def sdc_binop_impl(self, other, fill_value=None):

            left_index, right_index = self._index, other._index
            if use_index_methods == True:
                indexes_join_res = left_index.join(right_index, 'outer', return_indexers=True)
            else:
                indexes_join_res = sdc_indexes_join_outer(left_index, right_index)

            # FIXME_Numba#XXXX: remove sdc_fix_indexes_join call at all when issue is fixed
            joined_index, left_indexer, right_indexer = sdc_fix_indexes_join(*indexes_join_res)
            result_size = len(joined_index)
            result_data = numpy.empty(result_size, dtype=numpy.float64)

            _fill_value = numpy.nan if fill_value_is_none == True else fill_value  # noqa
            for i in numba.prange(result_size):
                left_pos = left_indexer[i]
                right_pos = right_indexer[i]

                left_nan = (left_pos == -1 or numpy.isnan(self._data[left_pos]))
                right_nan = (right_pos == -1 or numpy.isnan(other._data[right_pos]))
                _left = _fill_value if left_nan else self._data[left_pos]
                _right = _fill_value if right_nan else other._data[right_pos]
                result_data[i] = numpy.nan if (left_nan and right_nan) else _left + _right

            return pandas.Series(result_data, index=joined_index)

        return sdc_binop_impl


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

    if not isinstance(self, SeriesType):
        ty_checker.raise_exc(self, 'pandas.series', 'self')

    if not isinstance(other, (SeriesType, types.Number)):
        ty_checker.raise_exc(other, 'pandas.series or number', 'other')

    operands_are_series = self_is_series and other_is_series
    if operands_are_series:
        series_indexes_comparable = check_types_comparable(self.index, other.index)
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
    def series_binop_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_binop(self, other, fill_value)

    return series_binop_wrapper


def sdc_comp_binop(self, other, fill_value=None):
    pass


def sdc_comp_binop_ovld(self, other, fill_value=None):

    self_is_series, other_is_series = isinstance(self, SeriesType), isinstance(other, SeriesType)
    operands_are_series = self_is_series and other_is_series
    fill_value_is_none = isinstance(fill_value, (types.NoneType, types.Omitted)) or fill_value is None

    if not operands_are_series:
        def _series_lt_scalar_impl(self, other, fill_value=None):

            _self = numpy_like.fillna(self._data, inplace=False, value=fill_value)
            return pandas.Series(_self < other, index=self._index, name=self._name)

        return _series_lt_scalar_impl

    else:

        index_api_supported = not (isinstance(self.index, sdc_old_index_types)
                                 and isinstance(other.index, sdc_old_index_types))
        def _series_lt_common_impl(self, other, fill_value=None):

            left_index, right_index = self.index, other.index
            if index_api_supported == True:  # noqa
                if not (left_index is right_index or left_index.equals(right_index)):
                    raise ValueError("Can only compare identically-labeled Series objects")
            else:
                if not (left_index is right_index or numpy_like.array_equal(left_index, right_index)):
                    raise ValueError("Can only compare identically-labeled Series objects")

            res_size = len(left_index)
            if fill_value_is_none == True:  # noqa
                res_data = self._data < other._data
            else:
                res_data = numpy.empty(res_size, dtype=types.bool_)
                for i in numba.prange(res_size):
                    left_nan = isna(self._data, i)
                    right_nan = isna(other._data, i)
                    _left = fill_value if left_nan else self._data[i]
                    _right = fill_value if right_nan else other._data[i]
                    res_data[i] = False if (left_nan and right_nan) else _left < _right

            res_index = sdc_unify_index_types(left_index, right_index)
            return pandas.Series(res_data, index=res_index)

        return _series_lt_common_impl

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
        series_indexes_comparable = check_types_comparable(self.index, other.index)
        if not series_indexes_comparable:
            raise TypingError('{} Not implemented for series with not-comparable indexes. \
            Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    series_data_comparable = check_types_comparable(self, other)
    if not series_data_comparable:
        raise TypingError('{} Not supported for not-comparable operands. \
        Given: self={}, other={}'.format(_func_name, self, other))

    # specializations for both numeric and string series
    def series_comp_binop_wrapper(self, other, level=None, fill_value=None, axis=0):
        return sdc_comp_binop(self, other, fill_value)

    return series_comp_binop_wrapper


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

    _func_name = 'Method comp_binop().'
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
        series_indexes_comparable = check_types_comparable(self.index, other.index)
        if not series_indexes_comparable:
            raise TypingError('{} Not implemented for series with not-comparable indexes. \
            Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    series_data_comparable = check_types_comparable(self, other)
    if not series_data_comparable:
        raise TypingError('{} Not supported for not-comparable operands. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def series_operator_binop_wrapper(self, other):
        return sdc_binop(self, other)

    return series_operator_binop_wrapper


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
        series_indexes_comparable = check_types_comparable(self.index, other.index)
        if not series_indexes_comparable:
            raise TypingError('{} Not implemented for series with not-comparable indexes. \
            Given: self.index={}, other.index={}'.format(_func_name, self.index, other.index))

    series_data_comparable = check_types_comparable(self, other)
    if not series_data_comparable:
        raise TypingError('{} Not supported for not-comparable operands. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def series_operator_comp_binop_wrapper(self, other):
        return sdc_comp_binop(self, other)

    return series_operator_comp_binop_wrapper


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

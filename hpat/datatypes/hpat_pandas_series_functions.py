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

import operator

from numba import types
from numba.extending import (types, overload, overload_method, overload_attribute)
from numba.errors import TypingError

from hpat.hiframes.pd_series_ext import SeriesType


'''
Pandas Series (https://pandas.pydata.org/pandas-docs/stable/reference/series.html)
functions and operators definition in HPAT
Also, it contains Numba internal operators which are required for Series type handling

Implemented operators:
    at
    getitem
    iat
    iloc
    len
    loc
    ne

Implemented methods:

'''


@overload(operator.getitem)
def hpat_pandas_series_getitem(self, idx):
    '''
    Pandas Series opearator getitem implementation

    Algorithm: result = series[idx]
    Where:
        series: pandas.series
           idx: integer number or pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_static_getitem_series1
    '''

    _func_name = 'Operator getitem().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if isinstance(idx, types.Integer):
        def hpat_pandas_series_getitem_idx_integer_impl(self, idx):
            result = self._data[idx]
            return result

        return hpat_pandas_series_getitem_idx_integer_impl

    if isinstance(idx, types.SliceType):
        def hpat_pandas_series_getitem_idx_slice_impl(self, idx):
            result = hpat.hiframes.api.init_series(self._data[idx])
            return result

        return hpat_pandas_series_getitem_idx_slice_impl

    if isinstance(idx, SeriesType):
        def hpat_pandas_series_getitem_idx_series_impl(self, idx):
            super_index = idx._data
            result = self._data[super_index]
            return result

        return hpat_pandas_series_getitem_idx_series_impl

    raise TypingError('{} The index must be an Integer, Slice or a pandas.series. Given: {}'.format(_func_name, idx))


@overload_attribute(SeriesType, 'at')
@overload_attribute(SeriesType, 'iat')
@overload_attribute(SeriesType, 'iloc')
@overload_attribute(SeriesType, 'loc')
def hpat_pandas_series_iloc(self):
    '''
    Pandas Series opearator iloc implementation.

    Algorithm: result = series.iloc
    Where:
        series: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_list_str_unbox1
    '''

    _func_name = 'Operator iloc().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_iloc_impl(self):
        return self

    return hpat_pandas_series_iloc_impl


@overload(len)
def hpat_pandas_series_len(self):
    '''
    Pandas Series opearator len implementation
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.len.html#pandas.Series.str.len

    Algorithm: result = len(series)
    Where:
        series: pandas.series
    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_len
    '''

    _func_name = 'Operator len().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_len_impl(self):
        return len(self._data)

    return hpat_pandas_series_len_impl

@overload(operator.ne)
def hpat_pandas_series_not_equal(lhs, rhs):
    '''
    Pandas Series opearator != implementation.

    Algorithm: result = (A != B)
    Where:
        A: pandas.series
        B: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    '''

    _func_name = 'Operator ne().'

    if not isinstance(lhs, SeriesType) or not isinstance(rhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: lhs{} rhs{}'.format(_func_name, lhs, rhs))

    def hpat_pandas_series_not_equal_impl(lhs, rhs):
        return (lhs._data != rhs._data)

    return hpat_pandas_series_not_equal_impl

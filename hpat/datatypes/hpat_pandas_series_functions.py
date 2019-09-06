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
from numba.extending import (types, overload, overload_method)

from hpat.hiframes.pd_series_ext import SeriesType


'''
Pandas Series (https://pandas.pydata.org/pandas-docs/stable/reference/series.html)
functions and operators definition in HPAT
Also, it contains Numba internal operators which are required for Series type handling

Implemented methods:

Implemented operators:
    getitem
'''


@overload(operator.getitem)
def hpat_pandas_series_getitem(self, idx):
    '''
    Internal Numba operator getitem implementation
    Algorithm: result = series[idx]
    Where:
        series: pandas.series
        idx: integer number
    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_static_getitem_series1
    '''

    if not isinstance(self, SeriesType):
        raise TypingError('The object must be a pandas.series. Given: {}'.format(self))

    if not isinstance(idx, types.Integer):
        raise TypingError('The index must be an Integer. Given: {}'.format(idx))

    def hpat_pandas_series_getitem_impl(self, idx):
        result = self._data[idx]
        return result

    return hpat_pandas_series_getitem_impl

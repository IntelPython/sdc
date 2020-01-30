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

| :class:`pandas.SeriesGroupBy` functions and operators implementations in SDC
| Also, it contains Numba internal operators which are required for :class:`pandas.SeriesGroupBy` type handling

"""


import numpy
import pandas

from numba import types
from numba.extending import overload_method
from numba.errors import TypingError

from sdc.datatypes.hpat_pandas_seriesgroupby_types import SeriesGroupByType
from sdc.utils import sdc_overload_method


@sdc_overload_method(SeriesGroupByType, 'count')
def hpat_pandas_seriesgroupby_count(self):
    """
    Pandas Series method :meth:`pandas.core.groupby.GroupBy.count` implementation.

    .. only:: developer

       Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_groupby_count

    Parameters
    -----------
    self: :obj:`pandas.core.groupby.SeriesGroupBy`
               The object this method is working on

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object with count of values within each group
    """

    _func_name = 'Method seriesgroupby.count().'

    if not isinstance(self, SeriesGroupByType):
        raise TypingError('{} The object must be a pandas.seriesgroupby. Given: {}'.format(_func_name, self))

    def hpat_pandas_seriesgroupby_count_impl(self):
        """
        Pandas algorithm:
            https://github.com/pandas-dev/pandas/blob/b1049540fe207f8d8071ebfbd44e8f5224c98bad/pandas/core/groupby/generic.py#L1339
        """

        # is not implemented yet.
        # return self._data.value_counts()
        #
        # workaround
        freq = {}
        for x in self._data:
            if x not in freq:
                freq[x] = 1
            else:
                freq[x] += 1

        # Numba requires to translate dict() into list()
        keys = []
        values = []
        for key, value in freq.items():
            keys.append(key)
            values.append(value)

        return pandas.Series(values, keys)

    return hpat_pandas_seriesgroupby_count_impl

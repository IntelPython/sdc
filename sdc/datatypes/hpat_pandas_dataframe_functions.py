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

'''
| :class:`pandas.DataFrame` functions and operators implementations in Intel SDC
| Also, it contains Numba internal operators which are required for DataFrame type handling
'''

import operator
import pandas
import numpy

import sdc

from numba import types
from numba.extending import (overload, overload_method, overload_attribute)
from sdc.hiframes.pd_dataframe_ext import DataFrameType
from numba.errors import TypingError
from sdc.datatypes.hpat_pandas_series_functions import TypeChecker

if not sdc.config.use_default_dataframe:
    from sdc.datatypes.hpat_pandas_dataframe_types import DataFrameType


    @overload_method(DataFrameType, 'count')
    def sdc_pandas_dataframe_count(self, axis=0, level=None, numeric_only=False):
        """
        Pandas DataFrame method :meth:`pandas.DataFrame.count` implementation.

        .. only:: developer

            Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_count

        Parameters
        -----------
        self: :class:`pandas.DataFrame`
            input arg
        axis:
            *unsupported*
        level:
            *unsupported*
        numeric_only:
            *unsupported*

        Returns
        -------
        :obj:`pandas.Series` or `pandas.DataFrame`
                returns: For each column/row the number of non-NA/null entries. If level is specified returns a DataFrame.
        """

        _func_name = 'Method pandas.dataframe.count().'

        if not isinstance(self, DataFrameType):
            raise TypingError('{} The object must be a pandas.dataframe. Given: {}'.format(_func_name, self))

        if not (isinstance(axis, types.Omitted) or axis == 0):
            raise TypingError("{} 'axis' unsupported. Given: {}".format(_func_name, axis))

        if not (isinstance(level, types.Omitted) or level is None):
            raise TypingError("{} 'level' unsupported. Given: {}".format(_func_name, axis))

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is False):
            raise TypingError("{} 'numeric_only' unsupported. Given: {}".format(_func_name, axis))

        def sdc_pandas_dataframe_count_impl(self, axis=0, level=None, numeric_only=False):
            result_data = []
            result_index = []

            for dataframe_item in self._data:
                item_count = dataframe_item.count()
                item_name = dataframe_item._name
                result_data.append(item_count)
                result_index.append(item_name)

            return pandas.Series(data=result_data, index=result_index)

        return sdc_pandas_dataframe_count_impl

else:
    def sdc_pandas_dataframe_reduce_columns(df, name, *args, **kwargs):
        saved_columns = df.columns
        n_cols = len(saved_columns)
        data_args = tuple('data{}'.format(i) for i in range(n_cols))
        func_text = "def _reduce_impl(df, axis=None, skipna=None, level=None, numeric_only=None):\n"
        for i, d in enumerate(data_args):
            func_text += "  {} = hpat.hiframes.api.init_series(hpat.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}))\n".format(
                d + '_S', i)
            func_text += "  {} = {}.{}()\n".format(d + '_O', d + '_S', name)
        func_text += "  data = np.array(({},))\n".format(
            ", ".join(d + '_O' for d in data_args))
        func_text += "  index = hpat.str_arr_ext.StringArray(({},))\n".format(
            ", ".join("'{}'".format(c) for c in saved_columns))
        func_text += "  return hpat.hiframes.api.init_series(data, index)\n"
        loc_vars = {}

        exec(func_text, {'hpat': sdc, 'np': numpy}, loc_vars)
        _reduce_impl = loc_vars['_reduce_impl']

        return _reduce_impl


    @overload_method(DataFrameType, 'median')
    def median_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.median` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_median1

           Parameters
           -----------
           self: :class:`pandas.DataFrame`
               input arg
           axis:
                *unsupported*
           skipna:
               *unsupported*
           level:
               *unsupported*
           numeric_only:
               *unsupported*

           Returns
           -------
           :obj:`pandas.Series` or `pandas.DataFrame`
                   return the median of the values for the requested axis.
           """

        name = 'median'

        ty_checker = TypeChecker('Method {}().'.format(name))
        ty_checker.check(df, DataFrameType)

        if not (isinstance(axis, types.Omitted) or axis is None):
            ty_checker.raise_exc(axis, 'unsupported', 'axis')

        if not (isinstance(skipna, types.Omitted) or skipna is None):
            ty_checker.raise_exc(skipna, 'unsupported', 'skipna')

        if not (isinstance(level, types.Omitted) or level is None):
            ty_checker.raise_exc(level, 'unsupported', 'level')

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
            ty_checker.raise_exc(numeric_only, 'unsupported', 'numeric_only')

        return sdc_pandas_dataframe_reduce_columns(df, name)


    @overload_method(DataFrameType, 'mean')
    def mean_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.mean` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_mean1

           Parameters
           -----------
           self: :class:`pandas.DataFrame`
               input arg
           axis:
                *unsupported*
           skipna:
               *unsupported*
           level:
               *unsupported*
           numeric_only:
               *unsupported*

           Returns
           -------
           :obj:`pandas.Series` or `pandas.DataFrame`
                   return the mean of the values for the requested axis.
           """

        name = 'mean'

        ty_checker = TypeChecker('Method {}().'.format(name))
        ty_checker.check(df, DataFrameType)

        if not (isinstance(axis, types.Omitted) or axis is None):
            ty_checker.raise_exc(axis, 'unsupported', 'axis')

        if not (isinstance(skipna, types.Omitted) or skipna is None):
            ty_checker.raise_exc(skipna, 'unsupported', 'skipna')

        if not (isinstance(level, types.Omitted) or level is None):
            ty_checker.raise_exc(level, 'unsupported', 'level')

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
            ty_checker.raise_exc(numeric_only, 'unsupported', 'numeric_only')

        return sdc_pandas_dataframe_reduce_columns(df, name)


    @overload_method(DataFrameType, 'max')
    def max_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.max` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_max1

           Parameters
           -----------
           self: :class:`pandas.DataFrame`
               input arg
           axis:
                *unsupported*
           skipna:
               *unsupported*
           level:
               *unsupported*
           numeric_only:
               *unsupported*

           Returns
           -------
           :obj:`pandas.Series` or `pandas.DataFrame`
                   return the maximum of the values for the requested axis.
           """

        name = 'max'

        ty_checker = TypeChecker('Method {}().'.format(name))
        ty_checker.check(df, DataFrameType)

        if not (isinstance(axis, types.Omitted) or axis is None):
            ty_checker.raise_exc(axis, 'unsupported', 'axis')

        if not (isinstance(skipna, types.Omitted) or skipna is None):
            ty_checker.raise_exc(skipna, 'unsupported', 'skipna')

        if not (isinstance(level, types.Omitted) or level is None):
            ty_checker.raise_exc(level, 'unsupported', 'level')

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
            ty_checker.raise_exc(numeric_only, 'unsupported', 'numeric_only')

        return sdc_pandas_dataframe_reduce_columns(df, name)


    @overload_method(DataFrameType, 'min')
    def min_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.min` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_min1

           Parameters
           -----------
           self: :class:`pandas.DataFrame`
               input arg
           axis:
                *unsupported*
           skipna:
               *unsupported*
           level:
               *unsupported*
           numeric_only:
               *unsupported*

           Returns
           -------
           :obj:`pandas.Series` or `pandas.DataFrame`
                   returns: the minimum of the values for the requested axis.
           """

        name = 'min'

        ty_checker = TypeChecker('Method {}().'.format(name))
        ty_checker.check(df, DataFrameType)

        if not (isinstance(axis, types.Omitted) or axis is None):
            ty_checker.raise_exc(axis, 'unsupported', 'axis')

        if not (isinstance(skipna, types.Omitted) or skipna is None):
            ty_checker.raise_exc(skipna, 'unsupported', 'skipna')

        if not (isinstance(level, types.Omitted) or level is None):
            ty_checker.raise_exc(level, 'unsupported', 'level')

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
            ty_checker.raise_exc(numeric_only, 'unsupported', 'numeric_only')

        return sdc_pandas_dataframe_reduce_columns(df, name)


    @overload_method(DataFrameType, 'sum')
    def sum_overload(df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.sum` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_sum1

           Parameters
           -----------
           self: :class:`pandas.DataFrame`
               input arg
           axis:
                *unsupported*
           skipna:
               *unsupported*
           level:
               *unsupported*
           numeric_only:
               *unsupported*
           min_count:
                *unsupported*

           Returns
           -------
           :obj:`pandas.Series` or `pandas.DataFrame`
                   return the sum of the values for the requested axis.
           """

        name = 'sum'

        ty_checker = TypeChecker('Method {}().'.format(name))
        ty_checker.check(df, DataFrameType)

        if not (isinstance(axis, types.Omitted) or axis is None):
            ty_checker.raise_exc(axis, 'unsupported', 'axis')

        if not (isinstance(skipna, types.Omitted) or skipna is None):
            ty_checker.raise_exc(skipna, 'unsupported', 'skipna')

        if not (isinstance(level, types.Omitted) or level is None):
            ty_checker.raise_exc(level, 'unsupported', 'level')

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
            ty_checker.raise_exc(numeric_only, 'unsupported', 'numeric_only')

        if not (isinstance(min_count, types.Omitted) or min_count == 0):
            ty_checker.raise_exc(min_count, 'unsupported', 'min_count')

        return sdc_pandas_dataframe_reduce_columns(df, name)

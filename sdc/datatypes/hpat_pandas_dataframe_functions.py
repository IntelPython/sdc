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

def sdc_pandas_dataframe_reduce_columns(df, name, params):
    saved_columns = df.columns
    n_cols = len(saved_columns)
    data_args = tuple('data{}'.format(i) for i in range(n_cols))
    all_params = ['df']
    for key, value in params:
        all_params.append('{}={}'.format(key, value))
    func_definition = 'def _reduce_impl({}):'.format(', '.join(all_params))
    func_lines = [func_definition]
    for i, d in enumerate(data_args):
        line = '  {} = hpat.hiframes.api.init_series(hpat.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}))'
        func_lines.append(line.format(d + '_S', i))
        func_lines.append('  {} = {}.{}()'.format(d + '_O', d + '_S', name))
    func_lines.append('  data = np.array(({},))'.format(
        ", ".join(d + '_O' for d in data_args)))
    func_lines.append('  index = hpat.str_arr_ext.StringArray(({},))'.format(
        ', '.join('"{}"'.format(c) for c in saved_columns)))
    func_lines.append('  return hpat.hiframes.api.init_series(data, index)')
    loc_vars = {}
    func_text = '\n'.join(func_lines)

    exec(func_text, {'hpat': sdc, 'np': numpy}, loc_vars)
    _reduce_impl = loc_vars['_reduce_impl']

    return _reduce_impl

def check_type(name, df, axis=None, skipna=None, level=None, numeric_only=None, ddof=1, min_count=0):
    ty_checker = TypeChecker('Method {}().'.format(name))
    ty_checker.check(df, DataFrameType)

    if not (isinstance(axis, types.Omitted) or axis is None):
        ty_checker.raise_exc(axis, 'unsupported', 'axis')

    if not (isinstance(skipna, (types.Omitted, types.NoneType, types.Boolean)) or skipna is None):
        ty_checker.raise_exc(skipna, 'bool', 'skipna')

    if not (isinstance(level, types.Omitted) or level is None):
        ty_checker.raise_exc(level, 'unsupported', 'level')

    if not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
        ty_checker.raise_exc(numeric_only, 'unsupported', 'numeric_only')

    if not (isinstance(ddof, types.Omitted) or ddof == 1):
        ty_checker.raise_exc(ddof, 'unsupported', 'ddof')

    if not (isinstance(min_count, types.Omitted) or min_count == 0):
        ty_checker.raise_exc(min_count, 'unsupported', 'min_count')

@overload_method(DataFrameType, 'median')
def median_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.median` implementation.

       .. only:: developer

           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_median1
           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_median

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

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only)

    params = [('axis', None), ('skipna', skipna), ('level', None), ('numeric_only', numeric_only)]

    return sdc_pandas_dataframe_reduce_columns(df, name, params)

@overload_method(DataFrameType, 'mean')
def mean_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.mean` implementation.

       .. only:: developer

           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_mean1
           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_mean

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

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only)

    params = [('axis', None), ('skipna', skipna), ('level', None), ('numeric_only', numeric_only)]

    return sdc_pandas_dataframe_reduce_columns(df, name, params)

@overload_method(DataFrameType, 'std')
def std_overload(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.std` implementation.

       .. only:: developer

           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_std1
           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_std

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
       ddof:
           *unsupported*
       numeric_only:
           *unsupported*

       Returns
       -------
       :obj:`pandas.Series` or `pandas.DataFrame`
               return sample standard deviation over requested axis.
       """

    name = 'std'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, ddof=ddof)

    params = [('axis', None), ('skipna', skipna), ('level', None), ('ddof', ddof), ('numeric_only', numeric_only)]

    return sdc_pandas_dataframe_reduce_columns(df, name, params)

@overload_method(DataFrameType, 'var')
def var_overload(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.var` implementation.

       .. only:: developer

           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_var1
           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_var

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
       ddof:
           *unsupported*
       numeric_only:
           *unsupported*

       Returns
       -------
       :obj:`pandas.Series` or `pandas.DataFrame`
               return sample standard deviation over requested axis.
       """

    name = 'var'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, ddof=ddof)

    params = [('axis', None), ('skipna', skipna), ('level', None), ('ddof', ddof), ('numeric_only', numeric_only)]

    return sdc_pandas_dataframe_reduce_columns(df, name, params)

@overload_method(DataFrameType, 'max')
def max_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.max` implementation.

       .. only:: developer

           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_max1
           Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_max

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

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only)

    params = [('axis', None), ('skipna', skipna), ('level', None), ('numeric_only', numeric_only)]

    return sdc_pandas_dataframe_reduce_columns(df, name, params)

    @overload_method(DataFrameType, 'min')
    def min_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.min` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_min1
               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_min

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

        check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only)

        params = [('axis', None), ('skipna', skipna), ('level', None), ('numeric_only', numeric_only)]

        return sdc_pandas_dataframe_reduce_columns(df, name, params)

    @overload_method(DataFrameType, 'sum')
    def sum_overload(df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.sum` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_sum1
               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_sum

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

        check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, min_count=min_count)

        params = [('axis', None), ('skipna', skipna), ('level', None), ('numeric_only', numeric_only),
                  ('min_count', min_count)]

        return sdc_pandas_dataframe_reduce_columns(df, name, params)

    @overload_method(DataFrameType, 'prod')
    def prod_overload(df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.prod` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_prod1
               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_prod

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
                   return the product of the values for the requested axis.
           """

        name = 'prod'

        check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, min_count=min_count)

        params = [('axis', None), ('skipna', skipna), ('level', None), ('numeric_only', numeric_only),
                  ('min_count', min_count)]

        return sdc_pandas_dataframe_reduce_columns(df, name, params)


    @overload_method(DataFrameType, 'count')
    def count_overload(df, axis=0, level=None, numeric_only=False):
        """
           Pandas DataFrame method :meth:`pandas.DataFrame.count` implementation.

           .. only:: developer

               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_count
               Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_count1

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
                   for each column/row the number of non-NA/null entries. If level is specified returns a DataFrame.
           """

        name = 'count'

        ty_checker = TypeChecker('Method {}().'.format(name))
        ty_checker.check(df, DataFrameType)

        if not (isinstance(axis, types.Omitted) or axis == 0):
            ty_checker.raise_exc(axis, 'unsupported', 'axis')

        if not (isinstance(level, types.Omitted) or level is None):
            ty_checker.raise_exc(level, 'unsupported', 'level')

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is False):
            ty_checker.raise_exc(numeric_only, 'unsupported', 'numeric_only')

        params = [('axis', None), ('level', None), ('numeric_only', numeric_only)]

        return sdc_pandas_dataframe_reduce_columns(df, name, params)

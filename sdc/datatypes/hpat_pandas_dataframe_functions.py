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
import copy
import numpy

import sdc

from numba import types
from numba.extending import (overload, overload_method, overload_attribute)
from sdc.hiframes.pd_dataframe_ext import DataFrameType
from numba.errors import TypingError
import sdc.datatypes.hpat_pandas_dataframe_types

from sdc.datatypes.hpat_pandas_series_functions import TypeChecker


def _dataframe_reduce_columns_codegen(func_name, func_params, series_params, columns):
    result_name_list = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    for i, c in enumerate(columns):
        result_c = f'result_{c}'
        func_lines += [f'  series_{c} = init_series(get_dataframe_data({func_params[0]}, {i}))',
                       f'  {result_c} = series_{c}.{func_name}({series_params})']
        result_name_list.append(result_c)
    print(result_name_list)
    all_results = ', '.join(result_name_list)
    print(all_results)
    all_columns = ', '.join([f"'{c}'" for c in columns])

    func_lines += [f'  return pandas.Series([{all_results}], [{all_columns}])']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas, 'np': numpy,
                   'init_series': sdc.hiframes.api.init_series,
                   'get_dataframe_data': sdc.hiframes.pd_dataframe_ext.get_dataframe_data}

    return func_text, global_vars


def sdc_pandas_dataframe_reduce_columns(df, func_name, params):
    all_params = ['df']
    par1 = {'count': ['level']}

    if func_name in par1:
        for key, value in params:
            if key in par1[func_name]:
                all_params.append('{}={}'.format(key, value))
    else:
        for key, value in params:
            all_params.append('{}={}'.format(key, value))
    ap = all_params.copy()
    par = '{}'.format(', '.join(ap[1:]))

    df_func_name = f'_df_{func_name}_impl'

    func_text, global_vars = _dataframe_reduce_columns_codegen(func_name, all_params, par, df.columns)

    loc_vars = {}
    print(global_vars, loc_vars)
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars[df_func_name]

    return _reduce_impl

# param = 'level'
# if param in params
# par += 'level=' + param
# else
# par += 'level=' + 'None'


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

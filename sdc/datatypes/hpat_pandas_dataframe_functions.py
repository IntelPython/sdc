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

from sdc.datatypes.hpat_pandas_series_functions import TypeChecker

def sdc_pandas_dataframe_reduce_columns(df, name, series_call_params):

    saved_columns = df.columns
    data_args = tuple('data{}'.format(i) for i in range(len(saved_columns)))
    all_params = ['df']

    for key, value in series_call_params:
        all_params.append('{}={}'.format(key, value))
    # This relies on parameters part of the signature of Series method called below being the same
    # as for the corresponding DataFrame method
    series_call_params_str = '{}'.format(', '.join(all_params[1:]))
    func_definition = 'def _reduce_impl({}):'.format(', '.join(all_params))
    func_lines = [func_definition]
    for i, d in enumerate(data_args):
        line = '  {} = sdc.hiframes.api.init_series(sdc.hiframes.pd_dataframe_ext.get_dataframe_data(all_params[0], {}))'
        func_lines.append(line.format(d + '_S', i))
        func_lines.append(' {}_O = {}_S.{}({})'.format(d, d, name, series_call_params_str))
    func_lines.append('  data = np.array(({},))'.format(
        ", ".join(d + '_O' for d in data_args)))
    func_lines.append('  index = sdc.str_arr_ext.StringArray(({},))'.format(
        ', '.join('"{}"'.format(c) for c in saved_columns)))
    func_lines.append('  return sdc.hiframes.api.init_series(data, index)')
    loc_vars = {}
    func_text = '\n'.join(func_lines)

    exec(func_text, {'sdc': sdc, 'np': numpy}, loc_vars)
    _reduce_impl = loc_vars['_reduce_impl']

    return _reduce_impl


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

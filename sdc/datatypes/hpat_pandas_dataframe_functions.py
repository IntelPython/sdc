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
from sdc.datatypes.common_functions import TypeChecker
from numba.errors import TypingError



@overload_method(DataFrameType, 'append')
def sdc_pandas_dataframe_append(df, other, ignore_index=True, verify_integrity=False, sort=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.append

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe_append.py
       :language: python
       :lines: 27-
       :caption: Pad strings in the Series by prepending '0' characters
       :name: ex_dataframe_append
    .. code-block:: console
        > python ./dataframe_append.py
            A  B
         0  1  2
         1  3  4
         0  5  6
         1  7  8
        dtype: object

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.append` implementation.
    .. only:: developer
    Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_append_df_no_index
    Parameters
    -----------
    df: :obj:`pandas.DataFrame`
        input arg
    other: :obj:`pandas.DataFrame` object or :obj:`pandas.Series` or :obj:`dict`
        The data to append
    ignore_index: :obj:`bool`
        *unsupported*
    verify_integrity: :obj:`bool`
        *unsupported*
    sort: :obj:`bool`
        *unsupported*

    Returns
    -------
    :obj: `pandas.DataFrame`
        return DataFrame with appended rows to the end
    """

    name = 'append'

    ty_checker = TypeChecker('Method {}().'.format(name))
    ty_checker.check(df, DataFrameType)
    # TODO: support other array-like types
    ty_checker.check(other, DataFrameType)
    # TODO: support index in series from df-columns
    if not isinstance(ignore_index, (bool, types.Boolean, types.Omitted)) and not ignore_index:
        ty_checker.raise_exc(ignore_index, 'boolean', 'ignore_index')

    if not isinstance(verify_integrity, (bool, types.Boolean, types.Omitted)) and verify_integrity:
        ty_checker.raise_exc(verify_integrity, 'boolean', 'verify_integrity')

    if not isinstance(sort, (bool, types.Boolean, types.Omitted)) and verify_integrity:
        ty_checker.raise_exc(verify_integrity, 'boolean', 'sort')

    args = (('other', other), ('ignore_index', ignore_index), ('verify_integrity', False), ('sort', None))

    def sdc_pandas_dataframe_append_impl(df, name, args):
        df_columns = df.columns
        n_cols = len(df_columns)
        data_args = tuple('data{}'.format(i) for i in range(n_cols))
        func_args = ['df', 'other']

        for key, value in args:
            #TODO: improve check
            if key not in func_args:
                if isinstance(value, types.Literal):
                    value = value.literal_value
                func_args.append('{}={}'.format(key, value))

        func_definition = 'def sdc_pandas_dataframe_{}_impl({}):'.format(name, ', '.join(func_args))
        func_lines = [func_definition]
        for i, d in enumerate(data_args):
            line = '    {} = sdc.hiframes.api.init_series(sdc.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}))'
            line2 = '    {} = sdc.hiframes.api.init_series(sdc.hiframes.pd_dataframe_ext.get_dataframe_data(other, {}))'
            func_lines.append(line.format(d + '_S', i))
            func_lines.append(line2.format('to_append_{}'.format(i) + '_S', i))
            func_lines.append(
                '    {} = {}.{}({})._data'.format(d + '_O', d + '_S', name, 'to_append_{}'.format(i) + '_S'))
        data = ", ".join(d + '_O' for d in data_args)
        # TODO: Handle index
        index = None
        col_names = ", ".join("'{}'".format(c) for c in df_columns)
        func_lines.append("    return sdc.hiframes.pd_dataframe_ext.init_dataframe({}, {}, {})\n".format(
            data,
            index,
            col_names))
        loc_vars = {}
        func_text = '\n'.join(func_lines)

        exec(func_text, {'sdc': sdc, 'np': numpy}, loc_vars)
        _append_impl = loc_vars['sdc_pandas_dataframe_append_impl']

        return _append_impl

    return sdc_pandas_dataframe_append_impl(df, name, args)


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


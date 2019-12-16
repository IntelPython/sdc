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
       :caption: Appending rows of other to the end of caller, returning a new object.
       Columns in other that are not in the caller are added as new columns.
       :name: ex_dataframe_append
    .. code-block:: console
        > python ./dataframe_append.py
            A  B
         0  1  2
         1  3  4
         0  5  6
         1  7  8
        dtype: object

             A    B    C    D
        0  1.0  2.0  NaN  NaN
        1  3.0  4.0  NaN  NaN
        0  NaN  NaN  5.0  6.0
        1  NaN  NaN  7.0  8.0
        dtype: object

     .. note::
        Parameter ignore_index, verify_integrity, sort are currently unsupported by Intel Scalable Dataframe Compiler
        Currently only pandas.DataFrame is supported as "other" parameter

    .. seealso::
        :ref:`concat <pandas.concat>`
            General function to concatenate DataFrame or Series objects.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.append` implementation.
    .. only:: developer
    Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_append*
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

    args = (('ignore_index', ignore_index), ('verify_integrity', False), ('sort', None))

    def sdc_pandas_dataframe_append_impl(df, other, name, args):
        spaces = 4 * ' '
        func_args = ['df', 'other']

        for key, value in args:
            #TODO: improve check
            if key not in func_args:
                if isinstance(value, types.Literal):
                    value = value.literal_value
                func_args.append('{}={}'.format(key, value))

        df_columns_indx = {col_name: i for i, col_name in enumerate(df.columns)}
        other_columns_indx = {col_name: i for i, col_name in enumerate(other.columns)}

        def get_dataframe_column(df, column, idx):
            return f'new_col_{column}_data_{df} = get_dataframe_data({df}, {idx})'

        def get_append_result(df1, df2, column):
            return f'new_col_{column} = init_series(new_col_{column}_data_{df1}).append(init_series(new_col_{column}_data_{df2}))._data'

        func_definition = [f'def sdc_pandas_dataframe_{name}_impl({", ".join(func_args)}):']
        func_text = []
        column_list = []

        func_text.append(f'len_df = len(get_dataframe_data(df, {0}))')
        func_text.append(f'len_other = len(get_dataframe_data(other, {0}))')

        for col_name, i in df_columns_indx.items():
            if col_name in other_columns_indx:
                func_text.append(get_dataframe_column('df', col_name, i))
                func_text.append(get_dataframe_column('other', col_name, other_columns_indx.get(col_name)))
                func_text.append(get_append_result('df', 'other', col_name))
                column_list.append((f'new_col_{col_name}', col_name))
            else:
                func_text.append(get_dataframe_column('df', col_name, i))
                func_text.append(f'new_col_{col_name}_data = init_series(new_col_{col_name}_data_df)._data')
                func_text.append(f'new_col_{col_name} = fill_array(new_col_{col_name}_data, len_df+len_other)')
                column_list.append((f'new_col_{col_name}', col_name))

        for col_name, i in other_columns_indx.items():
            if col_name not in df_columns_indx:
                func_text.append(get_dataframe_column('other', col_name, i))
                func_text.append(f'new_col_{col_name}_data = init_series(new_col_{col_name}_data_other)._data')
                func_text.append(f'new_col_{col_name} = fill_array(new_col_{col_name}_data, len_df+len_other, push_back=False)')
                column_list.append((f'new_col_{col_name}', col_name))

        data = ', '.join(column for column, _ in column_list)
        # TODO: Handle index
        index = None
        col_names = ', '.join(f"'{column_name}'" for _, column_name in column_list)
        func_text.append(f"return sdc.hiframes.pd_dataframe_ext.init_dataframe({data}, {index}, {col_names})\n")

        func_definition.extend([spaces + func_line for func_line in func_text])

        func_def = '\n'.join(func_definition)

        loc_vars = {}
        exec(func_def, {'sdc': sdc, 'np': numpy, 'get_dataframe_data': sdc.hiframes.pd_dataframe_ext.get_dataframe_data,
                         'init_series': sdc.hiframes.api.init_series, 'fill_array': sdc.datatypes.common_functions.fill_array}, loc_vars)
        _append_impl = loc_vars['sdc_pandas_dataframe_append_impl']
        return _append_impl

    return sdc_pandas_dataframe_append_impl(df, other, name, args)


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


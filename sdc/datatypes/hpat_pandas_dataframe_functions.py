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
from numba.errors import TypingError
from sdc.hiframes.pd_dataframe_ext import DataFrameType
# from sdc.datatypes.hpat_pandas_dataframe_types import DataFrameType
from sdc.utils import sdc_overload_method
from sdc.datatypes.common_functions import TypeChecker
from sdc.str_arr_ext import cp_str_list_to_array


def sdc_pandas_dataframe_drop_codegen(func_name, func_args, df, drop_cols):
    """
    Func generated:

    def sdc_pandas_dataframe_drop_impl(df, labels=None, axis=0, index=None, columns=None, level=None, inplace=False,
    errors="raise"):
        new_col_B_data_df = get_dataframe_data(df, 1)
        new_col_C_data_df = get_dataframe_data(df, 2)
        return sdc.hiframes.pd_dataframe_ext.init_dataframe(new_col_B_data_df, new_col_C_data_df, None, 'B', 'C')

    """
    indent = 4 * ' '

    df_columns_indx = {col_name: i for i, col_name in enumerate(df.columns)}
    saved_df_columns = [column for column in df.columns if column not in drop_cols]

    func_definition = [f'def sdc_pandas_dataframe_{func_name}_impl({", ".join(func_args)}):']

    func_text = []
    column_list = []

    def get_dataframe_column(df, column, idx):
        return f'new_col_{column}_data_{df} = get_dataframe_data({df}, {idx})'

    for column in saved_df_columns:
        func_text.append(get_dataframe_column('df', column, df_columns_indx[column]))
        column_list.append((f'new_col_{column}_data_df', column))

    data = ', '.join(column for column, _ in column_list)
    # TODO: Handle index
    index = None
    col_names = ', '.join(f"'{column_name}'" for _, column_name in column_list)

    func_text.append(f"return sdc.hiframes.pd_dataframe_ext.init_dataframe({data}, {index}, {col_names})\n")
    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {'sdc': sdc, 'np': numpy, 'get_dataframe_data': sdc.hiframes.pd_dataframe_ext.get_dataframe_data}

    print(func_def)

    return func_def, global_vars


@sdc_overload_method(DataFrameType, 'drop')
def sdc_pandas_dataframe_drop(df, labels=None, axis=0, index=None, columns=None, level=None, inplace=False,
                              errors='raise'):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.drop
    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_drop.py
       :language: python
       :lines: 27-
       :caption: Drop specified columns from DataFrame
       Remove columns by specifying directly index or column names.
       :name: ex_dataframe_drop

    .. code-block:: console

        > python ./dataframe_drop.py
            B  C
        0  4  a
        1  5  b
        2  6  c
        3  7  d
        dtype: object

     .. note::
        Parameters axis, index, level, inplace, errors are currently unsupported by Intel Scalable Dataframe Compiler
        Currently multi-indexing is not supported.

    .. seealso::
        :ref:`DataFrame.loc <pandas.DataFrame.loc>`
            Label-location based indexer for selection by label.
        :ref:`DataFrame.dropna <pandas.DataFrame.dropna>`
            Return DataFrame with labels on given axis omitted where (all or any) data are missing.
        :ref:`DataFrame.drop_duplicates <pandas.DataFrame.drop_duplicates>`
            Return DataFrame with duplicate rows removed, optionally only considering certain columns.
        :ref:`Series.drop <pandas.Series.drop>`
            Return Series with specified index labels removed.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.drop` implementation.
    .. only:: developer
    Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_drop*
    Parameters
    -----------
    df: :obj:`pandas.DataFrame`
        input arg
    labels: single label or list-like
        Column labels to drop
        *unsupported*
    axis: :obj:`int` default 0
        *unsupported*
    index: single label or list-like
        *unsupported*
    columns: single label or list-like
    level: :obj:`int` or :obj:`str`
        For MultiIndex, level from which the labels will be removed.
        *unsupported*
    inplace: :obj:`bool` default False
        *unsupported*
    errors: :obj:`str` default 'raise'
        If 'ignore', suppress error and only existing labels are dropped.
        *unsupported*

    Returns
    -------
    :obj: `pandas.DataFrame`
        DataFrame without the removed index or column labels.

    Raises
    -------
    KeyError
        If any of the labels is not found in the selected axis.
    """

    _func_name = 'drop'

    ty_checker = TypeChecker(f'Method {_func_name}().')
    ty_checker.check(df, DataFrameType)

    if not isinstance(labels, types.Omitted) and labels is not None:
        ty_checker.raise_exc(labels, 'None', 'labels')

    if not isinstance(axis, (int, types.Omitted)):
        ty_checker.raise_exc(axis, 'int', 'axis')

    if not isinstance(index, types.Omitted) and index is not None:
        ty_checker.raise_exc(index, 'None', 'index')

    if not isinstance(columns, (types.Omitted,types.Literal, types.UnicodeType)):
        ty_checker.raise_exc(columns, 'str', 'columns')

    if not isinstance(level, (types.Omitted, types.Literal)) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(inplace, (bool, types.Omitted)) and inplace:
        ty_checker.raise_exc(inplace, 'bool', 'inplace')

    if not isinstance(errors, (str, types.Omitted, types.Literal, types.UnicodeType)):
        ty_checker.raise_exc(errors, 'str', 'errors')

    args = {'labels': None, 'axis': 0, 'index': None, 'columns': None, 'level': None, 'inplace': False,
            'errors': '"raise"'}

    def sdc_pandas_dataframe_drop_impl(df, _func_name, args, columns, errors):
        if args['axis'] != 0:
            raise ValueError('Method drop(). The object axis\n expected: 0')

        if isinstance(errors, types.Literal):
            errors = errors.literal_value
        errors_suppress = errors == 'raise'

        func_args = ['df']
        for key, value in args.items():
            if key not in func_args:
                if isinstance(value, types.Literal):
                    value = value.literal_value
                func_args.append('{}={}'.format(key, value))

        # Only drop by one column is supported
        if isinstance(columns, types.StringLiteral):
            drop_cols = (columns.literal_value,)
        else:
            raise ValueError('Constant list of columns is currently unsupported in df.drop()')

        for label_column in drop_cols:
            if label_column not in df.columns:
                if errors_suppress:
                    raise ValueError(f'The label {label_column} is not found in the selected axis')

        func_def, global_vars = sdc_pandas_dataframe_drop_codegen(_func_name, func_args, df, drop_cols)
        loc_vars = {}
        exec(func_def, global_vars, loc_vars)
        _drop_impl = loc_vars['sdc_pandas_dataframe_drop_impl']
        return _drop_impl

    return sdc_pandas_dataframe_drop_impl(df, _func_name, args, columns, errors)


@sdc_overload_method(DataFrameType, 'count')
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

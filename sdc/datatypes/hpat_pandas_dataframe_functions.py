# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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
import numba
import sdc


from pandas.core.indexing import IndexingError

from numba import types
from numba.special import literally
from numba.typed import List, Dict
from numba.errors import TypingError
from pandas.core.indexing import IndexingError

from sdc.hiframes.pd_dataframe_ext import DataFrameType
from sdc.hiframes.pd_series_type import SeriesType
from sdc.utilities.sdc_typing_utils import (TypeChecker, check_index_is_numeric,
                                            check_types_comparable, kwsparams2list,
                                            gen_impl_generator, find_common_dtype_from_numpy_dtypes)
from sdc.str_arr_ext import StringArrayType

from sdc.hiframes.pd_dataframe_type import DataFrameType
from sdc.hiframes.pd_series_type import SeriesType

from sdc.datatypes.hpat_pandas_dataframe_getitem_types import (DataFrameGetitemAccessorType,
                                                               dataframe_getitem_accessor_init)
from sdc.datatypes.common_functions import SDCLimitation
from sdc.datatypes.hpat_pandas_dataframe_rolling_types import _hpat_pandas_df_rolling_init
from sdc.datatypes.hpat_pandas_rolling_types import (
    gen_sdc_pandas_rolling_overload_body, sdc_pandas_rolling_docstring_tmpl)
from sdc.datatypes.hpat_pandas_groupby_functions import init_dataframe_groupby
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
from sdc.utilities.utils import sdc_overload, sdc_overload_method, sdc_overload_attribute
from sdc.hiframes.api import isna
from sdc.functions.numpy_like import getitem_by_mask
from sdc.datatypes.common_functions import _sdc_take, sdc_reindex_series

@sdc_overload_attribute(DataFrameType, 'index')
def hpat_pandas_dataframe_index(df):
    """
       Intel Scalable Dataframe Compiler User Guide
       ********************************************
       Pandas API: pandas.DataFrame.index

       Examples
       --------
       .. literalinclude:: ../../../examples/dataframe/dataframe_index.py
          :language: python
          :lines: 27-
          :caption: The index (row labels) of the DataFrame.
          :name: ex_dataframe_index

       .. command-output:: python ./dataframe/dataframe_index.py
           :cwd: ../../../examples

       Intel Scalable Dataframe Compiler Developer Guide
       *************************************************
       Pandas DataFrame attribute :attr:`pandas.DataFrame.index` implementation.
       .. only:: developer
       Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_index*
       Parameters
       -----------
       df: :obj:`pandas.DataFrame`
           input arg
       Returns
       -------
       :obj: `numpy.array`
           return the index of DataFrame
    """

    ty_checker = TypeChecker(f'Attribute index.')
    ty_checker.check(df, DataFrameType)

    if isinstance(df.index, types.NoneType) or df.index is None:
        empty_df = not df.columns

        def hpat_pandas_df_index_none_impl(df):
            df_len = len(df._data[0]) if empty_df == False else 0  # noqa

            return numpy.arange(df_len)

        return hpat_pandas_df_index_none_impl
    else:
        def hpat_pandas_df_index_impl(df):
            return df._index

        return hpat_pandas_df_index_impl


def sdc_pandas_dataframe_values_codegen(df, numba_common_dtype):
    """
    Input:
    column_len = 3
    numba_common_dtype = float64

    Func generated:
    def sdc_pandas_dataframe_values_impl(df):
        row_len = len(df._data[0])
        df_col_A = df._data[0]
        df_col_B = df._data[1]
        df_col_C = df._data[2]
        df_values = numpy.empty(row_len*3, numpy.dtype("float64"))
        for i in range(row_len):
            df_values[i * 3 + 0] = df_col_A[i]
            df_values[i * 3 + 1] = df_col_B[i]
            df_values[i * 3 + 2] = df_col_C[i]
        return df_values.reshape(row_len, 3)

    """

    indent = 4 * ' '
    func_args = ['df']

    func_definition = [f'def sdc_pandas_dataframe_values_impl({", ".join(func_args)}):']
    func_text = []
    column_list = []
    column_len = len(df.columns)
    func_text.append(f'row_len = len(df._data[0])')

    for index, column_name in enumerate(df.columns):
        func_text.append(f'df_col_{index} = df._data[{index}]')
        column_list.append(f'df_col_{index}')

    func_text.append(f'df_values = numpy.empty(row_len*{column_len}, numpy.dtype("{numba_common_dtype}"))')
    func_text.append('for i in range(row_len):')
    for j in range(column_len):
        func_text.append(indent + f'df_values[i * {column_len} + {j}] = {column_list[j]}[i]')

    func_text.append(f"return df_values.reshape(row_len, {column_len})\n")
    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {'pandas': pandas, 'numpy': numpy}

    return func_def, global_vars


@sdc_overload_attribute(DataFrameType, 'values')
def hpat_pandas_dataframe_values(df):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.values

    Limitations
    -----------
    Only numeric values supported as an output

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_values.py
      :language: python
      :lines: 27-
      :caption: The values data of the DataFrame.
      :name: ex_dataframe_values

    .. command-output:: python ./dataframe/dataframe_values.py
       :cwd: ../../../examples

    .. seealso::

        :ref:`DataFrame.to_numpy <pandas.DataFrame.to_numpy>`
            Recommended alternative to this method.
        :ref:`DataFrame.index <pandas.DataFrame.index>`
            Retrieve the index labels.
        :ref:`DataFrame.columns <pandas.DataFrame.columns>`
            Retrieving the column names.

    .. note::

        The dtype will be a lower-common-denominator dtype (implicit upcasting);
        that is to say if the dtypes (even of numeric types) are mixed, the one that accommodates all will be chosen.
        Use this with care if you are not dealing with the blocks.
        e.g. If the dtypes are float16 and float32, dtype will be upcast to float32. If dtypes are int32 and uint8,
        dtype will be upcast to int32. By numpy.find_common_type() convention,
        mixing int64 and uint64 will result in a float64 dtype.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame attribute :attr:`pandas.DataFrame.values` implementation.
    .. only:: developer
    Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_values*
    Parameters
    -----------
    df: :obj:`pandas.DataFrame`
       input arg
    Returns
    -------
    :obj: `numpy.ndarray`
       return a Numpy representation of the DataFrame
    """

    func_name = 'Attribute values.'
    ty_checker = TypeChecker(func_name)
    ty_checker.check(df, DataFrameType)

    # TODO: Handle StringArrayType
    for i, column in enumerate(df.data):
        if isinstance(column, StringArrayType):
            ty_checker.raise_exc(column, 'Numeric type', f'df.data["{df.columns[i]}"]')

    numba_common_dtype = find_common_dtype_from_numpy_dtypes([column.dtype for column in df.data], [])

    def hpat_pandas_df_values_impl(df, numba_common_dtype):
        loc_vars = {}
        func_def, global_vars = sdc_pandas_dataframe_values_codegen(df, numba_common_dtype)

        exec(func_def, global_vars, loc_vars)
        _values_impl = loc_vars['sdc_pandas_dataframe_values_impl']
        return _values_impl

    return hpat_pandas_df_values_impl(df, numba_common_dtype)


def sdc_pandas_dataframe_append_codegen(df, other, _func_name, args):
    """
    Input:
    df = pd.DataFrame({'A': ['cat', 'dog', np.nan], 'B': [.2, .3, np.nan]})
    other = pd.DataFrame({'A': ['bird', 'fox', 'mouse'], 'C': ['a', np.nan, '']})
    Func generated:
    def sdc_pandas_dataframe_append_impl(df, other, ignore_index=True, verify_integrity=False, sort=None):
        len_df = len(df._data[0])
        len_other = len(other._data[0])
        new_col_A_data_df = df._data[0]
        new_col_A_data_other = other._data[0]
        new_col_A = init_series(new_col_A_data_df).append(init_series(new_col_A_data_other))._data
        new_col_B_data_df = df._data[1]
        new_col_B_data = init_series(new_col_B_data_df)._data
        new_col_B = fill_array(new_col_B_data, len_df+len_other)
        new_col_C_data_other = other._data[0]
        new_col_C_data = init_series(new_col_C_data_other)._data
        new_col_C = fill_str_array(new_col_C_data, len_df+len_other, push_back=False)
        return pandas.DataFrame({"A": new_col_A, "B": new_col_B, "C": new_col_C)
    """
    indent = 4 * ' '
    func_args = ['df', 'other']

    for key, value in args:
        # TODO: improve check
        if key not in func_args:
            if isinstance(value, types.Literal):
                value = value.literal_value
            func_args.append(f'{key}={value}')

    df_columns_indx = {col_name: i for i, col_name in enumerate(df.columns)}
    other_columns_indx = {col_name: i for i, col_name in enumerate(other.columns)}

    # Keep columns that are StringArrayType
    string_type_columns = set(col_name for typ, col_name in zip(df.data, df.columns)
                              if isinstance(typ, StringArrayType))

    for typ, col_name in zip(other.data, other.columns):
        if isinstance(typ, StringArrayType):
            string_type_columns.add(col_name)

    func_definition = [f'def sdc_pandas_dataframe_{_func_name}_impl({", ".join(func_args)}):']
    func_text = []
    column_list = []

    func_text.append(f'len_df = len(df._data[0])')
    func_text.append(f'len_other = len(other._data[0])')

    for col_name, col_id in df_columns_indx.items():
        func_text.append(f'new_col_{col_id}_data_{"df"} = {"df"}._data[{col_id}]')
        if col_name in other_columns_indx:
            other_col_id = other_columns_indx.get(col_name)
            func_text.append(f'new_col_{col_id}_data_{"other"} = '
                             f'{"other"}._data[{other_columns_indx.get(col_name)}]')
            s1 = f'init_series(new_col_{col_id}_data_{"df"})'
            s2 = f'init_series(new_col_{col_id}_data_{"other"})'
            func_text.append(f'new_col_{col_id} = {s1}.append({s2})._data')
        else:
            func_text.append(f'new_col_{col_id}_data = init_series(new_col_{col_id}_data_df)._data')
            if col_name in string_type_columns:
                func_text.append(f'new_col_{col_id} = fill_str_array(new_col_{col_id}_data, len_df+len_other)')
            else:
                func_text.append(f'new_col_{col_id} = fill_array(new_col_{col_id}_data, len_df+len_other)')
        column_list.append((f'new_col_{col_id}', col_name))

    for col_name, col_id in other_columns_indx.items():
        if col_name not in df_columns_indx:
            func_text.append(f'new_col_{col_id}_data_{"other"} = {"other"}._data[{col_id}]')
            func_text.append(f'new_col_{col_id}_data = init_series(new_col_{col_id}_data_other)._data')
            if col_name in string_type_columns:
                func_text.append(
                    f'new_col_{col_id}_other = '
                    f'fill_str_array(new_col_{col_id}_data, len_df+len_other, push_back=False)')
            else:
                func_text.append(f'new_col_{col_id}_other = '
                                 f'fill_array(new_col_{col_id}_data, len_df+len_other, push_back=False)')
            column_list.append((f'new_col_{col_id}_other', col_name))

    data = ', '.join(f'"{column_name}": {column}' for column, column_name in column_list)
    # TODO: Handle index
    func_text.append(f"return pandas.DataFrame({{{data}}})\n")
    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {'pandas': pandas,
                   'init_series': sdc.hiframes.api.init_series,
                   'fill_array': sdc.datatypes.common_functions.fill_array,
                   'fill_str_array': sdc.datatypes.common_functions.fill_str_array}

    return func_def, global_vars


@sdc_overload_method(DataFrameType, 'append')
def sdc_pandas_dataframe_append(df, other, ignore_index=True, verify_integrity=False, sort=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.append

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_append.py
        :language: python
        :lines: 37-
        :caption: Appending rows of other to the end of caller, returning a new object. Columns in other that are not
                  in the caller are added as new columns.
        :name: ex_dataframe_append

    .. command-output:: python ./dataframe/dataframe_append.py
        :cwd: ../../../examples

    .. note::
        Parameter ignore_index, verify_integrity, sort are currently unsupported
        by Intel Scalable Dataframe Compiler
        Currently only pandas.DataFrame is supported as "other" parameter

    .. seealso::
        `pandas.concat <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html>`_
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

    _func_name = 'append'

    ty_checker = TypeChecker(f'Method {_func_name}().')
    ty_checker.check(df, DataFrameType)
    # TODO: support other array-like types
    ty_checker.check(other, DataFrameType)
    # TODO: support index in series from df-columns
    if not isinstance(ignore_index, (bool, types.Boolean, types.Omitted)) and not ignore_index:
        ty_checker.raise_exc(ignore_index, 'boolean', 'ignore_index')

    if not isinstance(verify_integrity, (bool, types.Boolean, types.Omitted)) and verify_integrity:
        ty_checker.raise_exc(verify_integrity, 'boolean', 'verify_integrity')

    if not isinstance(sort, (bool, types.Boolean, types.Omitted)) and sort is not None:
        ty_checker.raise_exc(sort, 'boolean, None', 'sort')

    args = (('ignore_index', True), ('verify_integrity', False), ('sort', None))

    def sdc_pandas_dataframe_append_impl(df, other, _func_name, args):
        loc_vars = {}
        func_def, global_vars = sdc_pandas_dataframe_append_codegen(df, other, _func_name, args)

        exec(func_def, global_vars, loc_vars)
        _append_impl = loc_vars['sdc_pandas_dataframe_append_impl']
        return _append_impl

    return sdc_pandas_dataframe_append_impl(df, other, _func_name, args)


# Example func_text for func_name='count' columns=('A', 'B'):
#
#         def _df_count_impl(df, axis=0, level=None, numeric_only=False):
#           series_A = init_series(df._data[0])
#           result_A = series_A.count(level=level)
#           series_B = init_series(df._data[1])
#           result_B = series_B.count(level=level)
#           return pandas.Series([result_A, result_B], ['A', 'B'])


def _dataframe_reduce_columns_codegen(func_name, func_params, series_params, columns):
    result_name_list = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    for i, c in enumerate(columns):
        result_c = f'result_{i}'
        func_lines += [f'  series_{i} = pandas.Series({func_params[0]}._data[{i}])',
                       f'  {result_c} = series_{i}.{func_name}({series_params})']
        result_name_list.append(result_c)
    all_results = ', '.join(result_name_list)
    all_columns = ', '.join([f"'{c}'" for c in columns])

    func_lines += [f'  return pandas.Series([{all_results}], [{all_columns}])']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas}

    return func_text, global_vars


def sdc_pandas_dataframe_reduce_columns(df, func_name, params, ser_params):
    all_params = ['df']
    ser_par = []

    for key, value in params.items():
        all_params.append('{}={}'.format(key, value))
    for key, value in ser_params.items():
        ser_par.append('{}={}'.format(key, value))

    s_par = '{}'.format(', '.join(ser_par[:]))

    df_func_name = f'_df_{func_name}_impl'

    func_text, global_vars = _dataframe_reduce_columns_codegen(func_name, all_params, s_par, df.columns)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars[df_func_name]

    return _reduce_impl


def _dataframe_reduce_columns_codegen_head(func_name, func_params, series_params, columns, df):
    """
    Example func_text for func_name='head' columns=('float', 'int', 'string'):

        def _df_head_impl(df, n=5):
            series_float = pandas.Series(df._data[0])
            result_float = series_float.head(n=n)
            series_int = pandas.Series(df._data[1])
            result_int = series_int.head(n=n)
            series_string = pandas.Series(df._data[2])
            result_string = series_string.head(n=n)
            return pandas.DataFrame({"float": result_float, "int": result_int, "string": result_string},
                                    index = df._index[:n])
    """
    results = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl(df, {joined}):']
    ind = df_index_codegen_head(df)
    for i, c in enumerate(columns):
        result_c = f'result_{c}'
        func_lines += [f'  series_{c} = pandas.Series(df._data[{i}])',
                       f'  {result_c} = series_{c}.{func_name}({series_params})']
        results.append((columns[i], result_c))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, {ind})']
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas}

    return func_text, global_vars


def sdc_pandas_dataframe_head_codegen(df, func_name, params, ser_params):
    all_params = kwsparams2list(params)
    ser_par = kwsparams2list(ser_params)
    s_par = ', '.join(ser_par)

    df_func_name = f'_df_{func_name}_impl'
    func_text, global_vars = _dataframe_reduce_columns_codegen_head(func_name, all_params, s_par, df.columns, df)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars[df_func_name]

    return _reduce_impl


def df_index_codegen_head(self):
    # TODO: Rewrite when DF constructor will be fixed with index=None
    if isinstance(self.index, types.NoneType):
        return ''

    return 'index=df._index[:n]'


@sdc_overload_method(DataFrameType, 'head')
def head_overload(df, n=5):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.head

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_head.py
       :language: python
       :lines: 37-
       :caption: Return the first n rows.
       :name: ex_dataframe_head

    .. command-output:: python ./dataframe/dataframe_head.py
       :cwd: ../../../examples

    .. seealso::

        :ref:`DataFrame.tail <pandas.DataFrame.tail>`

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas Series method :meth:`pandas.Series.head` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_head*
    """
    name = 'head'

    if isinstance(n, types.Omitted):
        n = n.value

    params = {'n': 5}
    ser_par = {'n': 'n'}
    return sdc_pandas_dataframe_head_codegen(df, name, params, ser_par)


def _dataframe_codegen_copy(func_params, series_params, df):
    """
    Example func_text for func_name='copy' columns=('A', 'B', 'C'):
        def _df_copy_impl(df, deep=True):
            series_0 = pandas.Series(df._data[0])
            result_0 = series_0.copy(deep=deep)
            series_1 = pandas.Series(df._data[1])
            result_1 = series_1.copy(deep=deep)
            return pandas.DataFrame({"A": result_0, "B": result_1}, index=df._index)
    """
    results = []
    series_params_str = ', '.join(kwsparams2list(series_params))
    func_params_str = ', '.join(kwsparams2list(func_params))
    func_lines = [f"def _df_copy_impl(df, {func_params_str}):"]
    index = df_index_codegen_all(df)
    for i, c in enumerate(df.columns):
        result_c = f"result_{i}"
        func_lines += [f"  series_{i} = pandas.Series(df._data[{i}], name='{c}')",
                       f"  {result_c} = series_{i}.copy({series_params_str})"]
        results.append((c, result_c))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f"  return pandas.DataFrame({{{data}}}, {index})"]
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas}

    return func_text, global_vars


def sdc_pandas_dataframe_copy_codegen(df, params, series_params):
    func_text, global_vars = _dataframe_codegen_copy(params, series_params, df)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars['_df_copy_impl']

    return _reduce_impl


def df_index_codegen_all(self):
    if isinstance(self.index, types.NoneType):
        return ''

    return 'index=df._index'


@sdc_overload_method(DataFrameType, 'copy')
def copy_overload(df, deep=True):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.copy

    Limitations
    -----------
    - Parameter deep=False is currently unsupported for indexes by Intel Scalable Dataframe Compiler

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_copy.py
       :language: python
       :lines: 36-
       :caption: Make a copy of this object’s indices and data.
       :name: ex_dataframe_copy

    .. command-output:: python ./dataframe/dataframe_copy.py
       :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.Series.copy` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_copy*
    """
    ty_checker = TypeChecker("Method copy().")
    ty_checker.check(df, DataFrameType)

    if not (isinstance(deep, (types.Omitted, types.Boolean, types.NoneType)) or deep is True):
        ty_checker.raise_exc(deep, 'boolean', 'deep')

    params = {'deep': True}
    series_params = {'deep': 'deep'}
    return sdc_pandas_dataframe_copy_codegen(df, params, series_params)


def _dataframe_apply_columns_codegen(func_name, func_params, series_params, columns):
    result_name = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    for i, c in enumerate(columns):
        result_c = f'result_{i}'
        func_lines += [f'  series_{i} = pandas.Series({func_params[0]}._data[{i}])',
                       f'  {result_c} = series_{i}.{func_name}({series_params})']
        result_name.append((result_c, c))

    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)

    func_lines += [f'  return pandas.DataFrame({{{data}}})']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas}

    return func_text, global_vars


def sdc_pandas_dataframe_apply_columns(df, func_name, params, ser_params):
    all_params = ['df']
    ser_par = []

    def kwsparams2list(params):
        return ['{}={}'.format(k, v) for k, v in params.items()]

    all_params = ['df'] + kwsparams2list(params)
    ser_par = kwsparams2list(ser_params)

    s_par = ', '.join(ser_par)

    df_func_name = f'_df_{func_name}_impl'

    func_text, global_vars = _dataframe_apply_columns_codegen(func_name, all_params, s_par, df.columns)

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _apply_impl = loc_vars[df_func_name]

    return _apply_impl


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


@sdc_overload_method(DataFrameType, 'median')
def median_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.median` implementation.

       .. only:: developer

           Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_median*

       Parameters
       -----------
       df: :class:`pandas.DataFrame`
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

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'mean')
def mean_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.mean` implementation.

       .. only:: developer

           Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_mean*

       Parameters
       -----------
       df: :class:`pandas.DataFrame`
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

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


sdc_pandas_dataframe_rolling = sdc_overload_method(DataFrameType, 'rolling')(
    gen_sdc_pandas_rolling_overload_body(_hpat_pandas_df_rolling_init, DataFrameType))
sdc_pandas_dataframe_rolling.__doc__ = sdc_pandas_rolling_docstring_tmpl.format(
    ty='DataFrame', ty_lower='dataframe')


@sdc_overload_method(DataFrameType, 'std')
def std_overload(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.std` implementation.

       .. only:: developer

           Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_std*

       Parameters
       -----------
       df: :class:`pandas.DataFrame`
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

    params = {'axis': None, 'skipna': None, 'level': None, 'ddof': 1, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level', 'ddof': 'ddof'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'var')
def var_overload(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.var` implementation.

       .. only:: developer

           Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_var*

       Parameters
       -----------
       df: :class:`pandas.DataFrame`
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

    params = {'axis': None, 'skipna': None, 'level': None, 'ddof': 1, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level', 'ddof': 'ddof'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'max')
def max_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.max` implementation.

       .. only:: developer

           Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_max*

       Parameters
       -----------
       df: :class:`pandas.DataFrame`
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

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'min')
def min_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.min` implementation.

       .. only:: developer

           Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_min*

       Parameters
       -----------
       df: :class:`pandas.DataFrame`
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

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'sum')
def sum_overload(df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.sum` implementation.

       .. only:: developer

           Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_sum*

       Parameters
       -----------
       df: :class:`pandas.DataFrame`
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

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None, 'min_count': 0}
    ser_par = {'skipna': 'skipna', 'level': 'level', 'min_count': 'min_count'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'prod')
def prod_overload(df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0):
    """
       Pandas DataFrame method :meth:`pandas.DataFrame.prod` implementation.

       .. only:: developer

           Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_prod*

       Parameters
       -----------
       df: :class:`pandas.DataFrame`
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

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None, 'min_count': 0}
    ser_par = {'skipna': 'skipna', 'level': 'level', 'min_count': 'min_count'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'count')
def count_overload(df, axis=0, level=None, numeric_only=False):
    """
    Pandas DataFrame method :meth:`pandas.DataFrame.count` implementation.
    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_count*

    Parameters
    -----------
    df: :class:`pandas.DataFrame`
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

    params = {'axis': 0, 'level': None, 'numeric_only': False}
    ser_par = {'level': 'level'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


def sdc_pandas_dataframe_drop_codegen(func_name, func_args, df, drop_cols):
    """
    Input:
    df.drop(columns='M', errors='ignore')

    Func generated:
    def sdc_pandas_dataframe_drop_impl(df, labels=None, axis=0, index=None, columns=None, level=None, inplace=False,
     errors="raise"):
        if errors == "raise":
          raise ValueError("The label M is not found in the selected axis")
        new_col_A_data_df = df._data[0]
        new_col_B_data_df = df._data[1]
        new_col_C_data_df = df._data[2]
        return pandas.DataFrame({"A": new_col_A_data_df, "B": new_col_B_data_df, "C": new_col_C_data_df})

    """
    indent = 4 * ' '
    df_columns_indx = {col_name: i for i, col_name in enumerate(df.columns)}
    saved_df_columns = [column for column in df.columns if column not in drop_cols]
    func_definition = [f'def sdc_pandas_dataframe_{func_name}_impl({", ".join(func_args)}):']
    func_text = []
    column_list = []

    for label in drop_cols:
        if label not in df.columns:
            func_text.append(f'if errors == "raise":')
            func_text.append(indent + f'raise ValueError("The label {label} is not found in the selected axis")')
            break

    for column_id, column_name in enumerate(saved_df_columns):
        func_text.append(f'new_col_{column_id}_data_{"df"} = {"df"}._data['
                         f'{df_columns_indx[column_name]}]')
        column_list.append((f'new_col_{column_id}_data_df', column_name))

    data = ', '.join(f'"{column_name}": {column}' for column, column_name in column_list)
    index = 'df.index'
    func_text.append(f"return pandas.DataFrame({{{data}}}, index={index})\n")
    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {'pandas': pandas}

    return func_def, global_vars


def _dataframe_codegen_isna(func_name, columns, df):
    """
    Example func_text for func_name='isna' columns=('float', 'int', 'string'):

        def _df_isna_impl(df):
            series_float = pandas.Series(df._data[0])
            result_float = series_float.isna()
            series_int = pandas.Series(df._data[1])
            result_int = series_int.isna()
            series_string = pandas.Series(df._data[2])
            result_string = series_string.isna()
            return pandas.DataFrame({"float": result_float, "int": result_int, "string": result_string},
                                    index = df._index)
    """
    results = []
    func_lines = [f'def _df_{func_name}_impl(df):']
    index = df_index_codegen_all(df)
    for i, c in enumerate(columns):
        result_c = f'result_{c}'
        func_lines += [f'  series_{c} = pandas.Series(df._data[{i}])',
                       f'  {result_c} = series_{c}.{func_name}()']
        results.append((columns[i], result_c))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, {index})']
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas}

    return func_text, global_vars


def sdc_pandas_dataframe_isna_codegen(df, func_name):
    df_func_name = f'_df_{func_name}_impl'
    func_text, global_vars = _dataframe_codegen_isna(func_name, df.columns, df)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars[df_func_name]

    return _reduce_impl


def df_index_codegen_all(self):
    if isinstance(self.index, types.NoneType):
        return ''
    return 'index=df._index'


@sdc_overload_method(DataFrameType, 'isna')
def isna_overload(df):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.isna

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_isna.py
       :language: python
       :lines: 35-
       :caption: Detect missing values.
       :name: ex_dataframe_isna

    .. command-output:: python ./dataframe/dataframe_isna.py
       :cwd: ../../../examples

    .. seealso::

        :ref:`DataFrame.isnull <pandas.DataFrame.isnull>`
            Alias of isna.

        :ref:`DataFrame.notna <pandas.DataFrame.notna>`
            Boolean inverse of isna.

        :ref:`DataFrame.dropna <pandas.DataFrame.dropna>`
            Omit axes labels with missing values.

        `pandas.absolute <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isna.html#pandas.isna>`_
            Top-level isna.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.isna` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_isna*
    """

    return sdc_pandas_dataframe_isna_codegen(df, 'isna')


@sdc_overload_method(DataFrameType, 'drop')
def sdc_pandas_dataframe_drop(df, labels=None, axis=0, index=None, columns=None, level=None, inplace=False,
                              errors='raise'):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.drop

    Limitations
    -----------
    Parameter columns is expected to be a Literal value with one column name or Tuple with columns names.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_drop.py
        :language: python
        :lines: 37-
        :caption: Drop specified columns from DataFrame.
        :name: ex_dataframe_drop

    .. command-output:: python ./dataframe/dataframe_drop.py
        :cwd: ../../../examples

     .. note::
        Parameters axis, index, level, inplace, errors are currently unsupported
        by Intel Scalable Dataframe Compiler
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
    Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_drop*
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

    if not isinstance(columns, (types.Omitted, types.Tuple, types.Literal)):
        ty_checker.raise_exc(columns, 'str, tuple of str', 'columns')

    if not isinstance(level, (types.Omitted, types.Literal)) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(inplace, (bool, types.Omitted)) and inplace:
        ty_checker.raise_exc(inplace, 'bool', 'inplace')

    if not isinstance(errors, (str, types.Omitted, types.Literal)):
        ty_checker.raise_exc(errors, 'str', 'errors')

    args = {'labels': None, 'axis': 0, 'index': None, 'columns': None, 'level': None, 'inplace': False,
            'errors': f'"raise"'}

    def sdc_pandas_dataframe_drop_impl(df, _func_name, args, columns):
        func_args = ['df']
        for key, value in args.items():
            if key not in func_args:
                if isinstance(value, types.Literal):
                    value = value.literal_value
                func_args.append(f'{key}={value}')

        if isinstance(columns, types.StringLiteral):
            drop_cols = (columns.literal_value,)
        elif isinstance(columns, types.Tuple):
            drop_cols = tuple(column.literal_value for column in columns)
        else:
            raise ValueError('Only drop by one column or tuple of columns is currently supported in df.drop()')

        func_def, global_vars = sdc_pandas_dataframe_drop_codegen(_func_name, func_args, df, drop_cols)
        loc_vars = {}
        exec(func_def, global_vars, loc_vars)
        _drop_impl = loc_vars['sdc_pandas_dataframe_drop_impl']
        return _drop_impl

    return sdc_pandas_dataframe_drop_impl(df, _func_name, args, columns)


def df_length_expr(self):
    """Generate expression to get length of DF"""
    if self.columns:
        return 'len(self._data[0])'

    return '0'


def df_index_expr(self, length_expr=None):
    """Generate expression to get or create index of DF"""
    if isinstance(self.index, types.NoneType):
        if length_expr is None:
            length_expr = df_length_expr(self)

        return f'numpy.arange({length_expr})'

    return 'self._index'


def df_getitem_slice_idx_main_codelines(self, idx):
    """Generate main code lines for df.getitem with idx of slice"""
    results = []
    func_lines = [f'  res_index = {df_index_expr(self)}']
    for i, col in enumerate(self.columns):
        res_data = f'res_data_{i}'
        func_lines += [
            f'  {res_data} = pandas.Series((self._data[{i}])[idx], index=res_index[idx], name="{col}")'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index[idx])']

    return func_lines


def df_getitem_tuple_idx_main_codelines(self, literal_idx):
    """Generate main code lines for df.getitem with idx of tuple"""
    results = []
    func_lines = [f'  res_index = {df_index_expr(self)}']
    needed_cols = {col: i for i, col in enumerate(self.columns) if col in literal_idx}
    for col, i in needed_cols.items():
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = self._data [{i}]',
            f'  {res_data} = pandas.Series(data_{i}, index=res_index, name="{col}")'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index)']

    return func_lines


def df_getitem_bool_series_idx_main_codelines(self, idx):
    """Generate main code lines for df.getitem"""

    # optimization for default indexes in df and idx when index alignment is trivial
    if (isinstance(self.index, types.NoneType) and isinstance(idx.index, types.NoneType)):
        func_lines = [f'  length = {df_length_expr(self)}',
                      f'  if length > len(idx):',
                      f'    msg = "Unalignable boolean Series provided as indexer " + \\',
                      f'          "(index of the boolean Series and of the indexed object do not match)."',
                      f'    raise IndexingError(msg)',
                      f'  # do not trim idx._data to length as getitem_by_mask handles such case',
                      f'  res_index = getitem_by_mask(self.index, idx._data)',
                      f'  # df index is default, same as positions so it can be used in take']
        results = []
        for i, col in enumerate(self.columns):
            res_data = f'res_data_{i}'
            func_lines += [
                f'  data_{i} = self._data[{i}]',
                f'  {res_data} = sdc_take(data_{i}, res_index)'
            ]
            results.append((col, res_data))

        data = ', '.join(f'"{col}": {data}' for col, data in results)
        func_lines += [
            f'  return pandas.DataFrame({{{data}}}, index=res_index)'
        ]
    else:
        func_lines = [f'  length = {df_length_expr(self)}',
                      f'  self_index = self.index',
                      f'  idx_reindexed = sdc_reindex_series(idx._data, idx.index, idx._name, self_index)',
                      f'  res_index = getitem_by_mask(self_index, idx_reindexed._data)',
                      f'  selected_pos = getitem_by_mask(numpy.arange(length), idx_reindexed._data)']

        results = []
        for i, col in enumerate(self.columns):
            res_data = f'res_data_{i}'
            func_lines += [
                f'  data_{i} = self._data[{i}]',
                f'  {res_data} = sdc_take(data_{i}, selected_pos)'
            ]
            results.append((col, res_data))

        data = ', '.join(f'"{col}": {data}' for col, data in results)
        func_lines += [
            f'  return pandas.DataFrame({{{data}}}, index=res_index)'
        ]

    return func_lines


def df_getitem_bool_array_idx_main_codelines(self, idx):
    """Generate main code lines for df.getitem"""
    func_lines = [f'  length = {df_length_expr(self)}',
                  f'  if length != len(idx):',
                  f'    raise ValueError("Item wrong length.")',
                  f'  taken_pos = getitem_by_mask(numpy.arange(length), idx)',
                  f'  res_index = sdc_take(self.index, taken_pos)']
    results = []
    for i, col in enumerate(self.columns):
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = self._data[{i}]',
            f'  {res_data} = sdc_take(data_{i}, taken_pos)'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [
        f'  return pandas.DataFrame({{{data}}}, index=res_index)'
    ]

    return func_lines


def df_getitem_key_error_codelines():
    """Generate code lines to raise KeyError"""
    return ['  raise KeyError("Column is not in the DataFrame")']


def df_getitem_slice_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_slice_idx_impl(self, idx)
          res_index = self._index
          data_0 = self._data[0]
          res_data_0 = pandas.Series(data_0[idx], index=res_index[idx], name="A")
          data_1 = self._data [1]
          res_data_1 = pandas.Series(data_1[idx], index=res_index, name="B")
          return pandas.DataFrame({"A": res_data_0, "B": res_data_1}, index=res_index[idx])
    """
    func_lines = ['def _df_getitem_slice_idx_impl(self, idx):']
    if self.columns:
        func_lines += df_getitem_slice_idx_main_codelines(self, idx)
    else:
        # raise KeyError if input DF is empty
        func_lines += df_getitem_key_error_codelines()
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy}

    return func_text, global_vars


def df_getitem_tuple_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_tuple_idx_impl(self, idx)
          res_index = self._index
          data_1 = self._data[1]
          res_data_1 = pandas.Series(data_1, index=res_index, name="B")
          data_2 = self._data[2]
          res_data_2 = pandas.Series(data_2, index=res_index, name="C")
          return pandas.DataFrame({"B": res_data_1, "C": res_data_2}, index=res_index)
    """
    func_lines = ['def _df_getitem_tuple_idx_impl(self, idx):']
    literal_idx = {col.literal_value for col in idx}
    key_error = any(i not in self.columns for i in literal_idx)

    if self.columns and not key_error:
        func_lines += df_getitem_tuple_idx_main_codelines(self, literal_idx)
    else:
        # raise KeyError if input DF is empty or idx is invalid
        func_lines += df_getitem_key_error_codelines()

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy}

    return func_text, global_vars


def df_getitem_bool_series_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_bool_series_idx_impl(self, idx):
          length = len(self._data[0])
          if length > len(idx):
            msg = "Unalignable boolean Series provided as indexer " + \
                  "(index of the boolean Series and of the indexed object do not match)."
            raise IndexingError(msg)
          # do not trim idx._data to length as getitem_by_mask handles such case
          res_index = getitem_by_mask(self.index, idx._data)
          # df index is default, same as positions so it can be used in take
          data_0 = self._data[0]
          res_data_0 = sdc_take(data_0, res_index)
          data_1 = self._data[1]
          res_data_1 = sdc_take(data_1, res_index)
          return pandas.DataFrame({"A": res_data_0, "B": res_data_1}, index=res_index)
    """
    func_lines = ['def _df_getitem_bool_series_idx_impl(self, idx):']
    func_lines += df_getitem_bool_series_idx_main_codelines(self, idx)
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'getitem_by_mask': getitem_by_mask,
                   'sdc_take': _sdc_take,
                   'sdc_reindex_series': sdc_reindex_series,
                   'IndexingError': IndexingError}

    return func_text, global_vars


def df_getitem_bool_array_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_bool_array_idx_impl(self, idx):
          length = len(self._data[0])
          if length != len(idx):
            raise ValueError("Item wrong length.")
          taken_pos = getitem_by_mask(numpy.arange(length), idx)
          res_index = sdc_take(self.index, taken_pos)
          data_0 = self._data[0]
          res_data_0 = sdc_take(data_0, taken_pos)
          data_1 = self._data[1]
          res_data_1 = sdc_take(data_1, taken_pos)
          return pandas.DataFrame({"A": res_data_0, "B": res_data_1}, index=res_index)
    """
    func_lines = ['def _df_getitem_bool_array_idx_impl(self, idx):']
    func_lines += df_getitem_bool_array_idx_main_codelines(self, idx)
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'getitem_by_mask': getitem_by_mask,
                   'sdc_take': _sdc_take}

    return func_text, global_vars


gen_df_getitem_slice_idx_impl = gen_impl_generator(
    df_getitem_slice_idx_codegen, '_df_getitem_slice_idx_impl')
gen_df_getitem_tuple_idx_impl = gen_impl_generator(
    df_getitem_tuple_idx_codegen, '_df_getitem_tuple_idx_impl')
gen_df_getitem_bool_series_idx_impl = gen_impl_generator(
    df_getitem_bool_series_idx_codegen, '_df_getitem_bool_series_idx_impl')
gen_df_getitem_bool_array_idx_impl = gen_impl_generator(
    df_getitem_bool_array_idx_codegen, '_df_getitem_bool_array_idx_impl')


@sdc_overload(operator.getitem)
def sdc_pandas_dataframe_getitem(self, idx):
    """
        Intel Scalable Dataframe Compiler User Guide
        ********************************************
        Pandas API: pandas.DataFrame.getitem

        Get data from a DataFrame by indexer.

        Limitations
        -----------
        Supported ``key`` can be one of the following:

        * String literal, e.g. :obj:`df['A']`
        * A slice, e.g. :obj:`df[2:5]`
        * A tuple of string, e.g. :obj:`df[('A', 'B')]`
        * An array of booleans, e.g. :obj:`df[True,False]`
        * A series of booleans, e.g. :obj:`df(series([True,False]))`

        Supported getting a column through getting attribute.

        Examples
        --------
        .. literalinclude:: ../../../examples/dataframe/getitem/df_getitem_attr.py
           :language: python
           :lines: 37-
           :caption: Getting Pandas DataFrame column through getting attribute.
           :name: ex_dataframe_getitem

        .. command-output:: python ./dataframe/getitem/df_getitem_attr.py
           :cwd: ../../../examples

        .. literalinclude:: ../../../examples/dataframe/getitem/df_getitem.py
           :language: python
           :lines: 37-
           :caption: Getting Pandas DataFrame column where key is a string.
           :name: ex_dataframe_getitem

        .. command-output:: python ./dataframe/getitem/df_getitem.py
           :cwd: ../../../examples

        .. literalinclude:: ../../../examples/dataframe/getitem/df_getitem_slice.py
           :language: python
           :lines: 34-
           :caption: Getting slice of Pandas DataFrame.
           :name: ex_dataframe_getitem

        .. command-output:: python ./dataframe/getitem/df_getitem_slice.py
           :cwd: ../../../examples

        .. literalinclude:: ../../../examples/dataframe/getitem/df_getitem_tuple.py
           :language: python
           :lines: 37-
           :caption: Getting Pandas DataFrame elements where key is a tuple of strings.
           :name: ex_dataframe_getitem

        .. command-output:: python ./dataframe/getitem/df_getitem_tuple.py
           :cwd: ../../../examples

        .. literalinclude:: ../../../examples/dataframe/getitem/df_getitem_array.py
           :language: python
           :lines: 34-
           :caption: Getting Pandas DataFrame elements where key is an array of booleans.
           :name: ex_dataframe_getitem

        .. command-output:: python ./dataframe/getitem/df_getitem_array.py
           :cwd: ../../../examples

        .. literalinclude:: ../../../examples/dataframe/getitem/df_getitem_series.py
           :language: python
           :lines: 34-
           :caption: Getting Pandas DataFrame elements where key is series of booleans.
           :name: ex_dataframe_getitem

        .. command-output:: python ./dataframe/getitem/df_getitem_series.py
           :cwd: ../../../examples

        .. seealso::
            :ref:`Series.getitem <pandas.Series.getitem>`
                Get value(s) of Series by key.
            :ref:`Series.setitem <pandas.Series.setitem>`
                Set value to Series by index
            :ref:`Series.loc <pandas.Series.loc>`
                Access a group of rows and columns by label(s) or a boolean array.
            :ref:`Series.iloc <pandas.Series.iloc>`
                Purely integer-location based indexing for selection by position.
            :ref:`Series.at <pandas.Series.at>`
                Access a single value for a row/column label pair.
            :ref:`Series.iat <pandas.Series.iat>`
                Access a single value for a row/column pair by integer position.
            :ref:`DataFrame.setitem <pandas.DataFrame.setitem>`
                Set value to DataFrame by index
            :ref:`DataFrame.loc <pandas.DataFrame.loc>`
                Access a group of rows and columns by label(s) or a boolean array.
            :ref:`DataFrame.iloc <pandas.DataFrame.iloc>`
                Purely integer-location based indexing for selection by position.
            :ref:`DataFrame.at <pandas.DataFrame.at>`
                Access a single value for a row/column label pair.
            :ref:`DataFrame.iat <pandas.DataFrame.iat>`
                Access a single value for a row/column pair by integer position.

        Intel Scalable Dataframe Compiler Developer Guide
        *************************************************

        Pandas DataFrame method :meth:`pandas.DataFrame.getitem` implementation.

        .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_getitem*
        """
    ty_checker = TypeChecker('Operator getitem().')

    if not isinstance(self, DataFrameType):
        return None

    if isinstance(idx, types.StringLiteral):
        try:
            col_idx = self.columns.index(idx.literal_value)
            key_error = False
        except ValueError:
            key_error = True

        def _df_getitem_str_literal_idx_impl(self, idx):
            if key_error == False:  # noqa
                data = self._data[col_idx]
                return pandas.Series(data, index=self._index, name=idx)
            else:
                raise KeyError('Column is not in the DataFrame')

        return _df_getitem_str_literal_idx_impl

    if isinstance(idx, types.UnicodeType):
        def _df_getitem_unicode_idx_impl(self, idx):
            # http://numba.pydata.org/numba-doc/dev/developer/literal.html#specifying-for-literal-typing
            # literally raises special exception to call getitem with literal idx value got from unicode
            return literally(idx)

        return _df_getitem_unicode_idx_impl

    if isinstance(idx, types.Tuple):
        if all([isinstance(item, types.StringLiteral) for item in idx]):
            return gen_df_getitem_tuple_idx_impl(self, idx)

    if isinstance(idx, types.SliceType):
        return gen_df_getitem_slice_idx_impl(self, idx)

    if isinstance(idx, SeriesType) and isinstance(idx.dtype, types.Boolean):
        self_index_is_none = isinstance(self.index, types.NoneType)
        idx_index_is_none = isinstance(idx.index, types.NoneType)

        if self_index_is_none and not idx_index_is_none:
            if not check_index_is_numeric(idx):
                ty_checker.raise_exc(idx.index.dtype, 'number', 'idx.index.dtype')

        if not self_index_is_none and idx_index_is_none:
            if not check_index_is_numeric(self):
                ty_checker.raise_exc(idx.index.dtype, self.index.dtype, 'idx.index.dtype')

        if not self_index_is_none and not idx_index_is_none:
            if not check_types_comparable(self.index, idx.index):
                ty_checker.raise_exc(idx.index.dtype, self.index.dtype, 'idx.index.dtype')

        return gen_df_getitem_bool_series_idx_impl(self, idx)

    if isinstance(idx, types.Array) and isinstance(idx.dtype, types.Boolean):
        return gen_df_getitem_bool_array_idx_impl(self, idx)

    ty_checker = TypeChecker('Operator getitem().')
    expected_types = 'str, tuple(str), slice, series(bool), array(bool)'
    ty_checker.raise_exc(idx, expected_types, 'idx')


@sdc_overload(operator.getitem)
def sdc_pandas_dataframe_accessor_getitem(self, idx):
    if not isinstance(self, DataFrameGetitemAccessorType):
        return None

    accessor = self.accessor.literal_value

    if accessor == 'iat':
        if isinstance(idx, types.Tuple) and isinstance(idx[1], types.Literal):
            col = idx[1].literal_value
            if -1 < col < len(self.dataframe.columns):
                def df_getitem_iat_tuple_impl(self, idx):
                    row, _ = idx
                    if -1 < row < len(self._dataframe.index):
                        data = self._dataframe._data[col]
                        res_data = pandas.Series(data)
                        return res_data.iat[row]

                    raise IndexingError('Index is out of bounds for axis')

                return df_getitem_iat_tuple_impl

            raise IndexingError('Index is out of bounds for axis')

        raise TypingError('Operator getitem(). The index must be a row and literal column. Given: {}'.format(idx))

    raise TypingError('Operator getitem(). Unknown accessor. Only "loc", "iloc", "at", "iat" are supported.\
                      Given: {}'.format(accessor))


@sdc_overload_attribute(DataFrameType, 'iat')
def sdc_pandas_dataframe_iat(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.iat

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_iat.py
       :language: python
       :lines: 28-
       :caption: Get value at specified index position.
       :name: ex_dataframe_iat

    .. command-output:: python ./dataframe/dataframe_iat.py
       :cwd: ../../../examples

    .. seealso::

        :ref:`DataFrame.at <pandas.DataFrame.at>`
            Access a single value for a row/column label pair.

        :ref:`DataFrame.loc <pandas.DataFrame.loc>`
            Purely label-location based indexer for selection by label.

        :ref:`DataFrame.iloc <pandas.DataFrame.iloc>`
            Access group of rows and columns by integer position(s).

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.iat` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_dataframe_iat*
    """

    ty_checker = TypeChecker('Attribute iat().')
    ty_checker.check(self, DataFrameType)

    def sdc_pandas_dataframe_iat_impl(self):
        return dataframe_getitem_accessor_init(self, 'iat')

    return sdc_pandas_dataframe_iat_impl


@sdc_overload_method(DataFrameType, 'pct_change')
def pct_change_overload(df, periods=1, fill_method='pad', limit=None, freq=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.pct_change

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_pct_change.py
        :language: python
        :lines: 36-
        :caption: Percentage change between the current and a prior element.
        :name: ex_dataframe_pct_change

    .. command-output:: python ./dataframe/dataframe_pct_change.py
        :cwd: ../../../examples


    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.pct_change` implementation.

    .. only:: developer

      Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_pct_change*

    Parameters
    -----------
    df: :class:`pandas.DataFrame`
      input arg
    periods: :obj:`int`, default 1
        Periods to shift for forming percent change.
    fill_method: :obj:`str`, default 'pad'
        How to handle NAs before computing percent changes.
    limit:
      *unsupported*
    freq:
      *unsupported*

    Returns
    -------
    :obj:`pandas.Series` or `pandas.DataFrame`
      Percentage change between the current and a prior element.
    """

    name = 'pct_change'

    ty_checker = TypeChecker('Method {}().'.format(name))
    ty_checker.check(df, DataFrameType)

    if not isinstance(periods, (types.Integer, types.Omitted)):
        ty_checker.raise_exc(periods, 'int64', 'periods')

    if not isinstance(fill_method, (str, types.UnicodeType, types.StringLiteral, types.NoneType, types.Omitted)):
        ty_checker.raise_exc(fill_method, 'string', 'fill_method')

    if not isinstance(limit, (types.Omitted, types.NoneType)):
        ty_checker.raise_exc(limit, 'None', 'limit')

    if not isinstance(freq, (types.Omitted, types.NoneType)):
        ty_checker.raise_exc(freq, 'None', 'freq')

    params = {'periods': 1, 'fill_method': '"pad"', 'limit': None, 'freq': None}
    ser_par = {'periods': 'periods', 'fill_method': 'fill_method', 'limit': 'limit', 'freq': 'freq'}

    return sdc_pandas_dataframe_apply_columns(df, name, params, ser_par)


def df_index_codegen_isin(df_type, df, data):
    if isinstance(df_type.index, types.NoneType):
        func_lines = [f'  return pandas.DataFrame({{{data}}})']
    else:
        func_lines = [f'  return pandas.DataFrame({{{data}}}, index={df}._index)']
    return func_lines


def sdc_pandas_dataframe_isin_dict_codegen(func_name, df_type, values, all_params):
    """
    Example of generated implementation:

    def _df_isin_impl(df, values):
      result_len=len(df)
      if "A" in list(values.keys()):
        series_A = pandas.Series(df._data[0])
        val = list(values["A"])
        result_A = series_A.isin(val)
      else:
        result = numpy.repeat(False, result_len)
        result_A = pandas.Series(result)
      result_len=len(df)
      if "C" in list(values.keys()):
        series_C = pandas.Series(df._data[1])
        val = list(values["C"])
        result_C = series_C.isin(val)
      else:
        result = numpy.repeat(False, result_len)
        result_C = pandas.Series(result)
      return pandas.DataFrame({"A": result_A, "C": result_C})
    """
    result_name = []
    joined = ', '.join(all_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    df = all_params[0]
    for i, c in enumerate(df_type.columns):
        result_c = f'result_{c}'
        func_lines += [
            f'  result_len=len({df})',
            f'  if "{c}" in list(values.keys()):',
            f'    series_{c} = pandas.Series({df}._data[{i}])',
            f'    val = list(values["{c}"])',
            f'    result_{c} = series_{c}.{func_name}(val)',
            f'  else:',
            f'    result = numpy.repeat(False, result_len)',
            f'    result_{c} = pandas.Series(result)'
        ]
        result_name.append((result_c, c))
    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)
    func_lines += df_index_codegen_isin(df_type, df, data)
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def sdc_pandas_dataframe_isin_ser_codegen(func_name, df_type, values, all_params):
    """
    Example of generated implementation:

    def _df_isin_impl(df, values):
      series_A = pandas.Series(df._data[0])
      result = numpy.empty(len(series_A._data), numpy.bool_)
      result_len = len(series_A._data)
      for i in range(result_len):
        idx = df.index[i]
        value = series_A._data[i]
        result[i] = False
        for j in numba.prange(len(values)):
          if idx == j:
            value_val = values._data[j]
            if value == value_val:
              result[i] = True
            else:
              result[i] = False
            break
      result_A = pandas.Series(result)
      series_B = pandas.Series(df._data[1])
      result = numpy.empty(len(series_B._data), numpy.bool_)
      result_len = len(series_B._data)
      for i in range(result_len):
        idx = df.index[i]
        value = series_B._data[i]
        result[i] = False
        for j in numba.prange(len(values)):
          if idx == j:
            value_val = values._data[j]
            if value == value_val:
              result[i] = True
            else:
              result[i] = False
            break
      result_B = pandas.Series(result)
      return pandas.DataFrame({"A": result_A, "B": result_B}, index=df._index)
    """
    result_name = []
    joined = ', '.join(all_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    df = all_params[0]
    for i, c in enumerate(df_type.columns):
        result_c = f'result_{c}'
        func_lines += [
            f'  series_{c} = pandas.Series({df}._data[{i}])',
            f'  result = numpy.empty(len(series_{c}._data), numpy.bool_)',
            f'  result_len = len(series_{c}._data)'
        ]
        if isinstance(values.index, types.NoneType) and isinstance(df_type.index, types.NoneType):
            func_lines += [
                f'  for i in range(result_len):',
                f'    if i <= len(values._data):',
                f'      if series_{c}._data[i] == values._data[i]:',
                f'        result[i] = True',
                f'      else:',
                f'        result[i] = False',
                f'    else:',
                f'      result[i] = False'
            ]
        elif isinstance(values.index, types.NoneType):
            func_lines += [
                f'  for i in range(result_len):',
                f'    idx = {df}.index[i]',
                f'    value = series_{c}._data[i]',
                f'    result[i] = False',
                f'    for j in numba.prange(len(values)):',
                f'      if idx == j:',
                f'        value_val = values._data[j]',
                f'        if value == value_val:',
                f'          result[i] = True',
                f'        else:',
                f'          result[i] = False',
                f'        break'
            ]
        elif isinstance(df_type.index, types.NoneType):
            func_lines += [
                f'  for i in range(result_len):',
                f'    value = series_{c}._data[i]',
                f'    result[i] = False',
                f'    for j in numba.prange(len(values)):',
                f'      idx_val = values.index[j]',
                f'      if i == idx_val:',
                f'        value_val = values._data[j]',
                f'        if value == value_val:',
                f'          result[i] = True',
                f'        else:',
                f'          result[i] = False',
                f'        break'
            ]
        else:
            func_lines += [
                f'  for i in range(result_len):',
                f'    idx = {df}.index[i]',
                f'    value = series_{c}._data[i]',
                f'    result[i] = False',
                f'    for j in numba.prange(len(values)):',
                f'      idx_val = values.index[j]',
                f'      if idx == idx_val:',
                f'        value_val = values._data[j]',
                f'        if value == value_val:',
                f'          result[i] = True',
                f'        else:',
                f'          result[i] = False',
                f'        break'
            ]

        func_lines += [f'  {result_c} = pandas.Series(result)']
        result_name.append((result_c, c))

    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)
    func_lines += df_index_codegen_isin(df_type, df, data)
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'numba': numba,
                   'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def sdc_pandas_dataframe_isin_df_codegen(func_name, df_type, in_df, all_params):
    """
    Example of generated implementation:

    def _df_isin_impl(df, values):
      series_A = pandas.Series(df._data[0])
      series_A_values = pandas.Series(values.A)
      result = numpy.empty(len(series_A._data), numpy.bool_)
      result_len = len(series_A._data)
      for i in range(result_len):
        idx = df.index[i]
        value = series_A._data[i]
        result[i] = False
        for j in numba.prange(len(series_A_values)):
          idx_val = values.index[j]
          if idx == idx_val:
            value_val = series_A_values._data[j]
            if value == value_val:
              result[i] = True
            else:
              result[i] = False
            break
      result_A = pandas.Series(result)
      series_B = pandas.Series(df._data[1])
      series_B_values = pandas.Series(values.B)
      result = numpy.empty(len(series_B._data), numpy.bool_)
      result_len = len(series_B._data)
      for i in range(result_len):
        idx = df.index[i]
        value = series_B._data[i]
        result[i] = False
        for j in numba.prange(len(series_B_values)):
          idx_val = values.index[j]
          if idx == idx_val:
            value_val = series_B_values._data[j]
            if value == value_val:
              result[i] = True
            else:
              result[i] = False
            break
      result_B = pandas.Series(result)
      return pandas.DataFrame({"A": result_A, "B": result_B}, index=df._index)
    """
    result_name = []
    joined = ', '.join(all_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    df = all_params[0]
    val = all_params[1]
    for i, c in enumerate(df_type.columns):
        result_c = f'result_{c}'
        func_lines += [f'  series_{c} = pandas.Series({df}._data[{i}])']
        if c in in_df.columns:
            func_lines += [
                f'  series_{c}_values = pandas.Series({val}.{c})',
                f'  result = numpy.empty(len(series_{c}._data), numpy.bool_)',
                f'  result_len = len(series_{c}._data)'
            ]
            if isinstance(in_df.index, types.NoneType) and isinstance(df_type.index, types.NoneType):
                func_lines += [
                    f'  for i in range(result_len):',
                    f'    if i <= len(series_{c}_values):',
                    f'      if series_{c}._data[i] == series_{c}_values._data[i]:',
                    f'        result[i] = True',
                    f'      else:',
                    f'        result[i] = False',
                    f'    else:',
                    f'      result[i] = False']
            elif isinstance(df_type.index, types.NoneType):
                func_lines += [
                    f'  for i in range(result_len):',
                    f'    value = series_{c}._data[i]',
                    f'    result[i] = False',
                    f'    for j in numba.prange(len(series_{c}_values)):',
                    f'      idx_val = {val}.index[j]',
                    f'      if i == idx_val:',
                    f'        value_val = series_{c}_values._data[j]',
                    f'        if value == value_val:',
                    f'          result[i] = True',
                    f'        else:',
                    f'          result[i] = False',
                    f'        break',
                ]
            elif isinstance(in_df.index, types.NoneType):
                func_lines += [
                    f'  for i in range(result_len):',
                    f'    idx = {df}.index[i]',
                    f'    value = series_{c}._data[i]',
                    f'    result[i] = False',
                    f'    for j in numba.prange(len(series_{c}_values)):',
                    f'      if idx == j:',
                    f'        value_val = series_{c}_values._data[j]',
                    f'        if value == value_val:',
                    f'          result[i] = True',
                    f'        else:',
                    f'          result[i] = False',
                    f'        break',
                    ]
            else:
                func_lines += [
                    f'  for i in range(result_len):',
                    f'    idx = {df}.index[i]',
                    f'    value = series_{c}._data[i]',
                    f'    result[i] = False',
                    f'    for j in numba.prange(len(series_{c}_values)):',
                    f'      idx_val = {val}.index[j]',
                    f'      if idx == idx_val:',
                    f'        value_val = series_{c}_values._data[j]',
                    f'        if value == value_val:',
                    f'          result[i] = True',
                    f'        else:',
                    f'          result[i] = False',
                    f'        break',
                    ]
        else:
            func_lines += [
                f'  result = [False] * len(series_{c}._data)']
        func_lines += [f'  {result_c} = pandas.Series(result)']
        result_name.append((result_c, c))
    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)
    func_lines += df_index_codegen_isin(df_type, df, data)
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'numba': numba,
                   'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def gen_codegen(func, name, df, values, all_params):
    func_text, global_vars = func(name, df, values, all_params)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _apply_impl = loc_vars[f'_df_{name}_impl']

    return _apply_impl


def sdc_pandas_dataframe_isin_df(name, df, values, all_params):
    return gen_codegen(sdc_pandas_dataframe_isin_df_codegen, name, df, values, all_params)


def sdc_pandas_dataframe_isin_ser(name, df, values, all_params):
    return gen_codegen(sdc_pandas_dataframe_isin_ser_codegen, name, df, values, all_params)


def sdc_pandas_dataframe_isin_dict(name, df, values, all_params):
    return gen_codegen(sdc_pandas_dataframe_isin_dict_codegen, name, df, values, all_params)


def sdc_pandas_dataframe_isin_iter(name, all_params, ser_par, columns):
    func_text, global_vars = _dataframe_apply_columns_codegen(name, all_params, ser_par, columns)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _apply_impl = loc_vars[f'_df_{name}_impl']

    return _apply_impl


@sdc_overload_method(DataFrameType, 'isin')
def isin_overload(df, values):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.isin

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_isin_df.py
        :language: python
        :lines: 36-
        :caption: Whether each element in the DataFrame is contained in values of another DataFrame.
        :name: ex_dataframe_isin

    .. command-output:: python ./dataframe/dataframe_isin_df.py
        :cwd: ../../../examples

    .. literalinclude:: ../../../examples/dataframe/dataframe_isin_ser.py
        :language: python
        :lines: 36-
        :caption: Whether each element in the DataFrame is contained in values of Series.
        :name: ex_dataframe_isin

    .. command-output:: python ./dataframe/dataframe_isin_ser.py
        :cwd: ../../../examples

    .. literalinclude:: ../../../examples/dataframe/dataframe_isin_dict.py
        :language: python
        :lines: 36-
        :caption: Whether each element in the DataFrame is contained in values of Dictionary.
        :name: ex_dataframe_isin

    .. command-output:: python ./dataframe/dataframe_isin_dict.py
        :cwd: ../../../examples

    .. literalinclude:: ../../../examples/dataframe/dataframe_isin.py
        :language: python
        :lines: 36-
        :caption: Whether each element in the DataFrame is contained in values of List.
        :name: ex_dataframe_isin

    .. command-output:: python ./dataframe/dataframe_isin.py
        :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.isin` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_isin*

    Parameters
    -----------
    df: :class:`pandas.DataFrame`
        input arg
    values: iterable, Series, DataFrame or dict

    Returns
    -------
    :obj:`pandas.Series` or `pandas.DataFrame`
            Whether each element in the DataFrame is contained in values.
    """

    name = 'isin'

    ty_checker = TypeChecker('Method {}().'.format(name))
    ty_checker.check(df, DataFrameType)

    if not isinstance(values, (SeriesType, types.List, types.Set, DataFrameType, types.DictType)):
        ty_checker.raise_exc(values, 'iterable, Series, DataFrame', 'values')

    all_params = ['df', 'values']

    if isinstance(values, (types.List, types.Set)):
        ser_par = 'values=values'
        return sdc_pandas_dataframe_isin_iter(name, all_params, ser_par, df.columns)

    if isinstance(values, types.DictType):
        return sdc_pandas_dataframe_isin_dict(name, df, values, all_params)

    if isinstance(values, SeriesType):
        return sdc_pandas_dataframe_isin_ser(name, df, values, all_params)

    if isinstance(values, DataFrameType):
        return sdc_pandas_dataframe_isin_df(name, df, values, all_params)


@sdc_overload_method(DataFrameType, 'groupby')
def sdc_pandas_dataframe_groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                                 group_keys=True, squeeze=False, observed=False):

    if not isinstance(by, types.StringLiteral):
        return None

    column_id = self.columns.index(by.literal_value)
    list_type = types.ListType(types.int64)
    by_type = self.data[column_id].dtype

    def sdc_pandas_dataframe_groupby_impl(self, by=None, axis=0, level=None, as_index=True, sort=True,
                                          group_keys=True, squeeze=False, observed=False):

        grouped = Dict.empty(by_type, list_type)
        by_column_data = self._data[column_id]
        for i in numpy.arange(len(by_column_data)):
            if isna(by_column_data, i):
                continue
            value = by_column_data[i]
            group_list = grouped.get(value, List.empty_list(types.int64))
            group_list.append(i)
            grouped[value] = group_list

        return init_dataframe_groupby(self, column_id, grouped, sort)

    return sdc_pandas_dataframe_groupby_impl


def df_set_column_index_codelines(self):
    """Generate code lines with definition of resulting index for DF set_column"""
    func_lines = []
    if self.columns:
        func_lines += [
            f'  length = {df_length_expr(self)}',
            f'  if length == 0:',
            f'    raise SDCLimitation("Could not set item for DataFrame with empty columns")',
            f'  elif length != len(value):',
            f'    raise ValueError("Length of values does not match length of index")',
        ]
    else:
        func_lines += ['  length = len(value)']
    func_lines += [f'  res_index = {df_index_expr(self, length_expr="length")}']

    return func_lines


def df_add_column_codelines(self, key):
    """Generate code lines to add new column to DF"""
    func_lines = df_set_column_index_codelines(self)  # provide res_index = ...

    results = []
    for i, col in enumerate(self.columns):
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = self._data[{i}]',
            f'  {res_data} = pandas.Series(data_{i}, index=res_index, name="{col}")',
        ]
        results.append((col, res_data))

    res_data = 'new_res_data'
    literal_key = key.literal_value
    func_lines += [f'  {res_data} = pandas.Series(value, index=res_index, name="{literal_key}")']
    results.append((literal_key, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index)']

    return func_lines


def df_replace_column_codelines(self, key):
    """Generate code lines to replace existing column in DF"""
    func_lines = df_set_column_index_codelines(self)  # provide res_index = ...

    results = []
    literal_key = key.literal_value
    for i, col in enumerate(self.columns):
        if literal_key == col:
            func_lines += [f'  data_{i} = value']
        else:
            func_lines += [f'  data_{i} = self._data[{i}]']

        res_data = f'res_data_{i}'
        func_lines += [
            f'  {res_data} = pandas.Series(data_{i}, index=res_index, name="{col}")',
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index)']

    return func_lines


def df_add_column_codegen(self, key):
    """
    Example of generated implementation:
    def _df_add_column_impl(self, key, value):
      length = len(self._data[0])
      if length == 0:
        raise SDCLimitation("Could not set item for empty DataFrame")
      elif length != len(value):
        raise ValueError("Length of values does not match length of index")
      res_index = numpy.arange(length)
      data_0 = self._data[0]
      res_data_0 = pandas.Series(data_0, index=res_index, name="A")
      data_1 = self._data[1]
      res_data_1 = pandas.Series(data_1, index=res_index, name="C")
      new_res_data = pandas.Series(value, index=res_index, name="B")
      return pandas.DataFrame({"A": res_data_0, "C": res_data_1, "B": new_res_data}, index=res_index)
    """
    func_lines = [f'def _df_add_column_impl(self, key, value):']
    func_lines += df_add_column_codelines(self, key)

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'SDCLimitation': SDCLimitation}

    return func_text, global_vars


def df_replace_column_codegen(self, key):
    """
    Example of generated implementation:
    def _df_replace_column_impl(self, key, value):
      length = len(self._data[0])
      if length == 0:
        raise SDCLimitation("Could not set item for DataFrame with empty columns")
      elif length != len(value):
        raise ValueError("Length of values does not match length of index")
      res_index = numpy.arange(length)
      data_0 = value
      res_data_0 = pandas.Series(data_0, index=res_index, name="A")
      data_1 = self._data[1]
      res_data_1 = pandas.Series(data_1, index=res_index, name="C")
      return pandas.DataFrame({"A": res_data_0, "C": res_data_1}, index=res_index)
    """
    func_lines = [f'def _df_replace_column_impl(self, key, value):']
    func_lines += df_replace_column_codelines(self, key)

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'SDCLimitation': SDCLimitation}

    return func_text, global_vars


gen_df_add_column_impl = gen_impl_generator(
    df_add_column_codegen, '_df_add_column_impl')
gen_df_replace_column_impl = gen_impl_generator(
    df_replace_column_codegen, '_df_replace_column_impl')


@sdc_overload_method(DataFrameType, '_set_column')
def df_set_column_overload(self, key, value):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Limitations
    -----------
    - Supported setting a columns in a non-empty DataFrame as a 1D array only.
    - Unsupported change of the Parent DataFrame, returned new DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_add_column
    Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_replace_column
    """
    if not isinstance(self, DataFrameType):
        return None

    if isinstance(key, types.StringLiteral):
        try:
            self.columns.index(key.literal_value)
        except ValueError:
            return gen_df_add_column_impl(self, key)
        else:
            return gen_df_replace_column_impl(self, key)

    if isinstance(key, types.UnicodeType):
        def _df_set_column_unicode_key_impl(self, key, value):
            # http://numba.pydata.org/numba-doc/dev/developer/literal.html#specifying-for-literal-typing
            # literally raises special exception to call df._set_column with literal idx value got from unicode
            return literally(key)

        return _df_set_column_unicode_key_impl

    ty_checker = TypeChecker('Method _set_column().')
    ty_checker.raise_exc(key, 'str', 'key')

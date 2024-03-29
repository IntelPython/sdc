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

import numba
import numpy
import operator
import pandas
import sdc

from pandas.core.indexing import IndexingError

from numba import types
from numba import literally
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from pandas.core.indexing import IndexingError

from sdc.datatypes.indexes import *
from sdc.hiframes.pd_dataframe_ext import DataFrameType
from sdc.hiframes.pd_series_type import SeriesType
from sdc.utilities.sdc_typing_utils import (TypeChecker, check_index_is_numeric,
                                            check_types_comparable, kwsparams2list,
                                            gen_impl_generator, find_common_dtype_from_numpy_dtypes)
from sdc.str_arr_ext import StringArrayType

from sdc.extensions.indexes.empty_index_ext import init_empty_index

from sdc.hiframes.pd_dataframe_type import DataFrameType
from sdc.hiframes.pd_dataframe_ext import init_dataframe_internal, get_structure_maps
from sdc.hiframes.pd_series_type import SeriesType

from sdc.datatypes.hpat_pandas_dataframe_getitem_types import (DataFrameGetitemAccessorType,
                                                               dataframe_getitem_accessor_init)
from sdc.utilities.sdc_typing_utils import SDCLimitation
from sdc.datatypes.hpat_pandas_dataframe_rolling_types import _hpat_pandas_df_rolling_init
from sdc.datatypes.hpat_pandas_rolling_types import (
    gen_sdc_pandas_rolling_overload_body, sdc_pandas_rolling_docstring_tmpl)
from sdc.datatypes.hpat_pandas_groupby_functions import init_dataframe_groupby, merge_groupby_dicts_inplace
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
from sdc.utilities.utils import sdc_overload, sdc_overload_method, sdc_overload_attribute
from sdc.hiframes.api import isna
from sdc.functions.numpy_like import getitem_by_mask, find_idx
from sdc.functions.numpy_like import take as nplike_take
from sdc.datatypes.common_functions import (sdc_reindex_series,
                                            fill_array,
                                            fill_str_array,)
from sdc.utilities.prange_utils import parallel_chunks


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
    """

    ty_checker = TypeChecker('Attribute index.')
    ty_checker.check(df, DataFrameType)

    def hpat_pandas_df_index_impl(df):
        return df._index

    return hpat_pandas_df_index_impl


@sdc_overload_attribute(DataFrameType, 'columns')
def hpat_pandas_dataframe_columns(df):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.columns

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_columns.py
        :language: python
        :lines: 27-
        :caption: The column names of the DataFrame.
        :name: ex_dataframe_columns

    .. command-output:: python ./dataframe/dataframe_columns.py
        :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame attribute :attr:`pandas.DataFrame.columns` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_dataframe_columns*
    """

    ty_checker = TypeChecker('Attribute columns.')
    ty_checker.check(df, DataFrameType)

    # no columns in DF model to avoid impact on DF ctor IR size (captured when needed only)
    df_columns = df.columns

    def hpat_pandas_df_columns_impl(df):
        return df_columns

    return hpat_pandas_df_columns_impl


def sdc_pandas_dataframe_values_codegen(self, numba_common_dtype):
    """
    Example of generated implementation:
        def sdc_pandas_dataframe_values_impl(self):
          length = len(self._data[0][0])
          col_data_0 = self._data[0][0]
          col_data_1 = self._data[1][0]
          col_data_2 = self._data[0][1]
          values = numpy.empty(length*3, numpy.dtype("float64"))
          for i in range(length):
            values[i*3+0] = col_data_0[i]
            values[i*3+1] = col_data_1[i]
            values[i*3+2] = col_data_2[i]
          return values.reshape(length, 3)
    """
    columns_data = []
    columns_num = len(self.columns)
    func_lines = [
        f'def sdc_pandas_dataframe_values_impl(self):',
        f'  length = len(self._index)',
    ]
    for i, col in enumerate(self.columns):
        col_loc = self.column_loc[col]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        func_lines += [
            f'  col_data_{i} = self._data[{type_id}][{col_id}]',
        ]
        columns_data.append(f'col_data_{i}')

    func_lines += [
        f'  values = numpy.empty(length*{columns_num}, numpy.dtype("{numba_common_dtype}"))',
        f'  for i in range(length):',
    ]
    func_lines += ['\n'.join([
        f'    values[i*{columns_num}+{j}] = {columns_data[j]}[i]',
    ]) for j in range(columns_num)]
    func_lines += [
        f'  return values.reshape(length, {columns_num})\n'
    ]
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy}

    return func_text, global_vars


@sdc_overload_attribute(DataFrameType, 'values')
def hpat_pandas_dataframe_values(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.values

    Limitations
    -----------
    - Only numeric values supported as an output
    - The dtype will be a lower-common-denominator dtype (implicit upcasting);
    that is to say if the dtypes (even of numeric types) are mixed, the one that accommodates all will be chosen.
    Use this with care if you are not dealing with the blocks.
    e.g. If the dtypes are float16 and float32, dtype will be upcast to float32. If dtypes are int32 and uint8,
    dtype will be upcast to int32. By numpy.find_common_type() convention,
    mixing int64 and uint64 will result in a float64 dtype.

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

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame attribute :attr:`pandas.DataFrame.values` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_values*
    """

    func_name = 'Attribute values.'
    ty_checker = TypeChecker(func_name)
    ty_checker.check(self, DataFrameType)

    # TODO: Handle StringArrayType
    for i, column in enumerate(self.data):
        if isinstance(column, StringArrayType):
            ty_checker.raise_exc(column, 'Numeric type', f'df.data["{self.columns[i]}"]')

    numba_common_dtype = find_common_dtype_from_numpy_dtypes([column.dtype for column in self.data], [])

    def hpat_pandas_df_values_impl(self, numba_common_dtype):
        loc_vars = {}
        func_text, global_vars = sdc_pandas_dataframe_values_codegen(self, numba_common_dtype)

        exec(func_text, global_vars, loc_vars)
        _values_impl = loc_vars['sdc_pandas_dataframe_values_impl']
        return _values_impl

    return hpat_pandas_df_values_impl(self, numba_common_dtype)


def sdc_pandas_dataframe_append_codegen(df, other, _func_name, ignore_index_value, indexes_comparable, args):
    """
    Example of generated implementation:
    def sdc_pandas_dataframe_append_impl(df, other, ignore_index=False, verify_integrity=False, sort=None):
        len_df = len(df._data[0][0])
        len_other = len(other._data[0][0])
        new_col_0_data_df = df._data[0][0]
        new_col_0_data_other = other._data[0][0]
        new_col_0 = init_series(new_col_0_data_df).append(init_series(new_col_0_data_other))._data
        new_col_1_data_df = df._data[0][1]
        new_col_1_data_other = other._data[0][1]
        new_col_1 = init_series(new_col_1_data_df).append(init_series(new_col_1_data_other))._data
        return pandas.DataFrame({"A": new_col_0, "B": new_col_1})
    """
    indent = 4 * ' '
    func_args = ['df', 'other'] + kwsparams2list(args)

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

    func_text.append(f'len_df = len(df._data[0][0])')
    func_text.append(f'len_other = len(other._data[0][0])')

    for col_name, idx in df_columns_indx.items():
        col_loc = df.column_loc[col_name]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        func_text.append(f'new_col_{idx}_data_df = df._data[{type_id}][{col_id}]')
        if col_name in other_columns_indx:
            other_col_loc = other.column_loc[col_name]
            other_type_id, other_col_id = other_col_loc.type_id, other_col_loc.col_id
            func_text.append(f'new_col_{idx}_data_other = '
                             f'other._data[{other_type_id}][{other_col_id}]')
            s1 = f'pandas.Series(new_col_{idx}_data_df)'
            s2 = f'pandas.Series(new_col_{idx}_data_other)'
            func_text.append(f'new_col_{idx} = {s1}.append({s2})._data')
        else:
            func_text.append(f'new_col_{idx}_data = pandas.Series(new_col_{idx}_data_df)._data')
            if col_name in string_type_columns:
                func_text.append(f'new_col_{idx} = fill_str_array(new_col_{idx}_data, len_df+len_other)')
            else:
                func_text.append(f'new_col_{idx} = fill_array(new_col_{idx}_data, len_df+len_other)')
        column_list.append((f'new_col_{idx}', col_name))

    for col_name, idx in other_columns_indx.items():
        if col_name not in df_columns_indx:
            other_col_loc = other.column_loc[col_name]
            other_type_id, other_col_id = other_col_loc.type_id, other_col_loc.col_id
            func_text.append(f'new_col_{idx}_data_other = other._data[{other_type_id}][{other_col_id}]')
            func_text.append(f'new_col_{idx}_data = pandas.Series(new_col_{idx}_data_other)._data')
            if col_name in string_type_columns:
                func_text.append(
                    f'new_col_{idx}_other = '
                    f'fill_str_array(new_col_{idx}_data, len_df+len_other, push_back=False)')
            else:
                func_text.append(f'new_col_{idx}_other = '
                                 f'fill_array(new_col_{idx}_data, len_df+len_other, push_back=False)')
            column_list.append((f'new_col_{idx}_other', col_name))

    data = ', '.join(f'"{column_name}": {column}' for column, column_name in column_list)

    if ignore_index_value == True:  # noqa
        func_text.append(f'return pandas.DataFrame({{{data}}})\n')
    else:
        if indexes_comparable == False:  # noqa
            func_text.append(f'raise SDCLimitation("Indexes of dataframes are expected to have comparable '
                             f'(both Numeric or String) types if parameter ignore_index is set to False.")')
        else:
            func_text += [f'joined_index = df._index.append(other._index)\n',
                          f'return pandas.DataFrame({{{data}}}, index=joined_index)\n']

    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {'pandas': pandas,
                   'fill_array': fill_array,
                   'fill_str_array': fill_str_array,
                   'SDCLimitation': SDCLimitation}

    return func_def, global_vars


@sdc_overload_method(DataFrameType, 'append')
def sdc_pandas_dataframe_append(df, other, ignore_index=False, verify_integrity=False, sort=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.append

    Limitations
    -----------
    - Parameters ``verify_integrity`` and ``sort`` are unsupported.
    - Parameter ``other`` can be only :obj:`pandas.DataFrame`.
    - Indexes of dataframes are expected to have comparable (both Numeric or String) types if parameter ignore_index
    is set to False.
    - This function may reveal slower performance than Pandas* on user system. Users should exercise a tradeoff
    between staying in JIT-region with that function or going back to interpreter mode.

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

    .. seealso::
        `pandas.concat <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html>`_
            General function to concatenate DataFrame or Series objects.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.append` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_append*
    """

    _func_name = 'append'

    ty_checker = TypeChecker(f'Method {_func_name}().')
    ty_checker.check(df, DataFrameType)
    # TODO: support other array-like types
    ty_checker.check(other, DataFrameType)

    if not isinstance(verify_integrity, (bool, types.Boolean, types.Omitted)) and verify_integrity:
        ty_checker.raise_exc(verify_integrity, 'boolean', 'verify_integrity')

    if not isinstance(sort, (bool, types.Boolean, types.Omitted)) and sort is not None:
        ty_checker.raise_exc(sort, 'boolean, None', 'sort')

    if not isinstance(ignore_index, (bool, types.Boolean, types.Omitted)):
        ty_checker.raise_exc(ignore_index, 'boolean', 'ignore_index')

    indexes_comparable = check_types_comparable(df.index, other.index)

    if isinstance(ignore_index, types.Literal):
        ignore_index = ignore_index.literal_value
    elif not (ignore_index is False or isinstance(ignore_index, types.Omitted)):
        raise SDCLimitation("Parameter ignore_index should be Literal")

    args = {'ignore_index': False, 'verify_integrity': False, 'sort': None}

    def sdc_pandas_dataframe_append_impl(df, other, _func_name, ignore_index, indexes_comparable, args):
        loc_vars = {}
        func_def, global_vars = sdc_pandas_dataframe_append_codegen(df, other, _func_name, ignore_index,
                                                                    indexes_comparable, args)
        exec(func_def, global_vars, loc_vars)
        _append_impl = loc_vars['sdc_pandas_dataframe_append_impl']
        return _append_impl

    return sdc_pandas_dataframe_append_impl(df, other, _func_name, ignore_index, indexes_comparable, args)

# Example func_text for func_name='count' columns=('A', 'B'):
#
#         def _df_count_impl(df, axis=0, level=None, numeric_only=False):
#           series_A = init_series(df._data[0])
#           result_A = series_A.count(level=level)
#           series_B = init_series(df._data[1])
#           result_B = series_B.count(level=level)
#           return pandas.Series([result_A, result_B], ['A', 'B'])


def _dataframe_reduce_columns_codegen(func_name, func_params, series_params, columns, column_loc):
    result_name_list = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    for i, c in enumerate(columns):
        col_loc = column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f'result_{i}'
        func_lines += [f'  series_{i} = pandas.Series({func_params[0]}._data[{type_id}][{col_id}])',
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

    func_text, global_vars = _dataframe_reduce_columns_codegen(func_name, all_params, s_par, df.columns,
                                                               df.column_loc)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars[df_func_name]

    return _reduce_impl


def _dataframe_reduce_columns_codegen_head(func_name, func_params, series_params, df):
    """
    Example func_text for func_name='head' columns=('float', 'string'):
        def _df_head_impl(df, n=5):
          data_0 = df._data[0][0]
          series_0 = pandas.Series(data_0)
          result_0 = series_0.head(n=n)
          data_1 = df._data[1][0]
          series_1 = pandas.Series(data_1)
          result_1 = series_1.head(n=n)
          return pandas.DataFrame({"float": result_0, "string": result_1}, index=df._index[:n])
    """
    results = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl(df, {joined}):']
    ind = 'index=df._index[:n]'
    for i, c in enumerate(df.columns):
        col_loc = df.column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f'result_{i}'
        func_lines += [f'  data_{i} = df._data[{type_id}][{col_id}]',
                       f'  series_{i} = pandas.Series(data_{i})',
                       f'  {result_c} = series_{i}.{func_name}({series_params})']
        results.append((df.columns[i], result_c))

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
    func_text, global_vars = _dataframe_reduce_columns_codegen_head(func_name, all_params, s_par, df)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars[df_func_name]

    return _reduce_impl


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
    Example func_text for func_name='copy' columns=('A', 'B'):
        def _df_copy_impl(df, deep=True):
          data_0 = df._data[0][0]
          series_0 = pandas.Series(data_0, name='A')
          result_0 = series_0.copy(deep=deep)
          data_1 = df._data[1][0]
          series_1 = pandas.Series(data_1, name='B')
          result_1 = series_1.copy(deep=deep)
          return pandas.DataFrame({"A": result_0, "B": result_1}, index=df._index)
    """
    results = []
    series_params_str = ', '.join(kwsparams2list(series_params))
    func_params_str = ', '.join(kwsparams2list(func_params))
    func_lines = [f"def _df_copy_impl(df, {func_params_str}):"]
    index = 'df._index'
    for i, c in enumerate(df.columns):
        col_loc = df.column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f"result_{i}"
        func_lines += [f"  data_{i} = df._data[{type_id}][{col_id}]",
                       f"  series_{i} = pandas.Series(data_{i}, name='{c}')",
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


def _dataframe_apply_columns_codegen(func_name, func_params, series_params, columns, column_loc):
    result_name = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    for i, c in enumerate(columns):
        col_loc = column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f'result_{i}'
        func_lines += [f'  series_{i} = pandas.Series({func_params[0]}._data[{type_id}][{col_id}])',
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

    func_text, global_vars = _dataframe_apply_columns_codegen(func_name, all_params, s_par,
                                                              df.columns, df.column_loc)
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

    if not (isinstance(ddof, (types.Omitted, types.Integer)) or ddof == 1):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    if not (isinstance(min_count, types.Omitted) or min_count == 0):
        ty_checker.raise_exc(min_count, 'unsupported', 'min_count')


@sdc_overload_method(DataFrameType, 'median')
def median_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.median

    Limitations
    -----------
    Parameters ``axis``, ``level`` and ``numeric_only`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_median.py
       :language: python
       :lines: 35-
       :caption: Return the median of the values for the columns.
       :name: ex_dataframe_median

    .. command-output:: python ./dataframe/dataframe_median.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.median <pandas.Series.median>`
            Returns the median of the values for the Series.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.median` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_median*
    """

    name = 'median'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only)

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'mean')
def mean_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.mean

    Limitations
    -----------
    Parameters ``axis``, ``level`` and ``numeric_only`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_mean.py
       :language: python
       :lines: 35-
       :caption: Return the mean of the values for the columns.
       :name: ex_dataframe_mean

    .. command-output:: python ./dataframe/dataframe_mean.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.mean <pandas.Series.mean>`
            Return the mean of the values for the Series.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.mean` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_mean*
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
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.std

    Limitations
    -----------
    Parameters ``axis``, ``level`` and ``numeric_only`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_std.py
       :language: python
       :lines: 35-
       :caption: Return sample standard deviation over columns.
       :name: ex_dataframe_std

    .. command-output:: python ./dataframe/dataframe_std.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.std <pandas.Series.std>`
            Returns sample standard deviation over Series.
        :ref:`Series.var <pandas.Series.var>`
            Returns unbiased variance over Series.
        :ref:`DataFrame.var <pandas.DataFrame.var>`
            Returns unbiased variance over DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.std` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_std*
    """

    name = 'std'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, ddof=ddof)

    params = {'axis': None, 'skipna': None, 'level': None, 'ddof': 1, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level', 'ddof': 'ddof'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'var')
def var_overload(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.var

    Limitations
    -----------
    Parameters ``axis``, ``level`` and ``numeric_only`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_var.py
       :language: python
       :lines: 35-
       :caption: Return unbiased variance over requested axis.
       :name: ex_dataframe_var

    .. command-output:: python ./dataframe/dataframe_var.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.std <pandas.Series.std>`
            Returns sample standard deviation over Series.
        :ref:`Series.var<pandas.Series.var>`
            Returns unbiased variance over Series.
        :ref:`DataFrame.std <pandas.DataFrame.std>`
            Returns sample standard deviation over DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.var` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_var*
    """

    name = 'var'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, ddof=ddof)

    params = {'axis': None, 'skipna': None, 'level': None, 'ddof': 1, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level', 'ddof': 'ddof'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'max')
def max_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.max

    Limitations
    -----------
    Parameters ``axis``, ``level`` and  ``numeric_only`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_max.py
       :language: python
       :lines: 35-
       :caption: Return the maximum of the values for the columns.
       :name: ex_dataframe_max

    .. command-output:: python ./dataframe/dataframe_max.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.sum <pandas.Series.sum>`
            Return the sum.
        :ref:`Series.max <pandas.Series.max>`
            Return the maximum.
        :ref:`Series.idxmin <pandas.Series.idxmin>`
            Return the index of the minimum.
        :ref:`Series.idxmax <pandas.Series.idxmax>`
            Return the index of the maximum.
        :ref:`DataFrame.sum <pandas.DataFrame.sum>`
            Return the sum over the requested axis.
        :ref:`DataFrame.min <pandas.DataFrame.min>`
            Return the minimum over the requested axis.
        :ref:`DataFrame.idxmin <pandas.DataFrame.idxmin>`
            Return the index of the minimum over the requested axis.
        :ref:`DataFrame.idxmax <pandas.DataFrame.idxmax>`
            Return the index of the maximum over the requested axis.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.max` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_max*
    """

    name = 'max'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only)

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'min')
def min_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.min

    Limitations
    -----------
    Parameters ``axis``, ``level`` and ``numeric_only`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_min.py
       :language: python
       :lines: 35-
       :caption: Return the minimum of the values for the columns.
       :name: ex_dataframe_min

    .. command-output:: python ./dataframe/dataframe_min.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.sum <pandas.Series.sum>`
            Return the sum.
        :ref:`Series.min <pandas.Series.min>`
            Return the minimum.
        :ref:`Series.max <pandas.Series.max>`
            Return the maximum.
        :ref:`Series.idxmin <pandas.Series.idxmin>`
            Return the index of the minimum.
        :ref:`Series.idxmax <pandas.Series.idxmax>`
            Return the index of the maximum.
        :ref:`DataFrame.sum <pandas.DataFrame.sum>`
            Return the sum over the requested axis.
        :ref:`DataFrame.min <pandas.DataFrame.min>`
            Return the minimum over the requested axis.
        :ref:`DataFrame.max <pandas.DataFrame.max>`
            Return the maximum over the requested axis.
        :ref:`DataFrame.idxmin <pandas.DataFrame.idxmin>`
            Return the index of the minimum over the requested axis.
        :ref:`DataFrame.idxmax <pandas.DataFrame.idxmax>`
            Return the index of the maximum over the requested axis.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.min` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_min*
    """

    name = 'min'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only)

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None}
    ser_par = {'skipna': 'skipna', 'level': 'level'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'sum')
def sum_overload(df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.sum

    Limitations
    -----------
    Parameters ``axis``, ``level``, ``numeric_only`` and ``min_count`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_sum.py
       :language: python
       :lines: 35-
       :caption: Return the sum of the values for the columns.
       :name: ex_dataframe_sum

    .. command-output:: python ./dataframe/dataframe_sum.py
       :cwd: ../../../examples

    .. seealso::

        :ref:`Series.sum <pandas.Series.sum>`
            Return the sum.

        :ref:`Series.min <pandas.Series.min>`
            Return the minimum.

        :ref:`Series.max <pandas.Series.max>`
            Return the maximum.

        :ref:`Series.idxmin <pandas.Series.idxmin>`
            Return the index of the minimum.

        :ref:`Series.idxmax <pandas.Series.idxmax>`
            Return the index of the maximum.

        :ref:`DataFrame.sum <pandas.DataFrame.sum>`
            Return the sum over the requested axis.

        :ref:`DataFrame.min <pandas.DataFrame.min>`
            Return the minimum over the requested axis.

        :ref:`DataFrame.max <pandas.DataFrame.max>`
            Return the maximum over the requested axis.

        :ref:`DataFrame.idxmin <pandas.DataFrame.idxmin>`
            Return the index of the minimum over the requested axis.

        :ref:`DataFrame.idxmax <pandas.DataFrame.idxmax>`
            Return the index of the maximum over the requested axis.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.sum` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_sum*
    """

    name = 'sum'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, min_count=min_count)

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None, 'min_count': 0}
    ser_par = {'skipna': 'skipna', 'level': 'level', 'min_count': 'min_count'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'prod')
def prod_overload(df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.prod

    Limitations
    -----------
    Parameters ``axis``, ``level``, ``numeric_only`` and ``min_count`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_prod.py
       :language: python
       :lines: 35-
       :caption: Return the product of the values for the columns.
       :name: ex_dataframe_prod

    .. command-output:: python ./dataframe/dataframe_prod.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.prod <pandas.Series.prod>`
            Returns the product of the values for the Series.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.prod` implementation.

    .. only:: developer

        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_prod*
    """

    name = 'prod'

    check_type(name, df, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, min_count=min_count)

    params = {'axis': None, 'skipna': None, 'level': None, 'numeric_only': None, 'min_count': 0}
    ser_par = {'skipna': 'skipna', 'level': 'level', 'min_count': 'min_count'}

    return sdc_pandas_dataframe_reduce_columns(df, name, params, ser_par)


@sdc_overload_method(DataFrameType, 'count')
def count_overload(df, axis=0, level=None, numeric_only=False):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.count

    Limitations
    -----------
    Parameters ``axis``, ``level`` and ``numeric_only`` are unsupported.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_count.py
       :language: python
       :lines: 33-
       :caption: Count non-NA cells for each column or row.
       :name: ex_dataframe_count

    .. command-output:: python ./dataframe/dataframe_count.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.count <pandas.Series.count>`
            Number of non-NA elements in a Series.
        :ref:`DataFrame.shape <pandas.DataFrame.shape>`
            Number of DataFrame rows and columns (including NA elements).
        :ref:`DataFrame.isna <pandas.DataFrame.isna>`
            Boolean same-sized DataFrame showing places of NA elements.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.count` implementation.

        .. only:: developer

            Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_count*
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


def _dataframe_codegen_isna(func_name, columns, df):
    """
    Example if generated implementation
        def _df_isna_impl(df):
          data_0 = df._data[0][0]
          series_0 = pandas.Series(data_0)
          result_0 = series_0.isna()
          data_1 = df._data[1][0]
          series_1 = pandas.Series(data_1)
          result_1 = series_1.isna()
          return pandas.DataFrame({"A": result_0, "B": result_1}, index=df._index)
    """
    results = []
    func_lines = [f'def _df_{func_name}_impl(df):']
    index = 'df._index'
    for i, c in enumerate(columns):
        col_loc = df.column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f'result_{i}'
        func_lines += [f'  data_{i} = df._data[{type_id}][{col_id}]',
                       f'  series_{i} = pandas.Series(data_{i})',
                       f'  {result_c} = series_{i}.{func_name}()']
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

        `pandas.isna <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isna.html#pandas.isna>`_
            Top-level isna.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.isna` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_isna*
    """

    return sdc_pandas_dataframe_isna_codegen(df, 'isna')


def sdc_pandas_dataframe_drop_codegen(func_name, func_args, df, drop_col_names):
    """
    Example of generated implementation:
        def sdc_pandas_dataframe_drop_impl(df, labels=None, axis=0, index=None, columns=None,
                                            level=None, inplace=False, errors="raise"):
            list_0 = df._data[0].copy()
            for col_id in old_scheme_drop_idxs_0[::-1]:
                list_0.pop(col_id)
            list_1 = df._data[1].copy()
            new_data = (list_1, list_0, )
            return init_dataframe_internal(new_data, df._index, df_type)
    """
    indent = 4 * ' '
    func_definition = [f'def {func_name}({", ".join(func_args)}):']
    func_text = []

    old_column_loc, old_data_typs_map, old_types_order = get_structure_maps(df.data, df.columns)

    new_data_typs = tuple(t for i, t in enumerate(df.data) if df.columns[i] not in drop_col_names)
    new_column_names = tuple(c for c in df.columns if c not in drop_col_names)
    new_column_loc, new_data_typs_map, new_types_order = get_structure_maps(new_data_typs, new_column_names)

    old_types_idxs_map = dict(zip(old_types_order, range(len(old_types_order))))
    reorder_scheme = tuple(old_types_idxs_map[t] for t in new_types_order)
    df_type = DataFrameType(new_data_typs, df.index, new_column_names, column_loc=new_column_loc)

    old_scheme_drop_idxs = []
    for i, k in enumerate(old_types_order):
        a = [j for j, x in enumerate(old_data_typs_map[k][1]) if df.columns[x] in drop_col_names]
        old_scheme_drop_idxs.append(tuple(a) or None)

    for label in drop_col_names:
        if label not in df.columns:
            func_text.append(f'if errors == "raise":')
            func_text.append(indent + f'raise ValueError("The label {label} is not found in the selected axis")')
            break

    old_ntypes = len(old_types_order)
    for type_id in range(old_ntypes):
        func_text.append(f'list_{type_id} = df._data[{type_id}].copy()')
        if old_scheme_drop_idxs[type_id]:
            func_text.append(f'for col_id in old_scheme_drop_idxs_{type_id}[::-1]:')
            func_text.append(indent + f'list_{type_id}.pop(col_id)')

    # in new df the order of array lists (i.e. types_order) can be different, so
    # making a new tuple of lists reorder as needed
    new_ntypes = len(new_types_order)
    data_lists_reordered = ', '.join(['list_' + str(reorder_scheme[i]) for i in range(new_ntypes)])
    data_val = '(' + data_lists_reordered + ', )' if new_ntypes > 0 else '()'

    data, index = 'new_data', 'df._index'
    func_text.append(f'{data} = {data_val}')
    func_text.append(f"return init_dataframe_internal({data}, {index}, df_type)\n")
    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {
        'pandas': pandas,
        'init_dataframe_internal': init_dataframe_internal,
        'df_type': df_type
    }

    global_vars.update({f'old_scheme_drop_idxs_{i}': old_scheme_drop_idxs[i] for i in range(old_ntypes)})

    return func_def, global_vars


@sdc_overload_method(DataFrameType, 'drop')
def sdc_pandas_dataframe_drop(df, labels=None, axis=0, index=None, columns=None, level=None, inplace=False,
                              errors='raise'):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.drop

    Limitations
    -----------
    - Parameters ``labels``, ``axis``, ``index``, ``level`` and ``inplace`` are currently unsupported.
    - Parameter ``columns`` is required and is expected to be a Literal value with one column name
    or List with columns names. Mutating a list of column names after it was defined and then using it as a
    columns argument results in an SDCLimitation exception at runtime.
    - Supported ``errors`` can be {``raise``, ``ignore``}, default ``raise``. If ``ignore``, suppress error and only
    existing labels are dropped.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_drop.py
        :language: python
        :lines: 37-
        :caption: Drop specified columns from DataFrame.
        :name: ex_dataframe_drop

    .. command-output:: python ./dataframe/dataframe_drop.py
        :cwd: ../../../examples

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

    """

    method_name = f'Method drop().'

    ty_checker = TypeChecker(method_name)
    ty_checker.check(df, DataFrameType)

    if not isinstance(labels, (types.Omitted, types.NoneType)) and labels is not None:
        ty_checker.raise_exc(labels, 'None', 'labels')

    if not isinstance(axis, (types.Omitted, types.Integer)) and axis != 0:
        ty_checker.raise_exc(axis, 'int', 'axis')

    if not isinstance(index, (types.Omitted, types.NoneType)) and index is not None:
        ty_checker.raise_exc(index, 'None', 'index')

    if not (isinstance(columns, (types.Omitted, types.StringLiteral))
            or (isinstance(columns, types.Tuple)
                and all(isinstance(c, types.StringLiteral) for c in columns))
            or (isinstance(columns, types.UniTuple) and isinstance(columns.dtype, types.StringLiteral))
            or isinstance(columns, types.List) and isinstance(columns.dtype, types.UnicodeType)
            ):
        ty_checker.raise_exc(columns, 'str, list of const str', 'columns')

    if not isinstance(level, (types.Omitted, types.NoneType, types.Literal)) and level is not None:
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(inplace, (types.Omitted, types.NoneType, types.Boolean)) and inplace:
        ty_checker.raise_exc(inplace, 'bool', 'inplace')

    if not isinstance(errors, (types.Omitted, types.UnicodeType, types.StringLiteral)) and errors != "raise":
        ty_checker.raise_exc(errors, 'str', 'errors')

    if isinstance(columns, types.List):
        if columns.initial_value is None:
            raise TypingError('{} Unsupported use of parameter columns:'
                              ' expected list of constant strings. Given: {}'.format(method_name, columns))
        else:
            # this works because global tuple of strings is captured as Tuple of StringLiterals
            columns_as_tuple = tuple(columns.initial_value)

            def _sdc_pandas_dataframe_drop_wrapper_impl(df, labels=None, axis=0, index=None,
                                                        columns=None, level=None, inplace=False, errors="raise"):

                # if at runtime columns list differs from it's initial value (known at compile time)
                # we cannot tell which columns to drop and what is the resulting DataFrameType, so raise exception
                if list(columns_as_tuple) != columns:
                    raise SDCLimitation("Unsupported use of parameter columns: non-const list was used.")

                return df.drop(labels=labels,
                               axis=axis,
                               index=index,
                               columns=columns_as_tuple,
                               level=level,
                               inplace=inplace,
                               errors=errors)

            return _sdc_pandas_dataframe_drop_wrapper_impl

    args = {'labels': None, 'axis': 0, 'index': None, 'columns': None, 'level': None, 'inplace': False,
            'errors': f'"raise"'}

    def sdc_pandas_dataframe_drop_impl(df, args, columns):
        func_args = ['df']
        for key, value in args.items():
            if key not in func_args:
                if isinstance(value, types.Literal):
                    value = value.literal_value
                func_args.append(f'{key}={value}')

        if isinstance(columns, types.StringLiteral):
            drop_cols = (columns.literal_value,)
        elif isinstance(columns, (types.Tuple, types.UniTuple)):
            drop_cols = tuple(column.literal_value for column in columns)
        else:
            raise ValueError('Only drop by one column or tuple of columns is currently supported in df.drop()')

        func_name = 'sdc_pandas_dataframe_drop_impl'
        func_def, global_vars = sdc_pandas_dataframe_drop_codegen(func_name, func_args, df, drop_cols)
        loc_vars = {}
        exec(func_def, global_vars, loc_vars)
        _drop_impl = loc_vars[func_name]
        return _drop_impl

    return sdc_pandas_dataframe_drop_impl(df, args, columns)


def df_getitem_slice_idx_main_codelines(self, idx):
    """Generate main code lines for df.getitem with idx of slice"""

    types_order = get_structure_maps(self.data, self.columns)[2]
    n_lists = len(types_order)

    results = []
    func_lines = []
    for i in range(n_lists):
        func_lines += [
            f'  list_{i} = self._data[{i}].copy()',
            f'  for i, item in enumerate(list_{i}):',
            f'    list_{i}[i] = item[idx]'
        ]

    all_lists_joined = ', '.join([f'list_{i}' for i in range(n_lists)]) + ', '
    res_data = f'({all_lists_joined})' if n_lists > 0 else '()'
    func_lines += [
        f'  res_data = {res_data}',
        f'  res_index = self._index[idx]',
        f'  return init_dataframe_internal(res_data, res_index, df_type)'
    ]

    return func_lines


def df_getitem_tuple_idx_main_codelines(self, literal_idx):
    """Generate main code lines for df.getitem with idx of tuple"""
    results = []
    func_lines = [f'  res_index = self.index']
    needed_cols = {col: i for i, col in enumerate(self.columns) if col in literal_idx}
    for col, i in needed_cols.items():
        col_loc = self.column_loc[col]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = self._data[{type_id}][{col_id}]',
            f'  {res_data} = pandas.Series(data_{i}, index=res_index, name="{col}")'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index)']

    return func_lines


def df_getitem_bool_series_idx_main_codelines(self, idx):
    """Generate main code lines for df.getitem"""

    # optimization for default indexes in df and idx when index alignment is trivial
    if (isinstance(self.index, PositionalIndexType) and isinstance(idx.index, PositionalIndexType)):
        func_lines = [f'  self_index = self._index',
                      f'  if len(self_index) > len(idx):',
                      f'    msg = "Unalignable boolean Series provided as indexer " + \\',
                      f'          "(index of the boolean Series and of the indexed object do not match)."',
                      f'    raise IndexingError(msg)',
                      f'  # do not trim idx._data to df length as getitem_by_mask handles such case',
                      f'  res_index = getitem_by_mask(self_index, idx._data)',
                      f'  # df index is default, same as positions so it can be used in take']
        results = []
        for i, col in enumerate(self.columns):
            col_loc = self.column_loc[col]
            type_id, col_id = col_loc.type_id, col_loc.col_id
            res_data = f'res_data_{i}'
            func_lines += [
                f'  data_{i} = self._data[{type_id}][{col_id}]',
                f'  {res_data} = sdc_take(data_{i}, res_index)'
            ]
            results.append((col, res_data))

        data = ', '.join(f'"{col}": {data}' for col, data in results)
        func_lines += [
            f'  return pandas.DataFrame({{{data}}}, index=res_index)'
        ]
    else:
        func_lines = [
            f'  self_index = self._index',
            f'  idx_reindexed_by_self = sdc_reindex_series(idx._data, idx._index, idx._name, self_index)',
            f'  final_mask = idx_reindexed_by_self._data',
            f'  res_index = self_index[final_mask]',
            f'  selected_pos = getitem_by_mask(numpy.arange(len(self_index)), final_mask)'
        ]
        results = []
        for i, col in enumerate(self.columns):
            col_loc = self.column_loc[col]
            type_id, col_id = col_loc.type_id, col_loc.col_id
            res_data = f'res_data_{i}'
            func_lines += [
                f'  data_{i} = self._data[{type_id}][{col_id}]',
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

    has_positional_index = isinstance(idx, PositionalIndexType)
    res_index_expr = 'taken_pos' if has_positional_index else 'self._index.take(taken_pos)'
    func_lines = [f'  length = len(self._index)',
                  f'  if length != len(idx):',
                  f'    raise ValueError("Item wrong length.")',
                  f'  taken_pos = getitem_by_mask(numpy.arange(length), idx)',
                  f'  res_index = {res_index_expr}'
                  ]
    results = []
    for i, col in enumerate(self.columns):
        col_loc = self.column_loc[col]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = self._data[{type_id}][{col_id}]',
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
        def _df_getitem_slice_idx_impl(self, idx):
          list_0 = self._data[0].copy()
          for i, item in enumerate(list_0):
            list_0[i] = item[idx]
          res_data = (list_0, )
          res_index = self._index[idx]
          return init_dataframe_internal(res_data, res_index, df_type)
    """
    func_lines = ['def _df_getitem_slice_idx_impl(self, idx):']
    if self.columns:
        func_lines += df_getitem_slice_idx_main_codelines(self, idx)
    else:
        # raise KeyError if input DF is empty
        func_lines += df_getitem_key_error_codelines()
    func_text = '\n'.join(func_lines)

    # since we need to know result df type to call init_dataframe_internal
    # deduce the resulting df index type
    index_getitem_sig = cpu_target.typing_context.resolve_function_type(
        operator.getitem,
        (self.index, idx),
        {}
    )
    new_index_type = index_getitem_sig.return_type
    df_type = DataFrameType(self.data, new_index_type, self.columns, column_loc=self.column_loc)

    global_vars = {'pandas': pandas,
                   'numpy': numpy,
                   'df_type': df_type,
                   'init_dataframe_internal': init_dataframe_internal}

    return func_text, global_vars


def df_getitem_tuple_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_tuple_idx_impl(self, idx):
          res_index = self.index
          data_0 = self._data[0][0]
          res_data_0 = pandas.Series(data_0, index=res_index, name="A")
          data_2 = self._data[0][2]
          res_data_2 = pandas.Series(data_2, index=res_index, name="C")
          return pandas.DataFrame({"A": res_data_0, "C": res_data_2}, index=res_index)
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
          self_index = self._index
          if len(self_index) > len(idx):
            msg = "Unalignable boolean Series provided as indexer " + \
                  "(index of the boolean Series and of the indexed object do not match)."
            raise IndexingError(msg)
          # do not trim idx._data to df length as getitem_by_mask handles such case
          res_index = getitem_by_mask(self_index, idx._data)
          # df index is default, same as positions so it can be used in take
          data_0 = self._data[0][0]
          res_data_0 = sdc_take(data_0, res_index)
          data_1 = self._data[0][1]
          res_data_1 = sdc_take(data_1, res_index)
          data_2 = self._data[0][2]
          res_data_2 = sdc_take(data_2, res_index)
          return pandas.DataFrame({"A": res_data_0, "B": res_data_1, "C": res_data_2}, index=res_index)
    """
    func_lines = ['def _df_getitem_bool_series_idx_impl(self, idx):']
    func_lines += df_getitem_bool_series_idx_main_codelines(self, idx)
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'getitem_by_mask': getitem_by_mask,
                   'sdc_take': nplike_take,
                   'sdc_reindex_series': sdc_reindex_series,
                   'IndexingError': IndexingError}

    return func_text, global_vars


def df_getitem_bool_array_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_bool_array_idx_impl(self, idx):
          length = len(self._index)
          if length != len(idx):
            raise ValueError("Item wrong length.")
          taken_pos = getitem_by_mask(numpy.arange(length), idx)
          res_index = self._index.take(taken_pos)
          data_0 = self._data[0][0]
          res_data_0 = sdc_take(data_0, taken_pos)
          data_1 = self._data[1][0]
          res_data_1 = sdc_take(data_1, taken_pos)
          data_2 = self._data[2][0]
          res_data_2 = sdc_take(data_2, taken_pos)
          return pandas.DataFrame({"A": res_data_0, "B": res_data_1, "C": res_data_2}, index=res_index)
    """
    func_lines = ['def _df_getitem_bool_array_idx_impl(self, idx):']
    func_lines += df_getitem_bool_array_idx_main_codelines(self, idx)
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'getitem_by_mask': getitem_by_mask,
                   'sdc_take': nplike_take}

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
        col_loc = self.column_loc.get(idx.literal_value)
        if col_loc is None:
            key_error = True
        else:
            type_id, col_id = col_loc.type_id, col_loc.col_id
            key_error = False

        def _df_getitem_str_literal_idx_impl(self, idx):
            if key_error == False:  # noqa
                data = self._data[type_id][col_id]
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
        if not check_types_comparable(self.index, idx.index):
            ty_checker.raise_exc(idx.index.dtype, self.index.dtype, 'idx.index.dtype')

        return gen_df_getitem_bool_series_idx_impl(self, idx)

    if isinstance(idx, types.Array) and isinstance(idx.dtype, types.Boolean):
        return gen_df_getitem_bool_array_idx_impl(self, idx)

    ty_checker = TypeChecker('Operator getitem().')
    expected_types = 'str, tuple(str), slice, series(bool), array(bool)'
    ty_checker.raise_exc(idx, expected_types, 'idx')


def df_getitem_tuple_at_codegen(self, row, col):
    """
    Example of generated implementation:
        def _df_getitem_tuple_at_impl(self, idx):
          row, _ = idx
          data = self._dataframe._data[2][0]
          res_data = pandas.Series(data, index=self._dataframe.index)
          return res_data.at[row]
    """
    func_lines = ['def _df_getitem_tuple_at_impl(self, idx):',
                  '  row, _ = idx']
    check = False
    for i in range(len(self.columns)):
        if self.columns[i] == col:
            col_loc = self.column_loc[col]
            type_id, col_id = col_loc.type_id, col_loc.col_id
            check = True
            func_lines += [
                f'  data = self._dataframe._data[{type_id}][{col_id}]',
                f'  res_data = pandas.Series(data, index=self._dataframe.index)',
                '  return res_data.at[row]',
            ]
    if check == False:  # noqa
        raise KeyError('Column is not in the DataFrame')

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas}

    return func_text, global_vars


def df_getitem_single_label_loc_codegen(self, idx):
    """
    Example of generated implementation:
        def _df_getitem_single_label_loc_impl(self, idx):
          idx_list = find_idx(self._dataframe._index, idx)
          data_0 = sdc_take(self._dataframe._data[0][0], idx_list)
          res_data_0 = pandas.Series(data_0)
          data_1 = sdc_take(self._dataframe._data[1][0], idx_list)
          res_data_1 = pandas.Series(data_1)
          data_2 = sdc_take(self._dataframe._data[0][1], idx_list)
          res_data_2 = pandas.Series(data_2)
          if len(idx_list) < 1:
            raise KeyError('Index is not in the DataFrame')
          new_index = self._dataframe._index.take(idx_list)
          return pandas.DataFrame({"A": res_data_0, "B": res_data_1, "C": res_data_2}, index=new_index)
    """
    if isinstance(self.index, PositionalIndexType):
        fill_list = ['  idx_list =  numpy.array([idx])']
        new_index = ['  new_index = numpy.array([idx])']

    else:
        fill_list = ['  idx_list = find_idx(self._dataframe._index, idx)']
        new_index = ['  new_index = self._dataframe._index.take(idx_list)']

    fill_list_text = '\n'.join(fill_list)
    new_index_text = '\n'.join(new_index)
    func_lines = ['def _df_getitem_single_label_loc_impl(self, idx):',
                  f'{fill_list_text}']
    results = []
    for i, c in enumerate(self.columns):
        col_loc = self.column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        data = f'data_{i}'
        res_data = f'res_data_{i}'
        func_lines += [f'  {data} = sdc_take(self._dataframe._data[{type_id}][{col_id}], idx_list)',
                       f'  {res_data} = pandas.Series({data})']
        results.append((c, res_data))

    func_lines += ['  if len(idx_list) < 1:',
                   "    raise KeyError('Index is not in the DataFrame')"]

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'{new_index_text}',
                   f'  return pandas.DataFrame({{{data}}}, index=new_index)']

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'numba': numba,
                   'sdc_take': nplike_take,
                   'find_idx': find_idx,
                   'KeyError': KeyError}

    return func_text, global_vars


def df_getitem_int_iloc_codegen(self, idx):
    """
    Example of generated implementation:
        def _df_getitem_int_iloc_impl(self, idx):
          if -1 < idx < len(self._dataframe.index):
            data_0 = pandas.Series(self._dataframe._data[0][0])
            result_0 = data_0.iat[idx]
            data_1 = pandas.Series(self._dataframe._data[0][1])
            result_1 = data_1.iat[idx]
            data_2 = pandas.Series(self._dataframe._data[1][0])
            result_2 = data_2.iat[idx]
            return pandas.Series(data=[result_0, result_1, result_2], index=['A', 'B', 'C'], name=str(idx))
          raise IndexingError('Index is out of bounds for axis')
    """
    func_lines = ['def _df_getitem_int_iloc_impl(self, idx):',
                  '  if -1 < idx < len(self._dataframe.index):']
    results = []
    index = []
    name = 'self._dataframe._index[idx]'
    if isinstance(self.index, PositionalIndexType):
        name = 'idx'
    for i, c in enumerate(self.columns):
        col_loc = self.column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f"result_{i}"
        func_lines += [f"    data_{i} = pandas.Series(self._dataframe._data[{type_id}][{col_id}])",
                       f"    {result_c} = data_{i}.iat[idx]"]
        results.append(result_c)
        index.append(c)
    data = ', '.join(col for col in results)
    func_lines += [f"    return pandas.Series(data=[{data}], index={index}, name=str({name}))",
                   f"  raise IndexingError('Index is out of bounds for axis')"]

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy, 'IndexingError': IndexingError}

    return func_text, global_vars


def df_getitem_slice_iloc_codegen(self, idx):
    """
    Example of generated implementation:
        def _df_getitem_slice_iloc_impl(self, idx):
          data_0 = pandas.Series(self._dataframe._data[0][0])
          result_0 = data_0.iloc[idx]
          data_1 = pandas.Series(self._dataframe._data[0][1])
          result_1 = data_1.iloc[idx]
          data_2 = pandas.Series(self._dataframe._data[1][0])
          result_2 = data_2.iloc[idx]
          return pandas.DataFrame(data={"A": result_0, "B": result_1, "C": result_2}, index=self._dataframe.index[idx])
    """
    func_lines = ['def _df_getitem_slice_iloc_impl(self, idx):']
    results = []
    for i, c in enumerate(self.columns):
        col_loc = self.column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f"result_{i}"
        func_lines += [f"  data_{i} = pandas.Series(self._dataframe._data[{type_id}][{col_id}])",
                       f"  {result_c} = data_{i}.iloc[idx]"]
        results.append((c, result_c))
    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f"  return pandas.DataFrame(data={{{data}}}, index=self._dataframe.index[idx])"]

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy}

    return func_text, global_vars


def df_getitem_list_iloc_codegen(self, idx):
    """
    Example of generated implementation:
        def _df_getitem_list_iloc_impl(self, idx):
          check_idx = False
          for i in idx:
            if -1 < i < len(self._dataframe.index):
              check_idx = True
          if check_idx == True:
            data_0 = pandas.Series(self._dataframe._data[0][0])
            result_0 = data_0.iloc[numpy.array(idx)]
            data_1 = pandas.Series(self._dataframe._data[0][1])
            result_1 = data_1.iloc[numpy.array(idx)]
            data_2 = pandas.Series(self._dataframe._data[1][0])
            result_2 = data_2.iloc[numpy.array(idx)]
            return pandas.DataFrame(data={"A": result_0, "B": result_1, "C": result_2}, index=idx)
          raise IndexingError('Index is out of bounds for axis')
    """
    func_lines = ['def _df_getitem_list_iloc_impl(self, idx):',
                  '  check_idx = False',
                  '  for i in idx:',
                  '    if -1 < i < len(self._dataframe.index):',
                  '      check_idx = True',
                  '  if check_idx == True:']
    results = []
    index = '[self._dataframe._index[i] for i in idx]'
    if isinstance(self.index, PositionalIndexType):
        index = 'idx'
    for i, c in enumerate(self.columns):
        col_loc = self.column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f"result_{i}"
        func_lines += [f"    data_{i} = pandas.Series(self._dataframe._data[{type_id}][{col_id}])",
                       f"    {result_c} = data_{i}.iloc[numpy.array(idx)]"]
        results.append((c, result_c))
    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f"    return pandas.DataFrame(data={{{data}}}, index={index})",
                   f"  raise IndexingError('Index is out of bounds for axis')"]

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy, 'IndexingError': IndexingError}

    return func_text, global_vars


def df_getitem_list_bool_iloc_codegen(self, idx):
    """
    Example of generated implementation:
        def _df_getitem_list_bool_iloc_impl(self, idx):
          if len(self._dataframe.index) == len(idx):
            data_0 = self._dataframe._data[0][0]
            result_0 = pandas.Series(data_0[numpy.array(idx)])
            data_1 = self._dataframe._data[0][1]
            result_1 = pandas.Series(data_1[numpy.array(idx)])
            data_2 = self._dataframe._data[1][0]
            result_2 = pandas.Series(data_2[numpy.array(idx)])
            return pandas.DataFrame(data={"A": result_0,
                                          "B": result_1,
                                          "C": result_2},
                                          index=self._dataframe.index[numpy.array(idx)])
          raise IndexingError('Item wrong length')
    """
    func_lines = ['def _df_getitem_list_bool_iloc_impl(self, idx):']
    results = []
    index = 'self._dataframe.index[numpy.array(idx)]'
    func_lines += ['  if len(self._dataframe.index) == len(idx):']
    for i, c in enumerate(self.columns):
        col_loc = self.column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f"result_{i}"
        func_lines += [f"    data_{i} = self._dataframe._data[{type_id}][{col_id}]",
                       f"    {result_c} = pandas.Series(data_{i}[numpy.array(idx)])"]
        results.append((c, result_c))
    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f"    return pandas.DataFrame(data={{{data}}}, index={index})",
                   f"  raise IndexingError('Item wrong length')"]

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy, 'IndexingError': IndexingError}

    return func_text, global_vars


def gen_df_getitem_tuple_at_impl(self, row, col):
    func_text, global_vars = df_getitem_tuple_at_codegen(self, row, col)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars['_df_getitem_tuple_at_impl']

    return _reduce_impl


gen_df_getitem_loc_single_label_impl = gen_impl_generator(
    df_getitem_single_label_loc_codegen, '_df_getitem_single_label_loc_impl')

gen_df_getitem_iloc_int_impl = gen_impl_generator(
    df_getitem_int_iloc_codegen, '_df_getitem_int_iloc_impl')

gen_df_getitem_iloc_slice_impl = gen_impl_generator(
    df_getitem_slice_iloc_codegen, '_df_getitem_slice_iloc_impl')

gen_df_getitem_iloc_list_impl = gen_impl_generator(
    df_getitem_list_iloc_codegen, '_df_getitem_list_iloc_impl')

gen_df_getitem_iloc_list_bool_impl = gen_impl_generator(
    df_getitem_list_bool_iloc_codegen, '_df_getitem_list_bool_iloc_impl')


@sdc_overload(operator.getitem)
def sdc_pandas_dataframe_accessor_getitem(self, idx):
    if not isinstance(self, DataFrameGetitemAccessorType):
        return None

    accessor = self.accessor.literal_value

    if accessor == 'at':
        num_idx = (isinstance(idx[0], types.Number)
                   and isinstance(self.dataframe.index, (PositionalIndexType, RangeIndexType, Int64IndexType)))
        str_idx = (isinstance(idx[0], (types.UnicodeType, types.StringLiteral))
                   and isinstance(self.dataframe.index, StringArrayType))
        if isinstance(idx, types.Tuple) and isinstance(idx[1], types.StringLiteral):
            if num_idx or str_idx:
                row = idx[0]
                col = idx[1].literal_value
                return gen_df_getitem_tuple_at_impl(self.dataframe, row, col)

            raise TypingError('Attribute at(). The row parameter type ({}) is different from the index type\
                              ({})'.format(type(idx[0]), type(self.dataframe.index)))

        raise TypingError('Attribute at(). The index must be a row and literal column. Given: {}'.format(idx))

    if accessor == 'loc':
        if isinstance(idx, (types.Integer, types.UnicodeType, types.StringLiteral)):
            return gen_df_getitem_loc_single_label_impl(self.dataframe, idx)

        ty_checker = TypeChecker('Attribute loc().')
        ty_checker.raise_exc(idx, 'int or str', 'idx')

    if accessor == 'iat':
        if isinstance(idx, types.Tuple) and isinstance(idx[1], types.Literal):
            col = idx[1].literal_value
            if -1 < col < len(self.dataframe.columns):
                col_loc = self.dataframe.column_loc[self.dataframe.columns[col]]
                type_id, col_id = col_loc.type_id, col_loc.col_id

                def df_getitem_iat_tuple_impl(self, idx):
                    row, _ = idx
                    if -1 < row < len(self._dataframe.index):
                        data = self._dataframe._data[type_id][col_id]
                        res_data = pandas.Series(data)
                        return res_data.iat[row]

                    raise IndexingError('Index is out of bounds for axis')

                return df_getitem_iat_tuple_impl

            raise IndexingError('Index is out of bounds for axis')

        raise TypingError('Operator getitem(). The index must be a row and literal column. Given: {}'.format(idx))

    if accessor == 'iloc':
        if isinstance(idx, types.SliceType):
            return gen_df_getitem_iloc_slice_impl(self.dataframe, idx)

        if (
            isinstance(idx, (types.List, types.Array)) and
            isinstance(idx.dtype, (types.Boolean, bool))
        ):
            return gen_df_getitem_iloc_list_bool_impl(self.dataframe, idx)

        if isinstance(idx, types.List):
            return gen_df_getitem_iloc_list_impl(self.dataframe, idx)

        if isinstance(idx, types.Integer):
            return gen_df_getitem_iloc_int_impl(self.dataframe, idx)

        if isinstance(idx, (types.Tuple, types.UniTuple)):

            def df_getitem_tuple_iat_impl(self, idx):
                return self._dataframe.iat[idx]

            return df_getitem_tuple_iat_impl

        raise TypingError('Attribute iloc(). The index must be an integer, a list or array of integers,\
                          a slice object with ints or a boolean array.\
                          Given: {}'.format(idx))

    raise TypingError('Operator getitem(). Unknown accessor. Only "loc", "iloc", "at", "iat" are supported.\
                      Given: {}'.format(accessor))


@sdc_overload_attribute(DataFrameType, 'iloc')
def sdc_pandas_dataframe_iloc(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.iloc

    Limitations
    -----------
    - Parameter ``'name'`` in new DataFrame can be String only
    - Column can be literal value only, in DataFrame.iloc[row, column]
    - Iloc works with basic cases only: an integer, a list or array of integers,
        a slice object with ints, a boolean array

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_iloc.py
       :language: python
       :lines: 36-
       :caption: Get value at specified index position.
       :name: ex_dataframe_iloc

    .. command-output:: python ./dataframe/dataframe_iloc.py
       :cwd: ../../../examples

    .. seealso::

        :ref:`DataFrame.iat <pandas.DataFrame.iat>`
            Fast integer location scalar accessor.

        :ref:`DataFrame.loc <pandas.DataFrame.loc>`
            Purely label-location based indexer for selection by label.

        :ref:`Series.iloc <pandas.Series.iloc>`
            Purely integer-location based indexing for selection by position.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.iloc` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_iloc*
    """

    ty_checker = TypeChecker('Attribute iloc().')
    ty_checker.check(self, DataFrameType)

    def sdc_pandas_dataframe_iloc_impl(self):
        return dataframe_getitem_accessor_init(self, 'iloc')

    return sdc_pandas_dataframe_iloc_impl


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


@sdc_overload_attribute(DataFrameType, 'at')
def sdc_pandas_dataframe_at(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Limitations
    -----------
    - ``Dataframe.at`` always returns ``array``.
    - Parameter ``column`` in ``idx`` must be a literal value.

    Pandas API: pandas.DataFrame.at

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_at.py
       :language: python
       :lines: 28-
       :caption: Access a single value for a row/column label pair.
       :name: ex_dataframe_at

    .. command-output:: python ./dataframe/dataframe_at.py
       :cwd: ../../../examples

    .. seealso::

        :ref:`DataFrame.iat <pandas.DataFrame.iat>`
            Access a single value for a row/column pair by integer position.

        :ref:`DataFrame.loc <pandas.DataFrame.loc>`
            Access a group of rows and columns by label(s).

        :ref:`Series.at <pandas.Series.at>`
            Access a single value using a label.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.at` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_at*
    """

    ty_checker = TypeChecker('Attribute at().')
    ty_checker.check(self, DataFrameType)

    def sdc_pandas_dataframe_at_impl(self):
        return dataframe_getitem_accessor_init(self, 'at')

    return sdc_pandas_dataframe_at_impl


@sdc_overload_attribute(DataFrameType, 'loc')
def sdc_pandas_dataframe_loc(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.loc

    Limitations
    -----------
    - Loc always returns Dataframe.
    - Parameter ``idx`` is supported only to be a single value, e.g. :obj:`df.loc['A']`.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_loc.py
       :language: python
       :lines: 36-
       :caption: Access a group of rows and columns by label(s) or a boolean array.
       :name: ex_dataframe_loc

    .. command-output:: python ./dataframe/dataframe_loc.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`DataFrame.at <pandas.DataFrame.at>`
            Access a single value for a row/column label pair.
        :ref:`DataFrame.iloc <pandas.DataFrame.iloc>`
            Access group of rows and columns by integer position(s).
        :ref:`DataFrame.xs <pandas.DataFrame.xs>`
            Returns a cross-section (row(s) or column(s)) from the Series/DataFrame.
        :ref:`Series.loc <pandas.Series.loc>`
            Access group of values using labels.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.loc` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_loc*
    """

    ty_checker = TypeChecker('Attribute loc().')
    ty_checker.check(self, DataFrameType)

    def sdc_pandas_dataframe_loc_impl(self):
        return sdc.datatypes.hpat_pandas_dataframe_getitem_types.dataframe_getitem_accessor_init(self, 'loc')

    return sdc_pandas_dataframe_loc_impl


@sdc_overload_method(DataFrameType, 'pct_change')
def pct_change_overload(df, periods=1, fill_method='pad', limit=None, freq=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.DataFrame.pct_change

    Limitations
    -----------
    Parameters ``limit`` and ``freq`` are supported only with default value ``None``.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_pct_change.py
        :language: python
        :lines: 36-
        :caption: Percentage change between the current and a prior element.
        :name: ex_dataframe_pct_change

    .. command-output:: python ./dataframe/dataframe_pct_change.py
        :cwd: ../../../examples

    .. seealso::

        :ref:`Series.diff <pandas.Series.diff>`
            Compute the difference of two elements in a Series.

        :ref:`DataFrame.diff <pandas.DataFrame.diff>`
            Compute the difference of two elements in a DataFrame.

        :ref:`Series.shift <pandas.Series.shift>`
            Shift the index by some number of periods.

        :ref:`DataFrame.shift <pandas.DataFrame.shift>`
            Shift the index by some number of periods.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.pct_change` implementation.

    .. only:: developer

      Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_pct_change*
    """

    name = 'pct_change'

    ty_checker = TypeChecker('Method {}().'.format(name))
    ty_checker.check(df, DataFrameType)

    if not isinstance(periods, (types.Integer, types.Omitted)) and periods != 1:
        ty_checker.raise_exc(periods, 'int64', 'periods')

    if not isinstance(fill_method, (str, types.UnicodeType, types.StringLiteral, types.NoneType, types.Omitted)):
        ty_checker.raise_exc(fill_method, 'string', 'fill_method')

    if not isinstance(limit, (types.Omitted, types.NoneType)) and limit is not None:
        ty_checker.raise_exc(limit, 'None', 'limit')

    if not isinstance(freq, (types.Omitted, types.NoneType)) and freq is not None:
        ty_checker.raise_exc(freq, 'None', 'freq')

    params = {'periods': 1, 'fill_method': '"pad"', 'limit': None, 'freq': None}
    ser_par = {'periods': 'periods', 'fill_method': 'fill_method', 'limit': 'limit', 'freq': 'freq'}

    return sdc_pandas_dataframe_apply_columns(df, name, params, ser_par)


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
    column_loc = df_type.column_loc
    for i, c in enumerate(df_type.columns):
        col_loc = column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f'result_{c}'
        func_lines += [
            f'  result_len=len({df})',
            f'  if "{c}" in list(values.keys()):',
            f'    series_{c} = pandas.Series({df}._data[{type_id}][{col_id}])',
            f'    val = list(values["{c}"])',
            f'    result_{c} = series_{c}.{func_name}(val)',
            f'  else:',
            f'    result = numpy.repeat(False, result_len)',
            f'    result_{c} = pandas.Series(result)'
        ]
        result_name.append((result_c, c))
    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)
    func_lines.append(f'  return pandas.DataFrame({{{data}}}, index={df}._index)')
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
    column_loc = df_type.column_loc
    for i, c in enumerate(df_type.columns):
        col_loc = column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f'result_{c}'
        func_lines += [
            f'  series_{c} = pandas.Series({df}._data[{type_id}][{col_id}])',
            f'  result = numpy.empty(len(series_{c}._data), numpy.bool_)',
            f'  result_len = len(series_{c}._data)'
        ]
        if isinstance(values.index, PositionalIndexType) and isinstance(df_type.index, PositionalIndexType):
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
        elif isinstance(values.index, PositionalIndexType):
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
        elif isinstance(df_type.index, PositionalIndexType):
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
    func_lines.append(f'  return pandas.DataFrame({{{data}}}, index={df}._index)')
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
    column_loc = df_type.column_loc
    for i, c in enumerate(df_type.columns):
        col_loc = column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f'result_{c}'
        func_lines += [f'  series_{c} = pandas.Series({df}._data[{type_id}][{col_id}])']
        if c in in_df.columns:
            func_lines += [
                f'  series_{c}_values = pandas.Series({val}.{c})',
                f'  result = numpy.empty(len(series_{c}._data), numpy.bool_)',
                f'  result_len = len(series_{c}._data)'
            ]
            if isinstance(df.index, PositionalIndexType) and isinstance(df_type.index, PositionalIndexType):
                func_lines += [
                    f'  for i in range(result_len):',
                    f'    if i <= len(series_{c}_values):',
                    f'      if series_{c}._data[i] == series_{c}_values._data[i]:',
                    f'        result[i] = True',
                    f'      else:',
                    f'        result[i] = False',
                    f'    else:',
                    f'      result[i] = False']
            elif isinstance(df_type.index, PositionalIndexType):
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
            elif isinstance(df.index, PositionalIndexType):
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
    func_lines.append(f'  return pandas.DataFrame({{{data}}}, index={df}._index)')
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


def sdc_pandas_dataframe_isin_iter(name, all_params, ser_par, columns, column_loc):
    func_text, global_vars = _dataframe_apply_columns_codegen(name, all_params, ser_par, columns, column_loc)
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
    """

    name = 'isin'

    ty_checker = TypeChecker('Method {}().'.format(name))
    ty_checker.check(df, DataFrameType)

    if not isinstance(values, (SeriesType, types.List, types.Set, DataFrameType, types.DictType)):
        ty_checker.raise_exc(values, 'iterable, Series, DataFrame', 'values')

    all_params = ['df', 'values']

    if isinstance(values, (types.List, types.Set)):
        ser_par = 'values=values'
        return sdc_pandas_dataframe_isin_iter(name, all_params, ser_par, df.columns, df.column_loc)

    if isinstance(values, types.DictType):
        return sdc_pandas_dataframe_isin_dict(name, df, values, all_params)

    if isinstance(values, SeriesType):
        return sdc_pandas_dataframe_isin_ser(name, df, values, all_params)

    if isinstance(values, DataFrameType):
        return sdc_pandas_dataframe_isin_df(name, df, values, all_params)


@sdc_overload_method(DataFrameType, 'groupby')
def sdc_pandas_dataframe_groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                                 group_keys=True, squeeze=False, observed=False):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.groupby

    Limitations
    -----------
    - Parameters ``axis``, ``level``, ``as_index``, ``group_keys``, ``squeeze`` and ``observed`` \
are currently unsupported by Intel Scalable Dataframe Compiler
    - Parameter ``by`` is supported as single literal column name only
    - Mutating the contents of a DataFrame between creating a groupby object and calling it's methods is unsupported

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/groupby/dataframe_groupby_min.py
       :language: python
       :lines: 27-
       :caption: Groupby and calculate the minimum in each group.
       :name: ex_dataframe_groupby

    .. command-output:: python ./dataframe/groupby/dataframe_groupby_min.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`resample <pandas.DataFrame.resample>`
            Resample time-series data.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame attribute :meth:`pandas.DataFrame.groupby` implementation
    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_groupby.TestGroupBy.test_dataframe_groupby*

    Parameters
    ----------

    self: :obj:`pandas.DataFrame`
        Input DataFrame.
    by: :obj:`mapping`, :obj:`function`, :obj:`string` or :obj:`list`
        Used to determine the groups for the groupby.
    axis : :obj:`int` or :obj:`string`, default 0
        Split along rows (0) or columns (1).
    level : :obj:`int` or :obj:`str`, default None
        If the axis is a MultiIndex (hierarchical), group by a particular
        level or levels.
    as_index : :obj:`bool`, default True
        For aggregated output, return object with group labels as the
        index.
    sort : :obj:`bool`, default True
        Sort group keys. Get better performance by turning this off.
        Note this does not influence the order of observations within each
        group. Groupby preserves the order of rows within each group.
    group_keys : :obj:`bool`, default True
        When calling apply, add group keys to index to identify pieces.
    squeeze : :obj:`bool`, default False
        Reduce the dimensionality of the return type if possible,
        otherwise return a consistent type.
    observed : :obj:`bool`, default False
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.

    Returns
    -------
    :class:`pandas.DataFrameGroupBy`
        Returns a groupby object that contains information about the groups.
"""

    if not isinstance(by, types.StringLiteral):
        return None

    column_id = self.columns.index(by.literal_value)
    list_type = types.ListType(types.int64)
    by_type = self.data[column_id].dtype

    col_loc = self.column_loc[by.literal_value]
    type_id, col_id = col_loc.type_id, col_loc.col_id

    def sdc_pandas_dataframe_groupby_impl(self, by=None, axis=0, level=None, as_index=True, sort=True,
                                          group_keys=True, squeeze=False, observed=False):

        by_column_data = self._data[type_id][col_id]
        chunks = parallel_chunks(len(by_column_data))
        dict_parts = [Dict.empty(by_type, list_type) for _ in range(len(chunks))]

        # filling separate dict of by_value -> positions for each chunk of initial array
        for i in numba.prange(len(chunks)):
            chunk = chunks[i]
            res = dict_parts[i]
            for j in range(chunk.start, chunk.stop):
                if isna(by_column_data, j):
                    continue
                value = by_column_data[j]
                group_list = res.get(value)
                if group_list is None:
                    new_group_list = List.empty_list(types.int64)
                    new_group_list.append(j)
                    res[value] = new_group_list
                else:
                    group_list.append(j)

        # merging all dict parts into a single resulting dict
        res_dict = dict_parts[0]
        for i in range(1, len(chunks)):
            res_dict = merge_groupby_dicts_inplace(res_dict, dict_parts[i])

        return init_dataframe_groupby(self, column_id, res_dict, sort)

    return sdc_pandas_dataframe_groupby_impl


def df_set_column_index_codelines(self):
    """Generate code lines with definition of resulting index for DF set_column"""
    index_param_expr = 'self._index' if not isinstance(self.index, EmptyIndexType) else 'None'
    func_lines = []
    if self.columns:
        func_lines += [
            f'  length = len(self._index)',
            f'  if length == 0:',
            f'    raise SDCLimitation("Could not set item for DataFrame with empty columns")',
            f'  elif length != len(value):',
            f'    raise ValueError("Length of values does not match length of index")',
        ]
    else:
        func_lines += ['  length = len(value)']
    func_lines += [f'  res_index = {index_param_expr}']

    return func_lines


def df_add_column_codelines(self, key):
    """Generate code lines to add new column to DF"""
    func_lines = df_set_column_index_codelines(self)  # provide res_index = ...

    results = []
    for i, col in enumerate(self.columns):
        col_loc = self.column_loc[col]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = self._data[{type_id}][{col_id}]',
            f'  {res_data} = data_{i}',
        ]
        results.append((col, res_data))

    res_data = 'new_res_data'
    literal_key = key.literal_value
    func_lines += [f'  {res_data} = value']
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
            col_loc = self.column_loc[col]
            type_id, col_id = col_loc.type_id, col_loc.col_id
            func_lines += [f'  data_{i} = self._data[{type_id}][{col_id}]']

        res_data = f'res_data_{i}'
        func_lines += [
            f'  {res_data} = data_{i}',
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=self._index)']

    return func_lines


def df_add_column_codegen(self, key):
    """
    Example of generated implementation:
        def _df_add_column_impl(self, key, value):
          length = len(self._index)
          if length == 0:
            raise SDCLimitation("Could not set item for DataFrame with empty columns")
          elif length != len(value):
            raise ValueError("Length of values does not match length of index")
          res_index = self._index
          data_0 = self._data[0][0]
          res_data_0 = data_0
          data_1 = self._data[1][0]
          res_data_1 = data_1
          new_res_data = value
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
      length = len(self._index)
      if length == 0:
        raise SDCLimitation("Could not set item for DataFrame with empty columns")
      elif length != len(value):
        raise ValueError("Length of values does not match length of index")
      res_index = self._index
      data_0 = value
      res_data_0 = data_0
      data_1 = self._data[1][0]
      res_data_1 = data_1
      return pandas.DataFrame({"A": res_data_0, "C": res_data_1}, index=self._index)
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
    Pandas API: pandas.DataFrame.setitem

    Set data to a DataFrame by indexer.

    Limitations
    -----------
    - Supported setting a column in a DataFrame through private method ``df._set_column(key, value)``.
    - DataFrame passed into jit region as a parameter is not changed outside of the region.
    New DataFrame should be returned from the region in this case.
    - Supported setting a column in a non-empty DataFrame as a 1D array only.

    .. literalinclude:: ../../../examples/dataframe/setitem/df_set_new_column.py
       :language: python
       :lines: 37-
       :caption: Setting new column to the DataFrame.
       :name: ex_dataframe_set_new_column

    .. command-output:: python ./dataframe/setitem/df_set_new_column.py
       :cwd: ../../../examples

    .. literalinclude:: ../../../examples/dataframe/setitem/df_set_existing_column.py
       :language: python
       :lines: 37-
       :caption: Setting data to existing column of the DataFrame.
       :name: ex_dataframe_set_existing_column

    .. command-output:: python ./dataframe/setitem/df_set_existing_column.py
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
        :ref:`DataFrame.getitem <pandas.DataFrame.getitem>`
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


def sdc_pandas_dataframe_reset_index_codegen(drop, all_params, columns, column_loc):
    """
    Example of generated implementation:
        def _df_reset_index_impl(self, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
          result_0 = self._data[0][0]
          result_1 = self._data[0][1]
          return pandas.DataFrame({"A": result_0, "B": result_1})
    """
    result_name = []
    all_params_str = ', '.join(all_params)
    func_lines = [f'def _df_reset_index_impl({all_params_str}):']
    if not drop:
        old_index = 'old_index'
        func_lines += [f'  {old_index} = self.index']
        result_name.append((old_index, 'index'))
    for i, c in enumerate(columns):
        col_loc = column_loc[c]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        result_c = f'result_{i}'
        func_lines += [
            f'  result_{i} = self._data[{type_id}][{col_id}]'
        ]
        result_name.append((result_c, c))
    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)
    func_lines += [f'  return pandas.DataFrame({{{data}}})']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'numpy': numpy}

    return func_text, global_vars


def sdc_pandas_dataframe_reset_index_impl(self, drop=False):
    all_params = ['self', 'level=None', 'drop=False', 'inplace=False', 'col_level=0', 'col_fill=""']
    func_text, global_vars = sdc_pandas_dataframe_reset_index_codegen(drop, all_params,
                                                                      self.columns, self.column_loc)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _apply_impl = loc_vars[f'_df_reset_index_impl']

    return _apply_impl


@sdc_overload_method(DataFrameType, 'reset_index')
def sdc_pandas_dataframe_reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.reset_index

    Limitations
    -----------
    - Reset the index of the DataFrame, and use the default one instead.
    - Parameters level, inplacem col_level, col_fill unsupported.
    - Parameter drop can be only literal value or default value.

    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/dataframe_reset_index_drop_False.py
        :language: python
        :lines: 36-
        :caption: Reset the index of the DataFrame, and use the default one instead.
                  The old index becomes the first column.
        :name: ex_dataframe_reset_index

    .. command-output:: python ./dataframe/dataframe_reset_index_drop_False.py
        :cwd: ../../../examples

    .. literalinclude:: ../../../examples/dataframe/dataframe_reset_index_drop_True.py
        :language: python
        :lines: 36-
        :caption: Reset the index of the DataFrame, and use the default one instead.
        :name: ex_dataframe_reset_index

    .. command-output:: python ./dataframe/dataframe_reset_index_drop_True.py
        :cwd: ../../../examples

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.reset_index` implementation.

   .. only:: developer

       Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_reset_index*
   """

    func_name = 'reset_index'

    ty_checker = TypeChecker('Method {}().'.format(func_name))
    ty_checker.check(self, DataFrameType)

    if not (level is None or isinstance(level, types.Omitted)):
        raise TypingError('{} Unsupported parameter level. Given: {}'.format(func_name, level))

    if not isinstance(drop, (types.Omitted, types.Boolean, bool)):
        ty_checker.raise_exc(drop, 'bool', 'drop')

    if not (inplace is False or isinstance(inplace, types.Omitted)):
        raise TypingError('{} Unsupported parameter inplace. Given: {}'.format(func_name, inplace))

    if not (col_level == 0 or isinstance(col_level, types.Omitted)):
        raise TypingError('{} Unsupported parameter col_level. Given: {}'.format(func_name, col_level))

    if not (col_fill == '' or isinstance(col_fill, types.Omitted)):
        raise TypingError('{} Unsupported parameter col_fill. Given: {}'.format(func_name, col_fill))

    if isinstance(drop, types.Literal):
        literal_drop = drop.literal_value
        return sdc_pandas_dataframe_reset_index_impl(self, drop=literal_drop)
    elif isinstance(drop, types.Omitted):
        return sdc_pandas_dataframe_reset_index_impl(self, drop=drop.value)
    elif isinstance(drop, bool):
        return sdc_pandas_dataframe_reset_index_impl(self, drop=drop)

    raise SDCLimitation('Method {}(). Parameter drop is only supported as a literal.'.format(func_name))


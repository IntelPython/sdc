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
from sdc.datatypes.hpat_pandas_dataframe_getitem_types import (DataFrameGetitemAccessorType,
                                                               dataframe_getitem_accessor_init)
from sdc.datatypes.common_functions import SDCLimitation
from sdc.datatypes.hpat_pandas_dataframe_rolling_types import _hpat_pandas_df_rolling_init
from sdc.datatypes.hpat_pandas_rolling_types import (
    gen_sdc_pandas_rolling_overload_body, sdc_pandas_rolling_docstring_tmpl)
from sdc.datatypes.hpat_pandas_groupby_functions import init_dataframe_groupby, merge_groupby_dicts_inplace
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
from sdc.utilities.utils import sdc_overload, sdc_overload_method, sdc_overload_attribute
from sdc.hiframes.api import isna
from sdc.functions.numpy_like import getitem_by_mask
from sdc.datatypes.common_functions import _sdc_take, sdc_reindex_series
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


def sdc_pandas_dataframe_append_codegen(df, other, _func_name, ignore_index_value, indexes_comparable, args):
    """
    Input:
    df = pd.DataFrame({'A': ['cat', 'dog', np.nan], 'B': [.2, .3, np.nan]})
    other = pd.DataFrame({'A': ['bird', 'fox', 'mouse'], 'C': ['a', np.nan, '']})
    ignore_index=True

    Func generated:
    def sdc_pandas_dataframe_append_impl(df, other, ignore_index=False, verify_integrity=False, sort=None):
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

    if ignore_index_value == True:  # noqa
        func_text.append(f'return pandas.DataFrame({{{data}}})\n')
    else:
        if indexes_comparable == False:  # noqa
            func_text.append(f'raise SDCLimitation("Indexes of dataframes are expected to have comparable '
                             f'(both Numeric or String) types if parameter ignore_index is set to False.")')
        else:
            func_text += [f'joined_index = hpat_arrays_append(df.index, other.index)\n',
                          f'return pandas.DataFrame({{{data}}}, index=joined_index)\n']

    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {'pandas': pandas,
                   'init_series': sdc.hiframes.api.init_series,
                   'fill_array': sdc.datatypes.common_functions.fill_array,
                   'fill_str_array': sdc.datatypes.common_functions.fill_str_array,
                   'hpat_arrays_append': sdc.datatypes.common_functions.hpat_arrays_append,
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

    none_or_numeric_indexes = ((isinstance(df.index, types.NoneType) or isinstance(df.index, types.Number)) and
                               (isinstance(other.index, types.NoneType) or isinstance(other.index, types.Number)))
    indexes_comparable = check_types_comparable(df.index, other.index) or none_or_numeric_indexes

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


def _dataframe_reduce_columns_codegen(func_name, func_params, series_params, columns, df_structure):
    result_name_list = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']

    for i, c in enumerate(columns):
        type_id = df_structure[c].type_id
        col_id = df_structure[c].col_type_id
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
                                                               df.df_structure)
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

    for key, value in params.items():
        all_params.append('{}={}'.format(key, value))
    for key, value in ser_params.items():
        ser_par.append('{}={}'.format(key, value))

    s_par = ', '.join(ser_par)

    df_func_name = f'_df_{func_name}_impl'

    func_text, global_vars = _dataframe_apply_columns_codegen(func_name, all_params, s_par, df.columns)

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _reduce_impl = loc_vars[df_func_name]

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

        `pandas.isna <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isna.html#pandas.isna>`_
            Top-level isna.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Pandas DataFrame method :meth:`pandas.DataFrame.isna` implementation.

    .. only:: developer
        Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_isna*
    """

    return sdc_pandas_dataframe_isna_codegen(df, 'isna')


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
        func_text.append(f'new_col_{column_id}_data_{"df"} = get_dataframe_data({"df"}, '
                         f'{df_columns_indx[column_name]})')
        column_list.append((f'new_col_{column_id}_data_df', column_name))

    data = ', '.join(f'"{column_name}": {column}' for column, column_name in column_list)
    index = 'df.index'
    func_text.append(f"return pandas.DataFrame({{{data}}}, index={index})\n")
    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {'pandas': pandas, 'get_dataframe_data': sdc.hiframes.pd_dataframe_ext.get_dataframe_data}

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
    or Tuple with columns names.
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


def df_index_expr(self, length_expr=None, as_range=False):
    """Generate expression to get or create index of DF"""
    if isinstance(self.index, types.NoneType):
        if length_expr is None:
            length_expr = df_length_expr(self)

        if as_range:
            return f'range({length_expr})'
        else:
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
                      f'  self_index = {df_index_expr(self, as_range=True)}',
                      f'  if length > len(idx):',
                      f'    msg = "Unalignable boolean Series provided as indexer " + \\',
                      f'          "(index of the boolean Series and of the indexed object do not match)."',
                      f'    raise IndexingError(msg)',
                      f'  # do not trim idx._data to length as getitem_by_mask handles such case',
                      f'  res_index = getitem_by_mask(self_index, idx._data)',
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
                      f'  reindexed_idx = sdc_reindex_series(idx._data, idx.index, idx._name, self_index)',
                      f'  res_index = getitem_by_mask(self_index, reindexed_idx._data)',
                      f'  selected_pos = getitem_by_mask(numpy.arange(length), reindexed_idx._data)']

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
                  f'  self_index = {df_index_expr(self, as_range=True)}',
                  f'  taken_pos = getitem_by_mask(self_index, idx)',
                  f'  res_index = sdc_take(self_index, taken_pos)']
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
          self_index = range(len(self._data[0]))
          if length > len(idx):
            msg = "Unalignable boolean Series provided as indexer " + \
                  "(index of the boolean Series and of the indexed object do not match)."
            raise IndexingError(msg)
          # do not trim idx._data to length as getitem_by_mask handles such case
          res_index = getitem_by_mask(self_index, idx._data)
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
          self_index = range(len(self._data[0]))
          taken_pos = getitem_by_mask(self_index, idx)
          res_index = sdc_take(self_index, taken_pos)
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


def df_getitem_int_iloc_codegen(self, idx):
    """
    Example of generated implementation:
        def _df_getitem_int_iloc_impl(self, idx):
        if -1 < idx < len(self._dataframe.index):
            data_0 = pandas.Series(self._dataframe._data[0])
            result_0 = data_0.iat[idx]
            data_1 = pandas.Series(self._dataframe._data[1])
            result_1 = data_1.iat[idx]
            return pandas.Series(data=[result_0, result_1], index=['A', 'B'], name=str(idx))
        raise IndexingError('Index is out of bounds for axis')
    """
    func_lines = ['def _df_getitem_int_iloc_impl(self, idx):',
                  '  if -1 < idx < len(self._dataframe.index):']
    results = []
    index = []
    name = 'self._dataframe._index[idx]'
    if isinstance(self.index, types.NoneType):
        name = 'idx'
    for i, c in enumerate(self.columns):
        result_c = f"result_{i}"
        func_lines += [f"    data_{i} = pandas.Series(self._dataframe._data[{i}])",
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
            data_0 = pandas.Series(self._dataframe._data[0])
            result_0 = data_0.iloc[idx]
            data_1 = pandas.Series(self._dataframe._data[1])
            result_1 = data_1.iloc[idx]
            return pandas.DataFrame(data={"A": result_0, "B": result_1}, index=self._dataframe.index[idx])
    """
    func_lines = ['def _df_getitem_slice_iloc_impl(self, idx):']
    results = []
    for i, c in enumerate(self.columns):
        result_c = f"result_{i}"
        func_lines += [f"  data_{i} = pandas.Series(self._dataframe._data[{i}])",
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
                data_0 = pandas.Series(self._dataframe._data[0])
                result_0 = data_0.iloc[numpy.array(idx)]
                data_1 = pandas.Series(self._dataframe._data[1])
                result_1 = data_1.iloc[numpy.array(idx)]
                return pandas.DataFrame(data={"A": result_0, "B": result_1}, index=idx)
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
    if isinstance(self.index, types.NoneType):
        index = 'idx'
    for i, c in enumerate(self.columns):
        result_c = f"result_{i}"
        func_lines += [f"    data_{i} = pandas.Series(self._dataframe._data[{i}])",
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
                data_0 = self._dataframe._data[0]
                result_0 = pandas.Series(data_0[numpy.array(idx)])
                data_1 = self._dataframe._data[1]
                result_1 = pandas.Series(data_1[numpy.array(idx)])
                return pandas.DataFrame(data={"A": result_0, "B": result_1},
                    index=self._dataframe.index[numpy.array(idx)])
            raise IndexingError('Item wrong length')
    """
    func_lines = ['def _df_getitem_list_bool_iloc_impl(self, idx):']
    results = []
    index = 'self._dataframe.index[numpy.array(idx)]'
    func_lines += ['  if len(self._dataframe.index) == len(idx):']
    for i, c in enumerate(self.columns):
        result_c = f"result_{i}"
        func_lines += [f"    data_{i} = self._dataframe._data[{i}]",
                       f"    {result_c} = pandas.Series(data_{i}[numpy.array(idx)])"]
        results.append((c, result_c))
    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f"    return pandas.DataFrame(data={{{data}}}, index={index})",
                   f"  raise IndexingError('Item wrong length')"]

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy, 'IndexingError': IndexingError}

    return func_text, global_vars


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

    def sdc_pandas_dataframe_groupby_impl(self, by=None, axis=0, level=None, as_index=True, sort=True,
                                          group_keys=True, squeeze=False, observed=False):

        by_column_data = self._data[column_id]
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


def sdc_pandas_dataframe_reset_index_codegen(drop, all_params, columns):
    """
    Example of generated implementation:
        def _df_reset_index_impl(self, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
          old_index = self.index
          result_0 = get_dataframe_data(self, 0)
          result_1 = get_dataframe_data(self, 1)
          result_2 = get_dataframe_data(self, 2)
          return pandas.DataFrame({"index": old_index, "A": result_0, "B": result_1, "C": result_2})
    """
    result_name = []
    all_params_str = ', '.join(all_params)
    func_lines = [f'def _df_reset_index_impl({all_params_str}):']
    df = all_params[0]
    if not drop.literal_value:
        old_index = 'old_index'
        func_lines += [f'  {old_index} = {df}.index']
        result_name.append((old_index, 'index'))
    for i, c in enumerate(columns):
        result_c = f'result_{i}'
        func_lines += [
            f'  result_{i} = get_dataframe_data({df}, {i})'
        ]
        result_name.append((result_c, c))
    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)
    func_lines += [f'  return pandas.DataFrame({{{data}}})']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def sdc_pandas_dataframe_reset_index_impl(self, drop=False):
    all_params = ['self', 'level=None', 'drop=False', 'inplace=False', 'col_level=0', 'col_fill=""']

    func_text, global_vars = sdc_pandas_dataframe_reset_index_codegen(drop, all_params, self.columns)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _apply_impl = loc_vars[f'_df_reset_index_impl']

    return _apply_impl


def sdc_pandas_dataframe_reset_index_default_codegen(drop, all_params, columns):
    """
    Example of generated implementation:
        def _df_reset_index_impl(self, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
          old_index = self.index
          result_0 = get_dataframe_data(self, 0)
          result_1 = get_dataframe_data(self, 1)
          return pandas.DataFrame({"index": old_index, "A": result_0, "B": result_1})
    """
    result_name = []
    all_params_str = ', '.join(all_params)
    func_lines = [f'def _df_reset_index_impl({all_params_str}):']
    df = all_params[0]
    if not drop:
        old_index = 'old_index'
        func_lines += [f'  {old_index} = {df}.index']
        result_name.append((old_index, 'index'))
    for i, c in enumerate(columns):
        result_c = f'result_{i}'
        func_lines += [
            f'  result_{i} = get_dataframe_data({df}, {i})'
        ]
        result_name.append((result_c, c))
    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)
    func_lines += [f'  return pandas.DataFrame({{{data}}})']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def sdc_pandas_dataframe_reset_index_impl_default(self, drop=False):
    all_params = ['self', 'level=None', 'drop=False', 'inplace=False', 'col_level=0', 'col_fill=""']

    func_text, global_vars = sdc_pandas_dataframe_reset_index_default_codegen(drop, all_params, self.columns)
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

    if not (isinstance(drop, (types.Omitted, types.Boolean)) or drop is False):
        ty_checker.raise_exc(drop, 'bool', 'drop')

    if isinstance(drop, types.Omitted):
        drop = False

    if not (inplace is False or isinstance(inplace, types.Omitted)):
        raise TypingError('{} Unsupported parameter inplace. Given: {}'.format(func_name, inplace))

    if not (col_level == 0 or isinstance(col_level, types.Omitted)):
        raise TypingError('{} Unsupported parameter col_level. Given: {}'.format(func_name, col_level))

    if not (col_fill == '' or isinstance(col_fill, types.Omitted)):
        raise TypingError('{} Unsupported parameter col_fill. Given: {}'.format(func_name, col_fill))

    if not isinstance(drop, types.Literal):
        if isinstance(drop, bool):
            return sdc_pandas_dataframe_reset_index_impl_default(self, drop=drop)
        else:
            raise SDCLimitation('{} only work with Boolean literals drop.'.format(func_name))

    return sdc_pandas_dataframe_reset_index_impl(self, drop=drop)

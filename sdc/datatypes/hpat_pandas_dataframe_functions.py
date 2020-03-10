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
import sdc


from numba import types
from numba.special import literally
from numba.typed import List, Dict

from sdc.hiframes.pd_dataframe_ext import DataFrameType
from sdc.hiframes.pd_series_type import SeriesType
from sdc.utilities.sdc_typing_utils import (TypeChecker, check_index_is_numeric,
                                            check_types_comparable, kwsparams2list,
                                            gen_df_impl_generator, find_common_dtype_from_numpy_dtypes)
from sdc.str_arr_ext import StringArrayType

from sdc.hiframes.pd_dataframe_type import DataFrameType

from sdc.datatypes.hpat_pandas_dataframe_rolling_types import _hpat_pandas_df_rolling_init
from sdc.datatypes.hpat_pandas_rolling_types import (
    gen_sdc_pandas_rolling_overload_body, sdc_pandas_rolling_docstring_tmpl)
from sdc.datatypes.hpat_pandas_groupby_functions import init_dataframe_groupby
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
from sdc.utilities.utils import sdc_overload, sdc_overload_method, sdc_overload_attribute
from sdc.hiframes.api import isna


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
            df_len = len(get_dataframe_data(df, 0)) if empty_df == False else 0  # noqa

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
        row_len = len(get_dataframe_data(df, 0))
        df_col_A = get_dataframe_data(df, 0)
        df_col_B = get_dataframe_data(df, 1)
        df_col_C = get_dataframe_data(df, 2)
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
    func_text.append(f'row_len = len(get_dataframe_data(df, 0))')

    for index, column_name in enumerate(df.columns):
        func_text.append(f'df_col_{index} = get_dataframe_data(df, {index})')
        column_list.append(f'df_col_{index}')

    func_text.append(f'df_values = numpy.empty(row_len*{column_len}, numpy.dtype("{numba_common_dtype}"))')
    func_text.append('for i in range(row_len):')
    for j in range(column_len):
        func_text.append(indent + f'df_values[i * {column_len} + {j}] = {column_list[j]}[i]')

    func_text.append(f"return df_values.reshape(row_len, {column_len})\n")
    func_definition.extend([indent + func_line for func_line in func_text])
    func_def = '\n'.join(func_definition)

    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'get_dataframe_data': sdc.hiframes.pd_dataframe_ext.get_dataframe_data}

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
        len_df = len(get_dataframe_data(df, 0))
        len_other = len(get_dataframe_data(other, 0))
        new_col_A_data_df = get_dataframe_data(df, 0)
        new_col_A_data_other = get_dataframe_data(other, 0)
        new_col_A = init_series(new_col_A_data_df).append(init_series(new_col_A_data_other))._data
        new_col_B_data_df = get_dataframe_data(df, 1)
        new_col_B_data = init_series(new_col_B_data_df)._data
        new_col_B = fill_array(new_col_B_data, len_df+len_other)
        new_col_C_data_other = get_dataframe_data(other, 1)
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

    func_text.append(f'len_df = len(get_dataframe_data(df, 0))')
    func_text.append(f'len_other = len(get_dataframe_data(other, 0))')

    for col_name, col_id in df_columns_indx.items():
        func_text.append(f'new_col_{col_id}_data_{"df"} = get_dataframe_data({"df"}, {col_id})')
        if col_name in other_columns_indx:
            other_col_id = other_columns_indx.get(col_name)
            func_text.append(f'new_col_{col_id}_data_{"other"} = '
                             f'get_dataframe_data({"other"}, {other_columns_indx.get(col_name)})')
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
            func_text.append(f'new_col_{col_id}_data_{"other"} = get_dataframe_data({"other"}, {col_id})')
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

    global_vars = {'pandas': pandas, 'get_dataframe_data': sdc.hiframes.pd_dataframe_ext.get_dataframe_data,
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
#           series_A = init_series(get_dataframe_data(df, 0))
#           result_A = series_A.count(level=level)
#           series_B = init_series(get_dataframe_data(df, 1))
#           result_B = series_B.count(level=level)
#           return pandas.Series([result_A, result_B], ['A', 'B'])


def _dataframe_reduce_columns_codegen(func_name, func_params, series_params, columns):
    result_name_list = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    for i, c in enumerate(columns):
        result_c = f'result_{i}'
        func_lines += [f'  series_{i} = pandas.Series(get_dataframe_data({func_params[0]}, {i}))',
                       f'  {result_c} = series_{i}.{func_name}({series_params})']
        result_name_list.append(result_c)
    all_results = ', '.join(result_name_list)
    all_columns = ', '.join([f"'{c}'" for c in columns])

    func_lines += [f'  return pandas.Series([{all_results}], [{all_columns}])']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'get_dataframe_data': get_dataframe_data}

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
            series_float = pandas.Series(get_dataframe_data(df, 0))
            result_float = series_float.head(n=n)
            series_int = pandas.Series(get_dataframe_data(df, 1))
            result_int = series_int.head(n=n)
            series_string = pandas.Series(get_dataframe_data(df, 2))
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
        func_lines += [f'  series_{c} = pandas.Series(get_dataframe_data(df, {i}))',
                       f'  {result_c} = series_{c}.{func_name}({series_params})']
        results.append((columns[i], result_c))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, {ind})']
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas,
                   'get_dataframe_data': get_dataframe_data}

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


def _dataframe_apply_columns_codegen(func_name, func_params, series_params, columns):
    result_name = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    for i, c in enumerate(columns):
        result_c = f'result_{i}'
        func_lines += [f'  series_{i} = pandas.Series(get_dataframe_data({func_params[0]}, {i}))',
                       f'  {result_c} = series_{i}.{func_name}({series_params})']
        result_name.append((result_c, c))

    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)

    func_lines += [f'  return pandas.DataFrame({{{data}}})']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'get_dataframe_data': get_dataframe_data}

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
        new_col_A_data_df = get_dataframe_data(df, 0)
        new_col_B_data_df = get_dataframe_data(df, 1)
        new_col_C_data_df = get_dataframe_data(df, 2)
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


def _dataframe_codegen_isna(func_name, columns, df):
    """
    Example func_text for func_name='isna' columns=('float', 'int', 'string'):

        def _df_isna_impl(df):
            series_float = pandas.Series(get_dataframe_data(df, 0))
            result_float = series_float.isna()
            series_int = pandas.Series(get_dataframe_data(df, 1))
            result_int = series_int.isna()
            series_string = pandas.Series(get_dataframe_data(df, 2))
            result_string = series_string.isna()
            return pandas.DataFrame({"float": result_float, "int": result_int, "string": result_string},
                                    index = df._index)
    """
    results = []
    func_lines = [f'def _df_{func_name}_impl(df):']
    index = df_index_codegen_all(df)
    for i, c in enumerate(columns):
        result_c = f'result_{c}'
        func_lines += [f'  series_{c} = pandas.Series(get_dataframe_data(df, {i}))',
                       f'  {result_c} = series_{c}.{func_name}()']
        results.append((columns[i], result_c))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, {index})']
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas,
                   'get_dataframe_data': get_dataframe_data}

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


def df_length_codelines(self):
    """Generate code lines to get length of DF"""
    if self.columns:
        return ['  length = len(get_dataframe_data(self, 0))']

    return ['  length = 0']


def df_index_codelines(self, with_length=False):
    """Generate code lines to get or create index of DF"""
    func_lines = []
    if isinstance(self.index, types.NoneType):
        if with_length:
            func_lines += df_length_codelines(self)

        func_lines += ['  res_index = numpy.arange(length)']
    else:
        func_lines += ['  res_index = self._index']

    return func_lines


def df_getitem_slice_idx_main_codelines(self, idx):
    """Generate main code lines for df.getitem with idx of slice"""
    results = []
    func_lines = df_index_codelines(self, with_length=True)
    for i, col in enumerate(self.columns):
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = get_dataframe_data(self, {i})',
            f'  {res_data} = pandas.Series(data_{i}[idx], index=res_index[idx], name="{col}")'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index[idx])']

    return func_lines


def df_getitem_tuple_idx_main_codelines(self, literal_idx):
    """Generate main code lines for df.getitem with idx of tuple"""
    results = []
    func_lines = df_index_codelines(self, with_length=True)
    needed_cols = {col: i for i, col in enumerate(self.columns) if col in literal_idx}
    for col, i in needed_cols.items():
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = get_dataframe_data(self, {i})',
            f'  {res_data} = pandas.Series(data_{i}, index=res_index, name="{col}")'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index)']

    return func_lines


def df_getitem_bool_series_idx_main_codelines(self, idx):
    """Generate main code lines for df.getitem"""
    func_lines = df_length_codelines(self)
    func_lines += ['  _idx_data = idx._data[:length]']
    func_lines += df_index_codelines(self)

    results = []
    for i, col in enumerate(self.columns):
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = get_dataframe_data(self, {i})',
            f'  series_{i} = pandas.Series(data_{i}, index=res_index, name="{col}")',
            f'  {res_data} = series_{i}[_idx_data]'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index[_idx_data])']

    return func_lines


def df_getitem_bool_array_idx_main_codelines(self, idx):
    """Generate main code lines for df.getitem"""
    func_lines = df_length_codelines(self)
    func_lines += ['  if length != len(idx):',
                   '    raise ValueError("Item wrong length.")']
    func_lines += df_index_codelines(self)

    results = []
    for i, col in enumerate(self.columns):
        res_data = f'res_data_{i}'
        func_lines += [
            f'  data_{i} = get_dataframe_data(self, {i})',
            f'  {res_data} = pandas.Series(data_{i}[idx], index=res_index[idx], name="{col}")'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=res_index[idx])']

    return func_lines


def df_getitem_key_error_codelines():
    """Generate code lines to raise KeyError"""
    return ['  raise KeyError("Column is not in the DataFrame")']


def df_getitem_slice_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_slice_idx_impl(self, idx)
          res_index = self._index
          data_0 = get_dataframe_data(self, 0)
          res_data_0 = pandas.Series(data_0[idx], index=res_index[idx], name="A")
          data_1 = get_dataframe_data(self, 1)
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
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def df_getitem_tuple_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_tuple_idx_impl(self, idx)
          res_index = self._index
          data_1 = get_dataframe_data(self, 1)
          res_data_1 = pandas.Series(data_1, index=res_index, name="B")
          data_2 = get_dataframe_data(self, 2)
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
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def df_getitem_bool_series_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_bool_series_idx_impl(self, idx):
          length = len(get_dataframe_data(self, 0))
          _idx_data = idx._data[:length]
          res_index = self._index
          data_0 = get_dataframe_data(self, 0)
          series_0 = pandas.Series(data_0, index=res_index, name="A")
          res_data_0 = series_0[_idx_data]
          data_1 = get_dataframe_data(self, 1)
          series_1 = pandas.Series(data_1, index=res_index, name="B")
          res_data_1 = series_1[_idx_data]
          return pandas.DataFrame({"A": res_data_0, "B": res_data_1}, index=res_index[_idx_data])
    """
    func_lines = ['def _df_getitem_bool_series_idx_impl(self, idx):']
    func_lines += df_getitem_bool_series_idx_main_codelines(self, idx)
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def df_getitem_bool_array_idx_codegen(self, idx):
    """
    Example of generated implementation with provided index:
        def _df_getitem_bool_array_idx_impl(self, idx):
          length = len(get_dataframe_data(self, 0))
          if length != len(idx):
            raise ValueError("Item wrong length.")
          res_index = numpy.arange(length)
          data_0 = get_dataframe_data(self, 0)
          res_data_0 = pandas.Series(data_0[idx], index=res_index[idx], name="A")
          data_1 = get_dataframe_data(self, 1)
          res_data_1 = pandas.Series(data_1[idx], index=res_index[idx], name="B")
          return pandas.DataFrame({"A": res_data_0, "B": res_data_1}, index=res_index[idx])
    """
    func_lines = ['def _df_getitem_bool_array_idx_impl(self, idx):']
    func_lines += df_getitem_bool_array_idx_main_codelines(self, idx)
    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas, 'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


gen_df_getitem_slice_idx_impl = gen_df_impl_generator(
    df_getitem_slice_idx_codegen, '_df_getitem_slice_idx_impl')
gen_df_getitem_tuple_idx_impl = gen_df_impl_generator(
    df_getitem_tuple_idx_codegen, '_df_getitem_tuple_idx_impl')
gen_df_getitem_bool_series_idx_impl = gen_df_impl_generator(
    df_getitem_bool_series_idx_codegen, '_df_getitem_bool_series_idx_impl')
gen_df_getitem_bool_array_idx_impl = gen_df_impl_generator(
    df_getitem_bool_array_idx_codegen, '_df_getitem_bool_array_idx_impl')


@sdc_overload(operator.getitem)
def sdc_pandas_dataframe_getitem(self, idx):
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
                data = get_dataframe_data(self, col_idx)
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
        by_column_data = get_dataframe_data(self, column_id)
        for i in numpy.arange(len(by_column_data)):
            if isna(by_column_data, i):
                continue
            value = by_column_data[i]
            group_list = grouped.get(value, List.empty_list(types.int64))
            group_list.append(i)
            grouped[value] = group_list

        return init_dataframe_groupby(self, column_id, grouped, sort)

    return sdc_pandas_dataframe_groupby_impl


def sdc_pandas_dataframe_reset_index_codegen(df_type, all_params, columns, result_name, func_line):
    joined = ', '.join(all_params)
    func_lines = [f'def _df_reset_index_impl({joined}):']
    df = all_params[0]
    func_lines += func_line
    for i, c in enumerate(columns):
        result_c = f'result_{c}'
        func_lines += [
            f'  result_{c} = get_dataframe_data({df}, {i})'
        ]
        result_name.append((result_c, c))
    data = ', '.join(f'"{column_name}": {column}' for column, column_name in result_name)
    func_lines += [f'  return pandas.DataFrame({{{data}}}, index=numpy.arange(len(result_{c})))']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas,
                   'numpy': numpy,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def sdc_pandas_dataframe_reset_index_drop_False(df_type, result_name, func_lines):
    codegen_ind = df_index_codelines(df_type, True)
    result_ind = f'res_index'
    for i in codegen_ind:
        func_lines += [f'{i}']
    result_name.append((result_ind, 'index'))

    return result_name, func_lines


def sdc_pandas_dataframe_reset_index_impl(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
    all_params = ['self', 'level=None', 'drop=False', 'inplace=False', 'col_level=0', 'col_fill=""']
    df_func_name = f'_df_reset_index_impl'

    result_name = []
    func_lines = []
    drop_val = str(drop)[14:-1]
    if drop_val == 'False':
        result_name, func_lines = sdc_pandas_dataframe_reset_index_drop_False(self, result_name, func_lines)

    func_text, global_vars = sdc_pandas_dataframe_reset_index_codegen(self, all_params, self.columns,
                                                                      result_name, func_lines)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _apply_impl = loc_vars[df_func_name]

    return _apply_impl


@sdc_overload_method(DataFrameType, 'reset_index')
def sdc_pandas_dataframe_reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
    """
   Pandas DataFrame method :meth:`pandas.DataFrame.reset_index` implementation.

   .. only:: developer

       Test: python -m sdc.runtests -k sdc.tests.test_dataframe.TestDataFrame.test_df_reset_index*

   Parameters
   -----------
   self: :class:`pandas.DataFrame`
       input arg
   level: :obj:`int`, `str`, `tuple`, or `list`, default None
       *unsupported*
   drop: :obj:`bool`, default False
       Just reset the index, without inserting it as a column in the new DataFrame.
   inplace: :obj:`bool`, default False
       *unsupported*
   col_level: :obj:`int`, `str`, default 0
       *unsupported*
   col_fill: :obj, default ''
       *unsupported*

   Returns
   -------
   :obj:`pandas.DataFrame`
        DataFrame with the new index or None if inplace=True.
   """

    func_name = 'reset_index'

    ty_checker = TypeChecker('Method {}().'.format(func_name))
    ty_checker.check(self, DataFrameType)

    if not (isinstance(level, (types.Omitted, types.NoneType)) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not isinstance(drop, (types.Omitted, types.Boolean)):
        ty_checker.raise_exc(drop, 'bool', 'drop')

    if not (isinstance(inplace, (types.Omitted, types.Boolean)) or inplace is False):
        ty_checker.raise_exc(inplace, 'False', 'inplace')

    if not (isinstance(col_level, (types.Omitted, types.Integer)) or col_level == 0):
        ty_checker.raise_exc(col_level, '0', 'col_level')

    if not (isinstance(col_fill, (types.Omitted, types.StringLiteral)) or col_fill == ""):
        ty_checker.raise_exc(col_fill, '""', 'col_fill')

    return sdc_pandas_dataframe_reset_index_impl(self, level=level, drop=drop, inplace=inplace,
                                                 col_level=col_level, col_fill=col_fill)

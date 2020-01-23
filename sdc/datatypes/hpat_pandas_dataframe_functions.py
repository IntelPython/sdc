# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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


import pandas
import numpy
import sdc

from numba import types, prange, literal_unroll
from sdc.hiframes.pd_dataframe_ext import DataFrameType
from sdc.datatypes.common_functions import TypeChecker
from sdc.str_arr_ext import StringArrayType

from sdc.hiframes.pd_dataframe_type import DataFrameType

from sdc.datatypes.hpat_pandas_dataframe_rolling_types import _hpat_pandas_df_rolling_init
from sdc.datatypes.hpat_pandas_rolling_types import (
    gen_sdc_pandas_rolling_overload_body, sdc_pandas_rolling_docstring_tmpl)
from sdc.datatypes.common_functions import TypeChecker, hpat_arrays_append_overload, find_common_dtype_from_numpy_dtypes
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
from sdc.utils import sdc_overload_method, sdc_overload_attribute


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


@sdc_overload_attribute(DataFrameType, 'values')
def hpat_pandas_dataframe_values(df):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame.values

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

    def hpat_pandas_df_values_impl(df):
        row_len = len(get_dataframe_data(df, 0))
        local_data = literal_unroll(df._data)
        column_len = len(local_data)

        df_values = numpy.empty(row_len*column_len, numba_common_dtype)

        for i in prange(row_len):
            for j in range(column_len):
                df_values[i*column_len + j] = numpy.array(local_data[j])[i]

        return df_values.reshape((row_len, column_len))

    return hpat_pandas_df_values_impl


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

    for col_name, i in df_columns_indx.items():
        func_text.append(f'new_col_{col_name}_data_{"df"} = get_dataframe_data({"df"}, {i})')
        if col_name in other_columns_indx:
            func_text.append(f'new_col_{col_name}_data_{"other"} = '
                             f'get_dataframe_data({"other"}, {other_columns_indx.get(col_name)})')
            s1 = f'init_series(new_col_{col_name}_data_{"df"})'
            s2 = f'init_series(new_col_{col_name}_data_{"other"})'
            func_text.append(f'new_col_{col_name} = {s1}.append({s2})._data')
        else:
            func_text.append(f'new_col_{col_name}_data = init_series(new_col_{col_name}_data_df)._data')
            if col_name in string_type_columns:
                func_text.append(f'new_col_{col_name} = fill_str_array(new_col_{col_name}_data, len_df+len_other)')
            else:
                func_text.append(f'new_col_{col_name} = fill_array(new_col_{col_name}_data, len_df+len_other)')
        column_list.append((f'new_col_{col_name}', col_name))

    for col_name, i in other_columns_indx.items():
        if col_name not in df_columns_indx:
            func_text.append(f'new_col_{col_name}_data_{"other"} = get_dataframe_data({"other"}, {i})')
            func_text.append(f'new_col_{col_name}_data = init_series(new_col_{col_name}_data_other)._data')
            if col_name in string_type_columns:
                func_text.append(
                    f'new_col_{col_name} = '
                    f'fill_str_array(new_col_{col_name}_data, len_df+len_other, push_back=False)')
            else:
                func_text.append(f'new_col_{col_name} = '
                                 f'fill_array(new_col_{col_name}_data, len_df+len_other, push_back=False)')
            column_list.append((f'new_col_{col_name}', col_name))

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
        result_c = f'result_{c}'
        func_lines += [f'  series_{c} = pandas.Series(get_dataframe_data({func_params[0]}, {i}))',
                       f'  {result_c} = series_{c}.{func_name}({series_params})']
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


def _dataframe_apply_columns_codegen(func_name, func_params, series_params, columns):
    result_name = []
    joined = ', '.join(func_params)
    func_lines = [f'def _df_{func_name}_impl({joined}):']
    for i, c in enumerate(columns):
        result_c = f'result_{c}'
        func_lines += [f'  series_{c} = pandas.Series(get_dataframe_data({func_params[0]}, {i}))',
                       f'  {result_c} = series_{c}.{func_name}({series_params})']
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

    for column in saved_df_columns:
        func_text.append(f'new_col_{column}_data_{"df"} = get_dataframe_data({"df"}, {df_columns_indx[column]})')
        column_list.append((f'new_col_{column}_data_df', column))

    data = ', '.join(f'"{column_name}": {column}' for column, column_name in column_list)
    # TODO: Handle index
    func_text.append(f"return pandas.DataFrame({{{data}}})\n")
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


@sdc_overload_method(DataFrameType, 'pct_change')
def pct_change_overload(df, periods=1, fill_method='pad', limit=None, freq=None):
    """
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

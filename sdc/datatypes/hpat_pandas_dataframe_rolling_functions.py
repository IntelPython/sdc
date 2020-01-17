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
import numpy
import pandas

from numba.types import float64, Boolean, Omitted, NoneType
from sdc.datatypes.common_functions import TypeChecker, params2list
from sdc.datatypes.hpat_pandas_dataframe_rolling_types import DataFrameRollingType
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
from sdc.hiframes.pd_dataframe_type import DataFrameType
from sdc.hiframes.pd_series_type import SeriesType
from sdc.utils import sdc_overload_method


sdc_pandas_dataframe_rolling_docstring_tmpl = """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.{method_name}
{limitations_block}
    Examples
    --------
    .. literalinclude:: ../../../examples/dataframe/rolling/dataframe_rolling_{method_name}.py
       :language: python
       :lines: 27-
       :caption: {example_caption}
       :name: ex_dataframe_rolling_{method_name}

    .. command-output:: python ./dataframe/rolling/dataframe_rolling_{method_name}.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`DataFrame.{method_name} <pandas.DataFrame.{method_name}>`
            Similar method for DataFrame.
        :ref:`DataFrame.{method_name} <pandas.DataFrame.{method_name}>`
            Similar method for DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas DataFrame method :meth:`pandas.DataFrame.rolling.{method_name}()` implementation.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_dataframe_rolling_{method_name}

    Parameters
    ----------
    self: :class:`pandas.DataFrame.rolling`
        input arg{extra_params}

    Returns
    -------
    :obj:`pandas.DataFrame`
         returns :obj:`pandas.DataFrame` object
"""


def df_rolling_params_codegen():
    """Generate rolling parameters"""
    params = ['window', 'min_periods', 'center', 'win_type', 'on', 'axis', 'closed']
    return ', '.join(f'self._{p}' for p in params)


def df_rolling_method_with_other_df_codegen(method_name, self, other, args=None, kws=None):
    args = args or []
    kwargs = kws or {}

    rolling_params = df_rolling_params_codegen()
    method_kws = {k: k for k in kwargs}
    impl_params = ['self'] + args + params2list(kwargs)
    impl_params_as_str = ', '.join(impl_params)

    data_columns = {col: idx for idx, col in enumerate(self.data.columns)}
    other_columns = {col: idx for idx, col in enumerate(other.columns)}

    # columns order matters
    common_columns = [col for col in data_columns if col in other_columns]
    all_columns = [col for col in data_columns]
    for col in other_columns:
        if col in all_columns:
            continue
        all_columns.append(col)

    results = []
    impl_name = f'_df_rolling_{method_name}_with_other_df_impl'
    func_lines = [f'def {impl_name}({impl_params_as_str}):']

    if 'pairwise' in kwargs:
        func_lines += [
            '  if pairwise is None:',
            '    _pairwise = False',
            '  else:',
            '    _pairwise = pairwise',
            '  if _pairwise:',
            '    raise ValueError("Method rolling.corr(). The object pairwise\\n expected: False, None")'
        ]

    data_length = 'len(get_dataframe_data(self._data, 0))' if data_columns else '0'
    other_length = 'len(get_dataframe_data(other, 0))' if other_columns else '0'
    func_lines += [f'  length = max([{data_length}, {other_length}])']

    for col in all_columns:
        res_data = f'result_data_{col}'
        if col in common_columns:
            other_series = f'other_series_{col}'
            method_kws['other'] = other_series
            method_params = ', '.join(args + params2list(method_kws))
            func_lines += [
                f'  data_{col} = get_dataframe_data(self._data, {data_columns[col]})',
                f'  other_data_{col} = get_dataframe_data(other, {other_columns[col]})',
                f'  series_{col} = pandas.Series(data_{col})',
                f'  {other_series} = pandas.Series(other_data_{col})',
                f'  rolling_{col} = series_{col}.rolling({rolling_params})',
                f'  result_{col} = rolling_{col}.corr({method_params})',
                f'  {res_data} = result_{col}._data[:length]'
            ]
        else:
            func_lines += [
                f'  {res_data} = numpy.empty(length, dtype=float64)',
                f'  {res_data}[:] = numpy.nan'
            ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}})']
    func_text = '\n'.join(func_lines)

    global_vars = {'numpy': numpy, 'pandas': pandas, 'float64': float64,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def df_rolling_method_tail_codegen(method_params, df_columns, method_name):
    rolling_params = df_rolling_params_codegen()
    method_params_as_str = ', '.join(method_params)

    results = []
    func_lines = []
    for idx, col in enumerate(df_columns):
        res_data = f'result_data_{col}'
        func_lines += [
            f'  data_{col} = get_dataframe_data(self._data, {idx})',
            f'  series_{col} = pandas.Series(data_{col})',
            f'  rolling_{col} = series_{col}.rolling({rolling_params})',
            f'  result_{col} = rolling_{col}.{method_name}({method_params_as_str})',
            f'  {res_data} = result_{col}._data[:len(data_{col})]'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}})']

    return func_lines


def df_rolling_method_codegen(method_name, self, with_other=False, args=None, kws=None):
    args = args or []
    kwargs = kws or {}

    impl_params = ['self'] + args + params2list(kwargs)
    impl_params_as_str = ', '.join(impl_params)

    impl_name = f'_df_rolling_{method_name}_impl'
    func_lines = [f'def {impl_name}({impl_params_as_str}):']

    if 'other' in kwargs and not with_other:
        if 'pairwise' in kwargs:
            func_lines += [
                '  if pairwise is None:',
                '    _pairwise = True',
                '  else:',
                '    _pairwise = pairwise',
                '  if _pairwise:',
                '    raise ValueError("Method rolling.corr(). The object pairwise\\n expected: False")'
            ]
        method_params = args + ['{}={}'.format(k, k) for k in kwargs if k != 'other']
    else:
        method_params = args + ['{}={}'.format(k, k) for k in kwargs]

    func_lines += df_rolling_method_tail_codegen(method_params, self.data.columns, method_name)
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas, 'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def gen_df_rolling_method_with_other_df_impl(method_name, self, other, args=None, kws=None):
    func_text, global_vars = df_rolling_method_with_other_df_codegen(method_name, self, other,
                                                                     args=args, kws=kws)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _impl = loc_vars[f'_df_rolling_{method_name}_with_other_df_impl']

    return _impl


def gen_df_rolling_method_impl(method_name, self, with_other=False, args=None, kws=None):
    func_text, global_vars = df_rolling_method_codegen(method_name, self, with_other=with_other,
                                                       args=args, kws=kws)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _impl = loc_vars[f'_df_rolling_{method_name}_impl']

    return _impl


@sdc_overload_method(DataFrameRollingType, 'corr')
def sdc_pandas_dataframe_rolling_corr(self, other=None, pairwise=None):

    ty_checker = TypeChecker('Method rolling.corr().')
    ty_checker.check(self, DataFrameRollingType)

    accepted_other = (Omitted, NoneType, DataFrameType, SeriesType)
    if not isinstance(other, accepted_other) and other is not None:
        ty_checker.raise_exc(other, 'DataFrame, Series', 'other')

    accepted_pairwise = (bool, Boolean, Omitted, NoneType)
    if not isinstance(pairwise, accepted_pairwise) and pairwise is not None:
        ty_checker.raise_exc(pairwise, 'bool', 'pairwise')

    kws = {'other': 'None', 'pairwise': 'None'}

    if isinstance(other, DataFrameType):
        return gen_df_rolling_method_with_other_df_impl('corr', self, other, kws=kws)

    with_other = not isinstance(other, (Omitted, NoneType)) and other is not None
    return gen_df_rolling_method_impl('corr', self, with_other=with_other, kws=kws)


@sdc_overload_method(DataFrameRollingType, 'min')
def sdc_pandas_dataframe_rolling_min(self):

    ty_checker = TypeChecker('Method rolling.min().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('min', self)


sdc_pandas_dataframe_rolling_corr.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'corr',
    'example_caption': 'Calculate rolling correlation.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    other: :obj:`Series` or :obj:`DataFrame`
        Other Series or DataFrame.
    pairwise: :obj:`bool`
        Calculate pairwise combinations of columns within a DataFrame.
    """
})

sdc_pandas_dataframe_rolling_min.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'min',
    'example_caption': 'Calculate the rolling minimum.',
    'limitations_block': '',
    'extra_params': ''
})

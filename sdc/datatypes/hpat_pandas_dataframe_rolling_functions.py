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

from numba.core.types import (float64, Boolean, Integer, Number, Omitted,
                         NoneType, StringLiteral, UnicodeType)
from sdc.utilities.sdc_typing_utils import TypeChecker, kwsparams2list
from sdc.datatypes.hpat_pandas_dataframe_rolling_types import DataFrameRollingType
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
from sdc.hiframes.pd_dataframe_type import DataFrameType
from sdc.hiframes.pd_series_type import SeriesType
from sdc.utilities.utils import sdc_overload_method


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


def df_rolling_method_other_df_codegen(method_name, self, other, args=None, kws=None):
    args = args or []
    kwargs = kws or {}

    rolling_params = df_rolling_params_codegen()
    method_kws = {k: k for k in kwargs}
    impl_params = ['self'] + args + kwsparams2list(kwargs)
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
    impl_name = f'_df_rolling_{method_name}_other_df_impl'
    func_lines = [f'def {impl_name}({impl_params_as_str}):']

    if 'pairwise' in kwargs:
        func_lines += [
            '  if pairwise is None:',
            '    _pairwise = False',
            '  else:',
            '    _pairwise = pairwise',
            '  if _pairwise:',
            f'    raise ValueError("Method rolling.{method_name}(). The object pairwise\\n expected: False, None")'
        ]

    data_length = 'len(self._data._data[0][0])' if data_columns else '0'
    other_length = 'len(other._data[0][0])' if other_columns else '0'
    func_lines += [f'  length = max([{data_length}, {other_length}])']

    for col in all_columns:
        res_data = f'result_data_{col}'
        if col in common_columns:
            col_loc = self.data.column_loc[col]
            type_id, col_id = col_loc.type_id, col_loc.col_id
            other_col_loc = other.column_loc[col]
            other_type_id = other_col_loc.type_id
            other_col_id = other_col_loc.col_id

            other_series = f'other_series_{col}'
            method_kws['other'] = other_series
            method_params = ', '.join(args + kwsparams2list(method_kws))
            func_lines += [
                f'  data_{col} = self._data._data[{type_id}][{col_id}]',
                f'  other_data_{col} = other._data[{other_type_id}][{other_col_id}]',
                f'  series_{col} = pandas.Series(data_{col})',
                f'  {other_series} = pandas.Series(other_data_{col})',
                f'  rolling_{col} = series_{col}.rolling({rolling_params})',
                f'  result_{col} = rolling_{col}.{method_name}({method_params})',
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

    global_vars = {'numpy': numpy, 'pandas': pandas, 'float64': float64}

    return func_text, global_vars


def df_rolling_method_main_codegen(method_params, df_columns, column_loc, method_name):
    rolling_params = df_rolling_params_codegen()
    method_params_as_str = ', '.join(method_params)

    results = []
    func_lines = []
    for idx, col in enumerate(df_columns):
        col_loc = column_loc[col]
        type_id, col_id = col_loc.type_id, col_loc.col_id
        res_data = f'result_data_{col}'
        func_lines += [
            f'  data_{col} = self._data._data[{type_id}][{col_id}]',
            f'  series_{col} = pandas.Series(data_{col})',
            f'  rolling_{col} = series_{col}.rolling({rolling_params})',
            f'  result_{col} = rolling_{col}.{method_name}({method_params_as_str})',
            f'  {res_data} = result_{col}._data[:len(data_{col})]'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}})']

    return func_lines


def gen_df_rolling_method_other_none_codegen(rewrite_name=None):
    """Generate df.rolling method code generator based on name of the method"""
    def df_rolling_method_other_none_codegen(method_name, self, args=None, kws=None):
        _method_name = rewrite_name or method_name
        args = args or []
        kwargs = kws or {}

        impl_params = ['self'] + args + kwsparams2list(kwargs)
        impl_params_as_str = ', '.join(impl_params)

        impl_name = f'_df_rolling_{_method_name}_other_none_impl'
        func_lines = [f'def {impl_name}({impl_params_as_str}):']

        if 'pairwise' in kwargs:
            func_lines += [
                '  if pairwise is None:',
                '    _pairwise = True',
                '  else:',
                '    _pairwise = pairwise',
                '  if _pairwise:',
                f'    raise ValueError("Method rolling.{_method_name}(). The object pairwise\\n expected: False")'
            ]
        method_params = args + ['{}={}'.format(k, k) for k in kwargs if k != 'other']
        func_lines += df_rolling_method_main_codegen(method_params, self.data.columns, self.data.column_loc,
                                                     method_name)

        func_text = '\n'.join(func_lines)

        global_vars = {'pandas': pandas}

        return func_text, global_vars

    return df_rolling_method_other_none_codegen


df_rolling_method_other_none_codegen = gen_df_rolling_method_other_none_codegen()
df_rolling_cov_other_none_codegen = gen_df_rolling_method_other_none_codegen('cov')


def df_rolling_method_codegen(method_name, self, args=None, kws=None):
    args = args or []
    kwargs = kws or {}

    impl_params = ['self'] + args + kwsparams2list(kwargs)
    impl_params_as_str = ', '.join(impl_params)

    impl_name = f'_df_rolling_{method_name}_impl'
    func_lines = [f'def {impl_name}({impl_params_as_str}):']

    method_params = args + ['{}={}'.format(k, k) for k in kwargs]
    func_lines += df_rolling_method_main_codegen(method_params, self.data.columns,
                                                 self.data.column_loc, method_name)
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas}

    return func_text, global_vars


def gen_df_rolling_method_other_df_impl(method_name, self, other, args=None, kws=None):
    func_text, global_vars = df_rolling_method_other_df_codegen(method_name, self, other,
                                                                args=args, kws=kws)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _impl = loc_vars[f'_df_rolling_{method_name}_other_df_impl']

    return _impl


def gen_df_rolling_method_other_none_impl(method_name, self, args=None, kws=None):
    func_text, global_vars = df_rolling_method_other_none_codegen(method_name, self,
                                                                  args=args, kws=kws)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _impl = loc_vars[f'_df_rolling_{method_name}_other_none_impl']

    return _impl


def gen_df_rolling_cov_other_none_impl(method_name, self, args=None, kws=None):
    func_text, global_vars = df_rolling_cov_other_none_codegen(method_name, self,
                                                               args=args, kws=kws)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _impl = loc_vars[f'_df_rolling_cov_other_none_impl']

    return _impl


def gen_df_rolling_method_impl(method_name, self, args=None, kws=None):
    func_text, global_vars = df_rolling_method_codegen(method_name, self,
                                                       args=args, kws=kws)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _impl = loc_vars[f'_df_rolling_{method_name}_impl']

    return _impl


@sdc_overload_method(DataFrameRollingType, 'apply')
def sdc_pandas_dataframe_rolling_apply(self, func, raw=None):

    ty_checker = TypeChecker('Method rolling.apply().')
    ty_checker.check(self, DataFrameRollingType)

    raw_accepted = (Omitted, NoneType, Boolean)
    if not isinstance(raw, raw_accepted) and raw is not None:
        ty_checker.raise_exc(raw, 'bool', 'raw')

    return gen_df_rolling_method_impl('apply', self, args=['func'],
                                      kws={'raw': 'None'})


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

    none_other = isinstance(other, (Omitted, NoneType)) or other is None
    kws = {'other': 'None', 'pairwise': 'None'}

    if none_other:
        return gen_df_rolling_method_other_none_impl('corr', self, kws=kws)

    if isinstance(other, DataFrameType):
        return gen_df_rolling_method_other_df_impl('corr', self, other, kws=kws)

    return gen_df_rolling_method_impl('corr', self, kws=kws)


@sdc_overload_method(DataFrameRollingType, 'count')
def sdc_pandas_dataframe_rolling_count(self):

    ty_checker = TypeChecker('Method rolling.count().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('count', self)


@sdc_overload_method(DataFrameRollingType, 'cov')
def sdc_pandas_dataframe_rolling_cov(self, other=None, pairwise=None, ddof=1):

    ty_checker = TypeChecker('Method rolling.cov().')
    ty_checker.check(self, DataFrameRollingType)

    accepted_other = (Omitted, NoneType, DataFrameType, SeriesType)
    if not isinstance(other, accepted_other) and other is not None:
        ty_checker.raise_exc(other, 'DataFrame, Series', 'other')

    accepted_pairwise = (bool, Boolean, Omitted, NoneType)
    if not isinstance(pairwise, accepted_pairwise) and pairwise is not None:
        ty_checker.raise_exc(pairwise, 'bool', 'pairwise')

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    none_other = isinstance(other, (Omitted, NoneType)) or other is None
    kws = {'other': 'None', 'pairwise': 'None', 'ddof': '1'}

    if none_other:
        # method _df_cov in comparison to method cov doesn't align input data
        # by replacing infinite and matched finite values with nans
        return gen_df_rolling_cov_other_none_impl('_df_cov', self, kws=kws)

    if isinstance(other, DataFrameType):
        return gen_df_rolling_method_other_df_impl('cov', self, other, kws=kws)

    return gen_df_rolling_method_impl('cov', self, kws=kws)


@sdc_overload_method(DataFrameRollingType, 'kurt')
def sdc_pandas_dataframe_rolling_kurt(self):

    ty_checker = TypeChecker('Method rolling.kurt().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('kurt', self)


@sdc_overload_method(DataFrameRollingType, 'max')
def sdc_pandas_dataframe_rolling_max(self):

    ty_checker = TypeChecker('Method rolling.max().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('max', self)


@sdc_overload_method(DataFrameRollingType, 'mean')
def sdc_pandas_dataframe_rolling_mean(self):

    ty_checker = TypeChecker('Method rolling.mean().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('mean', self)


@sdc_overload_method(DataFrameRollingType, 'median')
def sdc_pandas_dataframe_rolling_median(self):

    ty_checker = TypeChecker('Method rolling.median().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('median', self)


@sdc_overload_method(DataFrameRollingType, 'min')
def sdc_pandas_dataframe_rolling_min(self):

    ty_checker = TypeChecker('Method rolling.min().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('min', self)


@sdc_overload_method(DataFrameRollingType, 'quantile')
def sdc_pandas_dataframe_rolling_quantile(self, quantile, interpolation='linear'):

    ty_checker = TypeChecker('Method rolling.quantile().')
    ty_checker.check(self, DataFrameRollingType)

    if not isinstance(quantile, Number):
        ty_checker.raise_exc(quantile, 'float', 'quantile')

    str_types = (Omitted, StringLiteral, UnicodeType)
    if not isinstance(interpolation, str_types) and interpolation != 'linear':
        ty_checker.raise_exc(interpolation, 'str', 'interpolation')

    return gen_df_rolling_method_impl('quantile', self, args=['quantile'],
                                      kws={'interpolation': '"linear"'})


@sdc_overload_method(DataFrameRollingType, 'skew')
def sdc_pandas_dataframe_rolling_skew(self):

    ty_checker = TypeChecker('Method rolling.skew().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('skew', self)


@sdc_overload_method(DataFrameRollingType, 'std')
def sdc_pandas_dataframe_rolling_std(self, ddof=1):

    ty_checker = TypeChecker('Method rolling.std().')
    ty_checker.check(self, DataFrameRollingType)

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    return gen_df_rolling_method_impl('std', self, kws={'ddof': '1'})


@sdc_overload_method(DataFrameRollingType, 'sum')
def sdc_pandas_dataframe_rolling_sum(self):

    ty_checker = TypeChecker('Method rolling.sum().')
    ty_checker.check(self, DataFrameRollingType)

    return gen_df_rolling_method_impl('sum', self)


@sdc_overload_method(DataFrameRollingType, 'var')
def sdc_pandas_dataframe_rolling_var(self, ddof=1):

    ty_checker = TypeChecker('Method rolling.var().')
    ty_checker.check(self, DataFrameRollingType)

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    return gen_df_rolling_method_impl('var', self, kws={'ddof': '1'})


sdc_pandas_dataframe_rolling_apply.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'apply',
    'example_caption': 'Calculate the rolling apply.',
    'limitations_block':
    """
    Limitations
    -----------
    - This function may reveal slower performance than Pandas* on user system. Users should exercise a tradeoff
    between staying in JIT-region with that function or going back to interpreter mode.
    - Supported ``raw`` only can be `None` or `True`. Parameters ``args``, ``kwargs`` unsupported.
    - DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    func:
        A single value producer
    raw: :obj:`bool`
        False : passes each row or column as a Series to the function.
        True or None : the passed function will receive ndarray objects instead.
    """
})

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

sdc_pandas_dataframe_rolling_count.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'count',
    'example_caption': 'Count of any non-NaN observations inside the window.',
    'limitations_block': '',
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_cov.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'cov',
    'example_caption': 'Calculate rolling covariance.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    Different size of `self` and `other` can produce result different from the result of Pandas
    due to different float rounding in Python and SDC.
    """,
    'extra_params':
    """
    other: :obj:`Series` or :obj:`DataFrame`
        Other Series or DataFrame.
    pairwise: :obj:`bool`
        Calculate pairwise combinations of columns within a DataFrame.
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})

sdc_pandas_dataframe_rolling_kurt.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'kurt',
    'example_caption': 'Calculate unbiased rolling kurtosis.',
    'limitations_block': '',
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_max.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'max',
    'example_caption': 'Calculate the rolling maximum.',
    'limitations_block': '',
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_mean.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'mean',
    'example_caption': 'Calculate the rolling mean of the values.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_median.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'median',
    'example_caption': 'Calculate the rolling median.',
    'limitations_block':
    """
    Limitations
    -----------
    This function may reveal slower performance than Pandas* on user system. Users should exercise a tradeoff
    between staying in JIT-region with that function or going back to interpreter mode.
    """,
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_min.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'min',
    'example_caption': 'Calculate the rolling minimum.',
    'limitations_block': '',
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_quantile.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'quantile',
    'example_caption': 'Calculate the rolling quantile.',
    'limitations_block':
    """
    Limitations
    -----------
    - This function may reveal slower performance than Pandas* on user system. Users should exercise a tradeoff
    between staying in JIT-region with that function or going back to interpreter mode.
    - Supported ``interpolation`` only can be `'linear'`.
    - DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    quantile: :obj:`float`
        Quantile to compute. 0 <= quantile <= 1.
    interpolation: :obj:`str`
        This optional parameter specifies the interpolation method to use.
    """
})

sdc_pandas_dataframe_rolling_skew.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'skew',
    'example_caption': 'Unbiased rolling skewness.',
    'limitations_block': '',
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_std.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'std',
    'example_caption': 'Calculate rolling standard deviation.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})

sdc_pandas_dataframe_rolling_sum.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'sum',
    'example_caption': 'Calculate rolling sum of given Series.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_var.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'var',
    'example_caption': 'Calculate unbiased rolling variance.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})

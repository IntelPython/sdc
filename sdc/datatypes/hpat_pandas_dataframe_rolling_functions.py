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
import pandas

from sdc.datatypes.common_functions import TypeChecker
from sdc.datatypes.hpat_pandas_dataframe_rolling_types import DataFrameRollingType
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
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


def df_rolling_method_codegen(method_name, self, args=None, kws=None):
    args = args or []
    kwargs = kws or {}

    rolling_params = ['window', 'min_periods', 'center',
                      'win_type', 'on', 'axis', 'closed']
    rolling_params_as_str = ', '.join(f'self._{p}' for p in rolling_params)
    method_params = args + ['{}={}'.format(k, k) for k in kwargs]
    method_params_as_str = ', '.join(method_params)

    impl_params = ['self'] + args + ['{}={}'.format(k, v) for k, v in kwargs.items()]
    impl_params_as_str = ', '.join(impl_params)

    results = []
    impl_name = f'_df_rolling_{method_name}_impl'
    func_lines = [f'def {impl_name}({impl_params_as_str}):']

    for idx, col in enumerate(self.data.columns):
        res_data = f'result_data_{col}'
        func_lines += [
            f'  data_{col} = get_dataframe_data(self._data, {idx})',
            f'  series_{col} = pandas.Series(data_{col})',
            f'  rolling_{col} = series_{col}.rolling({rolling_params_as_str})',
            f'  result_{col} = rolling_{col}.{method_name}({method_params_as_str})',
            f'  {res_data} = result_{col}._data[:len(data_{col})]'
        ]
        results.append((col, res_data))

    data = ', '.join(f'"{col}": {data}' for col, data in results)
    func_lines += [f'  return pandas.DataFrame({{{data}}})']
    func_text = '\n'.join(func_lines)

    global_vars = {'pandas': pandas, 'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


def gen_df_rolling_method_impl(method_name, self, args=None, kws=None):
    func_text, global_vars = df_rolling_method_codegen(method_name, self,
                                                       args=args, kws=kws)
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _impl = loc_vars[f'_df_rolling_{method_name}_impl']

    return _impl


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


sdc_pandas_dataframe_rolling_median.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'median',
    'example_caption': 'Calculate the rolling median.',
    'limitations_block': '',
    'extra_params': ''
})

sdc_pandas_dataframe_rolling_min.__doc__ = sdc_pandas_dataframe_rolling_docstring_tmpl.format(**{
    'method_name': 'min',
    'example_caption': 'Calculate the rolling minimum.',
    'limitations_block': '',
    'extra_params': ''
})

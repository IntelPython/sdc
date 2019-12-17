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

import numpy
import pandas

from numba import prange
from numba.extending import register_jitable
from numba.types import float64

from sdc.datatypes.common_functions import TypeChecker
from sdc.datatypes.hpat_pandas_series_rolling_types import SeriesRollingType
from sdc.utils import sdc_overload_method


hpat_pandas_series_rolling_docstring_tmpl = """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.{method_name}
{limitations_block}
    Examples
    --------
    .. literalinclude:: ../../../examples/series/rolling/series_rolling_{method_name}.py
       :language: python
       :lines: 27-
       :caption: {example_caption}
       :name: ex_series_rolling_{method_name}

    .. code-block:: console

        > python ./series_rolling_{method_name}.py{example_result}

    .. seealso::
        :ref:`Series.rolling <pandas.Series.rolling>`
            Calling object with a Series.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`Series.{method_name} <pandas.Series.{method_name}>`
            Similar method for Series.
        :ref:`DataFrame.{method_name} <pandas.DataFrame.{method_name}>`
            Similar method for DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.Series.rolling.{method_name}()` implementation.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_series_rolling_{method_name}

    Parameters
    ----------
    self: :class:`pandas.Series.rolling`
        input arg

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
"""


@register_jitable
def arr_max(arr):
    """Calculate maximum of values"""
    if len(arr) == 0:
        return numpy.nan

    return arr.max()


@register_jitable
def arr_mean(arr):
    """Calculate mean of values"""
    if len(arr) == 0:
        return numpy.nan

    return arr.mean()


@register_jitable
def arr_min(arr):
    """Calculate minimum of values"""
    if len(arr) == 0:
        return numpy.nan

    return arr.min()


def gen_hpat_pandas_series_rolling_impl(rolling_func, output_type=None):
    """Generate series rolling methods implementations based on input func"""
    nan_out_type = output_type is None

    def impl(self):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        out_type = input_arr.dtype if nan_out_type == True else output_type  # noqa
        output_arr = numpy.empty(length, dtype=out_type)

        for i in prange(min(win, length)):
            arr_range = input_arr[:i + 1]
            finite_arr = arr_range[numpy.isfinite(arr_range)]
            if len(finite_arr) < minp:
                output_arr[i] = numpy.nan
            else:
                output_arr[i] = rolling_func(finite_arr)

        for i in prange(min(win, length), length):
            arr_range = input_arr[i + 1 - win:i + 1]
            finite_arr = arr_range[numpy.isfinite(arr_range)]
            if len(finite_arr) < minp:
                output_arr[i] = numpy.nan
            else:
                output_arr[i] = rolling_func(finite_arr)

        return pandas.Series(output_arr, input_series._index, name=input_series._name)

    return impl


hpat_pandas_rolling_series_max_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_max, float64))
hpat_pandas_rolling_series_mean_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_mean, float64))
hpat_pandas_rolling_series_min_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_min, float64))


@sdc_overload_method(SeriesRollingType, 'max')
def hpat_pandas_series_rolling_max(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.max

    Examples
    --------
    .. literalinclude:: ../../../examples/series/rolling/series_rolling_max.py
       :language: python
       :lines: 27-
       :caption: Calculate the rolling maximum.
       :name: ex_series_rolling_max

    .. code-block:: console

        > python ./series_rolling_max.py
        0    NaN
        1    NaN
        2    5.0
        3    5.0
        4    6.0
        dtype: float64

    .. seealso::
        :ref:`Series.rolling <pandas.Series.rolling>`
            Calling object with a Series.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`Series.max <pandas.Series.max>`
            Similar method for Series.
        :ref:`DataFrame.max <pandas.DataFrame.max>`
            Similar method for DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.Series.rolling.max()` implementation.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_series_rolling_max

    Parameters
    ----------
    self: :class:`pandas.Series.rolling`
        input arg

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method max().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_max_impl


@sdc_overload_method(SeriesRollingType, 'mean')
def hpat_pandas_series_rolling_mean(self):

    ty_checker = TypeChecker('Method mean().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_mean_impl


@sdc_overload_method(SeriesRollingType, 'min')
def hpat_pandas_series_rolling_min(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.min

    Examples
    --------
    .. literalinclude:: ../../../examples/series/rolling/series_rolling_min.py
       :language: python
       :lines: 27-
       :caption: Calculate the rolling minimum.
       :name: ex_series_rolling_min

    .. code-block:: console

        > python ./series_rolling_min.py
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        dtype: float64

    .. seealso::
        :ref:`Series.rolling <pandas.Series.rolling>`
            Calling object with a Series.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`Series.min <pandas.Series.min>`
            Similar method for Series.
        :ref:`DataFrame.min <pandas.DataFrame.min>`
            Similar method for DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.Series.rolling.min()` implementation.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_series_rolling_min

    Parameters
    ----------
    self: :class:`pandas.Series.rolling`
        input arg

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method min().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_min_impl


hpat_pandas_series_rolling_mean.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'mean',
    'example_caption': 'Calculate the rolling mean of the values.',
    'example_result':
    """
        0         NaN
        1         NaN
        2    4.000000
        3    3.333333
        4    4.333333
        dtype: float64
    """,
    'limitations_block':
    """
    Limitations
    -----------
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """
})

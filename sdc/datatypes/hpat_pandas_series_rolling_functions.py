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
from numba.types import float64, Number, Omitted, StringLiteral, UnicodeType

from sdc.datatypes.common_functions import TypeChecker
from sdc.datatypes.hpat_pandas_series_rolling_types import SeriesRollingType
from sdc.utils import sdc_overload_method


@register_jitable
def arr_nonnan_count(arr):
    """Count non-NaN values"""
    return len(arr) - numpy.isnan(arr).sum()


@register_jitable
def arr_max(arr):
    """Calculate maximum of values"""
    if len(arr) == 0:
        return numpy.nan

    return arr.max()


@register_jitable
def arr_min(arr):
    """Calculate minimum of values"""
    if len(arr) == 0:
        return numpy.nan

    return arr.min()


@register_jitable
def arr_quantile(arr, q):
    """Calculate quantile of values"""
    if len(arr) == 0:
        return numpy.nan

    return numpy.quantile(arr, q)


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


def gen_hpat_pandas_series_rolling_zerominp_impl(rolling_func, output_type=None):
    """Generate series rolling methods implementations with zero min_periods"""
    nan_out_type = output_type is None

    def impl(self):
        win = self._window

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        out_type = input_arr.dtype if nan_out_type == True else output_type  # noqa
        output_arr = numpy.empty(length, dtype=out_type)

        boundary = min(win, length)
        for i in prange(boundary):
            arr_range = input_arr[:i + 1]
            output_arr[i] = rolling_func(arr_range)

        for i in prange(boundary, length):
            arr_range = input_arr[i + 1 - win:i + 1]
            output_arr[i] = rolling_func(arr_range)

        return pandas.Series(output_arr, input_series._index, name=input_series._name)

    return impl


hpat_pandas_rolling_series_count_impl = register_jitable(
    gen_hpat_pandas_series_rolling_zerominp_impl(arr_nonnan_count, float64))
hpat_pandas_rolling_series_max_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_max, float64))
hpat_pandas_rolling_series_min_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_min, float64))


@sdc_overload_method(SeriesRollingType, 'count')
def hpat_pandas_series_rolling_count(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.count

    Examples
    --------
    .. literalinclude:: ../../../examples/series/rolling/series_rolling_count.py
       :language: python
       :lines: 27-
       :caption: Count of any non-NaN observations inside the window.
       :name: ex_series_rolling_count

    .. code-block:: console

        > python ./series_rolling_count.py
        0    1.0
        1    2.0
        2    3.0
        3    2.0
        4    2.0
        dtype: float64

    .. seealso::
        :ref:`Series.rolling <pandas.Series.rolling>`
            Calling object with a Series.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`Series.count <pandas.Series.count>`
            Similar method for Series.
        :ref:`DataFrame.count <pandas.DataFrame.count>`
            Similar method for DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.Series.rolling.count()` implementation.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_series_rolling_count

    Parameters
    ----------
    self: :class:`pandas.Series.rolling`
        input arg

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method rolling.count().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_count_impl


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

    ty_checker = TypeChecker('Method rolling.max().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_max_impl


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

    ty_checker = TypeChecker('Method rolling.min().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_min_impl


@sdc_overload_method(SeriesRollingType, 'quantile')
def hpat_pandas_series_rolling_quantile(self, quantile, interpolation='linear'):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.quantile

    Limitations
    -----------
    Supported ``interpolation`` only can be `'linear'`.
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/rolling/series_rolling_quantile.py
       :language: python
       :lines: 27-
       :caption: Calculate the rolling quantile.
       :name: ex_series_rolling_quantile

    .. code-block:: console

        > python ./series_rolling_quantile.py
        0    NaN
        1    NaN
        2    4.0
        3    3.0
        4    5.0
        dtype: float64

    .. seealso::
        :ref:`Series.rolling <pandas.Series.rolling>`
            Calling object with a Series.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`Series.quantile <pandas.Series.quantile>`
            Similar method for Series.
        :ref:`DataFrame.quantile <pandas.DataFrame.quantile>`
            Similar method for DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.Series.rolling.quantile()` implementation.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_series_rolling_quantile

    Parameters
    ----------
    self: :class:`pandas.Series.rolling`
        input arg
    quantile: :obj:`float`
        Quantile to compute. 0 <= quantile <= 1.
    interpolation: :obj:`str`
        This optional parameter specifies the interpolation method to use.

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method rolling.quantile().')
    ty_checker.check(self, SeriesRollingType)

    if not isinstance(quantile, Number):
        ty_checker.raise_exc(quantile, 'float', 'quantile')

    str_types = (Omitted, StringLiteral, UnicodeType)
    if not isinstance(interpolation, str_types) and interpolation != 'linear':
        ty_checker.raise_exc(interpolation, 'str', 'interpolation')

    def hpat_pandas_rolling_series_quantile_impl(self, quantile, interpolation='linear'):
        if quantile < 0 or quantile > 1:
            raise ValueError('quantile value not in [0, 1]')
        if interpolation != 'linear':
            raise ValueError('interpolation value not "linear"')

        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

        def culc_quantile(arr, quantile, minp):
            finite_arr = arr[numpy.isfinite(arr)]
            if len(finite_arr) < minp:
                return numpy.nan
            else:
                return arr_quantile(finite_arr, quantile)

        boundary = min(win, length)
        for i in prange(boundary):
            arr_range = input_arr[:i + 1]
            output_arr[i] = culc_quantile(arr_range, quantile, minp)

        for i in prange(min(win, length), length):
            arr_range = input_arr[i + 1 - win:i + 1]
            output_arr[i] = culc_quantile(arr_range, quantile, minp)

        return pandas.Series(output_arr, input_series._index, name=input_series._name)

    return hpat_pandas_rolling_series_quantile_impl

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
from numba.types import float64, Boolean, NoneType, Omitted

from sdc.datatypes.common_functions import TypeChecker
from sdc.datatypes.hpat_pandas_series_rolling_types import SeriesRollingType
from sdc.utils import sdc_overload_method


@register_jitable
def arr_corr(x, y):
    """Calculate correlation of values"""
    if len(x) == 0:
        return numpy.nan

    return numpy.corrcoef(x, y)[0, 1]


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
hpat_pandas_rolling_series_min_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_min, float64))


@sdc_overload_method(SeriesRollingType, 'corr')
def hpat_pandas_series_rolling_corr(self, other=None, pairwise=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.corr

    Limitations
    -----------
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    Resulting Series has default index and name.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/rolling/series_rolling_corr.py
       :language: python
       :lines: 27-
       :caption: Calculate rolling correlation.
       :name: ex_series_rolling_corr

    .. code-block:: console

        > python ./series_rolling_corr.py
        0         NaN
        1         NaN
        2         NaN
        3    0.333333
        4    0.916949
        dtype: float64

    .. seealso::
        :ref:`Series.rolling <pandas.Series.rolling>`
            Calling object with a Series.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`Series.corr <pandas.Series.corr>`
            Similar method for Series.
        :ref:`DataFrame.corr <pandas.DataFrame.corr>`
            Similar method for DataFrame.
        :ref:`rolling.cov <pandas.core.window.Rolling.cov>`
            Similar method to calculate covariance.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.Series.rolling.corr()` implementation.

    .. only:: developer
    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_series_rolling_corr

    Parameters
    ----------
    self: :class:`pandas.Series.rolling`
        input arg
    other: :obj:`Series`
        Other Series.
    pairwise: :obj:`bool`
        Not relevant for Series.

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method rolling.corr().')
    ty_checker.check(self, SeriesRollingType)

    # TODO: check `other` is Series after a circular import of SeriesType fixed
    # accepted_other = (bool, Omitted, NoneType, SeriesType)
    # if not isinstance(other, accepted_other) and other is not None:
    #     ty_checker.raise_exc(other, 'Series', 'other')

    accepted_pairwise = (bool, Boolean, Omitted, NoneType)
    if not isinstance(pairwise, accepted_pairwise) and pairwise is not None:
        ty_checker.raise_exc(pairwise, 'bool', 'pairwise')

    nan_other = isinstance(other, (Omitted, NoneType)) or other is None

    def hpat_pandas_rolling_series_std_impl(self, other=None, pairwise=None):
        win = self._window
        minp = self._min_periods

        main_series = self._data
        main_arr = main_series._data
        main_arr_length = len(main_arr)

        if nan_other == True:  # noqa
            other_arr = main_arr
        else:
            other_arr = other._data

        other_arr_length = len(other_arr)
        length = max(main_arr_length, other_arr_length)
        output_arr = numpy.empty(length, dtype=float64)

        def calc_corr(main, other, minp):
            # align arrays `main` and `other` by size and finiteness
            min_length = min(len(main), len(other))
            main_valid_indices = numpy.isfinite(main[:min_length])
            other_valid_indices = numpy.isfinite(other[:min_length])
            valid = main_valid_indices & other_valid_indices

            if len(main[valid]) < minp:
                return numpy.nan
            else:
                return arr_corr(main[valid], other[valid])

        for i in prange(win):
            main_arr_range = main_arr[:i + 1]
            other_arr_range = other_arr[:i + 1]
            output_arr[i] = calc_corr(main_arr_range, other_arr_range, minp)

        for i in prange(win, length):
            main_arr_range = main_arr[i + 1 - win:i + 1]
            other_arr_range = other_arr[i + 1 - win:i + 1]
            output_arr[i] = calc_corr(main_arr_range, other_arr_range, minp)

        return pandas.Series(output_arr)

    return hpat_pandas_rolling_series_std_impl


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

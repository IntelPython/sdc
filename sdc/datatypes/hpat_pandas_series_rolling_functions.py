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
from numba.types import float64, Boolean, Integer, NoneType, Omitted

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
        input arg{extra_params}

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
"""


@register_jitable
def arr_corr(x, y):
    """Calculate correlation of values"""
    if len(x) == 0:
        return numpy.nan

    return numpy.corrcoef(x, y)[0, 1]


@register_jitable
def arr_nonnan_count(arr):
    """Count non-NaN values"""
    return len(arr) - numpy.isnan(arr).sum()


@register_jitable
def arr_cov(x, y, ddof):
    """Calculate covariance of values"""
    if len(x) == 0:
        return numpy.nan

    return numpy.cov(x, y, ddof=ddof)[0, 1]


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
def arr_median(arr):
    """Calculate median of values"""
    if len(arr) == 0:
        return numpy.nan

    return numpy.median(arr)


@register_jitable
def arr_min(arr):
    """Calculate minimum of values"""
    if len(arr) == 0:
        return numpy.nan

    return arr.min()


@register_jitable
def arr_std(arr, ddof):
    """Calculate standard deviation of values"""
    return arr_var(arr, ddof) ** 0.5


@register_jitable
def arr_sum(arr):
    """Calculate sum of values"""
    return arr.sum()


@register_jitable
def arr_var(arr, ddof):
    """Calculate unbiased variance of values"""
    length = len(arr)
    if length in [0, ddof]:
        return numpy.nan

    return numpy.var(arr) * length / (length - ddof)


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

        def apply_minp(arr, minp):
            finite_arr = arr[numpy.isfinite(arr)]
            if len(finite_arr) < minp:
                return numpy.nan
            else:
                return rolling_func(finite_arr)

        boundary = min(win, length)
        for i in prange(boundary):
            arr_range = input_arr[:i + 1]
            output_arr[i] = apply_minp(arr_range, minp)

        for i in prange(boundary, length):
            arr_range = input_arr[i + 1 - win:i + 1]
            output_arr[i] = apply_minp(arr_range, minp)

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
hpat_pandas_rolling_series_mean_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_mean, float64))
hpat_pandas_rolling_series_median_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_median, float64))
hpat_pandas_rolling_series_min_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_min, float64))
hpat_pandas_rolling_series_sum_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_sum, float64))


@sdc_overload_method(SeriesRollingType, 'corr')
def hpat_pandas_series_rolling_corr(self, other=None, pairwise=None):

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

        for i in prange(min(win, length)):
            main_arr_range = main_arr[:i + 1]
            other_arr_range = other_arr[:i + 1]
            output_arr[i] = calc_corr(main_arr_range, other_arr_range, minp)

        for i in prange(win, length):
            main_arr_range = main_arr[i + 1 - win:i + 1]
            other_arr_range = other_arr[i + 1 - win:i + 1]
            output_arr[i] = calc_corr(main_arr_range, other_arr_range, minp)

        return pandas.Series(output_arr)

    return hpat_pandas_rolling_series_std_impl


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


@sdc_overload_method(SeriesRollingType, 'cov')
def hpat_pandas_series_rolling_cov(self, other=None, pairwise=None, ddof=1):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.cov

    Limitations
    -----------
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    Resulting Series has default index and name.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/rolling/series_rolling_cov.py
       :language: python
       :lines: 27-
       :caption: Calculate rolling covariance.
       :name: ex_series_rolling_cov

    .. code-block:: console

        > python ./series_rolling_cov.py
        0         NaN
        1         NaN
        2         NaN
        3    0.166667
        4    4.333333
        dtype: float64

    .. seealso::
        :ref:`Series.rolling <pandas.Series.rolling>`
            Calling object with a Series.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`Series.cov <pandas.Series.cov>`
            Similar method for Series.
        :ref:`DataFrame.cov <pandas.DataFrame.cov>`
            Similar method for DataFrame.
        :ref:`rolling.cov <pandas.core.window.Rolling.cov>`
            Similar method to calculate covariance.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.Series.rolling.cov()` implementation.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_series_rolling_cov

    Parameters
    ----------
    self: :class:`pandas.Series.rolling`
        input arg
    other: :obj:`Series`
        Other Series.
    pairwise: :obj:`bool`
        Not relevant for Series.
    ddof: :obj:`int`
        Delta Degrees of Freedom.

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method rolling.cov().')
    ty_checker.check(self, SeriesRollingType)

    # TODO: check `other` is Series after a circular import of SeriesType fixed
    # accepted_other = (bool, Omitted, NoneType, SeriesType)
    # if not isinstance(other, accepted_other) and other is not None:
    #     ty_checker.raise_exc(other, 'Series', 'other')

    accepted_pairwise = (bool, Boolean, Omitted, NoneType)
    if not isinstance(pairwise, accepted_pairwise) and pairwise is not None:
        ty_checker.raise_exc(pairwise, 'bool', 'pairwise')

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    nan_other = isinstance(other, (Omitted, NoneType)) or other is None

    def hpat_pandas_rolling_series_std_impl(self, other=None, pairwise=None, ddof=1):
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

        def calc_cov(main, other, ddof, minp):
            # align arrays `main` and `other` by size and finiteness
            min_length = min(len(main), len(other))
            main_valid_indices = numpy.isfinite(main[:min_length])
            other_valid_indices = numpy.isfinite(other[:min_length])
            valid = main_valid_indices & other_valid_indices

            if len(main[valid]) < minp:
                return numpy.nan
            else:
                return arr_cov(main[valid], other[valid], ddof)

        for i in prange(min(win, length)):
            main_arr_range = main_arr[:i + 1]
            other_arr_range = other_arr[:i + 1]
            output_arr[i] = calc_cov(main_arr_range, other_arr_range, ddof, minp)

        for i in prange(win, length):
            main_arr_range = main_arr[i + 1 - win:i + 1]
            other_arr_range = other_arr[i + 1 - win:i + 1]
            output_arr[i] = calc_cov(main_arr_range, other_arr_range, ddof, minp)

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

    ty_checker = TypeChecker('Method rolling.max().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_max_impl


@sdc_overload_method(SeriesRollingType, 'mean')
def hpat_pandas_series_rolling_mean(self):

    ty_checker = TypeChecker('Method rolling.mean().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_mean_impl


@sdc_overload_method(SeriesRollingType, 'median')
def hpat_pandas_series_rolling_median(self):

    ty_checker = TypeChecker('Method rolling.median().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_median_impl


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


@sdc_overload_method(SeriesRollingType, 'std')
def hpat_pandas_series_rolling_std(self, ddof=1):

    ty_checker = TypeChecker('Method rolling.std().')
    ty_checker.check(self, SeriesRollingType)

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    def hpat_pandas_rolling_series_std_impl(self, ddof=1):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

        def culc_std(arr, ddof, minp):
            finite_arr = arr[numpy.isfinite(arr)]
            if len(finite_arr) < minp:
                return numpy.nan
            else:
                return arr_std(finite_arr, ddof)

        boundary = min(win, length)
        for i in prange(boundary):
            arr_range = input_arr[:i + 1]
            output_arr[i] = culc_std(arr_range, ddof, minp)

        for i in prange(min(win, length), length):
            arr_range = input_arr[i + 1 - win:i + 1]
            output_arr[i] = culc_std(arr_range, ddof, minp)

        return pandas.Series(output_arr, input_series._index, name=input_series._name)

    return hpat_pandas_rolling_series_std_impl


@sdc_overload_method(SeriesRollingType, 'sum')
def hpat_pandas_series_rolling_sum(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.core.window.Rolling.sum

    Limitations
    -----------
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/rolling/series_rolling_sum.py
       :language: python
       :lines: 27-
       :caption: Calculate rolling sum of given Series.
       :name: ex_series_rolling_sum

    .. code-block:: console

        > python ./series_rolling_sum.py
        0     NaN
        1     NaN
        2    12.0
        3    10.0
        4    13.0
        dtype: float64

    .. seealso::
        :ref:`Series.rolling <pandas.Series.rolling>`
            Calling object with a Series.
        :ref:`DataFrame.rolling <pandas.DataFrame.rolling>`
            Calling object with a DataFrame.
        :ref:`Series.sum <pandas.Series.sum>`
            Similar method for Series.
        :ref:`DataFrame.sum <pandas.DataFrame.sum>`
            Similar method for DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.Series.rolling.sum()` implementation.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_rolling.TestRolling.test_series_rolling_sum

    Parameters
    ----------
    self: :class:`pandas.Series.rolling`
        input arg

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method rolling.sum().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_sum_impl


@sdc_overload_method(SeriesRollingType, 'var')
def hpat_pandas_series_rolling_var(self, ddof=1):

    ty_checker = TypeChecker('Method rolling.var().')
    ty_checker.check(self, SeriesRollingType)

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    def hpat_pandas_rolling_series_var_impl(self, ddof=1):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

        def culc_var(arr, ddof, minp):
            finite_arr = arr[numpy.isfinite(arr)]
            if len(finite_arr) < minp:
                return numpy.nan
            else:
                return arr_var(finite_arr, ddof)

        boundary = min(win, length)
        for i in prange(boundary):
            arr_range = input_arr[:i + 1]
            output_arr[i] = culc_var(arr_range, ddof, minp)

        for i in prange(boundary, length):
            arr_range = input_arr[i + 1 - win:i + 1]
            output_arr[i] = culc_var(arr_range, ddof, minp)

        return pandas.Series(output_arr, input_series._index, name=input_series._name)

    return hpat_pandas_rolling_series_var_impl


hpat_pandas_series_rolling_corr.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'corr',
    'example_caption': 'Calculate rolling correlation.',
    'example_result':
    """
        0         NaN
        1         NaN
        2         NaN
        3    0.333333
        4    0.916949
        dtype: float64
    """,
    'limitations_block':
    """
    Limitations
    -----------
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    Resulting Series has default index and name.
    """,
    'extra_params':
    """
    other: :obj:`Series`
        Other Series.
    pairwise: :obj:`bool`
        Not relevant for Series.
    """
})

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
    """,
    'extra_params': ''
})

hpat_pandas_series_rolling_median.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'median',
    'example_caption': 'Calculate the rolling median.',
    'example_result':
    """
        0    NaN
        1    NaN
        2    4.0
        3    3.0
        4    5.0
        dtype: float64
    """,
    'limitations_block': '',
    'extra_params': ''
})

hpat_pandas_series_rolling_std.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'std',
    'example_caption': 'Calculate rolling standard deviation.',
    'example_result':
    """
        0         NaN
        1         NaN
        2    1.000000
        3    1.527525
        4    2.081666
        dtype: float64
    """,
    'limitations_block':
    """
    Limitations
    -----------
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})

hpat_pandas_series_rolling_var.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'var',
    'example_caption': 'Calculate unbiased rolling variance.',
    'example_result':
    """
        0         NaN
        1         NaN
        2    1.000000
        3    2.333333
        4    4.333333
        dtype: float64
    """,
    'limitations_block':
    """
    Limitations
    -----------
    Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})

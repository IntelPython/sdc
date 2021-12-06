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

from functools import partial

from numba import prange
from numba.extending import register_jitable
from numba.core.types import (float64, Boolean, Integer, NoneType, Number,
                         Omitted, StringLiteral, UnicodeType)

from sdc.datatypes.common_functions import _almost_equal
from sdc.datatypes.hpat_pandas_series_rolling_types import SeriesRollingType
from sdc.functions.statistics import skew_formula
from sdc.hiframes.pd_series_type import SeriesType
from sdc.utilities.prange_utils import parallel_chunks
from sdc.utilities.sdc_typing_utils import TypeChecker
from sdc.utilities.utils import sdc_overload_method, sdc_register_jitable


# disabling parallel execution for rolling due to numba issue https://github.com/numba/numba/issues/5098
sdc_rolling_overload = partial(sdc_overload_method, parallel=False)


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

    .. command-output:: python ./series/rolling/series_rolling_{method_name}.py
       :cwd: ../../../examples

    .. literalinclude:: ../../../examples/dataframe/rolling/dataframe_rolling_{method_name}.py
       :language: python
       :lines: 27-
       :caption: {example_caption}
       :name: ex_dataframe_rolling_{method_name}

    .. command-output:: python ./dataframe/rolling/dataframe_rolling_{method_name}.py
       :cwd: ../../../examples

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


@sdc_register_jitable
def arr_apply(arr, func):
    """Apply function for values"""
    return func(arr)


@sdc_register_jitable
def arr_mean(arr):
    """Calculate mean of values"""
    if len(arr) == 0:
        return numpy.nan

    return arr.mean()


@sdc_register_jitable
def arr_median(arr):
    """Calculate median of values"""
    if len(arr) == 0:
        return numpy.nan

    return numpy.median(arr)


@sdc_register_jitable
def arr_quantile(arr, q):
    """Calculate quantile of values"""
    if len(arr) == 0:
        return numpy.nan

    return numpy.quantile(arr, q)


def gen_hpat_pandas_series_rolling_impl(rolling_func):
    """Generate series rolling methods implementations based on input func"""
    def impl(self):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

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


def gen_hpat_pandas_series_rolling_ddof_impl(rolling_func):
    """Generate series rolling methods implementations with parameter ddof"""
    def impl(self, ddof=1):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

        def apply_minp(arr, ddof, minp):
            finite_arr = arr[numpy.isfinite(arr)]
            if len(finite_arr) < minp:
                return numpy.nan
            else:
                return rolling_func(finite_arr, ddof)

        boundary = min(win, length)
        for i in prange(boundary):
            arr_range = input_arr[:i + 1]
            output_arr[i] = apply_minp(arr_range, ddof, minp)

        for i in prange(boundary, length):
            arr_range = input_arr[i + 1 - win:i + 1]
            output_arr[i] = apply_minp(arr_range, ddof, minp)

        return pandas.Series(output_arr, input_series._index, name=input_series._name)

    return impl


hpat_pandas_rolling_series_median_impl = register_jitable(
    gen_hpat_pandas_series_rolling_impl(arr_median))


@sdc_register_jitable
def pop_corr(x, y, nfinite, result):
    """Calculate the window sums for corr without old value."""
    sum_x, sum_y, sum_xy, sum_xx, sum_yy = result
    if numpy.isfinite(x) and numpy.isfinite(y):
        nfinite -= 1
        sum_x -= x
        sum_y -= y
        sum_xy -= x * y
        sum_xx -= x * x
        sum_yy -= y * y

    return nfinite, (sum_x, sum_y, sum_xy, sum_xx, sum_yy)


@sdc_register_jitable
def put_corr(x, y, nfinite, result):
    """Calculate the window sums for corr with new value."""
    sum_x, sum_y, sum_xy, sum_xx, sum_yy = result
    if numpy.isfinite(x) and numpy.isfinite(y):
        nfinite += 1
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_xx += x * x
        sum_yy += y * y

    return nfinite, (sum_x, sum_y, sum_xy, sum_xx, sum_yy)


@sdc_register_jitable
def pop_count(value, counter, result):
    """Calculate the window count without old value."""
    if numpy.isnan(value):
        return counter, result

    return counter, result - 1


@sdc_register_jitable
def put_count(value, counter, result):
    """Calculate the window count with new value."""
    # rolling.count() fills front of the resulting array by nans according to min_periods using counter
    # Ex: s = pd.Series([1, 2, np.nan]) ->  s.rolling(window=3, min_periods=2).count() -> pd.Series([np.nan, 2.0, 2.0])
    counter += 1
    if numpy.isnan(value):
        return counter, result

    return counter, result + 1


@sdc_register_jitable
def result(nfinite, minp, result):
    """Get result."""
    return result


@sdc_register_jitable
def pop_cov(x, y, nfinite, result, align_finiteness=False):
    """Calculate the window sums for cov without old value."""
    sum_x, sum_y, sum_xy, count = result
    if numpy.isfinite(x) and numpy.isfinite(y):
        nfinite -= 1
        sum_x -= x
        sum_y -= y
        sum_xy -= x * y

        return nfinite, (sum_x, sum_y, sum_xy, count - 1)

    if align_finiteness or numpy.isnan(x) or numpy.isnan(y):
        # alignment by finiteness means replacement of all the infinite values with nans
        # count should NOT be recalculated in case of nans
        return nfinite, result

    return nfinite, (sum_x, sum_y, sum_xy, count - 1)


@sdc_register_jitable
def put_cov(x, y, nfinite, result, align_finiteness=False):
    """Calculate the window sums for cov with new value."""
    sum_x, sum_y, sum_xy, count = result
    if numpy.isfinite(x) and numpy.isfinite(y):
        nfinite += 1
        sum_x += x
        sum_y += y
        sum_xy += x * y

        return nfinite, (sum_x, sum_y, sum_xy, count + 1)

    if align_finiteness or numpy.isnan(x) or numpy.isnan(y):
        # alignment by finiteness means replacement of all the infinite values with nans
        # count should NOT be recalculated in case of nans
        return nfinite, result

    return nfinite, (sum_x, sum_y, sum_xy, count + 1)


@sdc_register_jitable
def put_kurt(value, nfinite, result):
    """Calculate the window sums for kurt with new value."""
    _sum, square_sum, cube_sum, fourth_degree_sum = result
    if numpy.isfinite(value):
        nfinite += 1
        _sum += value
        square_sum += value * value
        cube_sum += value * value * value
        fourth_degree_sum += value * value * value * value

    return nfinite, (_sum, square_sum, cube_sum, fourth_degree_sum)


@sdc_register_jitable
def pop_kurt(value, nfinite, result):
    """Calculate the window sums for kurt without old value."""
    _sum, square_sum, cube_sum, fourth_degree_sum = result
    if numpy.isfinite(value):
        nfinite -= 1
        _sum -= value
        square_sum -= value * value
        cube_sum -= value * value * value
        fourth_degree_sum -= value * value * value * value

    return nfinite, (_sum, square_sum, cube_sum, fourth_degree_sum)


@sdc_register_jitable
def calc_max(arr, idx, win_size):
    """Recalculate the window max based on data, index and window size."""
    start = max(0, idx - win_size + 1)
    nfinite = 0
    result = numpy.nan
    for i in range(start, idx + 1):
        value = arr[i]
        nfinite, result = put_max(value, nfinite, result)

    return nfinite, result


@sdc_register_jitable
def pop_max(value, nfinite, result, arr, idx, win_size):
    """Calculate the window max without old value."""
    if numpy.isfinite(value):
        nfinite -= 1
        if nfinite:
            if value == result:
                return calc_max(arr, idx, win_size)
        else:
            result = numpy.nan

    return nfinite, result


@sdc_register_jitable
def put_max(value, nfinite, result):
    """Calculate the window max with new value."""
    if numpy.isfinite(value):
        nfinite += 1
        if numpy.isnan(result) or value > result:
            result = value

    return nfinite, result


@sdc_register_jitable
def calc_min(arr, idx, win_size):
    """Recalculate the window min based on data, index and window size."""
    start = max(0, idx - win_size + 1)
    nfinite = 0
    result = numpy.nan
    for i in range(start, idx + 1):
        value = arr[i]
        nfinite, result = put_min(value, nfinite, result)

    return nfinite, result


@sdc_register_jitable
def pop_min(value, nfinite, result, arr, idx, win_size):
    """Calculate the window min without old value."""
    if numpy.isfinite(value):
        nfinite -= 1
        if nfinite:
            if value == result:
                return calc_min(arr, idx, win_size)
        else:
            result = numpy.nan

    return nfinite, result


@sdc_register_jitable
def put_min(value, nfinite, result):
    """Calculate the window min with new value."""
    if numpy.isfinite(value):
        nfinite += 1
        if numpy.isnan(result) or value < result:
            result = value

    return nfinite, result


@sdc_register_jitable
def put_skew(value, nfinite, result):
    """Calculate the window sums for skew with new value."""
    _sum, square_sum, cube_sum = result
    if numpy.isfinite(value):
        nfinite += 1
        _sum += value
        square_sum += value * value
        cube_sum += value * value * value

    return nfinite, (_sum, square_sum, cube_sum)


@sdc_register_jitable
def pop_skew(value, nfinite, result):
    """Calculate the window sums for skew without old value."""
    _sum, square_sum, cube_sum = result
    if numpy.isfinite(value):
        nfinite -= 1
        _sum -= value
        square_sum -= value * value
        cube_sum -= value * value * value

    return nfinite, (_sum, square_sum, cube_sum)


@sdc_register_jitable
def pop_sum(value, nfinite, result):
    """Calculate the window sum without old value."""
    if numpy.isfinite(value):
        nfinite -= 1
        result -= value

    return nfinite, result


@sdc_register_jitable
def put_sum(value, nfinite, result):
    """Calculate the window sum with new value."""
    if numpy.isfinite(value):
        nfinite += 1
        result += value

    return nfinite, result


@sdc_register_jitable
def pop_sum2(value, nfinite, result):
    """Calculate the window sums of first/second degree without old value."""
    _sum, square_sum = result
    if numpy.isfinite(value):
        nfinite -= 1
        _sum -= value
        square_sum -= value * value

    return nfinite, (_sum, square_sum)


@sdc_register_jitable
def put_sum2(value, nfinite, result):
    """Calculate the window sums of first/second degree with new value."""
    _sum, square_sum = result
    if numpy.isfinite(value):
        nfinite += 1
        _sum += value
        square_sum += value * value

    return nfinite, (_sum, square_sum)


@sdc_register_jitable
def result_or_nan(nfinite, minp, result):
    """Get result taking into account min periods."""
    if nfinite < minp:
        return numpy.nan

    return result


@sdc_register_jitable
def corr_result_or_nan(nfinite, minp, result):
    """Get result corr taking into account min periods."""
    if nfinite < max(1, minp):
        return numpy.nan

    sum_x, sum_y, sum_xy, sum_xx, sum_yy = result

    var_x = sum_xx - sum_x * sum_x / nfinite
    if _almost_equal(var_x, 0.):
        return numpy.nan

    var_y = sum_yy - sum_y * sum_y / nfinite
    if _almost_equal(var_y, 0.):
        return numpy.nan

    cov_xy = sum_xy - sum_x * sum_y / nfinite

    return cov_xy / numpy.sqrt(var_x * var_y)


@sdc_register_jitable
def cov_result_or_nan(nfinite, minp, result, ddof):
    """Get result of covariance taking into account min periods."""
    if nfinite < max(1, minp):
        return numpy.nan

    sum_x, sum_y, sum_xy, count = result
    res = (sum_xy - sum_x * sum_y / nfinite) / nfinite

    return ddof_result(count, minp, res, ddof)


@sdc_register_jitable
def kurt_result_or_nan(nfinite, minp, result):
    """Get result kurt taking into account min periods."""
    if nfinite < max(4, minp):
        return numpy.nan

    _sum, square_sum, cube_sum, fourth_degree_sum = result

    n = nfinite
    m2 = (square_sum - _sum * _sum / n) / n
    m4 = (fourth_degree_sum - 4*_sum*cube_sum/n + 6*_sum*_sum*square_sum/n/n - 3*_sum*_sum*_sum*_sum/n/n/n) / n
    res = 0 if m2 == 0 else m4 / m2 ** 2.0

    if (n > 2) & (m2 > 0):
        res = 1.0/(n-2)/(n-3) * ((n**2-1.0)*m4/m2**2.0 - 3*(n-1)**2.0)

    return res


@sdc_register_jitable
def mean_result_or_nan(nfinite, minp, result):
    """Get result mean taking into account min periods."""
    if nfinite == 0 or nfinite < minp:
        return numpy.nan

    return result / nfinite


@sdc_register_jitable
def skew_result_or_nan(nfinite, minp, result):
    """Get result skew taking into account min periods."""
    if nfinite < max(3, minp):
        return numpy.nan

    _sum, square_sum, cube_sum = result

    return skew_formula(nfinite, _sum, square_sum, cube_sum)


@sdc_register_jitable
def ddof_result(nfinite, minp, result, ddof):
    """Get result taking into account ddof."""
    if nfinite - ddof < 1:
        return numpy.nan

    return result * nfinite / (nfinite - ddof)


@sdc_register_jitable
def var_result_or_nan(nfinite, minp, result, ddof):
    """Get result var taking into account min periods."""
    if nfinite < max(1, minp):
        return numpy.nan

    _sum, square_sum = result
    res = (square_sum - _sum * _sum / nfinite) / nfinite

    return ddof_result(nfinite, minp, res, ddof)


@sdc_register_jitable
def std_result_or_nan(nfinite, minp, result, ddof):
    """Get result std taking into account min periods."""
    return var_result_or_nan(nfinite, minp, result, ddof) ** 0.5


def gen_sdc_pandas_series_rolling_impl(pop, put, get_result=result_or_nan,
                                       init_result=numpy.nan):
    """Generate series rolling methods implementations based on pop/put funcs"""
    def impl(self):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

        chunks = parallel_chunks(length)
        for i in prange(len(chunks)):
            chunk = chunks[i]
            nfinite = 0
            result = init_result

            if win == 0:
                for idx in range(chunk.start, chunk.stop):
                    output_arr[idx] = get_result(nfinite, minp, result)
                continue

            prelude_start = max(0, chunk.start - win + 1)
            prelude_stop = chunk.start

            interlude_start = prelude_stop
            interlude_stop = min(prelude_start + win, chunk.stop)

            for idx in range(prelude_start, prelude_stop):
                value = input_arr[idx]
                nfinite, result = put(value, nfinite, result)

            for idx in range(interlude_start, interlude_stop):
                value = input_arr[idx]
                nfinite, result = put(value, nfinite, result)
                output_arr[idx] = get_result(nfinite, minp, result)

            for idx in range(interlude_stop, chunk.stop):
                put_value = input_arr[idx]
                pop_value = input_arr[idx - win]
                nfinite, result = put(put_value, nfinite, result)
                nfinite, result = pop(pop_value, nfinite, result)
                output_arr[idx] = get_result(nfinite, minp, result)

        return pandas.Series(output_arr, input_series._index,
                             name=input_series._name)
    return impl


def gen_sdc_pandas_series_rolling_minmax_impl(pop, put, init_result=numpy.nan):
    """Generate series rolling min/max implementations based on pop/put funcs"""
    def impl(self):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

        chunks = parallel_chunks(length)
        for i in prange(len(chunks)):
            chunk = chunks[i]
            nfinite = 0
            result = init_result

            if win == 0:
                for idx in range(chunk.start, chunk.stop):
                    output_arr[idx] = result_or_nan(nfinite, minp, result)
                continue

            prelude_start = max(0, chunk.start - win + 1)
            prelude_stop = chunk.start

            interlude_start = prelude_stop
            interlude_stop = min(prelude_start + win, chunk.stop)

            for idx in range(prelude_start, prelude_stop):
                value = input_arr[idx]
                nfinite, result = put(value, nfinite, result)

            for idx in range(interlude_start, interlude_stop):
                value = input_arr[idx]
                nfinite, result = put(value, nfinite, result)
                output_arr[idx] = result_or_nan(nfinite, minp, result)

            for idx in range(interlude_stop, chunk.stop):
                put_value = input_arr[idx]
                pop_value = input_arr[idx - win]
                nfinite, result = put(put_value, nfinite, result)
                nfinite, result = pop(pop_value, nfinite, result,
                                      input_arr, idx, win)
                output_arr[idx] = result_or_nan(nfinite, minp, result)

        return pandas.Series(output_arr, input_series._index,
                             name=input_series._name)
    return impl


def gen_sdc_pandas_series_rolling_ddof_impl(pop, put, get_result=ddof_result,
                                            init_result=numpy.nan):
    """Generate series rolling ddof implementations based on pop/put funcs"""
    def impl(self, ddof=1):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

        chunks = parallel_chunks(length)
        for i in prange(len(chunks)):
            chunk = chunks[i]
            nfinite = 0
            result = init_result

            if win == 0:
                for idx in range(chunk.start, chunk.stop):
                    output_arr[idx] = get_result(nfinite, minp, result, ddof)
                continue

            prelude_start = max(0, chunk.start - win + 1)
            prelude_stop = chunk.start

            interlude_start = prelude_stop
            interlude_stop = min(prelude_start + win, chunk.stop)

            for idx in range(prelude_start, prelude_stop):
                value = input_arr[idx]
                nfinite, result = put(value, nfinite, result)

            for idx in range(interlude_start, interlude_stop):
                value = input_arr[idx]
                nfinite, result = put(value, nfinite, result)
                output_arr[idx] = get_result(nfinite, minp, result, ddof)

            for idx in range(interlude_stop, chunk.stop):
                put_value = input_arr[idx]
                pop_value = input_arr[idx - win]
                nfinite, result = put(put_value, nfinite, result)
                nfinite, result = pop(pop_value, nfinite, result)
                output_arr[idx] = get_result(nfinite, minp, result, ddof)

        return pandas.Series(output_arr, input_series._index,
                             name=input_series._name)
    return impl


sdc_pandas_series_rolling_count_impl = gen_sdc_pandas_series_rolling_impl(
    pop_count, put_count, init_result=0.)
sdc_pandas_series_rolling_kurt_impl = gen_sdc_pandas_series_rolling_impl(
    pop_kurt, put_kurt, get_result=kurt_result_or_nan,
    init_result=(0., 0., 0., 0.))
sdc_pandas_series_rolling_max_impl = gen_sdc_pandas_series_rolling_minmax_impl(
    pop_max, put_max)
sdc_pandas_series_rolling_mean_impl = gen_sdc_pandas_series_rolling_impl(
    pop_sum, put_sum, get_result=mean_result_or_nan, init_result=0.)
sdc_pandas_series_rolling_min_impl = gen_sdc_pandas_series_rolling_minmax_impl(
    pop_min, put_min)
sdc_pandas_series_rolling_skew_impl = gen_sdc_pandas_series_rolling_impl(
    pop_skew, put_skew, get_result=skew_result_or_nan, init_result=(0., 0., 0.))
sdc_pandas_series_rolling_sum_impl = gen_sdc_pandas_series_rolling_impl(
    pop_sum, put_sum, init_result=0.)
sdc_pandas_series_rolling_var_impl = gen_sdc_pandas_series_rolling_ddof_impl(
    pop_sum2, put_sum2, get_result=var_result_or_nan, init_result=(0., 0.))
sdc_pandas_series_rolling_std_impl = gen_sdc_pandas_series_rolling_ddof_impl(
    pop_sum2, put_sum2, get_result=std_result_or_nan, init_result=(0., 0.))


@sdc_rolling_overload(SeriesRollingType, 'apply')
def hpat_pandas_series_rolling_apply(self, func, raw=None):

    ty_checker = TypeChecker('Method rolling.apply().')
    ty_checker.check(self, SeriesRollingType)

    raw_accepted = (Omitted, NoneType, Boolean)
    if not isinstance(raw, raw_accepted) and raw is not None:
        ty_checker.raise_exc(raw, 'bool', 'raw')

    def hpat_pandas_rolling_series_apply_impl(self, func, raw=None):
        win = self._window
        minp = self._min_periods

        input_series = self._data
        input_arr = input_series._data
        length = len(input_arr)
        output_arr = numpy.empty(length, dtype=float64)

        def culc_apply(arr, func, minp):
            finite_arr = arr[numpy.isfinite(arr)]
            if len(finite_arr) < minp:
                return numpy.nan
            else:
                return arr_apply(finite_arr, func)

        boundary = min(win, length)
        for i in prange(boundary):
            arr_range = input_arr[:i + 1]
            output_arr[i] = culc_apply(arr_range, func, minp)

        for i in prange(boundary, length):
            arr_range = input_arr[i + 1 - win:i + 1]
            output_arr[i] = culc_apply(arr_range, func, minp)

        return pandas.Series(output_arr, input_series._index, name=input_series._name)

    return hpat_pandas_rolling_series_apply_impl


@sdc_overload_method(SeriesRollingType, 'corr')
def hpat_pandas_series_rolling_corr(self, other=None, pairwise=None):

    ty_checker = TypeChecker('Method rolling.corr().')
    ty_checker.check(self, SeriesRollingType)

    accepted_other = (bool, Omitted, NoneType, SeriesType)
    if not isinstance(other, accepted_other) and other is not None:
        ty_checker.raise_exc(other, 'Series', 'other')

    accepted_pairwise = (bool, Boolean, Omitted, NoneType)
    if not isinstance(pairwise, accepted_pairwise) and pairwise is not None:
        ty_checker.raise_exc(pairwise, 'bool', 'pairwise')

    nan_other = isinstance(other, (Omitted, NoneType)) or other is None

    def hpat_pandas_rolling_series_corr_impl(self, other=None, pairwise=None):
        win = self._window
        minp = self._min_periods

        main_series = self._data
        main_arr = main_series._data

        if nan_other == True:  # noqa
            other_arr = main_arr
        else:
            other_arr = other._data

        main_arr_length = len(main_arr)
        other_arr_length = len(other_arr)
        min_length = min(main_arr_length, other_arr_length)
        length = max(main_arr_length, other_arr_length)
        output_arr = numpy.empty(length, dtype=float64)

        chunks = parallel_chunks(length)
        for i in prange(len(chunks)):
            chunk = chunks[i]
            nfinite = 0
            result = (0., 0., 0., 0., 0.)

            if win == 0:
                for idx in range(chunk.start, chunk.stop):
                    output_arr[idx] = corr_result_or_nan(nfinite, minp, result)
                continue

            prelude_start = max(0, chunk.start - win + 1)
            prelude_stop = min(chunk.start, min_length)

            interlude_start = chunk.start
            interlude_stop = min(prelude_start + win, chunk.stop, min_length)

            postlude_start = min(prelude_start + win, chunk.stop)
            postlude_stop = min(chunk.stop, min_length)

            for idx in range(prelude_start, prelude_stop):
                x, y = main_arr[idx], other_arr[idx]
                nfinite, result = put_corr(x, y, nfinite, result)

            for idx in range(interlude_start, interlude_stop):
                x, y = main_arr[idx], other_arr[idx]
                nfinite, result = put_corr(x, y, nfinite, result)
                output_arr[idx] = corr_result_or_nan(nfinite, minp, result)

            for idx in range(postlude_start, postlude_stop):
                put_x, put_y = main_arr[idx], other_arr[idx]
                pop_x, pop_y = main_arr[idx - win], other_arr[idx - win]
                nfinite, result = put_corr(put_x, put_y, nfinite, result)
                nfinite, result = pop_corr(pop_x, pop_y, nfinite, result)
                output_arr[idx] = corr_result_or_nan(nfinite, minp, result)

            last_start = max(min_length, interlude_start)
            for idx in range(last_start, postlude_start):
                output_arr[idx] = corr_result_or_nan(nfinite, minp, result)

            last_start = max(min_length, postlude_start)
            last_stop = min(min_length + win, chunk.stop)
            for idx in range(last_start, last_stop):
                x, y = main_arr[idx - win], other_arr[idx - win]
                nfinite, result = pop_corr(x, y, nfinite, result)
                output_arr[idx] = corr_result_or_nan(nfinite, minp, result)

            for idx in range(last_stop, chunk.stop):
                output_arr[idx] = numpy.nan

        return pandas.Series(output_arr)

    return hpat_pandas_rolling_series_corr_impl


@sdc_overload_method(SeriesRollingType, 'count')
def hpat_pandas_series_rolling_count(self):

    ty_checker = TypeChecker('Method rolling.count().')
    ty_checker.check(self, SeriesRollingType)

    return sdc_pandas_series_rolling_count_impl


def _hpat_pandas_series_rolling_cov_check_types(self, other=None,
                                                pairwise=None, ddof=1):
    """Check types of parameters of series.rolling.cov()"""
    ty_checker = TypeChecker('Method rolling.cov().')
    ty_checker.check(self, SeriesRollingType)

    accepted_other = (bool, Omitted, NoneType, SeriesType)
    if not isinstance(other, accepted_other) and other is not None:
        ty_checker.raise_exc(other, 'Series', 'other')

    accepted_pairwise = (bool, Boolean, Omitted, NoneType)
    if not isinstance(pairwise, accepted_pairwise) and pairwise is not None:
        ty_checker.raise_exc(pairwise, 'bool', 'pairwise')

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')


def _gen_hpat_pandas_rolling_series_cov_impl(other, align_finiteness=False):
    """Generate series.rolling.cov() implementation based on series alignment"""
    nan_other = isinstance(other, (Omitted, NoneType)) or other is None

    def _impl(self, other=None, pairwise=None, ddof=1):
        win = self._window
        minp = self._min_periods

        main_series = self._data
        main_arr = main_series._data

        if nan_other == True:  # noqa
            other_arr = main_arr
        else:
            other_arr = other._data

        main_arr_length = len(main_arr)
        other_arr_length = len(other_arr)
        min_length = min(main_arr_length, other_arr_length)
        length = max(main_arr_length, other_arr_length)
        output_arr = numpy.empty(length, dtype=float64)

        chunks = parallel_chunks(length)
        for i in prange(len(chunks)):
            chunk = chunks[i]
            nfinite = 0
            result = (0., 0., 0., 0.)

            if win == 0:
                for idx in range(chunk.start, chunk.stop):
                    output_arr[idx] = cov_result_or_nan(nfinite, minp, result, ddof)
                continue

            prelude_start = max(0, chunk.start - win + 1)
            prelude_stop = min(chunk.start, min_length)

            interlude_start = chunk.start
            interlude_stop = min(prelude_start + win, chunk.stop, min_length)

            postlude_start = min(prelude_start + win, chunk.stop)
            postlude_stop = min(chunk.stop, min_length)

            for idx in range(prelude_start, prelude_stop):
                x, y = main_arr[idx], other_arr[idx]
                nfinite, result = put_cov(x, y, nfinite, result,
                                          align_finiteness=align_finiteness)

            for idx in range(interlude_start, interlude_stop):
                x, y = main_arr[idx], other_arr[idx]
                nfinite, result = put_cov(x, y, nfinite, result,
                                          align_finiteness=align_finiteness)
                output_arr[idx] = cov_result_or_nan(nfinite, minp, result, ddof)

            for idx in range(postlude_start, postlude_stop):
                put_x, put_y = main_arr[idx], other_arr[idx]
                pop_x, pop_y = main_arr[idx - win], other_arr[idx - win]
                nfinite, result = put_cov(put_x, put_y, nfinite, result,
                                          align_finiteness=align_finiteness)
                nfinite, result = pop_cov(pop_x, pop_y, nfinite, result,
                                          align_finiteness=align_finiteness)
                output_arr[idx] = cov_result_or_nan(nfinite, minp, result, ddof)

            last_start = max(min_length, interlude_start)
            for idx in range(last_start, postlude_start):
                output_arr[idx] = cov_result_or_nan(nfinite, minp, result, ddof)

            last_start = max(min_length, postlude_start)
            last_stop = min(min_length + win, chunk.stop)
            for idx in range(last_start, last_stop):
                x, y = main_arr[idx - win], other_arr[idx - win]
                nfinite, result = pop_cov(x, y, nfinite, result,
                                          align_finiteness=align_finiteness)
                output_arr[idx] = cov_result_or_nan(nfinite, minp, result, ddof)

            for idx in range(last_stop, chunk.stop):
                output_arr[idx] = numpy.nan

        return pandas.Series(output_arr)

    return _impl


@sdc_overload_method(SeriesRollingType, 'cov')
def hpat_pandas_series_rolling_cov(self, other=None, pairwise=None, ddof=1):
    _hpat_pandas_series_rolling_cov_check_types(self, other=other,
                                                pairwise=pairwise, ddof=ddof)

    return _gen_hpat_pandas_rolling_series_cov_impl(other, align_finiteness=True)


@sdc_overload_method(SeriesRollingType, '_df_cov')
def hpat_pandas_series_rolling_cov(self, other=None, pairwise=None, ddof=1):
    _hpat_pandas_series_rolling_cov_check_types(self, other=other,
                                                pairwise=pairwise, ddof=ddof)

    # prior to pandas_#39388 df.rolling.cov was different from series cov in handling inf values
    # so this specific overload had align_finiteness=False
    return _gen_hpat_pandas_rolling_series_cov_impl(other, align_finiteness=True)


@sdc_overload_method(SeriesRollingType, 'kurt')
def hpat_pandas_series_rolling_kurt(self):

    ty_checker = TypeChecker('Method rolling.kurt().')
    ty_checker.check(self, SeriesRollingType)

    return sdc_pandas_series_rolling_kurt_impl


@sdc_overload_method(SeriesRollingType, 'max')
def hpat_pandas_series_rolling_max(self):

    ty_checker = TypeChecker('Method rolling.max().')
    ty_checker.check(self, SeriesRollingType)

    return sdc_pandas_series_rolling_max_impl


@sdc_overload_method(SeriesRollingType, 'mean')
def hpat_pandas_series_rolling_mean(self):

    ty_checker = TypeChecker('Method rolling.mean().')
    ty_checker.check(self, SeriesRollingType)

    return sdc_pandas_series_rolling_mean_impl


@sdc_rolling_overload(SeriesRollingType, 'median')
def hpat_pandas_series_rolling_median(self):

    ty_checker = TypeChecker('Method rolling.median().')
    ty_checker.check(self, SeriesRollingType)

    return hpat_pandas_rolling_series_median_impl


@sdc_overload_method(SeriesRollingType, 'min')
def hpat_pandas_series_rolling_min(self):

    ty_checker = TypeChecker('Method rolling.min().')
    ty_checker.check(self, SeriesRollingType)

    return sdc_pandas_series_rolling_min_impl

@sdc_rolling_overload(SeriesRollingType, 'quantile')
def hpat_pandas_series_rolling_quantile(self, quantile, interpolation='linear'):

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

        def calc_quantile(arr, quantile, minp):
            finite_arr = arr[numpy.isfinite(arr)]
            if len(finite_arr) < minp:
                return numpy.nan
            else:
                return arr_quantile(finite_arr, quantile)

        boundary = min(win, length)
        for i in prange(boundary):
            arr_range = input_arr[:i + 1]
            output_arr[i] = calc_quantile(arr_range, quantile, minp)

        for i in prange(boundary, length):
            arr_range = input_arr[i + 1 - win:i + 1]
            output_arr[i] = calc_quantile(arr_range, quantile, minp)

        return pandas.Series(output_arr, input_series._index, name=input_series._name)

    return hpat_pandas_rolling_series_quantile_impl


@sdc_overload_method(SeriesRollingType, 'skew')
def hpat_pandas_series_rolling_skew(self):

    ty_checker = TypeChecker('Method rolling.skew().')
    ty_checker.check(self, SeriesRollingType)

    return sdc_pandas_series_rolling_skew_impl


@sdc_overload_method(SeriesRollingType, 'std')
def hpat_pandas_series_rolling_std(self, ddof=1):

    ty_checker = TypeChecker('Method rolling.std().')
    ty_checker.check(self, SeriesRollingType)

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    return sdc_pandas_series_rolling_std_impl


@sdc_overload_method(SeriesRollingType, 'sum')
def hpat_pandas_series_rolling_sum(self):

    ty_checker = TypeChecker('Method rolling.sum().')
    ty_checker.check(self, SeriesRollingType)

    return sdc_pandas_series_rolling_sum_impl


@sdc_overload_method(SeriesRollingType, 'var')
def hpat_pandas_series_rolling_var(self, ddof=1):

    ty_checker = TypeChecker('Method rolling.var().')
    ty_checker.check(self, SeriesRollingType)

    if not isinstance(ddof, (int, Integer, Omitted)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    return sdc_pandas_series_rolling_var_impl


hpat_pandas_series_rolling_apply.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'apply',
    'example_caption': 'Calculate the rolling apply.',
    'limitations_block':
    """
    Limitations
    -----------
    - This function may reveal slower performance than Pandas* on user system. Users should exercise a tradeoff
    between staying in JIT-region with that function or going back to interpreter mode.
    - Supported ``raw`` only can be `None` or `True`. Parameters ``args``, ``kwargs`` unsupported.
    - DataFrame/Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
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

hpat_pandas_series_rolling_corr.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'corr',
    'example_caption': 'Calculate rolling correlation.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame/Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    Resulting DataFrame/Series has default index and name.
    """,
    'extra_params':
    """
    other: :obj:`DataFrame` or :obj:`Series`
        Other DataFrame/Series.
    pairwise: :obj:`bool`
        Not relevant for Series.
    """
})

hpat_pandas_series_rolling_count.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'count',
    'example_caption': 'Count of any non-NaN observations inside the window.',
    'limitations_block': '',
    'extra_params': ''
})

hpat_pandas_series_rolling_cov.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'cov',
    'example_caption': 'Calculate rolling covariance.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame/Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    Resulting DataFrame/Series has default index and name.
    """,
    'extra_params':
    """
    other: :obj:`DataFrame` or :obj:`Series`
        Other DataFrame/Series.
    pairwise: :obj:`bool`
        Not relevant for Series.
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})

hpat_pandas_series_rolling_kurt.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'kurt',
    'example_caption': 'Calculate unbiased rolling kurtosis.',
    'limitations_block': '',
    'extra_params': ''
})

hpat_pandas_series_rolling_max.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'max',
    'example_caption': 'Calculate the rolling maximum.',
    'limitations_block': '',
    'extra_params': ''
})

hpat_pandas_series_rolling_mean.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'mean',
    'example_caption': 'Calculate the rolling mean of the values.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame/Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params': ''
})

hpat_pandas_series_rolling_median.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
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

hpat_pandas_series_rolling_min.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'min',
    'example_caption': 'Calculate the rolling minimum.',
    'limitations_block': '',
    'extra_params': ''
})

hpat_pandas_series_rolling_quantile.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'quantile',
    'example_caption': 'Calculate the rolling quantile.',
    'limitations_block':
    """
    Limitations
    -----------
    This function may reveal slower performance than Pandas* on user system. Users should exercise a tradeoff
    between staying in JIT-region with that function or going back to interpreter mode.
    Supported ``interpolation`` only can be `'linear'`.
    DataFrame/Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    quantile: :obj:`float`
        Quantile to compute. 0 <= quantile <= 1.
    interpolation: :obj:`str`
        This optional parameter specifies the interpolation method to use.
    """
})

hpat_pandas_series_rolling_skew.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'skew',
    'example_caption': 'Unbiased rolling skewness.',
    'limitations_block': '',
    'extra_params': ''
})

hpat_pandas_series_rolling_std.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'std',
    'example_caption': 'Calculate rolling standard deviation.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame/Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})

hpat_pandas_series_rolling_sum.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'sum',
    'example_caption': 'Calculate rolling sum',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame/Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params': ''
})

hpat_pandas_series_rolling_var.__doc__ = hpat_pandas_series_rolling_docstring_tmpl.format(**{
    'method_name': 'var',
    'example_caption': 'Calculate unbiased rolling variance.',
    'limitations_block':
    """
    Limitations
    -----------
    DataFrame/Series elements cannot be max/min float/integer. Otherwise SDC and Pandas results are different.
    """,
    'extra_params':
    """
    ddof: :obj:`int`
        Delta Degrees of Freedom.
    """
})

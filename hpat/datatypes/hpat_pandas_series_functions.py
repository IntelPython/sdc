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

import operator
import pandas

from numba import types
from numba.extending import (types, overload, overload_method, overload_attribute)
from numba.errors import TypingError

from hpat.hiframes.pd_series_ext import SeriesType

'''
Pandas Series (https://pandas.pydata.org/pandas-docs/stable/reference/series.html)
functions and operators definition in HPAT
Also, it contains Numba internal operators which are required for Series type handling

Implemented operators:
    add
    at
    div
    eq
    ge
    getitem
    gt
    iat
    iloc
    le
    len
    loc
    lt
    mod
    mul
    pow
    sub

Implemented methods:
    append
    ne
'''


@overload(operator.getitem)
def hpat_pandas_series_getitem(self, idx):
    """
    Pandas Series opearator 'getitem' implementation

    Algorithm: result = series[idx]
    Where:
        series: pandas.series
           idx: integer number, slice or pandas.series
        result: pandas.series or an element of the underneath type

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_static_getitem_series1
    """

    _func_name = 'Operator getitem().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if isinstance(idx, types.Integer):
        def hpat_pandas_series_getitem_idx_integer_impl(self, idx):
            """
            Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_iloc1
            """

            result = self._data[idx]
            return result

        return hpat_pandas_series_getitem_idx_integer_impl

    if isinstance(idx, types.SliceType):
        def hpat_pandas_series_getitem_idx_slice_impl(self, idx):
            """
            Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_iloc2
            """

            result = pandas.Series(self._data[idx])
            return result

        return hpat_pandas_series_getitem_idx_slice_impl

    if isinstance(idx, SeriesType):
        def hpat_pandas_series_getitem_idx_series_impl(self, idx):
            """
            Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_setitem_series_bool2
            """
            super_index = idx._data
            result = self._data[super_index]
            return result

        return hpat_pandas_series_getitem_idx_series_impl

    raise TypingError('{} The index must be an Integer, Slice or a pandas.series. Given: {}'.format(_func_name, idx))


@overload_attribute(SeriesType, 'at')
@overload_attribute(SeriesType, 'iat')
@overload_attribute(SeriesType, 'iloc')
@overload_attribute(SeriesType, 'loc')
def hpat_pandas_series_iloc(self):
    """
    Pandas Series opearator 'iloc' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.at.html#pandas.Series.at
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.iat.html#pandas.Series.iat
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.iloc.html#pandas.Series.iloc
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.loc.html#pandas.Series.loc

    Algorithm: result = series.iloc
    Where:
        series: pandas.series
        result: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_iloc2
    """

    _func_name = 'Operator at/iat/iloc/loc().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_iloc_impl(self):
        return self

    return hpat_pandas_series_iloc_impl


@overload(len)
def hpat_pandas_series_len(self):
    """
    Pandas Series operator 'len' implementation
        https://docs.python.org/2/library/functions.html#len

    Algorithm: result = len(series)
    Where:
        series: pandas.series
        result: number of items in the object

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_len
    """

    _func_name = 'Operator len().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_len_impl(self):
        return len(self._data)

    return hpat_pandas_series_len_impl


@overload_method(SeriesType, 'append')
def hpat_pandas_series_append(self, to_append):
    """
    Pandas Series method 'append' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.append.html#pandas.Series.append

    Algorithm: result = S.append(self, to_append, ignore_index=False, verify_integrity=False)

    Where:
                   S: pandas.series
           to_append: pandas.series
        ignore_index: unsupported
    verify_integrity: unsupported
              result: new pandas.series object

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_append1
    """

    _func_name = 'Method append().'

    if not isinstance(self, SeriesType) or not isinstance(to_append, SeriesType):
        raise TypingError(
            '{} The object must be a pandas.series. Given self: {}, to_append: {}'.format(_func_name, self, to_append))

    def hpat_pandas_series_append_impl(self, to_append):
        return pandas.Series(self._data + to_append._data)

    return hpat_pandas_series_append_impl


@overload_method(SeriesType, 'ne')
def hpat_pandas_series_not_equal(lhs, rhs):
    """
    Pandas Series method 'ne' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ne.html#pandas.Series.ne

    Algorithm: result = lhs.ne(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: boolean result

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method ne().'

    if not isinstance(lhs, SeriesType) or not isinstance(rhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}, rhs: {}'.format(_func_name, lhs, rhs))

    def hpat_pandas_series_not_equal_impl(lhs, rhs):
        return pandas.Series(lhs._data != rhs._data)

    return hpat_pandas_series_not_equal_impl


@overload_method(SeriesType, 'add')
def hpat_pandas_series_add(lhs, rhs):
    """
    Pandas Series method 'add' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.add.html#pandas.Series.add

    Algorithm: result = lhs.add(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series result of the operation

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method add().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_add_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(lhs._data + rhs._data)

        return hpat_pandas_series_add_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_add_number_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(lhs._data + rhs)

        return hpat_pandas_series_add_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given rhs: {}'.format(_func_name, rhs))


@overload_method(SeriesType, 'sub')
def hpat_pandas_series_sub(lhs, rhs):
    """
    Pandas Series method 'sub' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.sub.html#pandas.Series.sub

    Algorithm: result = lhs.sub(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series result of the operation

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method sub().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_sub_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(lhs._data - rhs._data)

        return hpat_pandas_series_sub_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_sub_number_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(lhs._data - rhs)

        return hpat_pandas_series_sub_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given rhs: {}'.format(_func_name, rhs))


@overload_method(SeriesType, 'mul')
def hpat_pandas_series_mul(lhs, rhs):
    """
    Pandas Series method 'mul' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mul.html#pandas.Series.mul

    Algorithm: result = lhs.mul(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series result of the operation

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method mul().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_mul_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(lhs._data * rhs._data)

        return hpat_pandas_series_mul_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_mul_number_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(lhs._data * rhs)

        return hpat_pandas_series_mul_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given rhs: {}'.format(_func_name, rhs))


@overload_method(SeriesType, 'div')
@overload_method(SeriesType, 'truediv')
def hpat_pandas_series_div(lhs, rhs):
    """
    Pandas Series method 'div' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.div.html#pandas.Series.div
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.truediv.html#pandas.Series.truediv

    Algorithm: result = lhs.div(other, level=None, fill_value=None, axis=0)
    Algorithm: result = lhs.truediv(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series result of the operation

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method div()/truediv().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_div_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(lhs._data / rhs._data)

        return hpat_pandas_series_div_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_div_number_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(lhs._data / rhs)

        return hpat_pandas_series_div_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given rhs: {}'.format(_func_name, rhs))


@overload_method(SeriesType, 'floordiv')
def hpat_pandas_series_floordiv(lhs, rhs):
    """
    Pandas Series method 'floordiv' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.floordiv.html#pandas.Series.floordiv

    Algorithm: result = lhs.floordiv(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series result of the operation

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method floordiv().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_floordiv_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(lhs._data // rhs._data)

        return hpat_pandas_series_floordiv_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_floordiv_number_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(lhs._data // rhs)

        return hpat_pandas_series_floordiv_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given rhs: {}'.format(_func_name, rhs))


@overload_method(SeriesType, 'pow')
def hpat_pandas_series_pow(lhs, rhs, level=None, fill_value=None, axis=0):
    """
    Pandas Series method 'pow' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.pow.html#pandas.Series.pow

    Algorithm: result = lhs.pow(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method pow().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_pow_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(lhs._data ** rhs._data)

        return hpat_pandas_series_pow_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_pow_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(lhs._data ** rhs)

        return hpat_pandas_series_pow_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given lhs: {} and rhs: {}'.format(
            _func_name, lhs, rhs))


@overload_method(SeriesType, 'mod')
def hpat_pandas_series_mod(lhs, rhs, level=None, fill_value=None, axis=0):
    """
    Pandas Series method 'mod' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mod.html#pandas.Series.mod

    Algorithm: result = lhs.mod(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    """

    _func_name = 'Method mod().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_mod_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(lhs._data % rhs._data)

        return hpat_pandas_series_mod_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_mod_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(lhs._data % rhs)

        return hpat_pandas_series_mod_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given lhs: {} and rhs: {}'.format(
            _func_name, lhs, rhs))


@overload_method(SeriesType, 'eq')
def hpat_pandas_series_eq(lhs, rhs, level=None, fill_value=None, axis=0):
    """
    Pandas Series method 'eq' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.eq.html#pandas.Series.eq

    Algorithm: result = lhs.eq(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method eq().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_eq_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(lhs._data == rhs._data)

        return hpat_pandas_series_eq_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_eq_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(lhs._data == rhs)

        return hpat_pandas_series_eq_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given lhs: {} and rhs: {}'.format(
            _func_name, lhs, rhs))


@overload_method(SeriesType, 'ge')
def hpat_pandas_series_ge(lhs, rhs, level=None, fill_value=None, axis=0):
    """
    Pandas Series method 'ge' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ge.html#pandas.Series.ge

    Algorithm: result = lhs.ge(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method ge().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_ge_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(lhs._data >= rhs._data)

        return hpat_pandas_series_ge_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_ge_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(lhs._data >= rhs)

        return hpat_pandas_series_ge_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given lhs: {} and rhs: {}'.format(
            _func_name, lhs, rhs))


@overload_method(SeriesType, 'lt')
def hpat_pandas_series_lt(lhs, rhs, level=None, fill_value=None, axis=0):
    """
    Pandas Series method 'lt' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.lt.html#pandas.Series.lt

    Algorithm: result = lhs.lt(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method lt().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_lt_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(lhs._data < rhs._data)

        return hpat_pandas_series_lt_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_lt_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(lhs._data < rhs)

        return hpat_pandas_series_lt_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given lhs: {} and rhs: {}'.format(
            _func_name, lhs, rhs))


@overload_method(SeriesType, 'gt')
def hpat_pandas_series_gt(lhs, rhs, level=None, fill_value=None, axis=0):
    """
    Pandas Series method 'gt' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.gt.html#pandas.Series.gt

    Algorithm: result = lhs.gt(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method gt().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_gt_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(lhs._data > rhs._data)

        return hpat_pandas_series_gt_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_gt_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(lhs._data > rhs)

        return hpat_pandas_series_gt_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given lhs: {} and rhs: {}'.format(
            _func_name, lhs, rhs))


@overload_method(SeriesType, 'le')
def hpat_pandas_series_le(lhs, rhs, level=None, fill_value=None, axis=0):
    """
    Pandas Series method 'le' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.le.html#pandas.Series.le

    Algorithm: result = lhs.le(other, level=None, fill_value=None, axis=0)

    Where:
               lhs: pandas.series
             other: pandas.series or scalar value
             level: unsupported
        fill_value: unsupported
              axis: unsupported
            result: pandas.series

    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    """

    _func_name = 'Method le().'

    if not isinstance(lhs, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given lhs: {}'.format(_func_name, lhs))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(rhs, SeriesType):
        def hpat_pandas_series_le_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(lhs._data <= rhs._data)

        return hpat_pandas_series_le_impl

    if isinstance(rhs, types.Integer) or isinstance(rhs, types.Float):
        def hpat_pandas_series_le_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(lhs._data <= rhs)

        return hpat_pandas_series_le_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given lhs: {} and rhs: {}'.format(
            _func_name, lhs, rhs))


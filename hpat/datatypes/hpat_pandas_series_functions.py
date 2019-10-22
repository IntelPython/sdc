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

"""
| :class:`pandas.Series` functions and operators implementations in HPAT
| Also, it contains Numba internal operators which are required for Series type handling
"""

import numpy
import operator
import pandas

from numba.errors import TypingError
from numba.extending import (types, overload, overload_method, overload_attribute)
from numba import types

import hpat
from hpat.hiframes.pd_series_ext import SeriesType
from hpat.str_arr_ext import StringArrayType
from hpat.utils import to_array


@overload(operator.getitem)
def hpat_pandas_series_getitem(self, idx):
    """
    Pandas Series operator :attr:`pandas.Series.get` implementation
    **Algorithm**: result = series[idx]

    **Test**: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_static_getitem_series1

    Parameters
    ----------
    series: :obj:`pandas.Series`
           input series
    idx: :obj:`int`, :obj:`slice` or :obj:`pandas.Series`
        input index

    Returns
    -------
    :class:`pandas.Series` or an element of the underneath type
            object of :class:`pandas.Series`
    """

    _func_name = 'Operator getitem().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if isinstance(idx, types.Integer):
        def hpat_pandas_series_getitem_idx_integer_impl(self, idx):
            """
            **Test**: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_iloc1
            """

            result = self._data[idx]
            return result

        return hpat_pandas_series_getitem_idx_integer_impl

    if isinstance(idx, types.SliceType):
        def hpat_pandas_series_getitem_idx_slice_impl(self, idx):
            """
            **Test**: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_iloc2
            """

            result = pandas.Series(self._data[idx])
            return result

        return hpat_pandas_series_getitem_idx_slice_impl

    if isinstance(idx, SeriesType):
        def hpat_pandas_series_getitem_idx_series_impl(self, idx):
            """
            **Test**: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_setitem_series_bool2
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
    Pandas Series operators :attr:`pandas.Series.at`, :attr:`pandas.Series.iat`, :attr:`pandas.Series.iloc`, :attr:`pandas.Series.loc` implementation.
    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_iloc2

    Parameters
    ----------
    series: :class:`pandas.Series`
           input series

    Returns
    -------
    :obj:`pandas.Series`
         returns an object of :obj:`pandas.Series`
    """

    _func_name = 'Operator at/iat/iloc/loc().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_iloc_impl(self):
        return self

    return hpat_pandas_series_iloc_impl


@overload_attribute(SeriesType, 'shape')
def hpat_pandas_series_shape(self):
    """
    Pandas Series attribute :attr:`pandas.Series.shape` implementation
    **Algorithm**: result = series.shape
    **Test**: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_shape1
    Parameters
    ----------
    series: :obj:`pandas.Series`
          input series
    Returns
    -------
    :obj:`tuple`
        a tuple of the shape of the underlying data
    """

    _func_name = 'Attribute shape.'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_shape_impl(self):
        return self._data.shape

    return hpat_pandas_series_shape_impl


@overload_attribute(SeriesType, 'values')
def hpat_pandas_series_iloc(self):
    """
    Pandas Series attribute 'values' implementation.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.values.html#pandas.Series.values
    Algorithm: result = series.values
    Where:
        series: pandas.series
        result: pandas.series as ndarray
    Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_values
    """

    _func_name = 'Attribute values.'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_values_impl(self):
        return self._data

    return hpat_pandas_series_values_impl


@overload_attribute(SeriesType, 'index')
def hpat_pandas_series_index(self):
    """
    Pandas Series attribute :attr:`pandas.Series.index` implementation
    **Algorithm**: result = series.index
    **Test**: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_index1
              python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_index2
    Parameters
    ----------
    series: :obj:`pandas.Series`
           input series
    Returns
    -------
    :class:`pandas.Series`
           the index of the Series
    """

    _func_name = 'Attribute index.'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if isinstance(self.index, types.NoneType) or self.index is None:
        def hpat_pandas_series_index_none_impl(self):
            return numpy.arange(len(self._data))

        return hpat_pandas_series_index_none_impl
    else:
        def hpat_pandas_series_index_impl(self):
            return self._index

        return hpat_pandas_series_index_impl


@overload_attribute(SeriesType, 'size')
def hpat_pandas_series_size(self):
    """
    Pandas Series attribute :attr:`pandas.Series.size` implementation

    .. only:: developer

        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_size

    Parameters
    ----------
    series: :obj:`pandas.Series`
        input series

    Returns
    -------
    :class:`pandas.Series`
        Return the number of elements in the underlying data.
    """

    _func_name = 'Attribute size.'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_size_impl(self):
        return len(self._data)

    return hpat_pandas_series_size_impl


@overload_attribute(SeriesType, 'ndim')
def hpat_pandas_series_ndim(self):
    """
    Pandas Series attribute :attr:`pandas.Series.ndim` implementation

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_getattr_ndim

    Parameters
    ----------
    self: :obj:`pandas.Series`
           input series

    Returns
    -------
    :obj:`int`
           Number of dimensions of the underlying data, by definition 1
    """

    _func_name = 'Attribute ndim.'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_ndim_impl(self):
        return 1

    return hpat_pandas_series_ndim_impl


@overload_attribute(SeriesType, 'T')
def hpat_pandas_series_T(self):
    """
    Pandas Series attribute :attr:`pandas.Series.T` implementation

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_getattr_T

    Parameters
    ----------
    self: :obj:`pandas.Series`
           input series

    Returns
    -------
    :obj:`numpy.ndarray`
         An array representing the underlying data
    """

    _func_name = 'Attribute T.'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_T_impl(self):
        return self._data

    return hpat_pandas_series_T_impl



@overload(len)
def hpat_pandas_series_len(self):
    """
    Pandas Series operator :func:`len` implementation

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_len

    Parameters
    ----------
    series: :class:`pandas.Series`

    Returns
    -------
    :obj:`int`
        number of items in the object
    """

    _func_name = 'Operator len().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    def hpat_pandas_series_len_impl(self):
        return len(self._data)

    return hpat_pandas_series_len_impl


@overload_method(SeriesType, 'shift')
def hpat_pandas_series_shift(self, periods=1, freq=None, axis=0, fill_value=None):
    """
    Pandas Series method :meth:`pandas.Series.shift` implementation.

    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_shift
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_shift_unboxing
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_shift_full
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_shift_str
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_shift_fill_str
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_shift_unsupported_params

    Parameters
    ----------
    self: :obj:`pandas.Series`
        input series
    periods: :obj:`int`
        Number of periods to shift. Can be positive or negative.
    freq: :obj:`DateOffset`, :obj:`tseries.offsets`, :obj:`timedelta`, :obj:`str`
        Offset to use from the tseries module or time rule (e.g. ‘EOM’).
        *unsupported*
    axis: :obj:`int`, :obj:`str`
        Axis along which the operation acts
        0/None/'index' - row-wise operation
        1/'columns'    - column-wise operation
        *unsupported*
    fill_value : :obj:`int`, :obj:`float`
        The scalar value to use for newly introduced missing values.

    Returns
    -------
    :obj:`scalar`
         returns :obj:`series` object
    """

    _func_name = 'Method shift().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(self.data.dtype, types.Number):
        msg = '{} The object must be a number. Given self.data.dtype: {}'
        raise TypingError(msg.format(_func_name, self.data.dtype))

    if not isinstance(fill_value, (types.Omitted, types.Number, types.NoneType)) and fill_value is not None:
        raise TypingError('{} The object must be a number. Given fill_value: {}'.format(_func_name, fill_value))

    if not isinstance(freq, (types.Omitted, types.NoneType)) and freq is not None:
        raise TypingError('{} Unsupported parameters. Given freq: {}'.format(_func_name, freq))

    if not isinstance(axis, (types.Omitted, int, types.Integer)):
        raise TypingError('{} Unsupported parameters. Given axis: {}'.format(_func_name, axis))

    def hpat_pandas_series_shift_impl(self, periods=1, freq=None, axis=0, fill_value=None):
        if axis != 0:
            raise TypingError('Method shift(). Unsupported parameters. Given axis != 0')

        arr = numpy.empty_like(self._data)
        if periods > 0:
            arr[:periods] = fill_value or numpy.nan
            arr[periods:] = self._data[:-periods]
        elif periods < 0:
            arr[periods:] = fill_value or numpy.nan
            arr[:periods] = self._data[-periods:]
        else:
            arr[:] = self._data

        return pandas.Series(arr)

    return hpat_pandas_series_shift_impl


@overload_method(SeriesType, 'isin')
def hpat_pandas_series_isin(self, values):
    """
    Pandas Series method :meth:`pandas.Series.isin` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_isin_list1
    Parameters
    -----------
    values : :obj:`list` or :obj:`set` object
               specifies values to look for in the series
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object indicating if each element of self is in values
    """

    _func_name = 'Method isin().'

    if not isinstance(self, SeriesType):
        raise TypingError(
            '{} The object must be a pandas.series. Given self: {}'.format(_func_name, self))

    if not isinstance(values, (types.Set, types.List)):
        raise TypingError(
            '{} The argument must be set or list-like object. Given values: {}'.format(_func_name, values))

    def hpat_pandas_series_isin_impl(self, values):
        # TODO: replace with below line when Numba supports np.isin in nopython mode
        # return pandas.Series(np.isin(self._data, values))
        return pandas.Series([(x in values) for x in self._data])

    return hpat_pandas_series_isin_impl


@overload_method(SeriesType, 'append')
def hpat_pandas_series_append(self, to_append):
    """
    Pandas Series method :meth:`pandas.Series.append` implementation.

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_append1
    Parameters
    -----------
    to_append : :obj:`pandas.Series` object
               input argument
    ignore_index:
                 *unsupported*
    verify_integrity:
                     *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method append().'

    if not isinstance(self, SeriesType) or not isinstance(to_append, SeriesType):
        raise TypingError(
            '{} The object must be a pandas.series. Given self: {}, to_append: {}'.format(_func_name, self, to_append))

    def hpat_pandas_series_append_impl(self, to_append):
        return pandas.Series(self._data + to_append._data)

    return hpat_pandas_series_append_impl


@overload_method(SeriesType, 'copy')
def hpat_pandas_series_copy(self, deep=True):
    """
    Pandas Series method :meth:`pandas.Series.copy` implementation.

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_copy_str1
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_copy_int1
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_copy_deep

    Parameters
    -----------
    self: :class:`pandas.Series`
        input arg
    deep: :obj:`bool`, default :obj:`True`
        Make a deep copy, including a copy of the data and the indices.
        With deep=False neither the indices nor the data are copied.
        [HPAT limitations]:
            - deep=False: shallow copy of index is not supported

    Returns
    -------
    :obj:`pandas.Series` or :obj:`pandas.DataFrame`
        Object type matches caller.
    """
    _func_name = 'Method Series.copy().'

    if (isinstance(self, SeriesType) and
        (isinstance(deep, (types.Omitted, types.Boolean)) or deep == True)
    ):
        if isinstance(self.index, types.NoneType):
            def hpat_pandas_series_copy_impl(self, deep=True):
                if deep:
                    return pandas.Series(self._data.copy())
                else:
                    return pandas.Series(self._data)
            return hpat_pandas_series_copy_impl
        else:
            def hpat_pandas_series_copy_impl(self, deep=True):
                if deep:
                    return pandas.Series(self._data.copy(), index=self._index.copy())
                else:
                    return pandas.Series(self._data, index=self._index.copy())
            return hpat_pandas_series_copy_impl


@overload_method(SeriesType, 'groupby')
def hpat_pandas_series_groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        observed=False):
    """
    Pandas Series method :meth:`pandas.Series.groupby` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_groupby_count
    Parameters
    -----------
    self: :class:`pandas.Series`
        input arg
    by: :obj:`pandas.Series` object
        Used to determine the groups for the groupby
    axis:
        *unsupported*
    level:
        *unsupported*
    as_index:
        *unsupported*
    sort:
        *unsupported*
    group_keys:
        *unsupported*
    squeeze:
        *unsupported*
    observed:
        *unsupported*
    Returns
    -------
    :obj:`pandas.SeriesGroupBy`
         returns :obj:`pandas.SeriesGroupBy` object
    """

    _func_name = 'Method Series.groupby().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if by is None and axis is None:
        raise TypingError("{} You have to supply one of 'by' or 'axis' parameters".format(_func_name))

    if level is not None and not isinstance(level, (types.Integer, types.NoneType, types.Omitted)):
        raise TypingError("{} 'level' must be an Integer. Given: {}".format(_func_name, level))

    def hpat_pandas_series_groupby_impl(
            self,
            by=None,
            axis=0,
            level=None,
            as_index=True,
            sort=True,
            group_keys=True,
            squeeze=False,
            observed=False):
        # TODO Needs to implement parameters value check
        # if level is not None and (level < -1 or level > 0):
        #     raise ValueError("Method Series.groupby(). level > 0 or level < -1 only valid with MultiIndex")

        return pandas.core.groupby.SeriesGroupBy(self)

    return hpat_pandas_series_groupby_impl


@overload_method(SeriesType, 'isna')
def hpat_pandas_series_isna(self):
    """
    Pandas Series method :meth:`pandas.Series.isna` implementation.

    .. only:: developer

        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_isna1
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_str_isna1

    Parameters
    -----------
    self : :obj:`pandas.Series` object
               input argument

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method isna().'

    if not isinstance(self, SeriesType):
        raise TypingError(
            '{} The object must be a pandas.series. Given self: {}'.format(_func_name, self))

    if isinstance(self.dtype, (types.Integer, types.Float)):

        def hpat_pandas_series_isna_impl(self):

            return pandas.Series(numpy.isnan(self._data))

        return hpat_pandas_series_isna_impl

    if isinstance(self.dtype, types.UnicodeType):

        def hpat_pandas_series_isna_impl(self):
            result = numpy.empty(len(self._data), numpy.bool_)
            byte_size = 8
            # iterate over bits in StringArrayType null_bitmap and fill array indicating if array's element are NaN
            for i in range(len(self._data)):
                bmap_idx = i // byte_size
                bit_idx = i % byte_size
                bmap = self._data.null_bitmap[bmap_idx]
                bit_value = (bmap >> bit_idx) & 1
                result[i] = bit_value == 0
            return pandas.Series(result)

        return hpat_pandas_series_isna_impl


@overload_method(SeriesType, 'ne')
def hpat_pandas_series_ne(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.ne` implementation.
    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8

    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method ne().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_ne_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(self._data != other._data)

        return hpat_pandas_series_ne_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_ne_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(self._data != other)

        return hpat_pandas_series_ne_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'add')
def hpat_pandas_series_add(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.add` implementation.
    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5

    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method add().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_add_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(lhs._data + rhs._data)

        return hpat_pandas_series_add_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_add_number_impl(lhs, rhs):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(lhs._data + rhs)

        return hpat_pandas_series_add_number_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'sub')
def hpat_pandas_series_sub(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.sub` implementation.
    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5

    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method sub().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_sub_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(self._data - other._data)

        return hpat_pandas_series_sub_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_sub_number_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(self._data - other)

        return hpat_pandas_series_sub_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given other: {}'.format(_func_name, other))


@overload_method(SeriesType, 'sum')
def hpat_pandas_series_sum(
    self,
    axis=None,
    skipna=None,
    level=None,
    numeric_only=None,
    min_count=0,
):
    """
    Pandas Series method :meth:`pandas.Series.sum` implementation.

    .. only:: developer

        Tests:
            python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_sum1
            # python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_sum2

    Parameters
    ----------
    self: :class:`pandas.Series`
        input series
    axis:
        *unsupported*
    skipna: :obj:`bool`, default :obj:`True`
        Exclude NA/null values when computing the result.
    level:
        *unsupported*
    numeric_only:
        *unsupported*
    min_count:
        *unsupported*

    Returns
    -------
    :obj:`float`
        scalar or Series (if level specified)
    """

    _func_name = 'Method sum().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not (isinstance(axis, (types.Integer, types.Omitted)) or axis == None):
        raise TypingError('{} The axis must be an Integer. Currently unsupported. Given: {}'.format(_func_name, axis))

    if not (isinstance(skipna, (types.Boolean, types.Omitted)) or skipna == None):
        raise TypingError('{} The skipna must be a Boolean. Given: {}'.format(_func_name, skipna))

    if not (isinstance(level, (types.Integer, types.StringLiteral, types.Omitted)) or level == None):
        raise TypingError('{} The level must be an Integer or level name. Currently unsupported. Given: {}'.format(_func_name, level))

    if not (isinstance(numeric_only, (types.Boolean, types.Omitted)) or numeric_only == None):
        raise TypingError('{} The numeric_only must be a Boolean. Currently unsupported. Given: {}'.format(_func_name, numeric_only))

    if not (isinstance(min_count, (types.Integer, types.Omitted)) or min_count == 0):
        raise TypingError('{} The min_count must be an Integer. Currently unsupported. Given: {}'.format(_func_name, min_count))

    def hpat_pandas_series_sum_impl(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
    ):
        """
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_sum1
        """
        if skipna is None:
            skipna = True
        if skipna:
            return numpy.nansum(self._data)
        return numpy.sum(self._data)

    return hpat_pandas_series_sum_impl


@overload_method(SeriesType, 'take')
def hpat_pandas_series_take(self, indices, axis=0, is_copy=False):
    """
    Pandas Series method :meth:`pandas.Series.take` implementation.
    .. only:: developer
       Tests: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_take_index_default
              python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_take_index_default_unboxing
              python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_take_index_int
              python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_take_index_int_unboxing
              python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_take_index_str
              python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_take_index_str_unboxing
    Parameters
    ----------
    self: :obj:`pandas.Series`
        input series
    indices: :obj:`array-like`
         An array of ints indicating which positions to take
    axis: {0 or `index`, 1 or `columns`, None}, default 0
        The axis on which to select elements. 0 means that we are selecting rows,
        1 means that we are selecting columns.
        *unsupported*
    is_copy: :obj:`bool`, default True
        Whether to return a copy of the original object or not.
        *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object containing the elements taken from the object
    """

    _func_name = 'Method take().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(indices, types.List):
        raise TypingError('{} The indices must be a List. Given: {}'.format(_func_name, indices))

    if not (isinstance(axis, (types.Integer, types.Omitted)) or axis == 0):
        raise TypingError('{} The axis must be an Integer. Currently unsupported. Given: {}'.format(_func_name, axis))

    if not (isinstance(is_copy, (types.Boolean, types.Omitted)) or is_copy == False):
        raise TypingError('{} The is_copy must be a boolean. Given: {}'.format(_func_name, is_copy))

    if self.index is not types.none:
        def hpat_pandas_series_take_impl(self, indices, axis=0, is_copy=False):
            local_data = [self._data[i] for i in indices]
            local_index = [self._index[i] for i in indices]

            return pandas.Series(local_data, local_index)

        return hpat_pandas_series_take_impl
    else:
        def hpat_pandas_series_take_noindex_impl(self, indices, axis=0, is_copy=False):
            local_data = [self._data[i] for i in indices]

            return pandas.Series(local_data, indices)

        return hpat_pandas_series_take_noindex_impl


@overload_method(SeriesType, 'mul')
def hpat_pandas_series_mul(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.mul` implementation.
    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5

    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method mul().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_mul_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(self._data * other._data)

        return hpat_pandas_series_mul_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_mul_number_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(self._data * other)

        return hpat_pandas_series_mul_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given other: {}'.format(_func_name, other))


@overload_method(SeriesType, 'div')
@overload_method(SeriesType, 'truediv')
def hpat_pandas_series_div(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.div` and :meth:`pandas.Series.truediv` implementation.
    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5

    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method div() or truediv().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_div_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(self._data / other._data)

        return hpat_pandas_series_div_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_div_number_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(self._data / other)

        return hpat_pandas_series_div_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given other: {}'.format(_func_name, other))


@overload_method(SeriesType, 'floordiv')
def hpat_pandas_series_floordiv(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.floordiv` implementation.
    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5

    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method floordiv().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_floordiv_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(self._data // other._data)

        return hpat_pandas_series_floordiv_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_floordiv_number_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(self._data // other)

        return hpat_pandas_series_floordiv_number_impl

    raise TypingError('{} The object must be a pandas.series or scalar. Given other: {}'.format(_func_name, other))


@overload_method(SeriesType, 'pow')
def hpat_pandas_series_pow(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.pow` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method pow().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_pow_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(self._data ** other._data)

        return hpat_pandas_series_pow_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_pow_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(self._data ** other)

        return hpat_pandas_series_pow_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'prod')
def hpat_pandas_series_prod(self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0):
    """
    Pandas Series method :meth:`pandas.Series.prod` implementation.

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_prod

    Parameters
    -----------
    self: :obj:`pandas.Series`
        input series
    axis: {index (0)}
        Axis for the function to be applied on.
        *unsupported*
    skipna: :obj:`bool`, default :obj:`True`
        Exclude nan values when computing the result
    level: :obj:`int`, :obj:`str`, default :obj:`None`
        If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a scalar.
        *unsupported*
    numeric_only: :obj:`bool`, default :obj:`None`
        Include only float, int, boolean columns.
        If None, will attempt to use everything, then use only numeric data.
        Not implemented for Series.
        *unsupported*
    min_count: :obj:`int`, default 0
        The required number of valid values to perform the operation.
        If fewer than min_count non-NA values are present the result will be NA.
        *unsupported*

    Returns
    -------
    :obj:
        Returns scalar or Series (if level specified)
    """

    _func_name = 'Method prod().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(self.data.dtype, (types.Integer, types.Float)):
        raise TypingError('{} Non numeric values unsupported. Given: {}'.format(_func_name, self.data.data.dtype))

    if not (isinstance(skipna, (types.Omitted, types.Boolean)) or skipna is True):
        raise TypingError("{} 'skipna' must be a boolean type. Given: {}".format(_func_name, skipna))

    if not (isinstance(axis, types.Omitted) or axis is None) \
            or not (isinstance(level, types.Omitted) or level is None) \
            or not (isinstance(numeric_only, types.Omitted) or numeric_only is None) \
            or not (isinstance(min_count, types.Omitted) or min_count == 0):
        raise TypingError(
            '{} Unsupported parameters. Given axis: {}, level: {}, numeric_only: {}, min_count: {}'.format(
                _func_name, axis, level, numeric_only, min_count))

    def hpat_pandas_series_prod_impl(self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0):
        if skipna:
            return numpy.nanprod(self._data)
        else:
            return numpy.prod(self._data)

    return hpat_pandas_series_prod_impl


@overload_method(SeriesType, 'quantile')
def hpat_pandas_series_quantile(self, q=0.5, interpolation='linear'):
    """
    Pandas Series method :meth:`pandas.Series.quantile` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_quantile
             python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_quantile_q_vector
    Parameters
    -----------
    q : :obj: float or array-like object, default 0.5
        the quantile(s) to compute
    interpolation: 'linear', 'lower', 'higher', 'midpoint', 'nearest', default `linear`
        *unsupported* by Numba
    Returns
    -------
    :obj:`pandas.Series` or float
    """

    _func_name = 'Method quantile().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(interpolation, types.Omitted) and interpolation is not 'linear':
        raise TypingError('{} Unsupported parameters. Given interpolation: {}'.format(_func_name, interpolation))

    if not isinstance(q, (types.Number, types.Omitted, types.List)) and q != 0.5:
        raise TypingError('{} The parameter must be float. Given type q: {}'.format(_func_name, type(q)))

    def hpat_pandas_series_quantile_impl(self, q=0.5, interpolation='linear'):
        return numpy.quantile(self._data, q)

    return hpat_pandas_series_quantile_impl


@overload_method(SeriesType, 'min')
def hpat_pandas_series_min(self, axis=None, skipna=True, level=None, numeric_only=None):
    """
    Pandas Series method :meth:`pandas.Series.min` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_min
             python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_min_param
    Parameters
    -----------
    axis:
        *unsupported*
    skipna: :obj:`bool` object
        Exclude nan values when computing the result
    level:
        *unsupported*
    numeric_only:
        *unsupported*
    Returns
    -------
    :obj:
         returns :obj: scalar
    """

    _func_name = 'Method min().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(self.data.dtype, (types.Integer, types.Float)):
        raise TypingError('{} Currently function supports only numeric values. Given data type: {}'.format(_func_name,
                                                                                                           self.data.dtype))

    if not isinstance(skipna, (types.Omitted, types.Boolean)) and skipna is not True:
        raise TypingError(
            '{} The parameter must be a boolean type. Given type skipna: {}'.format(_func_name, skipna))

    if not (isinstance(axis, types.Omitted) or axis is None) \
            or not (isinstance(level, types.Omitted) or level is None) \
            or not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
        raise TypingError(
            '{} Unsupported parameters. Given axis: {}, level: {}, numeric_only: {}'.format(_func_name, axis, level,
                                                                                            numeric_only))

    def hpat_pandas_series_min_impl(self, axis=None, skipna=True, level=None, numeric_only=None):
        if skipna:
            return numpy.nanmin(self._data)

        return self._data.min()

    return hpat_pandas_series_min_impl


@overload_method(SeriesType, 'max')
def hpat_pandas_series_max(self, axis=None, skipna=True, level=None, numeric_only=None):
    """
    Pandas Series method :meth:`pandas.Series.max` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_max
             python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_max_param
    Parameters
    -----------
    axis:
        *unsupported*
    skipna: :obj:`bool` object
        Exclude nan values when computing the result
    level:
        *unsupported*
    numeric_only:
        *unsupported*
    Returns
    -------
    :obj:
         returns :obj: scalar
    """

    _func_name = 'Method max().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(self.data.dtype, (types.Integer, types.Float)):
        raise TypingError('{} Currently function supports only numeric values. Given data type: {}'.format(_func_name,
                                                                                                           self.data.dtype))

    if not isinstance(skipna, (types.Omitted, types.Boolean)) and skipna is not True:
        raise TypingError(
            '{} The parameter must be a boolean type. Given type skipna: {}'.format(_func_name, skipna))

    if not (isinstance(axis, types.Omitted) or axis is None) \
            or not (isinstance(level, types.Omitted) or level is None) \
            or not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
        raise TypingError(
            '{} Unsupported parameters. Given axis: {}, level: {}, numeric_only: {}'.format(_func_name, axis, level,
                                                                                            numeric_only))

    def hpat_pandas_series_max_impl(self, axis=None, skipna=True, level=None, numeric_only=None):
        if skipna:
            return numpy.nanmax(self._data)

        return self._data.max()

    return hpat_pandas_series_max_impl


@overload_method(SeriesType, 'mean')
def hpat_pandas_series_mean(self, axis=None, skipna=None, level=None, numeric_only=None):
    """
    Pandas Series method :meth:`pandas.Series.mean` implementation.

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_mean

    Parameters
    -----------
    axis: {index (0)}
        Axis for the function to be applied on.
        *unsupported*
    skipna: :obj:`bool`, default True
        Exclude NA/null values when computing the result.
    level: :obj:`int` or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a scalar.
        *unsupported*
    numeric_only: :obj:`bool`, default None
        Include only float, int, boolean columns.
        If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
        *unsupported*

    Returns
    -------
    :obj:
         Return the mean of the values for the requested axis.
    """

    _func_name = 'Method mean().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(self.data.dtype, types.Number):
        raise TypingError('{} Currently function supports only numeric values. Given data type: {}'.format(_func_name, self.data.dtype))

    if not isinstance(skipna, (types.Omitted, types.Boolean)) and skipna is not None:
        raise TypingError(
            '{} The parameter must be a boolean type. Given type skipna: {}'.format(_func_name, skipna))

    if not (isinstance(axis, types.Omitted) or axis is None) \
            or not (isinstance(level, types.Omitted) or level is None) \
            or not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
        raise TypingError(
            '{} Unsupported parameters. Given axis: {}, level: {}, numeric_only: {}'.format(_func_name, axis, level,
                                                                                            numeric_only))

    def hpat_pandas_series_mean_impl(self, axis=None, skipna=None, level=None, numeric_only=None):
        if skipna is None:
            skipna = True

        if skipna:
            return numpy.nanmean(self._data)

        return self._data.mean()

    return hpat_pandas_series_mean_impl


@overload_method(SeriesType, 'mod')
def hpat_pandas_series_mod(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.mod` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method mod().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_mod_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5
            """

            return pandas.Series(self._data % other._data)

        return hpat_pandas_series_mod_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_mod_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op5_float_scalar
            """

            return pandas.Series(self._data % other)

        return hpat_pandas_series_mod_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'eq')
def hpat_pandas_series_eq(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.eq` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method eq().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_eq_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(self._data == other._data)

        return hpat_pandas_series_eq_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_eq_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(self._data == other)

        return hpat_pandas_series_eq_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'ge')
def hpat_pandas_series_ge(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.ge` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method ge().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_ge_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(self._data >= other._data)

        return hpat_pandas_series_ge_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_ge_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(self._data >= other)

        return hpat_pandas_series_ge_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'idxmin')
def hpat_pandas_series_idxmin(self, axis=None, skipna=True, *args):
    """
    Pandas Series method :meth:`pandas.Series.idxmin` implementation.

    .. only:: developer

        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_idxmin1
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_idxmin_str
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_idxmin_str_idx
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_idxmin_no
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_idxmin_int
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_idxmin_noidx
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_idxmin_idx

    Parameters
    -----------
    axis :  :obj:`int`, :obj:`str`, default: None
            Axis along which the operation acts
            0/None - row-wise operation
            1      - column-wise operation
            *unsupported*
    skipna:  :obj:`bool`, default: True
            exclude NA/null values
            *unsupported*

    Returns
    -------
    :obj:`pandas.Series.index` or nan
            returns: Label of the minimum value.
    """

    _func_name = 'Method idxmin().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(self.data.dtype, types.Number):
        raise TypingError('{} Numeric values supported only. Given: {}'.format(_func_name, self.data.dtype))

    if not (isinstance(skipna, (types.Omitted, types.Boolean, bool)) or skipna is True):
        raise TypingError("{} 'skipna' must be a boolean type. Given: {}".format(_func_name, skipna))

    if not (isinstance(axis, types.Omitted) or axis is None):
        raise TypingError("{} 'axis' unsupported. Given: {}".format(_func_name, axis))

    if not (isinstance(skipna, types.Omitted) or skipna is True):
        raise TypingError("{} 'skipna' unsupported. Given: {}".format(_func_name, skipna))

    if isinstance(self.index, types.NoneType) or self.index is None:
        def hpat_pandas_series_idxmin_impl(self, axis=None, skipna=True):

            return numpy.argmin(self._data)

        return hpat_pandas_series_idxmin_impl
    else:
        def hpat_pandas_series_idxmin_index_impl(self, axis=None, skipna=True):
            # no numpy.nanargmin is supported by Numba at this time
            result = numpy.argmin(self._data)
            return self._index[int(result)]

        return hpat_pandas_series_idxmin_index_impl


@overload_method(SeriesType, 'lt')
def hpat_pandas_series_lt(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.lt` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method lt().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_lt_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(self._data < other._data)

        return hpat_pandas_series_lt_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_lt_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(self._data < other)

        return hpat_pandas_series_lt_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'gt')
def hpat_pandas_series_gt(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.gt` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method gt().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_gt_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(self._data > other._data)

        return hpat_pandas_series_gt_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_gt_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(self._data > other)

        return hpat_pandas_series_gt_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'le')
def hpat_pandas_series_le(self, other, level=None, fill_value=None, axis=0):
    """
    Pandas Series method :meth:`pandas.Series.le` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
    Parameters
    ----------
    self: :class:`pandas.Series`
        input arg
    other: :obj:`pandas.Series`, :obj:`int` or :obj:`float`
        input arg
    level: :obj:`int` or name
         *unsupported*
    fill_value: :obj:`float` or None, default None
              *unsupported*
    axis: default 0
         *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method le().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if level is not None or fill_value is not None or axis != 0:
        raise TypingError(
            '{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,
                                                                                          axis))

    if isinstance(other, SeriesType):
        def hpat_pandas_series_le_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8
            """

            return pandas.Series(self._data <= other._data)

        return hpat_pandas_series_le_impl

    if isinstance(other, types.Integer) or isinstance(other, types.Float):
        def hpat_pandas_series_le_impl(self, other):
            """
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_integer_scalar
            Test:  python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_op8_float_scalar
            """

            return pandas.Series(self._data <= other)

        return hpat_pandas_series_le_impl

    raise TypingError(
        '{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(
            _func_name, self, other))


@overload_method(SeriesType, 'abs')
def hpat_pandas_series_abs(self):
    """
    Pandas Series method :meth:`pandas.Series.abs` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_abs1
    Parameters
    -----------
    self: :obj:`pandas.Series`
          input series
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` containing the absolute value of elements
    """

    _func_name = 'Method abs().'

    if not isinstance(self, SeriesType):
        raise TypingError(
            '{} The object must be a pandas.series. Given self: {}'.format(_func_name, self))

    if not isinstance(self.dtype, (types.Integer, types.Float)):
        raise TypingError(
            '{} The function only applies to elements that are all numeric. Given data type: {}'.format(_func_name,
                                                                                                        self.dtype))

    def hpat_pandas_series_abs_impl(self):
        return pandas.Series(numpy.abs(self._data))

    return hpat_pandas_series_abs_impl


@overload_method(SeriesType, 'unique')
def hpat_pandas_series_unique(self):
    """
    Pandas Series method :meth:`pandas.Series.unique` implementation.
    Note: Return values order is unspecified
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_unique_sorted
    Parameters
    -----------
    self: :class:`pandas.Series`
        input arg
    Returns
    -------
    :obj:`numpy.array`
         returns :obj:`numpy.array` ndarray
    """

    _func_name = 'Method unique().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if isinstance(self.data, StringArrayType):
        def hpat_pandas_series_unique_str_impl(self):
            '''
            Returns sorted unique elements of an array
            Note: Can't use Numpy due to StringArrayType has no ravel() for noPython mode.
            Also, NotImplementedError: unicode_type cannot be represented as a Numpy dtype

            Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_unique_str
            '''

            str_set = set(self._data)
            return to_array(str_set)

        return hpat_pandas_series_unique_str_impl

    def hpat_pandas_series_unique_impl(self):
        '''
        Returns sorted unique elements of an array

        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_unique
        '''

        return numpy.unique(self._data)

    return hpat_pandas_series_unique_impl


@overload_method(SeriesType, 'nunique')
def hpat_pandas_series_nunique(self, dropna=True):
    """
    Pandas Series method :meth:`pandas.Series.nunique` implementation.

    Note: Unsupported mixed numeric and string data
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_nunique
    Parameters
    -----------
    self: :obj:`pandas.Series`
        input series
    dropna: :obj:`bool`, default True
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method nunique().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if isinstance(self.data, StringArrayType):
        def hpat_pandas_series_nunique_str_impl(self, dropna=True):
            """
            It is better to merge with Numeric branch
            Unsupported:
                ['aa', np.nan, 'b', 'b', 'cccc', np.nan, 'ddd', 'dd']
                [3, np.nan, 'b', 'b', 5.1, np.nan, 'ddd', 'dd']
            """

            str_set = set(self._data)
            return len(str_set)

        return hpat_pandas_series_nunique_str_impl

    def hpat_pandas_series_nunique_impl(self, dropna=True):
        """
        This function for Numeric data because NumPy dosn't support StringArrayType
        Algo looks a bit ambigous because, currently, set() can not be used with NumPy with Numba JIT
        """

        data_mask_for_nan = numpy.isnan(self._data)
        nan_exists = numpy.any(data_mask_for_nan)
        data_no_nan = self._data[~data_mask_for_nan]
        data_set = set(data_no_nan)
        if dropna or not nan_exists:
            return len(data_set)
        else:
            return len(data_set) + 1

    return hpat_pandas_series_nunique_impl


@overload_method(SeriesType, 'count')
def hpat_pandas_series_count(self, level=None):

    """
    Pandas Series method :meth:`pandas.Series.count` implementation.
    .. only:: developer
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_count
    Parameters

    -----------
    self: :obj:`pandas.Series`
          input series
    level:  :obj:`int` or name
           *unsupported*
    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    _func_name = 'Method count().'

    if not isinstance(self, SeriesType):
        raise TypingError(
            '{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(level, (types.Omitted, types.NoneType)) and level is not None:
        raise TypingError(
            '{} The function only applies with level is None. Given level: {}'.format(_func_name, level))

    if isinstance(self.data, StringArrayType):
        def hpat_pandas_series_count_str_impl(self, level=None):

            return len(self._data)

        return hpat_pandas_series_count_str_impl

    def hpat_pandas_series_count_impl(self, level=None):
        """
        Return number of non-NA/null observations in the object
        Returns number of unique elements in the object
        Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_count
        """
        data_no_nan = self._data[~numpy.isnan(self._data)]
        return len(data_no_nan)

    return hpat_pandas_series_count_impl


@overload_method(SeriesType, 'median')
def hpat_pandas_series_median(self, axis=None, skipna=True, level=None, numeric_only=None):
    """
    Pandas Series method :meth:`pandas.Series.median` implementation.

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_median1
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_median_skipna_default1
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_median_skipna_false1

    Parameters
    -----------
    self: :obj:`pandas.Series`
          input series
    axis: :obj:`int` or :obj:`string` {0 or `index`, None}, default None
        The axis for the function to be applied on.
        *unsupported*
    skipna: :obj:`bool`, default True
        exclude NA/null values when computing the result
    level: :obj:`int` or :obj:`string`, default None
         *unsupported*
    numeric_only: :obj:`bool` or None, default None
         *unsupported*

    Returns
    -------
    :obj:`float` or :obj:`pandas.Series` (if level is specified)
         median of values in the series

    """

    _func_name = 'Method median().'

    if not isinstance(self, SeriesType):
        raise TypingError(
            '{} The object must be a pandas.series. Given self: {}'.format(_func_name, self))

    if not isinstance(self.dtype, types.Number):
        raise TypingError(
            '{} The function only applies to elements that are all numeric. Given data type: {}'.format(_func_name, self.dtype))

    if not (isinstance(axis, (types.Integer, types.UnicodeType, types.Omitted)) or axis is None):
        raise TypingError('{} The axis must be an Integer or a String. Currently unsupported. Given: {}'.format(_func_name, axis))

    if not (isinstance(skipna, (types.Boolean, types.Omitted)) or skipna == True):
        raise TypingError('{} The is_copy must be a boolean. Given: {}'.format(_func_name, skipna))

    if not ((level is None or isinstance(level, types.Omitted))
            or (numeric_only is None or isinstance(numeric_only, types.Omitted))
            or (axis is None or isinstance(axis, types.Omitted))
    ):
        raise TypingError('{} Unsupported parameters. Given level: {}, numeric_only: {}, axis: {}'.format(_func_name, level, numeric_only, axis))


    def hpat_pandas_series_median_impl(self, axis=None, skipna=True, level=None, numeric_only=None):
        if skipna:
            return numpy.nanmedian(self._data)

        return numpy.median(self._data)

    return hpat_pandas_series_median_impl


@overload_method(SeriesType, 'argsort')
def hpat_pandas_series_argsort(self, axis=0, kind='quicksort', order=None):
    """
    Pandas Series method :meth:`pandas.Series.argsort` implementation.

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_argsort1
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_argsort2
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_argsort_noidx
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_argsort_idx
       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_argsort_parallel

    Parameters
    -----------
    self: :class:`pandas.Series`
        input arg
    axis: :obj:`int`
        Has no effect but is accepted for compatibility with numpy.
    kind: {‘mergesort’, ‘quicksort’, ‘heapsort’}, default ‘quicksort’
        Choice of sorting algorithm. See np.sort for more information. ‘mergesort’ is the only stable algorithm
    order: None
        Has no effect but is accepted for compatibility with numpy.

    Returns
    -------
    :obj:`pandas.Series`
         returns: Positions of values within the sort order with -1 indicating nan values.
    """

    _func_name = 'Method argsort().'

    if not isinstance(self, SeriesType):
        raise TypingError('{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

    if not isinstance(self.data.dtype, types.Number):
        raise TypingError('{} Currently function supports only numeric values. Given data type: {}'.format(_func_name,
                                                                                             self.data.dtype))

    if not (isinstance(axis, types.Omitted) or isinstance(axis, types.Integer) or axis == 0):
        raise TypingError('{} Unsupported parameters. Given axis: {}'.format(_func_name, axis))

    if not isinstance(self.index, types.NoneType):
        def hpat_pandas_series_argsort_impl(self, axis=0, kind='quicksort', order=None):

            sort = numpy.argsort(self._data)
            series_data = pandas.Series(self._data)
            na = 0
            for i in series_data.isna():
                if i:
                    na += 1
            id = 0
            i = 0
            list_no_nan = numpy.empty(len(self._data) - na)
            for bool_value in series_data.isna():
                if not bool_value:
                    list_no_nan[id] = self._data[i]
                    id += 1
                i += 1
            sort_no_nan = numpy.argsort(list_no_nan)
            ne_na = sort[:len(sort) - na]
            num = 0
            result = numpy.full((len(self._data)), -1)
            for i in numpy.sort(ne_na):
                result[i] = sort_no_nan[num]
                num += 1

            return pandas.Series(result, self._index)

        return hpat_pandas_series_argsort_impl

    def hpat_pandas_series_argsort_impl(self, axis=0, kind='quicksort', order=None):

        sort = numpy.argsort(self._data)
        series_data = pandas.Series(self._data)
        na = 0
        for i in series_data.isna():
            if i:
                na += 1
        id = 0
        i = 0
        list_no_nan = numpy.empty(len(self._data) - na)
        for bool_value in series_data.isna():
            if not bool_value:
                list_no_nan[id] = self._data[i]
                id += 1
            i += 1
        sort_no_nan = numpy.argsort(list_no_nan)
        ne_na = sort[:len(sort) - na]
        num = 0
        result = numpy.full((len(self._data)), -1)
        for i in numpy.sort(ne_na):
            result[i] = sort_no_nan[num]
            num += 1

        return pandas.Series(result)

    return hpat_pandas_series_argsort_impl



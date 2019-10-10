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

    def hpat_pandas_series_index_impl(self):
        return self._index

    return hpat_pandas_series_index_impl


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value,axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
        raise TypingError('{} Unsupported parameters. Given level: {}, fill_value: {}, axis: {}'.format(_func_name, level, fill_value, axis))

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

    raise TypingError('{} The object must be a pandas.series and argument must be a number. Given: {} and other: {}'.format(_func_name, self, other))


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
            '{} The function only applies to elements that are all numeric. Given data type: {}'.format(_func_name, self.dtype))

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
        raise TypingError(
            '{} The object must be a pandas.series. Given: {}'.format(_func_name, self))

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

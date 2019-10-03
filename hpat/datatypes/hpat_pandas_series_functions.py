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

import operator
import pandas
import numpy

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
    getitem
    iat
    iloc
    len
    loc
    mul
    sub

Implemented methods:
    append
    ne

Implemented attributes:
    values
'''


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


@overload_method(SeriesType, 'min')
def hpat_pandas_series_min(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
    """
    Pandas Series method :meth:`pandas.Series.min` implementation.

    .. only:: developer

       Test: python -m hpat.runtests hpat.tests.test_series.TestSeries.test_series_attr3

    Parameters
    -----------
    axis :  {index (0)}, default: None
               Axis for the function to be applied on.
    skipna:  :obj:`bool`, default: True
                *unsupported*
    level:  :obj:`int` or level name, default: None
                *unsupported*
    numeric_only:  :obj:`bool`, default: None
                *unsupported*
    **kwargs:
                *unsupported*
    Returns
    -------
    :obj:`pandas.Series` or :obj:`int` or :obj:`float`
         returns :obj:`pandas.Series` object or :obj:`int` or :obj:`float`
    """

    _func_name = 'Method min().'

    if not isinstance(self, SeriesType):
        raise TypingError(
            '{} The object must be a pandas.series. Given self: {}'.format(_func_name, self))

    if not isinstance(self.dtype, (types.Integer, types.Float)):
        raise TypingError(
            '{} Currently function supports only numeric values. Given data type: {}'.format(_func_name, self.dtype))

    if not isinstance(skipna, (types.Omitted, bool)):
        raise TypingError(
            '{} The parameter must be a boolean type. Given type skipna: {}'.format(_func_name, type(skipna)))

    if not (isinstance(axis, types.Omitted) or axis is None) \
            or not (isinstance(level, types.Omitted) or level is None) \
            or not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
        raise TypingError(
            '{} Unsupported parameters. Given axis: {}, level: {}, numeric_only: {}'.format(_func_name, axis, level,
                                                                                            numeric_only))

    def hpat_pandas_series_min_impl(self, axis=None, skipna=True, level=None, numeric_only=None):
        if skipna:
            return numpy.nanmin(self._data)
        else:
            return self._data.min()

    return hpat_pandas_series_min_impl


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

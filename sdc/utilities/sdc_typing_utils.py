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

"""

| This file contains SDC utility functions related to typing compilation phase

"""

import numpy
import numba
import sdc

from numba import types
from numba.errors import TypingError
from numba import numpy_support

from sdc.str_arr_type import string_array_type


class SDCLimitation(Exception):
    """Exception to be raised in case of SDC limitation"""
    pass


class TypeChecker:
    """
        Validate object type and raise TypingError if the type is invalid, e.g.:
            Method nsmallest(). The object n
             given: bool
             expected: int
    """
    default_tmpl = '{} The object {}\n given: {}\n expected: {}'

    def __init__(self, func_name):
        """
        Parameters
        ----------
        func_name: :obj:`str`
            name of the function where types checking
        """
        self.func_name = func_name

    def _raise_exc(self, exc_cls, tmpl, *args):
        """
        Generic for raising exception
        Parameters
        ----------
        tmpl: :obj:`any`
            template for exception message
        exc_cls: :obj:`Exception`
            class of the exception to be raised
        args: :obj:`list`
            arguments to render template
        """
        raise exc_cls(tmpl.format(*args))

    def raise_exc(self, data, expected_types, name='', exc_cls=TypingError):
        """
        Raise exception with message about invalid type of parameter
        Parameters
        ----------
        data: :obj:`any`
            real type of the data
        expected_types: :obj:`str`
            expected types inserting directly to the exception
        name: :obj:`str`
            name of the parameter
        exc_cls: :obj:`Exception`
            class of the exception to be raised
        """
        self._raise_exc(self, exc_cls, self.default_tmpl, self.func_name,
                        name, data, expected_types)

    def raise_unsupported_exc(self, data, name='', exc_cls=SDCLimitation):
        """
        Raise exception with message about unsupported parameter
        Parameters
        ----------
        data: :obj:`any`
            real type of the data
        name: :obj:`str`
            name of the parameter
        exc_cls: :obj:`Exception`
            class of the exception to be raised
        """
        tmpl = '{} Unsupported object {}\n given: {}'
        self._raise_exc(self, exc_cls, tmpl, self.func_name, name, data)

    def check(self, data, accepted_type, name=''):
        """
        Check data type belongs to specified type
        Parameters
        ----------
        data: :obj:`any`
            real type of the data
        accepted_type: :obj:`type`
            accepted type
        name: :obj:`str`
            name of the parameter
        """
        if not isinstance(data, accepted_type):
            self.raise_exc(data, accepted_type.__name__, name=name)


def kwsparams2list(params):
    """Convert parameters dict to a list of string of a format 'key=value'"""
    return ['{}={}'.format(k, v) for k, v in params.items()]


def sigparams2list(param_names, defaults):
    """Creates a list of strings of a format 'key=value' from parameter names and default values"""
    return [(f'{param}' if param not in defaults else f'{param}={defaults[param]}') for param in param_names]


def has_literal_value(var, value):
    """Used during typing to check that variable var is a Numba literal value equal to value"""

    if not isinstance(var, types.Literal):
        return False

    if value is None or isinstance(value, type(bool)):
        return var.literal_value is value
    else:
        return var.literal_value == value


def has_python_value(var, value):
    """Used during typing to check that variable var was resolved as Python type and has specific value"""

    if not isinstance(var, type(value)):
        return False

    if value is None or isinstance(value, type(bool)):
        return var is value
    else:
        return var == value


def check_is_numeric_array(type_var):
    """Used during typing to check that type_var is a numeric numpy arrays"""
    return check_is_array_of_dtype(type_var, types.Number)


def check_index_is_numeric(ty_series):
    """Used during typing to check that series has numeric index"""
    return check_is_numeric_array(ty_series.index)


def check_types_comparable(ty_left, ty_right):
    """Used during typing to check that specified types can be compared"""

    if hasattr(ty_left, 'dtype'):
        ty_left = ty_left.dtype

    if hasattr(ty_right, 'dtype'):
        ty_right = ty_right.dtype

    # add the rest of supported types here
    if isinstance(ty_left, types.Number):
        return isinstance(ty_right, types.Number)
    if isinstance(ty_left, types.UnicodeType):
        return isinstance(ty_right, types.UnicodeType)
    if isinstance(ty_left, types.Boolean):
        return isinstance(ty_right, types.Boolean)

    return False


def check_arrays_comparable(ty_left, ty_right):
    """Used during typing to check that underlying arrays of specified types can be compared"""
    return ((ty_left == string_array_type and ty_right == string_array_type)
            or (check_is_numeric_array(ty_left) and check_is_numeric_array(ty_right)))


def check_is_array_of_dtype(type_var, dtype):
    """Used during typing to check that type_var is a numeric numpy array of specific dtype"""
    return isinstance(type_var, types.Array) and isinstance(type_var.dtype, dtype)


def find_common_dtype_from_numpy_dtypes(array_types, scalar_types):
    """Used to find common numba dtype for a sequences of numba dtypes each representing some numpy dtype"""
    np_array_dtypes = [numpy_support.as_dtype(dtype) for dtype in array_types]
    np_scalar_dtypes = [numpy_support.as_dtype(dtype) for dtype in scalar_types]
    np_common_dtype = numpy.find_common_type(np_array_dtypes, np_scalar_dtypes)
    numba_common_dtype = numpy_support.from_dtype(np_common_dtype)

    return numba_common_dtype


def gen_impl_generator(codegen, impl_name):
    """Generate generator of an implementation"""
    def _df_impl_generator(*args, **kwargs):
        func_text, global_vars = codegen(*args, **kwargs)

        loc_vars = {}
        exec(func_text, global_vars, loc_vars)
        _impl = loc_vars[impl_name]

        return _impl

    return _df_impl_generator

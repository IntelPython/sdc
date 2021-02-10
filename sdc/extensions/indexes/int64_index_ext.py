# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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

import numba
import numpy as np
import operator
import pandas as pd

from numba import types, prange
from numba.core import cgutils
from numba.extending import (typeof_impl, NativeValue, intrinsic, box, unbox, lower_builtin, )
from numba.core.errors import TypingError
from numba.core.typing.templates import signature
from numba.core.imputils import impl_ret_untracked, call_getiter

from sdc.datatypes.range_index_type import RangeIndexType
from sdc.datatypes.int64_index_type import Int64IndexType
from sdc.utilities.utils import sdc_overload, sdc_overload_attribute, sdc_overload_method
from sdc.utilities.sdc_typing_utils import TypeChecker, check_is_numeric_array, check_signed_integer
from sdc.functions import numpy_like
from numba.core.boxing import box_array, unbox_array
from sdc.hiframes.api import fix_df_index
from sdc.extensions.indexes.indexes_generic import _check_dtype_param_type


@intrinsic
def init_int64_index(typingctx, data, name=None):

    if not (isinstance(data, types.Array) and data.dtype is types.int64):
        return None
    assert data.ndim == 1, "Index data must be 1-dimensional"

    name = types.none if name is None else name
    is_named = False if name is types.none else True

    def codegen(context, builder, sig, args):
        data_val, name_val = args
        # create series struct and store values
        int64_index = cgutils.create_struct_proxy(
            sig.return_type)(context, builder)

        int64_index.data = data_val

        if is_named:
            if isinstance(name, types.StringLiteral):
                int64_index.name = numba.cpython.unicode.make_string_from_constant(
                    context, builder, types.unicode_type, name.literal_value)
            else:
                int64_index.name = name_val

        if context.enable_nrt:
            context.nrt.incref(builder, sig.args[0], data_val)
            if is_named:
                context.nrt.incref(builder, sig.args[1], name_val)

        return int64_index._getvalue()

    ret_typ = Int64IndexType(data, is_named)
    sig = signature(ret_typ, data, name)
    return sig, codegen


@sdc_overload(pd.Int64Index)
def pd_int64_index_overload(data, dtype=None, copy=False, name=None):

    _func_name = 'pd.Int64Index().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(data, (types.Array, types.List)) and isinstance(data.dtype, types.Integer)
            or isinstance(data, (RangeIndexType, Int64IndexType))):
        ty_checker.raise_exc(data, 'array/list of integers or integer index', 'data')

    dtype_is_number_class = isinstance(dtype, types.NumberClass)
    dtype_is_numpy_signed_int = (check_signed_integer(dtype)
                                 or dtype_is_number_class and check_signed_integer(dtype.dtype))
    dtype_is_unicode_str = isinstance(dtype, (types.UnicodeType, types.StringLiteral))
    if not _check_dtype_param_type(dtype):
        ty_checker.raise_exc(dtype, 'int64 dtype', 'dtype')

    if not (isinstance(copy, (types.NoneType, types.Omitted, types.Boolean)) or copy is False):
        ty_checker.raise_exc(copy, 'bool', 'copy')

    if not (isinstance(name, (types.NoneType, types.Omitted, types.StringLiteral, types.UnicodeType)) or name is None):
        ty_checker.raise_exc(name, 'string or none', 'name')

    is_data_array = isinstance(data, types.Array)
    is_data_index = isinstance(data, (RangeIndexType, Int64IndexType))
    data_dtype_is_int64 = data.dtype is types.int64

    def pd_int64_index_ctor_impl(data, dtype=None, copy=False, name=None):

        if not (dtype is None
                or dtype_is_numpy_signed_int
                or dtype_is_unicode_str and dtype in ('int8', 'int16', 'int32', 'int64')):
            raise ValueError("Incorrect `dtype` passed: expected signed integer")

        if is_data_array == True:  # noqa
            _data = data
        elif is_data_index == True:  # noqa
            _data = data.values
        else:
            _data = fix_df_index(data)._data

        if data_dtype_is_int64 == False:  # noqa
            _data = numpy_like.astype(_data, dtype=types.int64)
        else:
            if copy:
                _data = np.copy(_data)
        return init_int64_index(_data, name)

    return pd_int64_index_ctor_impl


@typeof_impl.register(pd.Int64Index)
def typeof_int64_index(val, c):
    index_data_ty = numba.typeof(val._data)
    is_named = val.name is not None
    return Int64IndexType(index_data_ty, is_named=is_named)


@box(Int64IndexType)
def box_int64_index(typ, val, c):

    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    int64_index = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data = box_array(typ.data, int64_index.data, c)

    # dtype and copy params are not stored so use default values
    dtype = c.pyapi.make_none()
    copy = c.pyapi.bool_from_bool(
        c.context.get_constant(types.bool_, False)
    )

    if typ.is_named:
        name = c.pyapi.from_native_value(types.unicode_type, int64_index.name)
    else:
        name = c.pyapi.make_none()

    res = c.pyapi.call_method(pd_class_obj, "Int64Index", (data, dtype, copy, name))

    c.pyapi.decref(data)
    c.pyapi.decref(dtype)
    c.pyapi.decref(copy)
    c.pyapi.decref(name)
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(Int64IndexType)
def unbox_int64_index(typ, val, c):

    # TODO: support index unboxing with reference to parent in Numba?
    int64_index = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_data = c.pyapi.object_getattr_string(val, "_data")
    int64_index.data = unbox_array(typ.data, index_data, c).value
    c.pyapi.decref(index_data)

    if typ.is_named:
        name_obj = c.pyapi.object_getattr_string(val, "name")
        int64_index.name = numba.cpython.unicode.unbox_unicode_str(
            types.unicode_type, name_obj, c).value
        c.pyapi.decref(name_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(int64_index._getvalue(), is_error=is_error)


@sdc_overload_attribute(Int64IndexType, 'name')
def pd_int64_index_name_overload(self):
    if not isinstance(self, Int64IndexType):
        return None

    is_named_index = self.is_named

    def pd_int64_index_name_impl(self):
        if is_named_index == True:  # noqa
            return self._name
        else:
            return None

    return pd_int64_index_name_impl


@sdc_overload_attribute(Int64IndexType, 'dtype')
def pd_int64_index_dtype_overload(self):
    if not isinstance(self, Int64IndexType):
        return None

    range_index_dtype = self.dtype

    def pd_int64_index_dtype_impl(self):
        return range_index_dtype

    return pd_int64_index_dtype_impl


@sdc_overload_attribute(Int64IndexType, 'values')
def pd_int64_index_values_overload(self):
    if not isinstance(self, Int64IndexType):
        return None

    def pd_int64_index_values_impl(self):
        return self._data

    return pd_int64_index_values_impl


@sdc_overload(len)
def pd_int64_index_len_overload(self):
    if not isinstance(self, Int64IndexType):
        return None

    def pd_int64_index_len_impl(self):
        return len(self._data)

    return pd_int64_index_len_impl


@sdc_overload(operator.contains)
def pd_int64_index_contains_overload(self, val):
    if not isinstance(self, Int64IndexType):
        return None

    _func_name = 'Operator contains().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(val, types.Integer)):
        ty_checker.raise_exc(val, 'integer scalar', 'val')

    def pd_int64_index_contains_impl(self, val):
        # TO-DO: add operator.contains support for arrays in Numba
        found = 0
        for i in prange(len(self._data)):
            if val == self._data[i]:
                found += 1

        return found > 0

    return pd_int64_index_contains_impl


@sdc_overload_method(Int64IndexType, 'copy')
def pd_int64_index_copy_overload(self, name=None, deep=False, dtype=None):
    if not isinstance(self, Int64IndexType):
        return None

    _func_name = 'Method copy().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(name, (types.NoneType, types.Omitted, types.UnicodeType)) or name is None):
        ty_checker.raise_exc(name, 'string or none', 'name')

    if not (isinstance(deep, (types.Omitted, types.Boolean)) or deep is False):
        ty_checker.raise_exc(deep, 'boolean', 'deep')

    if not _check_dtype_param_type(dtype):
        ty_checker.raise_exc(dtype, 'int64 dtype', 'dtype')

    name_is_none = isinstance(name, (types.NoneType, types.Omitted)) or name is None
    keep_name = name_is_none and self.is_named

    def pd_int64_index_copy_impl(self, name=None, deep=False, dtype=None):

        _name = self._name if keep_name == True else name  # noqa
        new_index_data = self._data if not deep else numpy_like.copy(self._data)
        return init_int64_index(new_index_data, _name)

    return pd_int64_index_copy_impl


@sdc_overload(operator.getitem)
def pd_int64_index_getitem_overload(self, idx):
    if not isinstance(self, Int64IndexType):
        return None

    _func_name = 'Operator getitem().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(idx, (types.Integer, types.SliceType))
            or isinstance(idx, (types.Array, types.List)) and isinstance(idx.dtype, (types.Integer, types.Boolean))):
        ty_checker.raise_exc(idx, 'integer, slice, integer array or list', 'idx')

    if isinstance(idx, types.Integer):
        def pd_int64_index_getitem_impl(self, idx):
            index_len = len(self._data)
            # FIXME_Numba#5801: Numba type unification rules make this float
            idx = types.int64((index_len + idx) if idx < 0 else idx)
            if (idx < 0 or idx >= index_len):
                raise IndexError("Int64Index.getitem: index is out of bounds")

            return self._data[idx]

        return pd_int64_index_getitem_impl

    else:
        def pd_int64_index_getitem_impl(self, idx):
            index_data = self._data[idx]
            return pd.Int64Index(index_data, name=self._name)

        return pd_int64_index_getitem_impl


# TO-DO: this and many other impls are generic and should be moved to indexes_generic.py
@sdc_overload(operator.eq)
def pd_int64_index_eq_overload(self, other):

    self_is_index = isinstance(self, Int64IndexType)
    other_is_index = isinstance(other, Int64IndexType)

    if not (self_is_index and other_is_index
            or (self_is_index and (check_is_numeric_array(other) or isinstance(other, types.Number)))
            or ((check_is_numeric_array(self) or isinstance(self, types.Number)) and other_is_index)):
        return None
    one_operand_is_scalar = isinstance(self, types.Number) or isinstance(other, types.Number)

    def pd_int64_index_eq_impl(self, other):

        if one_operand_is_scalar == False:  # noqa
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")

        # names do not matter when comparing pd.Int64Index
        left = self.values if self_is_index == True else self  # noqa
        right = other.values if other_is_index == True else other  # noqa
        return list(left == right)  # FIXME_Numba#5157: result must be np.array, remove list when Numba is fixed

    return pd_int64_index_eq_impl


@sdc_overload(operator.ne)
def pd_int64_index_ne_overload(self, other):

    self_is_index = isinstance(self, Int64IndexType)
    other_is_index = isinstance(other, Int64IndexType)

    if not (self_is_index and other_is_index
            or (self_is_index and (check_is_numeric_array(other) or isinstance(other, types.Number)))
            or ((check_is_numeric_array(self) or isinstance(self, types.Number)) and other_is_index)):
        return None

    def pd_int64_index_ne_impl(self, other):

        eq_res = np.asarray(self == other)  # FIXME_Numba#5157: remove np.asarray and return as list
        return list(~eq_res)

    return pd_int64_index_ne_impl


@lower_builtin(operator.is_, Int64IndexType, Int64IndexType)
def pd_int64_index_is_overload(context, builder, sig, args):

    ty_lhs, ty_rhs = sig.args
    if ty_lhs != ty_rhs:
        return cgutils.false_bit

    lhs, rhs = args
    lhs_ptr = builder.ptrtoint(lhs.operands[0], cgutils.intp_t)
    rhs_ptr = builder.ptrtoint(rhs.operands[0], cgutils.intp_t)
    return builder.icmp_signed('==', lhs_ptr, rhs_ptr)


@lower_builtin('getiter', Int64IndexType)
def pd_int64_index_getiter(context, builder, sig, args):
    """ Returns a new iterator object for Int64IndexType by delegating to array __iter__ """
    (value,) = args
    int64_index = cgutils.create_struct_proxy(sig.args[0])(context, builder, value)
    res = call_getiter(context, builder, sig.args[0].data, int64_index.data)
    return impl_ret_untracked(context, builder, Int64IndexType, res)


@sdc_overload_method(Int64IndexType, 'ravel')
def pd_int64_index_ravel_overload(self, order='C'):
    if not isinstance(self, Int64IndexType):
        return None

    _func_name = 'Method ravel().'

    # np.ravel argument order is not supported in Numba
    if not (isinstance(order, (types.Omitted, types.StringLiteral, types.UnicodeType)) or order == 'C'):
        raise TypingError('{} Unsupported parameters. Given order: {}'.format(_func_name, order))

    def pd_int64_index_ravel_impl(self, order='C'):
        # np.ravel argument order is not supported in Numba
        if order != 'C':
            raise ValueError(f"Unsupported value for argument 'order' (only default 'C' is supported)")

        return self.values

    return pd_int64_index_ravel_impl

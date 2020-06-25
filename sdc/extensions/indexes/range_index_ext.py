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

from numba import types
from numba.core import cgutils
from numba.extending import (typeof_impl, NativeValue, intrinsic, box, unbox, lower_builtin, )

from numba.core.typing.templates import signature
from numba.core.imputils import impl_ret_untracked, call_getiter

from sdc.datatypes.range_index_type import RangeIndexType, RangeIndexDataType
from sdc.datatypes.common_functions import SDCLimitation, _sdc_take
from sdc.utilities.utils import sdc_overload, sdc_overload_attribute, sdc_overload_method
from sdc.utilities.sdc_typing_utils import TypeChecker, check_is_numeric_array
from sdc.functions.numpy_like import getitem_by_mask


def _check_dtype_param_type(dtype):
    """ Returns True is dtype is a valid type for dtype parameter and False otherwise.
        Used in RangeIndex ctor and other methods that take dtype parameter. """

    valid_dtype_types = (types.NoneType, types.Omitted, types.UnicodeType, types.NumberClass)
    return isinstance(dtype, valid_dtype_types) or dtype is None


@intrinsic
def init_range_index(typingctx, data, name=None):
    name = types.none if name is None else name
    is_named = False if name is types.none else True

    def codegen(context, builder, sig, args):
        data_val, name_val = args
        # create series struct and store values
        range_index = cgutils.create_struct_proxy(
            sig.return_type)(context, builder)

        range_index.data = data_val

        if is_named:
            if isinstance(name, types.StringLiteral):
                range_index.name = numba.cpython.unicode.make_string_from_constant(
                    context, builder, types.unicode_type, name.literal_value)
            else:
                range_index.name = name_val

        if context.enable_nrt:
            context.nrt.incref(builder, sig.args[0], data_val)
            if is_named:
                context.nrt.incref(builder, sig.args[1], name_val)

        return range_index._getvalue()

    ret_typ = RangeIndexType(is_named)
    sig = signature(ret_typ, data, name)
    return sig, codegen


@sdc_overload(pd.RangeIndex)
def pd_range_index_overload(start=None, stop=None, step=None, dtype=None, copy=False, name=None, fastpath=None):

    _func_name = 'pd.RangeIndex().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(copy, types.Omitted) or copy is False):
        raise SDCLimitation(f"{_func_name} Unsupported parameter. Given 'copy': {copy}")

    if not (isinstance(copy, types.Omitted) or fastpath is None):
        raise SDCLimitation(f"{_func_name} Unsupported parameter. Given 'fastpath': {fastpath}")

    dtype_is_np_int64 = dtype is types.NumberClass(types.int64)
    dtype_is_unicode_str = isinstance(dtype, (types.UnicodeType, types.StringLiteral))
    if not _check_dtype_param_type(dtype):
        ty_checker.raise_exc(dtype, 'int64 dtype', 'dtype')

    # TODO: support ensure_python_int from pandas.core.dtype.common to handle integers as float params
    if not (isinstance(start, (types.NoneType, types.Omitted, types.Integer)) or start is None):
        ty_checker.raise_exc(start, 'number or none', 'start')
    if not (isinstance(stop, (types.NoneType, types.Omitted, types.Integer)) or stop is None):
        ty_checker.raise_exc(stop, 'number or none', 'stop')
    if not (isinstance(step, (types.NoneType, types.Omitted, types.Integer)) or step is None):
        ty_checker.raise_exc(step, 'number or none', 'step')

    if not (isinstance(name, (types.NoneType, types.Omitted, types.StringLiteral, types.UnicodeType)) or name is None):
        ty_checker.raise_exc(name, 'string or none', 'name')

    if ((isinstance(start, (types.NoneType, types.Omitted)) or start is None)
            and (isinstance(stop, (types.NoneType, types.Omitted)) or stop is None)
            and (isinstance(step, (types.NoneType, types.Omitted)) or step is None)):
        def pd_range_index_ctor_dummy_impl(
                start=None, stop=None, step=None, dtype=None, copy=False, name=None, fastpath=None):
            raise TypeError("RangeIndex(...) must be called with integers")

        return pd_range_index_ctor_dummy_impl

    def pd_range_index_ctor_impl(start=None, stop=None, step=None, dtype=None, copy=False, name=None, fastpath=None):

        if not (dtype is None
                or dtype_is_unicode_str and dtype == 'int64'
                or dtype_is_np_int64):
            raise TypeError("Invalid to pass a non-int64 dtype to RangeIndex")

        _start = types.int64(start) if start is not None else types.int64(0)

        if stop is None:
            _start, _stop = types.int64(0), types.int64(start)
        else:
            _stop = types.int64(stop)

        _step = types.int64(step) if step is not None else types.int64(1)
        if _step == 0:
            raise ValueError("Step must not be zero")

        return init_range_index(range(_start, _stop, _step), name)

    return pd_range_index_ctor_impl


@typeof_impl.register(pd.RangeIndex)
def typeof_range_index(val, c):
    is_named = val.name is not None
    return RangeIndexType(is_named=is_named)


@box(RangeIndexType)
def box_range_index(typ, val, c):

    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    range_index = cgutils.create_struct_proxy(
        typ)(c.context, c.builder, val)
    range_index_data = cgutils.create_struct_proxy(
        RangeIndexDataType)(c.context, c.builder, range_index.data)

    start = c.pyapi.from_native_value(types.int64, range_index_data.start)
    stop = c.pyapi.from_native_value(types.int64, range_index_data.stop)
    step = c.pyapi.from_native_value(types.int64, range_index_data.step)

    # dtype and copy params are not stored so use default values
    dtype = c.pyapi.make_none()
    copy = c.pyapi.bool_from_bool(
        c.context.get_constant(types.bool_, False)
    )

    if typ.is_named:
        name = c.pyapi.from_native_value(types.unicode_type, range_index.name)
    else:
        name = c.pyapi.make_none()

    res = c.pyapi.call_method(pd_class_obj, "RangeIndex", (start, stop, step, dtype, copy, name))

    c.pyapi.decref(start)
    c.pyapi.decref(stop)
    c.pyapi.decref(step)
    c.pyapi.decref(dtype)
    c.pyapi.decref(copy)
    c.pyapi.decref(name)
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(RangeIndexType)
def unbox_range_index(typ, val, c):
    start_obj = c.pyapi.object_getattr_string(val, "start")
    stop_obj = c.pyapi.object_getattr_string(val, "stop")
    step_obj = c.pyapi.object_getattr_string(val, "step")

    # TODO: support range unboxing with reference to parent in Numba?
    range_index = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    range_index_data = cgutils.create_struct_proxy(RangeIndexDataType)(c.context, c.builder)
    range_index_data.start = c.pyapi.long_as_longlong(start_obj)
    range_index_data.stop = c.pyapi.long_as_longlong(stop_obj)
    range_index_data.step = c.pyapi.long_as_longlong(step_obj)
    range_index.data = range_index_data._getvalue()

    if typ.is_named:
        name_obj = c.pyapi.object_getattr_string(val, "name")
        range_index.name = numba.cpython.unicode.unbox_unicode_str(
            types.unicode_type, name_obj, c).value
        c.pyapi.decref(name_obj)

    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)
    c.pyapi.decref(step_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(range_index._getvalue(), is_error=is_error)


@sdc_overload_attribute(RangeIndexType, 'start')
def pd_range_index_start_overload(self):
    if not isinstance(self, RangeIndexType):
        return None

    def pd_range_index_start_impl(self):
        return self._data.start

    return pd_range_index_start_impl


@sdc_overload_attribute(RangeIndexType, 'stop')
def pd_range_index_stop_overload(self):
    if not isinstance(self, RangeIndexType):
        return None

    def pd_range_index_stop_impl(self):
        return self._data.stop

    return pd_range_index_stop_impl


@sdc_overload_attribute(RangeIndexType, 'step')
def pd_range_index_step_overload(self):
    if not isinstance(self, RangeIndexType):
        return None

    def pd_range_index_step_impl(self):
        return self._data.step

    return pd_range_index_step_impl


@sdc_overload_attribute(RangeIndexType, 'name')
def pd_range_index_name_overload(self):
    if not isinstance(self, RangeIndexType):
        return None

    is_named_index = self.is_named
    def pd_range_index_name_impl(self):
        if is_named_index == True:  # noqa
            return self._name
        else:
            return None

    return pd_range_index_name_impl


@sdc_overload_attribute(RangeIndexType, 'dtype')
def pd_range_index_dtype_overload(self):
    if not isinstance(self, RangeIndexType):
        return None

    range_index_dtype = self.dtype

    def pd_range_index_dtype_impl(self):
        return range_index_dtype

    return pd_range_index_dtype_impl


@sdc_overload_attribute(RangeIndexType, 'values')
def pd_range_index_values_overload(self):
    if not isinstance(self, RangeIndexType):
        return None

    def pd_range_index_values_impl(self):
        # TO-DO: add caching when Numba supports writable attributes?
        return np.array(self)

    return pd_range_index_values_impl


@sdc_overload(len)
def pd_range_index_len_overload(self):
    if not isinstance(self, RangeIndexType):
        return None

    def pd_range_index_len_impl(self):
        return len(self._data)

    return pd_range_index_len_impl


@sdc_overload(operator.contains)
def pd_range_index_contains_overload(self, val):
    if not isinstance(self, RangeIndexType):
        return None

    def pd_range_index_contains_impl(self, val):
        return val in self._data

    return pd_range_index_contains_impl


@sdc_overload_method(RangeIndexType, 'copy')
def pd_range_index_copy_overload(self, name=None, deep=False, dtype=None):
    if not isinstance(self, RangeIndexType):
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
    def pd_range_index_copy_impl(self, name=None, deep=False, dtype=None):

        _name = self._name if keep_name == True else name  # noqa
        return init_range_index(self._data, _name)

    return pd_range_index_copy_impl


@sdc_overload(operator.getitem)
def pd_range_index_getitem_overload(self, idx):
    if not isinstance(self, RangeIndexType):
        return None

    _func_name = 'Operator getitem().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(idx, (types.Integer, types.SliceType))
            or isinstance(idx, (types.Array, types.List)) and isinstance(idx.dtype, (types.Integer, types.Boolean))):
        ty_checker.raise_exc(idx, 'integer, slice, integer array or list', 'idx')

    if isinstance(idx, types.Integer):
        def pd_range_index_getitem_impl(self, idx):
            range_len = len(self._data)
            idx = (range_len + idx) if idx < 0 else idx
            if (idx < 0 or idx >= range_len):
                raise IndexError("RangeIndex.getitem: index is out of bounds")
            return self.start + self.step * idx

        return pd_range_index_getitem_impl

    if isinstance(idx, types.SliceType):
        def pd_range_index_getitem_impl(self, idx):
            fix_start, fix_stop, fix_step = idx.indices(len(self._data))
            return pd.RangeIndex(
                self.start + self.step * fix_start,
                self.start + self.step * fix_stop,
                self.step * fix_step,
                name=self._name
            )

        return pd_range_index_getitem_impl

    # returns np.array which is used to represent pandas Int64Index now
    if isinstance(idx, (types.Array, types.List)):

        if isinstance(idx.dtype, types.Integer):
            def pd_range_index_getitem_impl(self, idx):
                return _sdc_take(self, idx)

            return pd_range_index_getitem_impl
        elif isinstance(idx.dtype, types.Boolean):
            def pd_range_index_getitem_impl(self, idx):
                return getitem_by_mask(self, idx)

            return pd_range_index_getitem_impl


@sdc_overload(operator.eq)
def pd_range_index_eq_overload(self, other):

    self_is_range_index = isinstance(self, RangeIndexType)
    other_is_range_index = isinstance(other, RangeIndexType)

    if not (self_is_range_index and other_is_range_index
            or (self_is_range_index and (check_is_numeric_array(other) or isinstance(other, types.Number)))
            or ((check_is_numeric_array(self) or isinstance(self, types.Number) and other_is_range_index))):
        return None
    one_operand_is_scalar = isinstance(self, types.Number) or isinstance(other, types.Number)

    def pd_range_index_eq_impl(self, other):

        if one_operand_is_scalar == False:  # noqa
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")

        # names do not matter when comparing pd.RangeIndex
        left = self.values if self_is_range_index == True else self  # noqa
        right = other.values if other_is_range_index == True else other  # noqa
        return list(left == right)  # FIXME_Numba#5157: result must be np.array, remove list when Numba is fixed

    return pd_range_index_eq_impl


@sdc_overload(operator.ne)
def pd_range_index_ne_overload(self, other):

    self_is_range_index = isinstance(self, RangeIndexType)
    other_is_range_index = isinstance(other, RangeIndexType)

    if not (self_is_range_index and other_is_range_index
            or (self_is_range_index and (check_is_numeric_array(other) or isinstance(other, types.Number)))
            or ((check_is_numeric_array(self) or isinstance(self, types.Number) and other_is_range_index))):
        return None

    def pd_range_index_ne_impl(self, other):

        eq_res = np.asarray(self == other)  # FIXME_Numba#5157: remove np.asarray and return as list
        return list(~eq_res)

    return pd_range_index_ne_impl


@lower_builtin(operator.is_, RangeIndexType, RangeIndexType)
def pd_range_index_is_overload(context, builder, sig, args):

    ty_lhs, ty_rhs = sig.args
    if ty_lhs != ty_rhs:
        return cgutils.false_bit

    lhs, rhs = args
    lhs_ptr = builder.ptrtoint(lhs.operands[0], cgutils.intp_t)
    rhs_ptr = builder.ptrtoint(rhs.operands[0], cgutils.intp_t)
    return builder.icmp_signed('==', lhs_ptr, rhs_ptr)


@lower_builtin('getiter', RangeIndexType)
def pd_range_index_getiter(context, builder, sig, args):
    """ Returns a new iterator object for RangeIndexType by delegating to range.__iter__ """
    (value,) = args
    range_index = cgutils.create_struct_proxy(sig.args[0])(context, builder, value)
    res = call_getiter(context, builder, types.range_state64_type, range_index.data)
    return impl_ret_untracked(context, builder, RangeIndexType, res)

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
from numba.core.errors import TypingError
from numba.core.typing.templates import signature
from numba.core.imputils import impl_ret_untracked, call_getiter

from sdc.datatypes.indexes import PositionalIndexType, RangeIndexType
from sdc.datatypes.indexes.range_index_type import RangeIndexDataType
from sdc.utilities.sdc_typing_utils import SDCLimitation
from sdc.utilities.utils import sdc_overload, sdc_overload_attribute, sdc_overload_method, BooleanLiteral
from sdc.utilities.sdc_typing_utils import (
                                    TypeChecker,
                                    check_signed_integer,
                                    sdc_pandas_index_types,
                                    check_types_comparable,
                                    _check_dtype_param_type,
                                    sdc_indexes_range_like,
                                )
from sdc.functions import numpy_like
from sdc.extensions.indexes.indexes_generic import *


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

    dtype_is_number_class = isinstance(dtype, types.NumberClass)
    dtype_is_numpy_signed_int = (check_signed_integer(dtype)
                                 or dtype_is_number_class and check_signed_integer(dtype.dtype))
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
                or dtype_is_numpy_signed_int
                or dtype_is_unicode_str and dtype in ('int8', 'int16', 'int32', 'int64')):
            raise ValueError("Incorrect `dtype` passed: expected signed integer")

        # TODO: add support of int32 type
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
    # Note: unboxing pd.RangeIndex creates instance of PositionalIndexType
    # if index values are trivial range, but creating pd.RangeIndex() with same
    # parameters via ctor will create instance of RangeIndexType.

    # This is needed for specializing of Series and DF methods on combination of
    # index types and preserving PositionalIndexType as result index type (when possible),
    # since in pandas operations on two range indexes may give:
    # either RangeIndex or Int64Index (in common case)
    is_named = val.name is not None
    if not (val.start == 0 and val.stop > 0 and val.step == 1):
        return RangeIndexType(is_named=is_named)
    else:
        return PositionalIndexType(is_named=is_named)


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

    def pd_range_index_dtype_impl(self):
        return sdc_indexes_attribute_dtype(self)

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

    if not (isinstance(deep, (types.NoneType, types.Omitted, types.Boolean)) or deep is False):
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
            # FIXME_Numba#5801: Numba type unification rules make this float
            idx = types.int64((range_len + idx) if idx < 0 else idx)
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

    if isinstance(idx, (types.Array, types.List)):

        if isinstance(idx.dtype, types.Integer):
            def pd_range_index_getitem_impl(self, idx):
                res_as_arr = self.take(idx)
                return pd.Int64Index(res_as_arr, name=self._name)

            return pd_range_index_getitem_impl
        elif isinstance(idx.dtype, types.Boolean):
            def pd_range_index_getitem_impl(self, idx):
                return numpy_like.getitem_by_mask(self, idx)

            return pd_range_index_getitem_impl


@sdc_overload(operator.eq)
def pd_range_index_eq_overload(self, other):

    _func_name = 'Operator eq.'
    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    self_is_range_index = isinstance(self, RangeIndexType)
    other_is_range_index = isinstance(other, RangeIndexType)

    possible_arg_types = (types.Array, types.Number) + sdc_pandas_index_types
    if not (self_is_range_index and other_is_range_index
            or (self_is_range_index and isinstance(other, possible_arg_types))
            or (isinstance(self, possible_arg_types) and other_is_range_index)):
        return None

    def pd_range_index_eq_impl(self, other):
        return sdc_indexes_operator_eq(self, other)

    return pd_range_index_eq_impl


@sdc_overload(operator.ne)
def pd_range_index_ne_overload(self, other):

    _func_name = 'Operator ne.'
    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    self_is_range_index = isinstance(self, RangeIndexType)
    other_is_range_index = isinstance(other, RangeIndexType)

    possible_arg_types = (types.Array, types.Number) + sdc_pandas_index_types
    if not (self_is_range_index and other_is_range_index
            or (self_is_range_index and isinstance(other, possible_arg_types))
            or (isinstance(self, possible_arg_types) and other_is_range_index)):
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
    res = call_getiter(context, builder, RangeIndexDataType, range_index.data)
    return impl_ret_untracked(context, builder, RangeIndexType, res)


@sdc_overload_method(RangeIndexType, 'ravel')
def pd_range_index_ravel_overload(self, order='C'):
    if not isinstance(self, RangeIndexType):
        return None

    _func_name = 'Method ravel().'

    if not (isinstance(order, (types.Omitted, types.StringLiteral, types.UnicodeType)) or order == 'C'):
        raise TypingError('{} Unsupported parameters. Given order: {}'.format(_func_name, order))

    def pd_range_index_ravel_impl(self, order='C'):
        # np.ravel argument order is not supported in Numba
        if order != 'C':
            raise ValueError(f"Unsupported value for argument 'order' (only default 'C' is supported)")

        return self.values

    return pd_range_index_ravel_impl


@sdc_overload_method(RangeIndexType, 'equals')
def pd_range_index_equals_overload(self, other):
    if not isinstance(self, RangeIndexType):
        return None

    _func_name = 'Method equals().'
    if not isinstance(other, sdc_pandas_index_types):
        raise SDCLimitation(f"{_func_name} Unsupported parameter. Given 'other': {other}")

    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    if isinstance(other, sdc_indexes_range_like):

        def pd_range_index_equals_impl(self, other):

            if len(self) != len(other):
                return False
            if len(self) == 0:
                return True

            if len(self) == 1:
                return self.start == other.start

            return self.start == other.start and self.step == other.step
    else:

        def pd_range_index_equals_impl(self, other):
            return sdc_numeric_indexes_equals(self, other)

    return pd_range_index_equals_impl


@sdc_overload_method(RangeIndexType, 'reindex')
def pd_range_index_reindex_overload(self, target, method=None, level=None, limit=None, tolerance=None):
    if not isinstance(self, RangeIndexType):
        return None

    _func_name = 'Method reindex().'
    if not isinstance(target, sdc_pandas_index_types):
        raise SDCLimitation(f"{_func_name} Unsupported parameter. Given 'target': {target}")

    if not check_types_comparable(self, target):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, target={}'.format(_func_name, self, target))

    def pd_range_index_reindex_impl(self, target, method=None, level=None, limit=None, tolerance=None):
        return sdc_indexes_reindex(self, target=target, method=method, level=level, tolerance=tolerance)

    return pd_range_index_reindex_impl


@sdc_overload_method(RangeIndexType, 'take')
def pd_range_index_take_overload(self, indexes):
    if not isinstance(self, RangeIndexType):
        return None

    _func_name = 'Method take().'
    ty_checker = TypeChecker(_func_name)

    valid_indexes_types = (types.Array, types.List) + sdc_pandas_index_types
    if not (isinstance(indexes, valid_indexes_types) and isinstance(indexes.dtype, types.Integer)):
        ty_checker.raise_exc(indexes, 'array/list of integers or integer index', 'indexes')

    def pd_range_index_take_impl(self, indexes):
        _self = pd.Int64Index(self.values, name=self._name)
        return _self.take(indexes)

    return pd_range_index_take_impl


@sdc_overload_method(RangeIndexType, 'append')
def pd_range_index_append_overload(self, other):
    if not isinstance(self, RangeIndexType):
        return None

    _func_name = 'Method append().'
    ty_checker = TypeChecker(_func_name)

    if not isinstance(other, sdc_pandas_index_types):
        ty_checker.raise_exc(other, 'pandas index', 'other')

    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def pd_range_index_append_impl(self, other):
        int64_index = pd.Int64Index(self.values, name=self._name)
        return int64_index.append(other)

    return pd_range_index_append_impl


@sdc_overload_method(RangeIndexType, 'join')
def pd_range_index_join_overload(self, other, how, level=None, return_indexers=False, sort=False):
    if not isinstance(self, RangeIndexType):
        return None

    _func_name = 'Method join().'
    ty_checker = TypeChecker(_func_name)

    if not isinstance(other, sdc_pandas_index_types):
        ty_checker.raise_exc(other, 'pandas index', 'other')

    if not isinstance(how, types.StringLiteral):
        ty_checker.raise_exc(how, 'string', 'how')
    if not how.literal_value == 'outer':
        raise SDCLimitation(f"{_func_name} Only supporting 'outer' now. Given 'how': {how.literal_value}")

    if not (isinstance(level, (types.Omitted, types.NoneType)) or level is None):
        ty_checker.raise_exc(level, 'None', 'level')

    if not (isinstance(return_indexers, (types.Omitted, BooleanLiteral)) or return_indexers is False):
        ty_checker.raise_exc(return_indexers, 'boolean', 'return_indexers')

    if not (isinstance(sort, (types.Omitted, types.Boolean)) or sort is False):
        ty_checker.raise_exc(sort, 'boolean', 'sort')

    _return_indexers = return_indexers.literal_value

    def pd_range_index_join_impl(self, other, how, level=None, return_indexers=False, sort=False):
        if _return_indexers == True:  # noqa
            return sdc_indexes_join_outer(self, other)
        else:
            joined_index, = sdc_indexes_join_outer(self, other)
            return joined_index

    return pd_range_index_join_impl

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
from numba.extending import (NativeValue, intrinsic, box, unbox, lower_builtin, )
from numba.core.errors import TypingError
from numba.core.typing.templates import signature
from numba.core.imputils import impl_ret_untracked, call_getiter

from sdc.datatypes.indexes import PositionalIndexType, RangeIndexType
from sdc.datatypes.indexes.range_index_type import RangeIndexDataType
from sdc.utilities.sdc_typing_utils import SDCLimitation
from sdc.utilities.utils import sdc_overload, sdc_overload_attribute, sdc_overload_method, BooleanLiteral
from sdc.extensions.indexes.range_index_ext import box_range_index, unbox_range_index
from sdc.utilities.sdc_typing_utils import (
                                TypeChecker,
                                _check_dtype_param_type,
                                check_types_comparable,
                                sdc_pandas_index_types,
                            )
from sdc.extensions.indexes.indexes_generic import *


@intrinsic
def init_positional_index(typingctx, size, name=None):
    name = types.none if name is None else name
    is_named = False if name is types.none else True

    ret_typ = PositionalIndexType(is_named)
    inner_sig = signature(ret_typ.data, size, name)
    def codegen(context, builder, sig, args):
        data_val, name_val = args

        # create positional_index struct and store created instance
        # of RangeIndexType as data member
        positional_index = cgutils.create_struct_proxy(
            sig.return_type)(context, builder)
        positional_index.data = context.compile_internal(
                    builder,
                    lambda size, name: pd.RangeIndex(size, name=name),
                    inner_sig,
                    [data_val, name_val]
        )

        return positional_index._getvalue()

    sig = signature(ret_typ, size, name)
    return sig, codegen


@box(PositionalIndexType)
def box_positional_index(typ, val, c):

    positional_index = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data_range_index = numba.core.cgutils.create_struct_proxy(typ.data)(
        c.context, c.builder, positional_index.data)
    return box_range_index(typ.data, data_range_index._getvalue(), c)


@unbox(PositionalIndexType)
def unbox_positional_index(typ, val, c):

    positional_index = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    res = unbox_range_index(typ.data, val, c)
    positional_index.data = res.value
    is_error = res.is_error

    return NativeValue(positional_index._getvalue(), is_error=is_error)


@sdc_overload_attribute(PositionalIndexType, 'start')
def pd_positional_index_start_overload(self):
    if not isinstance(self, PositionalIndexType):
        return None

    def pd_positional_index_start_impl(self):
        _self = self._data
        return _self.start

    return pd_positional_index_start_impl


@sdc_overload_attribute(PositionalIndexType, 'stop')
def pd_positional_index_stop_overload(self):
    if not isinstance(self, PositionalIndexType):
        return None

    def pd_positional_index_stop_impl(self):
        _self = self._data
        return _self.stop

    return pd_positional_index_stop_impl


@sdc_overload_attribute(PositionalIndexType, 'step')
def pd_positional_index_step_overload(self):
    if not isinstance(self, PositionalIndexType):
        return None

    def pd_positional_index_step_impl(self):
        _self = self._data
        return _self.step

    return pd_positional_index_step_impl


@sdc_overload_attribute(PositionalIndexType, 'name')
def pd_positional_index_name_overload(self):
    if not isinstance(self, PositionalIndexType):
        return None

    is_named_index = self.is_named
    def pd_positional_index_name_impl(self):
        _self = self._data
        if is_named_index == True:  # noqa
            return _self.name
        else:
            return None

    return pd_positional_index_name_impl


@sdc_overload_attribute(PositionalIndexType, 'dtype')
def pd_positional_index_dtype_overload(self):
    if not isinstance(self, PositionalIndexType):
        return None

    def pd_positional_index_dtype_impl(self):
        return sdc_indexes_attribute_dtype(self)

    return pd_positional_index_dtype_impl

@sdc_overload_attribute(PositionalIndexType, 'values')
def pd_positional_index_values_overload(self):
    if not isinstance(self, PositionalIndexType):
        return None

    def pd_positional_index_values_impl(self):
        # TO-DO: add caching when Numba supports writable attributes?
        return np.array(self)

    return pd_positional_index_values_impl

@sdc_overload(len)
def pd_positional_index_len_overload(self):
    if not isinstance(self, PositionalIndexType):
        return None

    def pd_positional_index_len_impl(self):
        return len(self._data)

    return pd_positional_index_len_impl


@sdc_overload(operator.contains)
def pd_range_index_contains_overload(self, val):
    if not isinstance(self, PositionalIndexType):
        return None

    def pd_range_index_contains_impl(self, val):
        _self = self._data
        return val in self._data

    return pd_range_index_contains_impl


@sdc_overload_method(PositionalIndexType, 'copy')
def pd_positional_index_copy_overload(self, name=None, deep=False, dtype=None):
    if not isinstance(self, PositionalIndexType):
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
    def pd_positional_index_copy_impl(self, name=None, deep=False, dtype=None):

        _name = self.name if keep_name == True else name  # noqa
        return init_positional_index(len(self), _name)

    return pd_positional_index_copy_impl


@sdc_overload(operator.getitem)
def pd_positional_index_getitem_overload(self, idx):
    if not isinstance(self, PositionalIndexType):
        return None

    _func_name = 'Operator getitem().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(idx, (types.Integer, types.SliceType))
            or isinstance(idx, (types.Array, types.List)) and isinstance(idx.dtype, (types.Integer, types.Boolean))):
        ty_checker.raise_exc(idx, 'integer, slice, integer array or list', 'idx')

    def pd_positional_index_getitem_impl(self, idx):
        _self = self._data
        return _self[idx]

    return pd_positional_index_getitem_impl


@sdc_overload(operator.eq)
def pd_positional_index_eq_overload(self, other):

    _func_name = 'Operator eq.'
    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    self_is_positional_index = isinstance(self, PositionalIndexType)
    other_is_positional_index = isinstance(other, PositionalIndexType)

    possible_arg_types = (types.Array, types.Number) + sdc_pandas_index_types
    if not (self_is_positional_index and other_is_positional_index
            or (self_is_positional_index and isinstance(other, possible_arg_types))
            or (isinstance(self, possible_arg_types) and other_is_positional_index)):
        return None

    def pd_positional_index_eq_impl(self, other):
        return sdc_indexes_operator_eq(self, other)

    return pd_positional_index_eq_impl


@sdc_overload(operator.ne)
def pd_positional_index_ne_overload(self, other):

    _func_name = 'Operator ne.'
    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    self_is_positional_index = isinstance(self, PositionalIndexType)
    other_is_positional_index = isinstance(other, PositionalIndexType)

    possible_arg_types = (types.Array, types.Number) + sdc_pandas_index_types
    if not (self_is_positional_index and other_is_positional_index
            or (self_is_positional_index and isinstance(other, possible_arg_types))
            or (isinstance(self, possible_arg_types) and other_is_positional_index)):
        return None

    def pd_positional_index_ne_impl(self, other):

        eq_res = np.asarray(self == other)  # FIXME_Numba#5157: remove np.asarray and return as list
        return list(~eq_res)

    return pd_positional_index_ne_impl


@lower_builtin(operator.is_, PositionalIndexType, PositionalIndexType)
def pd_positional_index_is_overload(context, builder, sig, args):

    ty_lhs, ty_rhs = sig.args
    if ty_lhs != ty_rhs:
        return cgutils.false_bit

    lhs, rhs = args
    lhs_ptr = builder.ptrtoint(lhs.operands[0], cgutils.intp_t)
    rhs_ptr = builder.ptrtoint(rhs.operands[0], cgutils.intp_t)
    return builder.icmp_signed('==', lhs_ptr, rhs_ptr)


@lower_builtin('getiter', PositionalIndexType)
def pd_positional_index_getiter(context, builder, sig, args):
    """ Returns a new iterator object for PositionalIndexType by delegating to range.__iter__ """
    (value,) = args
    positional_index = cgutils.create_struct_proxy(sig.args[0])(context, builder, value)
    range_index = cgutils.create_struct_proxy(sig.args[0].data)(context, builder, positional_index.data)
    res = call_getiter(context, builder, RangeIndexDataType, range_index.data)
    return impl_ret_untracked(context, builder, PositionalIndexType, res)





@sdc_overload_method(PositionalIndexType, 'ravel')
def pd_positional_index_ravel_overload(self, order='C'):
    if not isinstance(self, PositionalIndexType):
        return None

    _func_name = 'Method ravel().'
    # np.ravel argument order is not supported in Numba
    if not (isinstance(order, (types.Omitted, types.StringLiteral, types.UnicodeType)) or order == 'C'):
        raise TypingError('{} Unsupported parameters. Given order: {}'.format(_func_name, order))

    def pd_positional_index_ravel_impl(self, order='C'):
        _self = self._data
        return _self.values

    return pd_positional_index_ravel_impl


@sdc_overload_method(PositionalIndexType, 'equals')
def pd_positional_index_equals_overload(self, other):
    if not isinstance(self, PositionalIndexType):
        return None

    _func_name = 'Method equals().'
    if not isinstance(other, sdc_pandas_index_types):
        raise SDCLimitation(f"{_func_name} Unsupported parameter. Given 'other': {other}")

    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def pd_positional_index_equals_impl(self, other):

        _self = self._data
        return _self.equals(other)

    return pd_positional_index_equals_impl


@sdc_overload_method(PositionalIndexType, 'reindex')
def pd_positional_index_reindex_overload(self, target, method=None, level=None, limit=None, tolerance=None):
    if not isinstance(self, PositionalIndexType):
        return None

    _func_name = 'Method reindex().'
    if not isinstance(target, sdc_pandas_index_types):
        raise SDCLimitation(f"{_func_name} Unsupported parameter. Given 'target': {target}")

    if not check_types_comparable(self, target):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, target={}'.format(_func_name, self, target))


    def pd_positional_index_reindex_impl(self, target, method=None, level=None, limit=None, tolerance=None):
        return sdc_indexes_reindex(self, target=target, method=method, level=level, tolerance=tolerance)

    return pd_positional_index_reindex_impl


@sdc_overload_method(PositionalIndexType, 'take')
def pd_positional_index_take_overload(self, indexes):
    if not isinstance(self, PositionalIndexType):
        return None

    _func_name = 'Method take().'
    ty_checker = TypeChecker(_func_name)

    valid_indexes_types = (types.Array, types.List) + sdc_pandas_index_types
    if not (isinstance(indexes, valid_indexes_types) and isinstance(indexes.dtype, types.Integer)):
        ty_checker.raise_exc(indexes, 'array/list of integers or integer index', 'indexes')

    def pd_positional_index_take_impl(self, indexes):
        _self = self._data
        return _self.take(indexes)

    return pd_positional_index_take_impl


@sdc_overload_method(PositionalIndexType, 'append')
def pd_positional_index_append_overload(self, other):
    if not isinstance(self, PositionalIndexType):
        return None

    _func_name = 'Method append().'
    ty_checker = TypeChecker(_func_name)

    if not isinstance(other, sdc_pandas_index_types):
        ty_checker.raise_exc(other, 'pandas index', 'other')

    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def pd_positional_index_append_impl(self, other):
        _self = self._data
        return _self.append(other)

    return pd_positional_index_append_impl


@sdc_overload_method(PositionalIndexType, 'join')
def pd_positional_index_join_overload(self, other, how, level=None, return_indexers=False, sort=False):
    if not isinstance(self, PositionalIndexType):
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
    if isinstance(self, PositionalIndexType) and isinstance(other, PositionalIndexType):

        def pd_indexes_join_positional_impl(self, other, how, level=None, return_indexers=False, sort=False):
            self_size, other_size = len(self), len(other)
            min_size = min(len(self), len(other))
            max_size = max(self_size, other_size)

            joined_index = init_positional_index(max_size)
            if _return_indexers == True:  # noqa
                self_indexer = None if self_size == other_size else np.arange(max_size)
                other_indexer = None if self_size == other_size else np.arange(max_size)
                if self_size > other_size:
                    other_indexer[min_size:] = -1
                elif self_size < other_size:
                    self_indexer[min_size:] = -1

                result = joined_index, self_indexer, other_indexer
            else:
                result = joined_index

            return result

        return pd_indexes_join_positional_impl

    else:

        def pd_positional_index_join_common_impl(self, other, how, level=None, return_indexers=False, sort=False):
            if _return_indexers == True:
                return sdc_indexes_join_outer(self, other)
            else:
                return sdc_indexes_join_outer(self, other)[0]

        return pd_positional_index_join_common_impl

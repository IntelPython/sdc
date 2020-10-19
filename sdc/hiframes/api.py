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


import numpy as np
import pandas as pd

import numba
from numba.core import cgutils, types
from numba.core.typing import signature
from numba.core.typing.templates import infer_global, AbstractTemplate, CallableTemplate
from numba.extending import overload, intrinsic
from numba.core.imputils import (lower_builtin, impl_ret_borrowed)

import sdc
from sdc.str_ext import string_type, list_string_array_type
from sdc.str_arr_ext import (StringArrayType, string_array_type)
from sdc.datatypes.range_index_type import RangeIndexType
from sdc.hiframes.pd_series_ext import (
    SeriesType,
    if_series_to_array_type)
from numba.core.errors import TypingError
from sdc.datatypes.categorical.types import Categorical


def isna(arr, i):
    return False


@overload(isna)
def isna_overload(arr, i):
    if arr == string_array_type:
        return lambda arr, i: sdc.str_arr_ext.str_arr_is_na(arr, i)
    # TODO: support NaN in list(list(str))
    if arr == list_string_array_type:
        return lambda arr, i: False
    # TODO: extend to other types
    assert isinstance(arr, types.Array) or isinstance(arr, types.List)
    dtype = arr.dtype
    if isinstance(dtype, types.Float):
        return lambda arr, i: np.isnan(arr[i])

    # NaT for dt64
    if isinstance(dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = dtype('NaT')
        # TODO: replace with np.isnat
        return lambda arr, i: arr[i] == nat

    # XXX integers don't have nans, extend to boolean
    return lambda arr, i: False


def get_nan_mask(arr):
    return np.zeros(len(arr), np.bool_)


@overload(get_nan_mask)
def get_nan_mask_overload(arr):

    _func_name = "Function: get_nan_mask"
    def get_nan_mask_via_isna_impl(arr):
        len_arr = len(arr)
        res = np.empty(len_arr, dtype=np.bool_)
        for i in numba.prange(len_arr):
            res[i] = isna(arr, i)
        return res

    if isinstance(arr, types.Array):
        dtype = arr.dtype
        if isinstance(dtype, types.Float):
            return lambda arr: np.isnan(arr)
        elif isinstance(dtype, (types.Boolean, types.Integer)):
            return lambda arr: np.zeros(len(arr), np.bool_)
        elif isinstance(dtype, (types.NPDatetime, types.NPTimedelta)):
            return get_nan_mask_via_isna_impl
        else:
            raise TypingError('{} Not implemented for arrays with dtype: {}'.format(_func_name, dtype))
    else:
        # for StringArrayType and other cases rely on isna implementation
        return get_nan_mask_via_isna_impl


# the same as fix_df_array but can be parallel
@numba.generated_jit(nopython=True)
def parallel_fix_df_array(c):  # pragma: no cover
    return lambda c: fix_df_array(c)


def fix_rolling_array(c):  # pragma: no cover
    return c


def dummy_unbox_series(arr):
    return arr


@infer_global(dummy_unbox_series)
class DummyToSeriesType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = if_series_to_array_type(args[0], True)
        return signature(arr, *args)


@lower_builtin(dummy_unbox_series, types.Any)
def dummy_unbox_series_impl(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# this function should be used for getting S._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_data(S):
    return lambda S: S._data


def fix_df_array(column):
    return column


@overload(fix_df_array)
def fix_df_array_overload(column):

    if (isinstance(column, types.List)):
        dtype = column.dtype
        if isinstance(dtype, (types.Number, types.Boolean)):
            def fix_df_array_list_impl(column):
                return np.array(column)
            return fix_df_array_list_impl

        if (dtype == string_type or isinstance(dtype, types.Optional) and dtype.type == string_type):
            def fix_df_array_list_str_impl(column):  # pragma: no cover
                return sdc.str_arr_ext.StringArray(column)
            return fix_df_array_list_str_impl

    if isinstance(column, SeriesType):
        return lambda column: column._data

    if isinstance(column, RangeIndexType):
        return lambda column: np.array(column)

    if isinstance(column, (types.Array, StringArrayType, Categorical)):
        return lambda column: column


def fix_df_index(index, *columns):
    return index


@overload(fix_df_index)
def fix_df_index_overload(index):

    # TO-DO: replace types.none index with separate type, e.g. DefaultIndex
    if (index is None or isinstance(index, types.NoneType)):
        def fix_df_index_impl(index):
            return None

    elif isinstance(index, RangeIndexType):
        def fix_df_index_impl(index):
            return index

    else:
        # default case, transform index the same as df data
        def fix_df_index_impl(index):
            return fix_df_array(index)

    return fix_df_index_impl


@infer_global(fix_rolling_array)
class FixDfRollingArrayType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        column = args[0]
        dtype = column.dtype
        ret_typ = column
        if dtype == types.boolean or isinstance(dtype, types.Integer):
            ret_typ = types.Array(types.float64, 1, 'C')
        # TODO: add other types
        return signature(ret_typ, column)


@lower_builtin(fix_rolling_array, types.Any)  # TODO: replace Any with types
def lower_fix_rolling_array(context, builder, sig, args):
    func = fix_rolling_array_overload(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


def fix_rolling_array_overload(column):
    assert isinstance(column, types.Array)
    dtype = column.dtype
    # convert bool and integer to float64
    if dtype == types.boolean or isinstance(dtype, types.Integer):
        def fix_rolling_array_impl(column):  # pragma: no cover
            return column.astype(np.float64)
    else:
        def fix_rolling_array_impl(column):  # pragma: no cover
            return column
    return fix_rolling_array_impl


@intrinsic
def init_series(typingctx, data, index=None, name=None):
    """Create a Series with provided data, index and name values.
    Used as a single constructor for Series and assigning its data, so that
    optimization passes can look for init_series() to see if underlying
    data has changed, and get the array variables from init_series() args if
    not changed.
    """

    index = types.none if index is None else index
    name = types.none if name is None else name
    is_named = False if name is types.none else True

    def codegen(context, builder, signature, args):
        data_val, index_val, name_val = args
        # create series struct and store values
        series = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        series.data = data_val
        series.index = index_val
        if is_named:
            if isinstance(name, types.StringLiteral):
                series.name = numba.cpython.unicode.make_string_from_constant(
                    context, builder, string_type, name.literal_value)
            else:
                series.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], index_val)
            if is_named:
                context.nrt.incref(builder, signature.args[2], name_val)

        return series._getvalue()

    dtype = data.dtype
    # XXX pd.DataFrame() calls init_series for even Series since it's untyped
    data = if_series_to_array_type(data)
    ret_typ = SeriesType(dtype, data, index, is_named)
    sig = signature(ret_typ, data, index, name)
    return sig, codegen

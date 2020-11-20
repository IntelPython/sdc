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


import operator

import numba
from numba import types
from numba.core import cgutils
from numba.extending import (models, register_model, lower_cast, infer_getattr,
                             type_callable, infer, overload, intrinsic,
                             lower_builtin, overload_method)
from numba.core.typing.templates import (infer_global, AbstractTemplate, signature,
                                    AttributeTemplate, bound_function)
from numba.core.imputils import impl_ret_new_ref, impl_ret_borrowed

from sdc.hiframes.pd_series_ext import SeriesType
from sdc.hiframes.pd_dataframe_type import DataFrameType, ColumnLoc
from sdc.str_ext import string_type


@infer_getattr
class DataFrameAttribute(AttributeTemplate):
    key = DataFrameType

    def generic_resolve(self, df, attr):
        if attr in df.columns:
            ind = df.columns.index(attr)
            arr_typ = df.data[ind]
            return SeriesType(arr_typ.dtype, arr_typ, df.index, True)


def get_structure_maps(col_types, col_names):
    # Define map column name to column location ex. {'A': (0,0), 'B': (1,0), 'C': (0,1)}
    column_loc = {}
    # Store unique types of columns ex. {'int64': (0, [0, 2]), 'float64': (1, [1])}
    data_typs_map = {}
    types_order = []
    type_id = 0
    for i, col_typ in enumerate(col_types):
        col_name = col_names[i]

        if col_typ not in data_typs_map:
            data_typs_map[col_typ] = (type_id, [i])
            # The first column in each type always has 0 index
            column_loc[col_name] = ColumnLoc(type_id, 0)
            types_order.append(col_typ)
            type_id += 1
        else:
            # Get index of column in list of types
            existing_type_id, col_indices = data_typs_map[col_typ]
            col_id = len(col_indices)
            column_loc[col_name] = ColumnLoc(existing_type_id, col_id)
            col_indices.append(i)

    return column_loc, data_typs_map, types_order


# TODO: alias analysis
# this function should be used for getting df._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.njit(no_cpython_wrapper=True, inline='always')
def get_dataframe_data(df, i):
    return df._data[i]


@overload(len)  # TODO: avoid lowering?
def df_len_overload(df):
    if not isinstance(df, DataFrameType):
        return

    if len(df.columns) == 0:  # empty df
        return lambda df: 0
    return lambda df: len(df._data[0][0])


# handle getitem for Tuples because sometimes df._data[i] in
# get_dataframe_data() doesn't translate to 'static_getitem' which causes
# Numba to fail. See TestDataFrame.test_unbox1, TODO: find root cause in Numba
# adapted from typing/builtins.py
@infer_global(operator.getitem)
class GetItemTuple(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        tup, idx = args
        if (not isinstance(tup, types.BaseTuple) or not isinstance(idx, types.IntegerLiteral)):
            return
        idx_val = idx.literal_value
        if isinstance(idx_val, int):
            ret = tup.types[idx_val]
        elif isinstance(idx_val, slice):
            ret = types.BaseTuple.from_types(tup.types[idx_val])

        return signature(ret, *args)


# adapted from targets/tupleobj.py
@lower_builtin(operator.getitem, types.BaseTuple, types.IntegerLiteral)
@lower_builtin(operator.getitem, types.BaseTuple, types.SliceLiteral)
def getitem_tuple_lower(context, builder, sig, args):
    tupty, idx = sig.args
    idx = idx.literal_value
    tup, _ = args
    if isinstance(idx, int):
        if idx < 0:
            idx += len(tupty)
        if not 0 <= idx < len(tupty):
            raise IndexError("cannot index at %d in %s" % (idx, tupty))
        res = builder.extract_value(tup, idx)
    elif isinstance(idx, slice):
        items = cgutils.unpack_tuple(builder, tup)[idx]
        res = context.make_tuple(builder, sig.return_type, items)
    else:
        raise NotImplementedError("unexpected index %r for %s" % (idx, sig.args[0]))
    return impl_ret_borrowed(context, builder, sig.return_type, res)

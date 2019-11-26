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


import operator
from collections import namedtuple
import pandas as pd
import numpy as np

import numba
from numba import types, cgutils
from numba.extending import (models, register_model, lower_cast, infer_getattr,
                             type_callable, infer, overload, make_attribute_wrapper, intrinsic,
                             lower_builtin, overload_method)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
                                    AttributeTemplate, bound_function)
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed

import sdc
from sdc.hiframes.pd_series_ext import SeriesType
from sdc.str_ext import string_type
from sdc.str_arr_ext import string_array_type


class DataFrameType(types.Type):  # TODO: IterableType over column names
    """Temporary type class for DataFrame objects.
    """

    def __init__(self, data=None, index=None, columns=None, has_parent=False):
        self.data = data
        if index is None:
            index = types.none
        self.index = index
        self.columns = columns
        # keeping whether it is unboxed from Python to enable reflection of new
        # columns
        self.has_parent = has_parent
        super(DataFrameType, self).__init__(
            name="dataframe({}, {}, {}, {})".format(data, index, columns, has_parent))

    def copy(self, index=None, has_parent=None):
        # XXX is copy necessary?
        if index is None:
            index = types.none if self.index == types.none else self.index.copy()
        data = tuple(a.copy() for a in self.data)
        if has_parent is None:
            has_parent = self.has_parent
        return DataFrameType(data, index, self.columns, has_parent)

    @property
    def key(self):
        # needed?
        return self.data, self.index, self.columns, self.has_parent

    def unify(self, typingctx, other):
        if (isinstance(other, DataFrameType)
                and len(other.data) == len(self.data)
                and other.columns == self.columns
                and other.has_parent == self.has_parent):
            new_index = types.none
            if self.index != types.none and other.index != types.none:
                new_index = self.index.unify(typingctx, other.index)
            elif other.index != types.none:
                new_index = other.index
            elif self.index != types.none:
                new_index = self.index

            data = tuple(a.unify(typingctx, b) for a, b in zip(self.data, other.data))
            return DataFrameType(data, new_index, self.columns, self.has_parent)

    def is_precise(self):
        return all(a.is_precise() for a in self.data) and self.index.is_precise()

@register_model(DataFrameType)
class DataFrameModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        n_cols = len(fe_type.columns)
        members = [
            ('data', types.Tuple(fe_type.data)),
            ('index', fe_type.index),
            ('columns', types.UniTuple(string_type, n_cols)),
            # for lazy unboxing of df coming from Python (usually argument)
            # list of flags noting which columns and index are unboxed
            # index flag is last
            ('unboxed', types.UniTuple(types.int8, n_cols + 1)),
            ('parent', types.pyobject),
        ]
        super(DataFrameModel, self).__init__(dmm, fe_type, members)

make_attribute_wrapper(DataFrameType, 'data', '_data')
make_attribute_wrapper(DataFrameType, 'index', '_index')
make_attribute_wrapper(DataFrameType, 'columns', '_columns')
make_attribute_wrapper(DataFrameType, 'unboxed', '_unboxed')
make_attribute_wrapper(DataFrameType, 'parent', '_parent')

@infer_getattr
class DataFrameAttribute(AttributeTemplate):
    key = DataFrameType

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, 2)

    def resolve_iat(self, ary):
        return DataFrameIatType(ary)

    def resolve_iloc(self, ary):
        return DataFrameILocType(ary)

    def resolve_loc(self, ary):
        return DataFrameLocType(ary)

    def resolve_values(self, ary):
        # using np.stack(data, 1) for both typing and implementation
        stack_sig = self.context.resolve_function_type(
            np.stack, (types.Tuple(ary.data), types.IntegerLiteral(1)), {})
        return stack_sig.return_type

    @bound_function("df.apply")
    def resolve_apply(self, df, args, kws):
        kws = dict(kws)
        func = args[0] if len(args) > 0 else kws.get('func', None)
        # check axis
        axis = args[1] if len(args) > 1 else kws.get('axis', None)
        if (axis is None or not isinstance(axis, types.IntegerLiteral)
                or axis.literal_value != 1):
            raise ValueError("only apply() with axis=1 supported")

        # using NamedTuple instead of Series, TODO: pass Series
        Row = namedtuple('R', df.columns)

        # the data elements come from getitem of Series to perform conversion
        # e.g. dt64 to timestamp in TestDate.test_ts_map_date2
        dtypes = []
        for arr_typ in df.data:
            series_typ = SeriesType(arr_typ.dtype, arr_typ, df.index, True)
            el_typ = self.context.resolve_function_type(
                operator.getitem, (series_typ, types.int64), {}).return_type
            dtypes.append(el_typ)

        row_typ = types.NamedTuple(dtypes, Row)
        t = func.get_call_type(self.context, (row_typ,), {});
        return signature(SeriesType(t.return_type), *args)

    @bound_function("df.describe")
    def resolve_describe(self, df, args, kws):
        # TODO: use overload
        # TODO: return proper series output
        return signature(string_type, *args)

    def generic_resolve(self, df, attr):
        if attr in df.columns:
            ind = df.columns.index(attr)
            arr_typ = df.data[ind]
            return SeriesType(arr_typ.dtype, arr_typ, df.index, True)


@intrinsic
def init_dataframe(typingctx, *args):
    """Create a DataFrame with provided data, index and columns values.
    Used as a single constructor for DataFrame and assigning its data, so that
    optimization passes can look for init_dataframe() to see if underlying
    data has changed, and get the array variables from init_dataframe() args if
    not changed.
    """

    n_cols = len(args) // 2
    data_typs = tuple(args[:n_cols])
    index_typ = args[n_cols]
    column_names = tuple(a.literal_value for a in args[n_cols + 1:])

    def codegen(context, builder, signature, args):
        in_tup = args[0]
        data_arrs = [builder.extract_value(in_tup, i) for i in range(n_cols)]
        index = builder.extract_value(in_tup, n_cols)
        column_strs = [numba.unicode.make_string_from_constant(
            context, builder, string_type, c) for c in column_names]
        # create dataframe struct and store values
        dataframe = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)

        data_tup = context.make_tuple(
            builder, types.Tuple(data_typs), data_arrs)
        column_tup = context.make_tuple(
            builder, types.UniTuple(string_type, n_cols), column_strs)
        zero = context.get_constant(types.int8, 0)
        unboxed_tup = context.make_tuple(
            builder, types.UniTuple(types.int8, n_cols + 1), [zero] * (n_cols + 1))

        dataframe.data = data_tup
        dataframe.index = index
        dataframe.columns = column_tup
        dataframe.unboxed = unboxed_tup
        dataframe.parent = context.get_constant_null(types.pyobject)

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, index_typ, index)
            for var, typ in zip(data_arrs, data_typs):
                context.nrt.incref(builder, typ, var)
            for var in column_strs:
                context.nrt.incref(builder, string_type, var)

        return dataframe._getvalue()

    ret_typ = DataFrameType(data_typs, index_typ, column_names)
    sig = signature(ret_typ, types.Tuple(args))
    return sig, codegen


@intrinsic
def has_parent(typingctx, df=None):
    def codegen(context, builder, sig, args):
        dataframe = cgutils.create_struct_proxy(
            sig.args[0])(context, builder, value=args[0])
        return cgutils.is_not_null(builder, dataframe.parent)
    return signature(types.bool_, df), codegen


# TODO: alias analysis
# this function should be used for getting df._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_dataframe_data(df, i):

    def _impl(df, i):
        if has_parent(df) and df._unboxed[i] == 0:
            # TODO: make df refcounted to avoid repeated unboxing
            df = sdc.hiframes.boxing.unbox_dataframe_column(df, i)
        return df._data[i]

    return _impl


# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_dataframe_index(df):
    return lambda df: df._index


@intrinsic
def set_df_index(typingctx, df_t, index_t=None):
    """used in very limited cases like distributed to_csv() to create a new
    dataframe with index
    """
    # TODO: make inplace when dfs are full objects

    def codegen(context, builder, signature, args):
        in_df_arg = args[0]
        index = args[1]
        in_df = cgutils.create_struct_proxy(
            signature.args[0])(context, builder, value=in_df_arg)
        # create dataframe struct and store values
        dataframe = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)

        dataframe.data = in_df.data
        dataframe.index = index
        dataframe.columns = in_df.columns
        dataframe.unboxed = in_df.unboxed
        dataframe.parent = in_df.parent

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, index_t, index)
            # TODO: refcount
            context.nrt.incref(builder, types.Tuple(df_t.data), dataframe.data)
            context.nrt.incref(
                builder, types.UniTuple(string_type, len(df_t.columns)),
                dataframe.columns)

        return dataframe._getvalue()

    ret_typ = DataFrameType(df_t.data, index_t, df_t.columns)
    sig = signature(ret_typ, df_t, index_t)
    return sig, codegen


@intrinsic
def set_df_column_with_reflect(typingctx, df, cname, arr):
    """Set df column and reflect to parent Python object
    return a new df.
    """

    col_name = cname.literal_value
    n_cols = len(df.columns)
    new_n_cols = n_cols
    data_typs = df.data
    column_names = df.columns
    index_typ = df.index
    is_new_col = col_name not in df.columns
    col_ind = n_cols
    if is_new_col:
        data_typs += (arr,)
        column_names += (col_name,)
        new_n_cols += 1
    else:
        col_ind = df.columns.index(col_name)
        data_typs = tuple((arr if i == col_ind else data_typs[i])
                          for i in range(n_cols))

    def codegen(context, builder, signature, args):
        df_arg, _, arr_arg = args

        in_dataframe = cgutils.create_struct_proxy(df)(
            context, builder, value=df_arg)

        data_arrs = [builder.extract_value(in_dataframe.data, i)
                     if i != col_ind else arr_arg for i in range(n_cols)]
        if is_new_col:
            data_arrs.append(arr_arg)

        column_strs = [numba.unicode.make_string_from_constant(
            context, builder, string_type, c) for c in column_names]

        zero = context.get_constant(types.int8, 0)
        one = context.get_constant(types.int8, 1)
        unboxed_vals = [builder.extract_value(in_dataframe.unboxed, i)
                        if i != col_ind else one for i in range(n_cols)]

        if is_new_col:
            unboxed_vals.append(one)  # for new data array
        unboxed_vals.append(zero)  # for index

        index = in_dataframe.index
        # create dataframe struct and store values
        out_dataframe = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)

        data_tup = context.make_tuple(
            builder, types.Tuple(data_typs), data_arrs)
        column_tup = context.make_tuple(
            builder, types.UniTuple(string_type, new_n_cols), column_strs)
        unboxed_tup = context.make_tuple(
            builder, types.UniTuple(types.int8, new_n_cols + 1), unboxed_vals)

        out_dataframe.data = data_tup
        out_dataframe.index = index
        out_dataframe.columns = column_tup
        out_dataframe.unboxed = unboxed_tup
        out_dataframe.parent = in_dataframe.parent  # TODO: refcount of parent?

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, index_typ, index)
            for var, typ in zip(data_arrs, data_typs):
                context.nrt.incref(builder, typ, var)
            for var in column_strs:
                context.nrt.incref(builder, string_type, var)

        # set column of parent
        # get boxed array
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()  # acquire GIL
        env_manager = context.get_env_manager(builder)

        if context.enable_nrt:
            context.nrt.incref(builder, arr, arr_arg)

        # call boxing for array data
        # TODO: check complex data types possible for Series for dataframes set column here
        c = numba.pythonapi._BoxContext(context, builder, pyapi, env_manager)
        py_arr = sdc.hiframes.boxing._box_series_data(arr.dtype, arr, arr_arg, c)

        # get column as string obj
        cstr = context.insert_const_string(builder.module, col_name)
        cstr_obj = pyapi.string_from_string(cstr)

        # set column array
        pyapi.object_setitem(in_dataframe.parent, cstr_obj, py_arr)

        pyapi.decref(py_arr)
        pyapi.decref(cstr_obj)

        pyapi.gil_release(gil_state)    # release GIL

        return out_dataframe._getvalue()

    ret_typ = DataFrameType(data_typs, index_typ, column_names, True)
    sig = signature(ret_typ, df, cname, arr)
    return sig, codegen


@overload(len)  # TODO: avoid lowering?
def df_len_overload(df):
    if not isinstance(df, DataFrameType):
        return

    if len(df.columns) == 0:  # empty df
        return lambda df: 0
    return lambda df: len(df._data[0])


@overload(operator.getitem)  # TODO: avoid lowering?
def df_getitem_overload(df, ind):
    if isinstance(df, DataFrameType) and isinstance(ind, types.StringLiteral):
        index = df.columns.index(ind.literal_value)
        return lambda df, ind: sdc.hiframes.api.init_series(df._data[index])


@infer_global(operator.getitem)
class GetItemDataFrame(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        df, idx = args
        # df1 = df[df.A > .5]
        if (isinstance(df, DataFrameType)
                and isinstance(idx, (SeriesType, types.Array))
                and idx.dtype == types.bool_):
            return signature(df, *args)


@infer
class StaticGetItemDataFrame(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        df, idx = args
        if (isinstance(df, DataFrameType) and isinstance(idx, list)
                and all(isinstance(c, str) for c in idx)):
            data_typs = tuple(df.data[df.columns.index(c)] for c in idx)
            columns = tuple(idx)
            ret_typ = DataFrameType(data_typs, df.index, columns)
            return signature(ret_typ, *args)


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


# TODO: handle dataframe pass
# df.ia[] type
class DataFrameIatType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameIatType({})".format(df_type)
        super(DataFrameIatType, self).__init__(name)

# df.iloc[] type


class DataFrameILocType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameILocType({})".format(df_type)
        super(DataFrameILocType, self).__init__(name)

# df.loc[] type


class DataFrameLocType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameLocType({})".format(df_type)
        super(DataFrameLocType, self).__init__(name)


@infer
class StaticGetItemDataFrameIat(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        df, idx = args
        # TODO: handle df.at[]
        if isinstance(df, DataFrameIatType):
            # df.iat[3,1]
            if (isinstance(idx, tuple) and len(idx) == 2
                    and isinstance(idx[0], int)
                    and isinstance(idx[1], int)):
                col_no = idx[1]
                data_typ = df.df_type.data[col_no]
                return signature(data_typ.dtype, *args)


@infer_global(operator.getitem)
class GetItemDataFrameIat(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        df, idx = args
        # TODO: handle df.at[]
        if isinstance(df, DataFrameIatType):
            # df.iat[n,1]
            if (isinstance(idx, types.Tuple) and len(idx) == 2
                    and isinstance(idx.types[1], types.IntegerLiteral)):
                col_no = idx.types[1].literal_value
                data_typ = df.df_type.data[col_no]
                return signature(data_typ.dtype, *args)


@infer_global(operator.setitem)
class SetItemDataFrameIat(AbstractTemplate):
    key = operator.setitem

    def generic(self, args, kws):
        df, idx, val = args
        # TODO: handle df.at[]
        if isinstance(df, DataFrameIatType):
            # df.iat[n,1] = 3
            if (isinstance(idx, types.Tuple) and len(idx) == 2
                    and isinstance(idx.types[1], types.IntegerLiteral)):
                col_no = idx.types[1].literal_value
                data_typ = df.df_type.data[col_no]
                return signature(types.none, data_typ, idx.types[0], val)


@infer_global(operator.getitem)
class GetItemDataFrameLoc(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        df, idx = args
        # handling df.loc similar to df.iloc as temporary hack
        # TODO: handle proper labeled indexes
        if isinstance(df, DataFrameLocType):
            # df1 = df.loc[df.A > .5], df1 = df.loc[np.array([1,2,3])]
            if (isinstance(idx, (SeriesType, types.Array, types.List))
                    and (idx.dtype == types.bool_
                         or isinstance(idx.dtype, types.Integer))):
                return signature(df.df_type, *args)
            # df.loc[1:n]
            if isinstance(idx, types.SliceType):
                return signature(df.df_type, *args)
            # df.loc[1:n,'A']
            if (isinstance(idx, types.Tuple) and len(idx) == 2
                    and isinstance(idx.types[1], types.StringLiteral)):
                col_name = idx.types[1].literal_value
                col_no = df.df_type.columns.index(col_name)
                data_typ = df.df_type.data[col_no]
                # TODO: index
                ret_typ = SeriesType(data_typ.dtype, None, True)
                return signature(ret_typ, *args)


@infer_global(operator.getitem)
class GetItemDataFrameILoc(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        df, idx = args
        if isinstance(df, DataFrameILocType):
            # df1 = df.iloc[df.A > .5], df1 = df.iloc[np.array([1,2,3])]
            if (isinstance(idx, (SeriesType, types.Array, types.List))
                    and (idx.dtype == types.bool_
                         or isinstance(idx.dtype, types.Integer))):
                return signature(df.df_type, *args)
            # df.iloc[1:n]
            if isinstance(idx, types.SliceType):
                return signature(df.df_type, *args)
            # df.iloc[1:n,0]
            if (isinstance(idx, types.Tuple) and len(idx) == 2
                    and isinstance(idx.types[1], types.IntegerLiteral)):
                col_no = idx.types[1].literal_value
                data_typ = df.df_type.data[col_no]
                # TODO: index
                ret_typ = SeriesType(data_typ.dtype, None, True)
                return signature(ret_typ, *args)


@overload_method(DataFrameType, 'merge')
@overload(pd.merge)
def merge_overload(left, right, how='inner', on=None, left_on=None,
                   right_on=None, left_index=False, right_index=False, sort=False,
                   suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):

    # check if on's inferred type is NoneType and store the result,
    # use it later to branch based on the value available at compile time
    onHasNoneType = isinstance(numba.typeof(on), types.NoneType)

    def _impl(left, right, how='inner', on=None, left_on=None,
              right_on=None, left_index=False, right_index=False, sort=False,
              suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        if not onHasNoneType:
            left_on = right_on = on

        return sdc.hiframes.api.join_dummy(left, right, left_on, right_on, how)

    return _impl


@overload(pd.merge_asof)
def merge_asof_overload(left, right, on=None, left_on=None, right_on=None,
                        left_index=False, right_index=False, by=None, left_by=None,
                        right_by=None, suffixes=('_x', '_y'), tolerance=None,
                        allow_exact_matches=True, direction='backward'):

    # check if on's inferred type is NoneType and store the result,
    # use it later to branch based on the value available at compile time
    onHasNoneType = isinstance(numba.typeof(on), types.NoneType)

    def _impl(left, right, on=None, left_on=None, right_on=None,
              left_index=False, right_index=False, by=None, left_by=None,
              right_by=None, suffixes=('_x', '_y'), tolerance=None,
              allow_exact_matches=True, direction='backward'):
        if not onHasNoneType:
            left_on = right_on = on

        return sdc.hiframes.api.join_dummy(left, right, left_on, right_on, 'asof')

    return _impl


@overload_method(DataFrameType, 'pivot_table')
def pivot_table_overload(df, values=None, index=None, columns=None, aggfunc='mean',
                         fill_value=None, margins=False, dropna=True, margins_name='All',
                         _pivot_values=None):

    def _impl(df, values=None, index=None, columns=None, aggfunc='mean',
              fill_value=None, margins=False, dropna=True, margins_name='All',
              _pivot_values=None):

        return sdc.hiframes.pd_groupby_ext.pivot_table_dummy(
            df, values, index, columns, aggfunc, _pivot_values)

    return _impl


@overload(pd.crosstab)
def crosstab_overload(index, columns, values=None, rownames=None, colnames=None,
                      aggfunc=None, margins=False, margins_name='All', dropna=True,
                      normalize=False, _pivot_values=None):
    # TODO: hanlde multiple keys (index args)
    # TODO: handle values and aggfunc options
    def _impl(index, columns, values=None, rownames=None, colnames=None,
              aggfunc=None, margins=False, margins_name='All', dropna=True,
              normalize=False, _pivot_values=None):
        return sdc.hiframes.pd_groupby_ext.crosstab_dummy(
            index, columns, _pivot_values)
    return _impl


@overload(pd.concat)
def concat_overload(objs, axis=0, join='outer', join_axes=None,
                    ignore_index=False, keys=None, levels=None, names=None,
                    verify_integrity=False, sort=None, copy=True):
    # TODO: handle options
    return (lambda objs, axis=0, join='outer', join_axes=None,
            ignore_index=False, keys=None, levels=None, names=None,
            verify_integrity=False, sort=None, copy=True:
            sdc.hiframes.pd_dataframe_ext.concat_dummy(objs, axis))


def concat_dummy(objs):
    return pd.concat(objs)


@infer_global(concat_dummy)
class ConcatDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        objs = args[0]
        axis = 0

        if isinstance(args[1], types.IntegerLiteral):
            axis = args[1].literal_value

        if isinstance(objs, types.List):
            assert axis == 0
            assert isinstance(objs.dtype, (SeriesType, DataFrameType))
            ret_typ = objs.dtype.copy()
            if isinstance(ret_typ, DataFrameType):
                ret_typ = ret_typ.copy(has_parent=False)
            return signature(ret_typ, *args)

        if not isinstance(objs, types.BaseTuple):
            raise ValueError("Tuple argument for pd.concat expected")
        assert len(objs.types) > 0

        if axis == 1:
            data = []
            names = []
            col_no = 0
            for obj in objs.types:
                assert isinstance(obj, (SeriesType, DataFrameType))
                if isinstance(obj, SeriesType):
                    # TODO: handle names of SeriesTypes
                    data.append(obj.data)
                    names.append(str(col_no))
                    col_no += 1
                else:  # DataFrameType
                    # TODO: test
                    data.extend(obj.data)
                    names.extend(obj.columns)

            ret_typ = DataFrameType(tuple(data), None, tuple(names))
            return signature(ret_typ, *args)

        assert axis == 0
        # dataframe case
        if isinstance(objs.types[0], DataFrameType):
            assert all(isinstance(t, DataFrameType) for t in objs.types)
            # get output column names
            all_colnames = []
            for df in objs.types:
                all_colnames.extend(df.columns)
            # TODO: verify how Pandas sorts column names
            all_colnames = sorted(set(all_colnames))

            # get output data types
            all_data = []
            for cname in all_colnames:
                # arguments to the generated function
                arr_args = [df.data[df.columns.index(cname)]
                            for df in objs.types if cname in df.columns]
                # XXX we add arrays of float64 NaNs if a column is missing
                # so add a dummy array of float64 for accurate typing
                # e.g. int to float conversion
                # TODO: fix NA column additions for other types
                if len(arr_args) < len(objs.types):
                    arr_args.append(types.Array(types.float64, 1, 'C'))
                # use sdc.hiframes.api.concat() typer
                concat_typ = sdc.hiframes.api.ConcatType(
                    self.context).generic((types.Tuple(arr_args),), {})
                all_data.append(concat_typ.return_type)

            ret_typ = DataFrameType(tuple(all_data), None, tuple(all_colnames))
            return signature(ret_typ, *args)

        # series case
        elif isinstance(objs.types[0], SeriesType):
            assert all(isinstance(t, SeriesType) for t in objs.types)
            arr_args = [S.data for S in objs.types]
            concat_typ = sdc.hiframes.api.ConcatType(
                self.context).generic((types.Tuple(arr_args),), {})
            ret_typ = SeriesType(concat_typ.return_type.dtype)
            return signature(ret_typ, *args)
        # TODO: handle other iterables like arrays, lists, ...


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(concat_dummy, types.VarArg(types.Any))
def lower_concat_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'sort_values')
def sort_values_overload(df, by, axis=0, ascending=True, inplace=False,
                         kind='quicksort', na_position='last'):

    def _impl(df, by, axis=0, ascending=True, inplace=False, kind='quicksort',
              na_position='last'):

        return sdc.hiframes.pd_dataframe_ext.sort_values_dummy(
            df, by, ascending, inplace)

    return _impl


def sort_values_dummy(df, by, ascending, inplace):
    return df.sort_values(by, ascending=ascending, inplace=inplace)


@infer_global(sort_values_dummy)
class SortDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, by, ascending, inplace = args

        # inplace value
        if isinstance(inplace, sdc.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        ret_typ = df.copy()
        if inplace:
            ret_typ = types.none
        return signature(ret_typ, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(sort_values_dummy, types.VarArg(types.Any))
def lower_sort_values_dummy(context, builder, sig, args):
    if sig.return_type == types.none:
        return

    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


# dummy function to change the df type to have set_parent=True
# used in sort_values(inplace=True) hack
def set_parent_dummy(df):
    return df


@infer_global(set_parent_dummy)
class ParentDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, = args
        ret = DataFrameType(df.data, df.index, df.columns, True)
        return signature(ret, *args)


@lower_builtin(set_parent_dummy, types.VarArg(types.Any))
def lower_set_parent_dummy(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# TODO: jitoptions for overload_method and infer_global
# (no_cpython_wrapper to avoid error for iterator object)
@overload_method(DataFrameType, 'itertuples')
def itertuples_overload(df, index=True, name='Pandas'):

    def _impl(df, index=True, name='Pandas'):
        return sdc.hiframes.pd_dataframe_ext.itertuples_dummy(df)

    return _impl


def itertuples_dummy(df):
    return df


@infer_global(itertuples_dummy)
class ItertuplesDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, = args
        # XXX index handling, assuming implicit index
        assert "Index" not in df.columns
        columns = ('Index',) + df.columns
        arr_types = (types.Array(types.int64, 1, 'C'),) + df.data
        iter_typ = sdc.hiframes.api.DataFrameTupleIterator(columns, arr_types)
        return signature(iter_typ, *args)


@lower_builtin(itertuples_dummy, types.VarArg(types.Any))
def lower_itertuples_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'head')
def head_overload(df, n=5):

    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, n=5):
        return sdc.hiframes.pd_dataframe_ext.head_dummy(df, n)

    return _impl


def head_dummy(df, n):
    return df


@infer_global(head_dummy)
class HeadDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # copy type to sethas_parent False, TODO: data always copied?
        out_df = DataFrameType(df.data, df.index, df.columns)
        return signature(out_df, *args)


@lower_builtin(head_dummy, types.VarArg(types.Any))
def lower_head_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'fillna')
def fillna_overload(df, value=None, method=None, axis=None, inplace=False,
                    limit=None, downcast=None):
    # TODO: handle possible **kwargs options?

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent that has a string column (reflection)
    def _impl(df, value=None, method=None, axis=None, inplace=False,
              limit=None, downcast=None):
        return sdc.hiframes.pd_dataframe_ext.fillna_dummy(df, value, inplace)

    return _impl


def fillna_dummy(df, n):
    return df


@infer_global(fillna_dummy)
class FillnaDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, value, inplace = args
        # inplace value
        if isinstance(inplace, sdc.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        if not inplace:
            # copy type to sethas_parent False, TODO: data always copied?
            out_df = DataFrameType(df.data, df.index, df.columns)
            return signature(out_df, *args)
        return signature(types.none, *args)


@lower_builtin(fillna_dummy, types.VarArg(types.Any))
def lower_fillna_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'reset_index')
def reset_index_overload(df, level=None, drop=False, inplace=False,
                         col_level=0, col_fill=''):

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent (reflection)
    def _impl(df, level=None, drop=False, inplace=False,
              col_level=0, col_fill=''):
        return sdc.hiframes.pd_dataframe_ext.reset_index_dummy(df, inplace)

    return _impl


def reset_index_dummy(df, n):
    return df


@infer_global(reset_index_dummy)
class ResetIndexDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, inplace = args
        # inplace value
        if isinstance(inplace, sdc.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        if not inplace:
            out_df = DataFrameType(df.data, None, df.columns)
            return signature(out_df, *args)
        return signature(types.none, *args)


@lower_builtin(reset_index_dummy, types.VarArg(types.Any))
def lower_reset_index_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'dropna')
def dropna_overload(df, axis=0, how='any', thresh=None, subset=None,
                    inplace=False):

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent (reflection)
    def _impl(df, axis=0, how='any', thresh=None, subset=None, inplace=False):
        return sdc.hiframes.pd_dataframe_ext.dropna_dummy(df, inplace)

    return _impl


def dropna_dummy(df, n):
    return df


@infer_global(dropna_dummy)
class DropnaDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, inplace = args
        # inplace value
        if isinstance(inplace, sdc.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        if not inplace:
            # copy type to set has_parent False
            out_df = DataFrameType(df.data, df.index, df.columns)
            return signature(out_df, *args)
        return signature(types.none, *args)


@lower_builtin(dropna_dummy, types.VarArg(types.Any))
def lower_dropna_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'drop')
def drop_overload(df, labels=None, axis=0, index=None, columns=None,
                  level=None, inplace=False, errors='raise'):

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent (reflection)
    def _impl(df, labels=None, axis=0, index=None, columns=None,
              level=None, inplace=False, errors='raise'):
        return sdc.hiframes.pd_dataframe_ext.drop_dummy(
            df, labels, axis, columns, inplace)

    return _impl


def drop_dummy(df, labels, axis, columns, inplace):
    return df


@infer_global(drop_dummy)
class DropDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, labels, axis, columns, inplace = args

        if labels != types.none:
            if not isinstance(axis, types.IntegerLiteral) or not axis.literal_value == 1:
                raise ValueError("only axis=1 supported for df.drop()")
            if isinstance(labels, types.StringLiteral):
                drop_cols = (labels.literal_value,)
            elif hasattr(labels, 'consts'):
                drop_cols = labels.consts
            else:
                raise ValueError(
                    "constant list of columns expected for labels in df.drop()")
        else:
            assert columns != types.none
            if isinstance(columns, types.StringLiteral):
                drop_cols = (columns.literal_value,)
            elif hasattr(columns, 'consts'):
                drop_cols = columns.consts
            else:
                raise ValueError(
                    "constant list of columns expected for labels in df.drop()")

        assert all(c in df.columns for c in drop_cols)
        new_cols = tuple(c for c in df.columns if c not in drop_cols)
        new_data = tuple(df.data[df.columns.index(c)] for c in new_cols)

        # inplace value
        if isinstance(inplace, sdc.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        # TODO: reflection
        has_parent = False  # df.has_parent
        # if not inplace:
        #     has_parent = False  # data is copied

        out_df = DataFrameType(new_data, df.index, new_cols, has_parent)
        return signature(out_df, *args)


@lower_builtin(drop_dummy, types.VarArg(types.Any))
def lower_drop_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'isna')
def isna_overload(df):

    def _impl(df):
        return sdc.hiframes.pd_dataframe_ext.isna_dummy(df)

    return _impl


def isna_dummy(df):
    return df


@infer_global(isna_dummy)
class IsnaDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, = args
        bool_arr = types.Array(types.bool_, 1, 'C')
        n_cols = len(df.columns)
        out_df = DataFrameType((bool_arr,) * n_cols, df.index, df.columns)
        return signature(out_df, *args)


@lower_builtin(isna_dummy, types.VarArg(types.Any))
def lower_isna_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'astype')
def astype_overload(df, dtype, copy=True, errors='raise'):

    def _impl(df, dtype, copy=True, errors='raise'):
        return sdc.hiframes.pd_dataframe_ext.astype_dummy(df, dtype, copy, errors)

    return _impl


def astype_dummy(df, dtype, copy, errors):
    return df


@infer_global(astype_dummy)
class AstypeDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, dtype, copy, errors = args

        if isinstance(dtype, types.Function) and dtype.typing_key == str:
            out_arr = string_array_type
        else:
            out_arr = types.Array(dtype.dtype, 1, 'C')

        n_cols = len(df.columns)
        out_df = DataFrameType((out_arr,) * n_cols, df.index, df.columns)
        return signature(out_df, *args)


@lower_builtin(astype_dummy, types.VarArg(types.Any))
def lower_astype_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'isin')
def isin_overload(df, values):

    def _impl(df, values):
        return sdc.hiframes.pd_dataframe_ext.isin_dummy(df, values)

    return _impl


def isin_dummy(df, labels, axis, columns, inplace):
    return df


@infer_global(isin_dummy)
class IsinDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, values = args

        bool_arr = types.Array(types.bool_, 1, 'C')
        n_cols = len(df.columns)
        out_df = DataFrameType((bool_arr,) * n_cols, df.index, df.columns)
        return signature(out_df, *args)


@lower_builtin(isin_dummy, types.VarArg(types.Any))
def lower_isin_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'append')
def append_overload(df, other, ignore_index=False, verify_integrity=False,
                    sort=None):
    if isinstance(other, DataFrameType):
        return (lambda df, other, ignore_index=False, verify_integrity=False,
                sort=None: pd.concat((df, other)))

    # TODO: tuple case
    # TODO: non-homogenous build_list case
    if isinstance(other, types.List) and isinstance(other.dtype, DataFrameType):
        return (lambda df, other, ignore_index=False, verify_integrity=False,
                sort=None: pd.concat([df] + other))

    raise ValueError("invalid df.append() input. Only dataframe and list"
                     " of dataframes supported")


@overload_method(DataFrameType, 'pct_change')
def pct_change_overload(df, periods=1, fill_method='pad', limit=None, freq=None):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, periods=1, fill_method='pad', limit=None, freq=None):
        return sdc.hiframes.pd_dataframe_ext.pct_change_dummy(df, periods)

    return _impl


def pct_change_dummy(df, n):
    return df


@infer_global(pct_change_dummy)
class PctChangeDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        float_arr = types.Array(types.float64, 1, 'C')
        data = tuple(float_arr if isinstance(ary.dtype, types.Integer) else ary
                     for ary in df.data)
        out_df = DataFrameType(data, df.index, df.columns)
        return signature(out_df, *args)


@lower_builtin(pct_change_dummy, types.VarArg(types.Any))
def lower_pct_change_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'mean')
def mean_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, axis=None, skipna=None, level=None, numeric_only=None):
        return sdc.hiframes.pd_dataframe_ext.mean_dummy(df)

    return _impl


def mean_dummy(df, n):
    return df


@infer_global(mean_dummy)
class MeanDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # TODO: ignore non-numerics
        # output is float64 series with column names as string index
        out = SeriesType(
            types.float64, types.Array(types.float64, 1, 'C'),
            string_array_type)
        return signature(out, *args)


@lower_builtin(mean_dummy, types.VarArg(types.Any))
def lower_mean_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'median')
def median_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, axis=None, skipna=None, level=None, numeric_only=None):
        return sdc.hiframes.pd_dataframe_ext.median_dummy(df)

    return _impl


def median_dummy(df, n):
    return df


@infer_global(median_dummy)
class MedianDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # TODO: ignore non-numerics
        # output is float64 series with column names as string index
        out = SeriesType(
            types.float64, types.Array(types.float64, 1, 'C'),
            string_array_type)
        return signature(out, *args)


@lower_builtin(median_dummy, types.VarArg(types.Any))
def lower_median_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'std')
def std_overload(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: support ddof
    def _impl(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
        return sdc.hiframes.pd_dataframe_ext.std_dummy(df)

    return _impl


def std_dummy(df, n):
    return df


@infer_global(std_dummy)
class StdDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # TODO: ignore non-numerics
        # output is float64 series with column names as string index
        out = SeriesType(
            types.float64, types.Array(types.float64, 1, 'C'),
            string_array_type)
        return signature(out, *args)


@lower_builtin(std_dummy, types.VarArg(types.Any))
def lower_std_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'var')
def var_overload(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: support ddof
    def _impl(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None):
        return sdc.hiframes.pd_dataframe_ext.var_dummy(df)

    return _impl


def var_dummy(df, n):
    return df


@infer_global(var_dummy)
class VarDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # TODO: ignore non-numerics
        # output is float64 series with column names as string index
        out = SeriesType(
            types.float64, types.Array(types.float64, 1, 'C'),
            string_array_type)
        return signature(out, *args)


@lower_builtin(var_dummy, types.VarArg(types.Any))
def lower_var_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'max')
def max_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, axis=None, skipna=None, level=None, numeric_only=None):
        return sdc.hiframes.pd_dataframe_ext.max_dummy(df)

    return _impl


def max_dummy(df, n):
    return df


@infer_global(max_dummy)
class MaxDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # TODO: ignore non-numerics
        # output is series with column names as string index
        # unify types for output series, TODO: check Pandas unify rules
        dtype = self.context.unify_types(*tuple(d.dtype for d in df.data))
        out = SeriesType(dtype, types.Array(dtype, 1, 'C'), string_array_type)
        return signature(out, *args)


@lower_builtin(max_dummy, types.VarArg(types.Any))
def lower_max_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()

# TODO: refactor since copy of max
@overload_method(DataFrameType, 'min')
def min_overload(df, axis=None, skipna=None, level=None, numeric_only=None):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, axis=None, skipna=None, level=None, numeric_only=None):
        return sdc.hiframes.pd_dataframe_ext.min_dummy(df)

    return _impl


def min_dummy(df, n):
    return df


@infer_global(min_dummy)
class MinDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # TODO: ignore non-numerics
        # output is series with column names as string index
        # unify types for output series, TODO: check Pandas unify rules
        dtype = self.context.unify_types(*tuple(d.dtype for d in df.data))
        out = SeriesType(dtype, types.Array(dtype, 1, 'C'), string_array_type)
        return signature(out, *args)


@lower_builtin(min_dummy, types.VarArg(types.Any))
def lower_min_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'sum')
def sum_overload(df, axis=None, skipna=None, level=None, numeric_only=None,
                 min_count=0):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, axis=None, skipna=None, level=None, numeric_only=None,
              min_count=0):
        return sdc.hiframes.pd_dataframe_ext.sum_dummy(df)

    return _impl


def sum_dummy(df, n):
    return df


@infer_global(sum_dummy)
class SumDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # TODO: ignore non-numerics
        # get series sum output types
        dtypes = tuple(numba.typing.arraydecl.ArrayAttribute.resolve_sum(
            self, SeriesType(d.dtype)).get_call_type(self, (), {}).return_type
            for d in df.data)

        # output is series with column names as string index
        # unify types for output series, TODO: check Pandas unify rules
        dtype = self.context.unify_types(*dtypes)
        out = SeriesType(dtype, types.Array(dtype, 1, 'C'), string_array_type)
        return signature(out, *args)


@lower_builtin(sum_dummy, types.VarArg(types.Any))
def lower_sum_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'prod')
def prod_overload(df, axis=None, skipna=None, level=None, numeric_only=None,
                  min_count=0):
    # TODO: kwargs
    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, axis=None, skipna=None, level=None, numeric_only=None,
              min_count=0):
        return sdc.hiframes.pd_dataframe_ext.prod_dummy(df)

    return _impl


def prod_dummy(df, n):
    return df


@infer_global(prod_dummy)
class ProdDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df = args[0]
        # TODO: ignore non-numerics
        # get series prod output types
        dtypes = tuple(numba.typing.arraydecl.ArrayAttribute.resolve_prod(
            self, SeriesType(d.dtype)).get_call_type(self, (), {}).return_type
            for d in df.data)

        # output is series with column names as string index
        # unify types for output series, TODO: check Pandas unify rules
        dtype = self.context.unify_types(*dtypes)
        out = SeriesType(dtype, types.Array(dtype, 1, 'C'), string_array_type)
        return signature(out, *args)


@lower_builtin(prod_dummy, types.VarArg(types.Any))
def lower_prod_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, 'count')
def count_overload(df, axis=0, level=None, numeric_only=False):
    # TODO: avoid dummy and generate func here when inlining is possible
    def _impl(df, axis=0, level=None, numeric_only=False):
        return sdc.hiframes.pd_dataframe_ext.count_dummy(df)

    return _impl


def count_dummy(df, n):
    return df


@infer_global(count_dummy)
class CountDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        dtype = types.intp
        out = SeriesType(dtype, types.Array(dtype, 1, 'C'), string_array_type)
        return signature(out, *args)


@lower_builtin(count_dummy, types.VarArg(types.Any))
def lower_count_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(
        sig.return_type)(context, builder)
    return out_obj._getvalue()

# TODO: other Pandas versions (0.24 defaults are different than 0.23)
@overload_method(DataFrameType, 'to_csv')
def to_csv_overload(df, path_or_buf=None, sep=',', na_rep='', float_format=None,
                    columns=None, header=True, index=True, index_label=None, mode='w',
                    encoding=None, compression='infer', quoting=None, quotechar='"',
                    line_terminator=None, chunksize=None, tupleize_cols=None,
                    date_format=None, doublequote=True, escapechar=None, decimal='.'):

    # TODO: refactor when objmode() can understand global string constant
    # String output case
    if path_or_buf is None or path_or_buf == types.none:
        def _impl(df, path_or_buf=None, sep=',', na_rep='', float_format=None,
                  columns=None, header=True, index=True, index_label=None,
                  mode='w', encoding=None, compression='infer', quoting=None,
                  quotechar='"', line_terminator=None, chunksize=None,
                  tupleize_cols=None, date_format=None, doublequote=True,
                  escapechar=None, decimal='.'):
            with numba.objmode(D='unicode_type'):
                D = df.to_csv(path_or_buf, sep, na_rep, float_format,
                              columns, header, index, index_label, mode,
                              encoding, compression, quoting, quotechar,
                              line_terminator, chunksize, tupleize_cols,
                              date_format, doublequote, escapechar, decimal)
            return D

        return _impl

    def _impl(df, path_or_buf=None, sep=',', na_rep='', float_format=None,
              columns=None, header=True, index=True, index_label=None, mode='w',
              encoding=None, compression='infer', quoting=None, quotechar='"',
              line_terminator=None, chunksize=None, tupleize_cols=None,
              date_format=None, doublequote=True, escapechar=None, decimal='.'):
        with numba.objmode:
            df.to_csv(path_or_buf, sep, na_rep, float_format,
                      columns, header, index, index_label, mode,
                      encoding, compression, quoting, quotechar,
                      line_terminator, chunksize, tupleize_cols,
                      date_format, doublequote, escapechar, decimal)

    return _impl

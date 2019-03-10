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
import hpat
from hpat.hiframes.pd_series_ext import SeriesType
from hpat.str_ext import string_type


class DataFrameType(types.Type):  # TODO: IterableType over column names
    """Temporary type class for DataFrame objects.
    """
    def __init__(self, data=None, index=None, columns=None, has_parent=False):
        # data is tuple of Array types
        # index is Array type (TODO: Index obj)
        # columns is tuple of strings

        self.data = data
        if index is None:
            index = types.none
        self.index = index
        self.columns = columns
        # keeping whether it is unboxed from Python to enable reflection of new
        # columns
        self.has_parent = has_parent
        super(DataFrameType, self).__init__(
            name="dataframe({}, {}, {}, {})".format(
                data, index, columns, has_parent))

    def copy(self):
        # XXX is copy necessary?
        index = types.none if self.index == types.none else self.index.copy()
        data = tuple(a.copy() for a in self.data)
        return DataFrameType(data, index, self.columns, self.has_parent)

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

            data = tuple(a.unify(typingctx, b) for a,b in zip(self.data, other.data))
            return DataFrameType(
                data, new_index, self.columns, self.has_parent)

    def can_convert_to(self, typingctx, other):
        return
        # overload resolution tries to convert for even get_dataframe_data()
        # TODO: find valid conversion possibilities
        # if (isinstance(other, DataFrameType)
        #         and len(other.data) == len(self.data)
        #         and other.columns == self.columns):
        #     import pdb; pdb.set_trace()
        #     data_convert = max(a.can_convert_to(typingctx, b)
        #                         for a,b in zip(self.data, other.data))
        #     if self.index == types.none and other.index == types.none:
        #         return data_convert
        #     if self.index != types.none and other.index != types.none:
        #         return max(data_convert,
        #             self.index.can_convert_to(typingctx, other.index))

    def is_precise(self):
        return all(a.is_precise() for a in self.data) and self.index.is_precise()


# TODO: encapsulate in meminfo since dataframe is mutible, for example:
# df = pd.DataFrame({'A': A})
# df2 = df
# if cond:
#    df['A'] = B
# df2.A
# TODO: meminfo for reference counting of dataframes
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
        # check lambda
        if not isinstance(func, types.MakeFunctionLiteral):
            raise ValueError("df.apply(): lambda not found")

        # check axis
        axis = args[1] if len(args) > 1 else kws.get('axis', None)
        if (axis is None or not isinstance(axis, types.IntegerLiteral)
                or axis.literal_value != 1):
            raise ValueError("only apply() with axis=1 supported")

        # using NamedTuple instead of Series, TODO: pass Series
        Row = namedtuple('R', df.columns)
        dtype = types.NamedTuple([a.dtype for a in df.data], Row)
        code = func.literal_value.code
        f_ir = numba.ir_utils.get_ir_of_code({'np': np}, code)
        _, f_return_type, _ = numba.compiler.type_inference_stage(
                self.context, f_ir, (dtype,), None)

        return signature(SeriesType(f_return_type), *args)

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

    n_cols = len(args)//2
    data_typs = tuple(args[:n_cols])
    index_typ = args[n_cols]
    column_names = tuple(a.literal_value for a in args[n_cols+1:])

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
            builder, types.UniTuple(types.int8, n_cols+1), [zero]*(n_cols+1))

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
            df = hpat.hiframes.boxing.unbox_dataframe_column(df, i)
        return df._data[i]

    return _impl


# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_dataframe_index(df):
    return lambda df: df._index


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

        unboxed_vals = [builder.extract_value(in_dataframe.unboxed, i)
                        if i != col_ind else arr_arg for i in range(n_cols)]
        zero = context.get_constant(types.int8, 0)
        one = context.get_constant(types.int8, 1)
        if unboxed_vals:
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
            builder, types.UniTuple(types.int8, new_n_cols+1), unboxed_vals)

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
        py_arr = pyapi.from_native_value(arr, arr_arg, env_manager)    # calls boxing

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
    if len(df.columns) == 0:  # empty df
        return lambda df: 0
    return lambda df: len(df._data[0])


@overload(operator.getitem)  # TODO: avoid lowering?
def df_getitem_overload(df, ind):
    if isinstance(df, DataFrameType) and isinstance(ind, types.StringLiteral):
        index = df.columns.index(ind.literal_value)
        return lambda df, ind: hpat.hiframes.api.init_series(df._data[index])


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
        if (not isinstance(tup, types.BaseTuple) or
                not isinstance(idx, types.IntegerLiteral)):
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
        raise NotImplementedError("unexpected index %r for %s"
                                  % (idx, sig.args[0]))
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

    def _impl(left, right, how='inner', on=None, left_on=None,
            right_on=None, left_index=False, right_index=False, sort=False,
            suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        if on is not None:
            left_on = right_on = on

        return hpat.hiframes.api.join_dummy(
            left, right, left_on, right_on, how)

    return _impl

@overload(pd.merge_asof)
def merge_asof_overload(left, right, on=None, left_on=None, right_on=None,
        left_index=False, right_index=False, by=None, left_by=None,
        right_by=None, suffixes=('_x', '_y'), tolerance=None,
        allow_exact_matches=True, direction='backward'):

    def _impl(left, right, on=None, left_on=None, right_on=None,
            left_index=False, right_index=False, by=None, left_by=None,
            right_by=None, suffixes=('_x', '_y'), tolerance=None,
            allow_exact_matches=True, direction='backward'):
        if on is not None:
            left_on = right_on = on

        return hpat.hiframes.api.join_dummy(
            left, right, left_on, right_on, 'asof')

    return _impl

@overload_method(DataFrameType, 'pivot_table')
def pivot_table_overload(df, values=None, index=None, columns=None, aggfunc='mean',
        fill_value=None, margins=False, dropna=True, margins_name='All',
        _pivot_values=None):

    def _impl(df, values=None, index=None, columns=None, aggfunc='mean',
            fill_value=None, margins=False, dropna=True, margins_name='All',
            _pivot_values=None):

        return hpat.hiframes.pd_groupby_ext.pivot_table_dummy(
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
        return hpat.hiframes.pd_groupby_ext.crosstab_dummy(
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
            hpat.hiframes.pd_dataframe_ext.concat_dummy(objs))

def concat_dummy(objs):
    return pd.concat(objs)

@infer_global(concat_dummy)
class ConcatDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        objs = args[0]
        if not isinstance(objs, types.BaseTuple):
            raise ValueError("Tuple argument for pd.concat expected")
        assert len(objs.types) > 0

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
                # use hpat.hiframes.api.concat() typer
                concat_typ = hpat.hiframes.api.ConcatType(
                    self.context).generic((types.Tuple(arr_args),), {})
                all_data.append(concat_typ.return_type)

            ret_typ = DataFrameType(tuple(all_data), None, tuple(all_colnames))
            return signature(ret_typ, *args)
        # series case
        elif isinstance(objs.types[0], SeriesType):
            assert all(isinstance(t, SeriesType) for t in objs.types)
            arr_args = [S.data for S in objs.types]
            concat_typ = hpat.hiframes.api.ConcatType(
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

        return hpat.hiframes.pd_dataframe_ext.sort_values_dummy(
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
        if isinstance(inplace, hpat.utils.BooleanLiteral):
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

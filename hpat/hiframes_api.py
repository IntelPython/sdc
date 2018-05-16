from __future__ import print_function, division, absolute_import

import numba
from numba import ir, ir_utils
from numba import types
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.extending import overload

from hpat.str_ext import StringType, string_type
from hpat.str_arr_ext import StringArray, StringArrayType, string_array_type, unbox_str_series

from numba.typing.arraydecl import get_array_index_type
from numba.targets.imputils import lower_builtin, impl_ret_untracked, impl_ret_borrowed
import numpy as np
from hpat.pd_timestamp_ext import timestamp_series_type, pandas_timestamp_type
import hpat

# from numba.typing.templates import infer_getattr, AttributeTemplate, bound_function
# from numba import types
#
# @infer_getattr
# class PdAttribute(AttributeTemplate):
#     key = types.Array
#
#     @bound_function("array.rolling")
#     def resolve_rolling(self, ary, args, kws):
#         #assert not kws
#         #assert not args
#         return signature(ary.copy(layout='C'), types.intp)


def count(A):  # pragma: no cover
    return 0


def fillna(A):  # pragma: no cover
    return 0


def column_sum(A):  # pragma: no cover
    return 0


def var(A):  # pragma: no cover
    return 0


def std(A):  # pragma: no cover
    return 0


def mean(A):  # pragma: no cover
    return 0


def quantile(A, q):  # pragma: no cover
    return 0


def quantile_parallel(A, q):  # pragma: no cover
    return 0


def str_contains_regex(str_arr, pat):  # pragma: no cover
    return 0


def str_contains_noregex(str_arr, pat):  # pragma: no cover
    return 0

def nunique(A):  # pragma: no cover
    return len(set(A))

def nunique_parallel(A):  # pragma: no cover
    return len(set(A))

@overload(nunique)
def nunique_overload(arr_typ):
    # TODO: extend to other types like datetime?
    def nunique_seq(A):
        return len(set(A))
    return nunique_seq

@overload(nunique_parallel)
def nunique_overload_parallel(arr_typ):
    # TODO: extend to other types
    assert arr_typ == types.Array(types.int64, 1, 'C'), "only in64 for parallel nunique"
    sum_op = hpat.distributed_api.Reduce_Type.Sum.value
    def nunique_par(A):
        uniq_A = hpat.utils.to_array(set(A))
        (send_counts, recv_counts, send_disp, recv_disp, recv_size) = hpat.hiframes_join.get_sendrecv_counts(uniq_A)
        send_arr = np.empty_like(uniq_A)
        recv_arr = np.empty(recv_size, uniq_A.dtype)
        hpat.hiframes_join.shuffle_data(send_counts, recv_counts, send_disp, recv_disp, uniq_A, send_arr, recv_arr)
        loc_nuniq = len(set(recv_arr))
        return hpat.distributed_api.dist_reduce(loc_nuniq, np.int32(sum_op))
    return nunique_par

from numba.typing.arraydecl import _expand_integer


@infer_global(count)
class CountTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(types.intp, *args)


@infer_global(fillna)
class FillNaType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        # args: out_arr, in_arr, value
        return signature(types.none, *args)


@infer_global(column_sum)
class SumType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        # arg: in_arr
        return signature(_expand_integer(args[0].dtype), *args)


# copied from numba/numba/typing/arraydecl.py:563
@infer_global(mean)
@infer_global(var)
@infer_global(std)
class VarDdof1Type(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        if isinstance(args[0].dtype, (types.Integer, types.Boolean)):
            return signature(types.float64, *args)
        return signature(args[0].dtype, *args)


@infer_global(quantile)
@infer_global(quantile_parallel)
class QuantileType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) in [2, 3]
        return signature(types.float64, *args)


@infer_global(str_contains_regex)
@infer_global(str_contains_noregex)
class ContainsType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        # args: str_arr, pat
        return signature(types.Array(types.boolean, 1, 'C'), *args)

# @jit
# def describe(a_count, a_mean, a_std, a_min, q25, q50, q75, a_max):
#     s = "count    "+str(a_count)+"\n"\
#         "mean     "+str(a_mean)+"\n"\
#         "std      "+str(a_std)+"\n"\
#         "min      "+str(a_min)+"\n"\
#         "25%      "+str(q25)+"\n"\
#         "50%      "+str(q50)+"\n"\
#         "75%      "+str(q75)+"\n"\
#         "max      "+str(a_max)+"\n"

# import numba.typing.arraydecl
# from numba import types
# import numba.utils
# import numpy as np

# copied from numba/numba/typing/arraydecl.py:563
# def array_attribute_attachment(self, ary):
#     class VarType(AbstractTemplate):
#         key = "array.var"
#         def generic(self, args, kws):
#             assert not args
#             # only ddof keyword arg is supported
#             assert not kws or kws=={'ddof': types.int64}
#             if isinstance(self.this.dtype, (types.Integer, types.Boolean)):
#                 sig = signature(types.float64, recvr=self.this)
#             else:
#                 sig = signature(self.this.dtype, recvr=self.this)
#             sig.pysig = numba.utils.pysignature(np.var)
#             return sig
#     return types.BoundFunction(VarType, ary)
#
# numba.typing.arraydecl.ArrayAttribute.resolve_var = array_attribute_attachment


@lower_builtin(mean, types.Array)
def lower_column_mean_impl(context, builder, sig, args):
    zero = sig.return_type(0)

    def array_mean_impl(arr):  # pragma: no cover
        count = 0
        s = zero
        for val in arr:
            if not np.isnan(val):
                s += val
                count += 1
        if not count:
            s = np.nan
        else:
            s = s / count
        return s

    res = context.compile_internal(builder, array_mean_impl, sig, args,
                                   locals=dict(s=sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)


# copied from numba/numba/targets/arraymath.py:119
@lower_builtin(var, types.Array)
def array_var(context, builder, sig, args):
    def array_var_impl(arr):  # pragma: no cover
        # TODO: ignore NA
        # Compute the mean
        m = arr.mean()

        # Compute the sum of square diffs
        ssd = 0
        for v in np.nditer(arr):
            ssd += (v.item() - m) ** 2
        return ssd / (arr.size - 1)  # ddof=1 in pandas

    res = context.compile_internal(builder, array_var_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(std, types.Array)
def array_std(context, builder, sig, args):
    def array_std_impl(arry):  # pragma: no cover
        return var(arry) ** 0.5
    res = context.compile_internal(builder, array_std_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


from llvmlite import ir as lir
import quantile_alg
import llvmlite.binding as ll
ll.add_symbol('quantile_parallel', quantile_alg.quantile_parallel)
from numba.targets.arrayobj import make_array
from numba import cgutils
from hpat.distributed_lower import _h5_typ_table


@lower_builtin(quantile, types.npytypes.Array, types.float64)
@lower_builtin(quantile_parallel, types.npytypes.Array, types.float64, types.intp)
def lower_dist_quantile(context, builder, sig, args):

    # store an int to specify data type
    typ_enum = _h5_typ_table[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum))
    assert sig.args[0].ndim == 1

    arr = make_array(sig.args[0])(context, builder, args[0])
    local_size = builder.extract_value(arr.shape, 0)

    if len(args) == 3:
        total_size = args[2]
    else:
        # sequential case
        total_size = local_size

    call_args = [builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
                 local_size, total_size, args[1], builder.load(typ_arg)]

    # array, size, total_size, quantile, type enum
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(64),
                lir.DoubleType(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.DoubleType(), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="quantile_parallel")
    return builder.call(fn, call_args)


def fix_df_array(c):  # pragma: no cover
    return c


def fix_rolling_array(c):  # pragma: no cover
    return c

import pandas as pd
from numba.extending import typeof_impl, unbox, register_model, models, NativeValue
from numba import numpy_support

class PandasDataFrameType(types.Type):
    def __init__(self, col_names, col_types):
        self.col_names = col_names
        self.col_types = col_types
        super(PandasDataFrameType, self).__init__(
            name='PandasDataFrameType({}, {})'.format(col_names, col_types))


@typeof_impl.register(pd.DataFrame)
def typeof_pd_dataframe(val, c):
    col_names = val.columns.tolist()
    # TODO: support other types like string and timestamp
    col_types = get_hiframes_dtypes(val)
    return PandasDataFrameType(col_names, col_types)

register_model(PandasDataFrameType)(models.OpaqueModel)

@unbox(PandasDataFrameType)
def unbox_df(typ, val, c):
    """unbox dataframe to an Opaque pointer
    columns will be extracted later if necessary.
    """
    # XXX: refcount?
    return NativeValue(val)

def get_hiframes_dtypes(df):
    """get hiframe data types for a pandas dataframe
    """
    pd_typ_list = df.dtypes.tolist()
    col_names = df.columns.tolist()
    hi_typs = []
    for cname, typ in zip(col_names, pd_typ_list):
        if typ == np.dtype('O'):
            # XXX assuming the whole column is strings if 1st val is string
            first_val = df[cname][0]
            if isinstance(first_val, str):
                hi_typs.append(string_type)
                continue
            else:
                raise ValueError("data type for column {} not supported".format(cname))
        try:
            t = numpy_support.from_dtype(typ)
            hi_typs.append(t)
        except NotImplementedError:
            raise ValueError("data type for column {} not supported".format(cname))

    return hi_typs

def unbox_df_column(df, col_name, dtype):
    return df[col_name]

@infer_global(unbox_df_column)
class UnBoxDfCol(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        df_typ, col_ind_const, dtype_typ = args[0], args[1], args[2]
        if isinstance(dtype_typ, types.Const):
            if dtype_typ.value == 12:  # FIXME dtype for dt64
                out_typ = types.Array(types.NPDatetime('ns'), 1, 'C')
            elif dtype_typ.value == 11:  # FIXME dtype for str
                out_typ = string_array_type
            else:
                raise ValueError("invalid input dataframe dtype {}".format(dtype_typ.value))
        else:
            out_typ = types.Array(dtype_typ.dtype, 1, 'C')
        # FIXME: last arg should be types.DType?
        return signature(out_typ, *args)

UnBoxDfCol.support_literals = True

def set_df_col(df, cname, arr):
    df[cname] = arr

@infer_global(set_df_col)
class SetDfColInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        assert isinstance(args[1], types.Const)
        return signature(types.none, *args)

SetDfColInfer.support_literals = True

from numba.targets.boxing import box_array

@lower_builtin(set_df_col, PandasDataFrameType, types.Const, types.Array)
def set_df_col_lower(context, builder, sig, args):
    #
    col_name = sig.args[1].value
    arr_typ = sig.args[2]

    # get boxed array
    pyapi = context.get_python_api(builder)
    gil_state = pyapi.gil_ensure()  # acquire GIL
    env_manager = context.get_env_manager(builder)

    if context.enable_nrt:
        context.nrt.incref(builder, arr_typ, args[2])
    py_arr = pyapi.from_native_value(arr_typ, args[2], env_manager)    # calls boxing

    # get column as string obj
    cstr = context.insert_const_string(builder.module, col_name)
    cstr_obj = pyapi.string_from_string(cstr)

    # set column array
    pyapi.object_setitem(args[0], cstr_obj, py_arr)

    pyapi.decref(py_arr)
    pyapi.decref(cstr_obj)

    pyapi.gil_release(gil_state)    # release GIL

    return context.get_dummy_value()


from numba.targets.boxing import unbox_array

@lower_builtin(unbox_df_column, PandasDataFrameType, types.Const, types.Any)
def lower_unbox_df_column(context, builder, sig, args):
    # FIXME: last arg should be types.DType?
    pyapi = context.get_python_api(builder)
    c = numba.pythonapi._UnboxContext(context, builder, pyapi)

    # TODO: refcounts?
    col_ind = sig.args[1].value
    col_name = sig.args[0].col_names[col_ind]
    series_obj = c.pyapi.object_getattr_string(args[0], col_name)
    arr_obj = c.pyapi.object_getattr_string(series_obj, "values")

    if isinstance(sig.args[2], types.Const) and sig.args[2].value == 11:  # FIXME: str code
        native_val = unbox_str_series(string_array_type, arr_obj, c)
    else:
        if isinstance(sig.args[2], types.Const) and sig.args[2].value == 12:  # FIXME: dt64 code
            dtype = types.NPDatetime('ns')
        else:
            dtype = sig.args[2].dtype
        # TODO: error handling like Numba callwrappers.py
        native_val = unbox_array(types.Array(dtype=dtype, ndim=1, layout='C'), arr_obj, c)

    c.pyapi.decref(series_obj)
    c.pyapi.decref(arr_obj)
    return native_val.value



@overload(fix_df_array)
def fix_df_array_overload(column):
    # convert list of numbers/bools to numpy array
    if (isinstance(column, types.List)
            and (isinstance(column.dtype, types.Number)
                 or column.dtype == types.boolean)):
        def fix_df_array_impl(column):  # pragma: no cover
            return np.array(column)
        return fix_df_array_impl
    # convert list of strings to string array
    if isinstance(column, types.List) and isinstance(column.dtype, StringType):
        def fix_df_array_impl(column):  # pragma: no cover
            return StringArray(column)
        return fix_df_array_impl
    # column is array if not list
    assert isinstance(column, (types.Array, StringArrayType))
    def fix_df_array_impl(column):  # pragma: no cover
        return column
    # FIXME: np.array() for everything else?
    return fix_df_array_impl


@overload(fix_rolling_array)
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

# dummy function use to change type of timestamp series to array[dt64]
def ts_series_to_arr_typ(A):
    return A

@infer_global(ts_series_to_arr_typ)
class TsSeriesToArrType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        assert args[0] == timestamp_series_type
        return signature(types.Array(types.NPDatetime('ns'), 1, 'C'), *args)

@lower_builtin(ts_series_to_arr_typ, timestamp_series_type)
def lower_ts_series_to_arr_typ(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])

def ts_series_getitem(arr, ind):
    return arr[ind]

@infer_global(ts_series_getitem)
class TsSeriesGetItemType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        out = get_array_index_type(ary, idx)
        # check result to be dt64 since it might be sliced array
        # replace result with Timestamp
        if out is not None and out.result == types.NPDatetime('ns'):
            return signature(pandas_timestamp_type, ary, out.index)


# register series types for import
@typeof_impl.register(pd.Series)
def typeof_pd_str_series(val, c):
    if len(val) > 0 and isinstance(val[0], str):  # and isinstance(val[-1], str):
        return string_array_type
    if len(val) > 0 and isinstance(val[0], pd.Timestamp):
        return timestamp_series_type


@overload(np.array)
def np_array_array_overload(in_tp):
    if isinstance(in_tp, types.Array):
        return lambda a: a

    if isinstance(in_tp, types.containers.Set):
        # TODO: naive implementation, data from set can probably
        # be copied to array more efficienty
        dtype = in_tp.dtype
        def f(in_set):
            n = len(in_set)
            arr = np.empty(n, dtype)
            i = 0
            for a in in_set:
                arr[i] = a
                i += 1
            return arr
        return f

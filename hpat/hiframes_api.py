from __future__ import print_function, division, absolute_import

from collections import namedtuple

import numba
from numba import ir, ir_utils
from numba.ir_utils import require, mk_unique_var
from numba import types
import numba.array_analysis
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.extending import overload, intrinsic
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed, iternext_impl
from numba.targets.arrayobj import _getitem_array1d
import llvmlite.llvmpy.core as lc

from hpat.str_ext import StringType, string_type
from hpat.str_arr_ext import StringArray, StringArrayType, string_array_type, unbox_str_series, is_str_arr_typ

from numba.typing.arraydecl import get_array_index_type
from numba.targets.imputils import lower_builtin, impl_ret_untracked, impl_ret_borrowed
import numpy as np
from hpat.pd_timestamp_ext import timestamp_series_type, pandas_timestamp_type
import hpat
from hpat.pd_series_ext import SeriesType, BoxedSeriesType, string_series_type, arr_to_series_type, arr_to_boxed_series_type, series_to_array_type

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

def concat(arr_list):
    return pd.concat(arr_list)


# TODO: use infer_global to avoid lowering multiple versions?
@overload(concat)
def concat_overload(arr_list):
    # all string input case
    # TODO: handle numerics to string casting case
    if (isinstance(arr_list, types.UniTuple)
            and is_str_arr_typ(arr_list.dtype)):
        def string_concat_impl(in_arrs):
            # preallocate the output
            num_strs = 0
            num_chars = 0
            for A in in_arrs:
                arr = dummy_unbox_series(A)
                num_strs += len(arr)
                num_chars += hpat.str_arr_ext.num_total_chars(arr)
            out_arr = hpat.str_arr_ext.pre_alloc_string_array(num_strs, num_chars)
            # copy data to output
            curr_str_ind = 0
            curr_chars_ind = 0
            for A in in_arrs:
                arr = dummy_unbox_series(A)
                hpat.str_arr_ext.set_string_array_range(
                    out_arr, arr, curr_str_ind, curr_chars_ind)
                curr_str_ind += len(arr)
                curr_chars_ind += hpat.str_arr_ext.num_total_chars(arr)
            return out_arr

        return string_concat_impl
    for typ in arr_list:
        if not isinstance(typ, types.Array):
            raise ValueError("concat supports only numerical and string arrays")
    # numerical input
    return lambda a: np.concatenate(a)

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
    sum_op = hpat.distributed_api.Reduce_Type.Sum.value
    if is_str_arr_typ(arr_typ):
        int32_typ_enum = np.int32(_h5_typ_table[types.int32])
        char_typ_enum = np.int32(_h5_typ_table[types.uint8])
        def nunique_par_str(A):
            uniq_A = hpat.utils.to_array(set(A))
            n_strs = len(uniq_A)
            n_pes = hpat.distributed_api.get_size()
            # send recv counts for the number of strings
            send_counts, recv_counts = hpat.hiframes_join.send_recv_counts_new(uniq_A)
            send_disp = hpat.hiframes_join.calc_disp(send_counts)
            recv_disp = hpat.hiframes_join.calc_disp(recv_counts)
            recv_size = recv_counts.sum()
            # send recv counts for the number of chars
            send_chars_count, recv_chars_count = set_recv_counts_chars(uniq_A)
            send_disp_chars = hpat.hiframes_join.calc_disp(send_chars_count)
            recv_disp_chars = hpat.hiframes_join.calc_disp(recv_chars_count)
            recv_num_chars = recv_chars_count.sum()
            n_all_chars = hpat.str_arr_ext.num_total_chars(uniq_A)

            # allocate send recv arrays
            send_arr_lens = np.empty(n_strs, np.uint32)  # XXX offset type is uint32
            send_arr_chars = np.empty(n_all_chars, np.uint8)
            recv_arr = hpat.str_arr_ext.pre_alloc_string_array(recv_size, recv_num_chars)

            # populate send array
            tmp_offset = np.zeros(n_pes, dtype=np.int64)
            tmp_offset_chars = np.zeros(n_pes, dtype=np.int64)

            for i in range(n_strs):
                str = uniq_A[i]
                node_id = hash(str) % n_pes
                # lens
                ind = send_disp[node_id] + tmp_offset[node_id]
                send_arr_lens[ind] = len(str)
                tmp_offset[node_id] += 1
                # chars
                indc = send_disp_chars[node_id] + tmp_offset_chars[node_id]
                str_copy(send_arr_chars, indc, str.c_str(), len(str))
                tmp_offset_chars[node_id] += len(str)
                hpat.str_ext.del_str(str)

            # shuffle len values
            offset_ptr = hpat.str_arr_ext.get_offset_ptr(recv_arr)
            c_alltoallv(send_arr_lens.ctypes, offset_ptr, send_counts.ctypes, recv_counts.ctypes, send_disp.ctypes, recv_disp.ctypes, int32_typ_enum)
            data_ptr = hpat.str_arr_ext.get_data_ptr(recv_arr)
            # shuffle char values
            c_alltoallv(send_arr_chars.ctypes, data_ptr, send_chars_count.ctypes, recv_chars_count.ctypes, send_disp_chars.ctypes, recv_disp_chars.ctypes, char_typ_enum)
            convert_len_arr_to_offset(offset_ptr, recv_size)
            loc_nuniq = len(set(recv_arr))
            return hpat.distributed_api.dist_reduce(loc_nuniq, np.int32(sum_op))
        return nunique_par_str

    assert arr_typ == types.Array(types.int64, 1, 'C'), "only in64 for parallel nunique"
    def nunique_par(A):
        uniq_A = hpat.utils.to_array(set(A))
        send_counts, recv_counts = hpat.hiframes_join.send_recv_counts_new(uniq_A)
        send_disp = hpat.hiframes_join.calc_disp(send_counts)
        recv_disp = hpat.hiframes_join.calc_disp(recv_counts)
        recv_size = recv_counts.sum()
        # (send_counts, recv_counts, send_disp, recv_disp,
        #  recv_size) = hpat.hiframes_join.get_sendrecv_counts(uniq_A)
        send_arr = np.empty_like(uniq_A)
        recv_arr = np.empty(recv_size, uniq_A.dtype)
        hpat.hiframes_join.shuffle_data(send_counts, recv_counts, send_disp, recv_disp, uniq_A, send_arr, recv_arr)
        loc_nuniq = len(set(recv_arr))
        return hpat.distributed_api.dist_reduce(loc_nuniq, np.int32(sum_op))
    return nunique_par

c_alltoallv = types.ExternalFunction("c_alltoallv", types.void(types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.int32))
convert_len_arr_to_offset = types.ExternalFunction("convert_len_arr_to_offset", types.void(types.voidptr, types.intp))

# TODO: refactor with join
@numba.njit
def set_recv_counts_chars(key_arr):
    n_pes = hpat.distributed_api.get_size()
    send_counts = np.zeros(n_pes, np.int32)
    recv_counts = np.empty(n_pes, np.int32)
    for i in range(len(key_arr)):
        str = key_arr[i]
        node_id = hash(str) % n_pes
        send_counts[node_id] += len(str)
        hpat.str_ext.del_str(str)
    hpat.distributed_api.alltoall(send_counts, recv_counts, 1)
    return send_counts, recv_counts

@intrinsic
def str_copy(typingctx, buff_arr_typ, ind_typ, str_typ, len_typ=None):
    def codegen(context, builder, sig, args):
        buff_arr, ind, str, len_str = args
        buff_arr = make_array(sig.args[0])(context, builder, buff_arr)
        ptr = builder.gep(buff_arr.data, [ind])
        cgutils.raw_memcpy(builder, ptr, str, len_str, 1)
        return context.get_dummy_value()

    return types.void(types.Array(types.uint8, 1, 'C'), types.intp, types.voidptr, types.intp), codegen


@intrinsic
def str_copy_ptr(typingctx, ptr_typ, ind_typ, str_typ, len_typ=None):
    def codegen(context, builder, sig, args):
        ptr, ind, _str, len_str = args
        ptr = builder.gep(ptr, [ind])
        cgutils.raw_memcpy(builder, ptr, _str, len_str, 1)
        return context.get_dummy_value()

    return types.void(types.voidptr, types.intp, types.voidptr, types.intp), codegen

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

def sort_values(key_arr):  # pragma: no cover
    return

@infer_global(sort_values)
class SortTyping(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *args)


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


@unbox(BoxedSeriesType)
def unbox_series(typ, val, c):
    arr_obj = c.pyapi.object_getattr_string(val, "values")

    if typ.dtype == string_type:
        native_val = unbox_str_series(string_array_type, arr_obj, c)
    else:
        # TODO: error handling like Numba callwrappers.py
        native_val = unbox_array(types.Array(dtype=typ.dtype, ndim=1, layout='C'), arr_obj, c)

    c.pyapi.decref(arr_obj)
    return native_val


def to_series_type(arr):
    return arr

@infer_global(to_series_type)
class ToSeriesType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        if isinstance(arr, BoxedSeriesType):
            series_type = SeriesType(arr.dtype, 1, 'C')
        else:
            series_type = arr_to_series_type(arr)
        assert series_type is not None, "unknown type for pd.Series: {}".format(arr)
        return signature(series_type, arr)

@lower_builtin(to_series_type, types.Any)
def to_series_dummy_impl(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])

# dummy func to convert input series to array type
def dummy_unbox_series(arr):
    return arr

@infer_global(dummy_unbox_series)
class DummyToSeriesType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = series_to_array_type(args[0])
        return signature(arr, *args)

@lower_builtin(dummy_unbox_series, types.Any)
def dummy_unbox_series_impl(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


@overload(fix_df_array)
def fix_df_array_overload(column):
    # convert list of numbers/bools to numpy array
    if (isinstance(column, types.List)
            and (isinstance(column.dtype, types.Number)
                 or column.dtype == types.boolean)):
        def fix_df_array_list_impl(column):  # pragma: no cover
            return np.array(column)
        return fix_df_array_list_impl

    # convert list of strings to string array
    if isinstance(column, types.List) and isinstance(column.dtype, StringType):
        def fix_df_array_str_impl(column):  # pragma: no cover
            return StringArray(column)
        return fix_df_array_str_impl

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
        assert args[0] == timestamp_series_type or args[0] == types.Array(types.int64, 1, 'C')
        return signature(types.Array(types.NPDatetime('ns'), 1, 'C'), *args)

@lower_builtin(ts_series_to_arr_typ, timestamp_series_type)
@lower_builtin(ts_series_to_arr_typ, types.Array(types.int64, 1, 'C'))
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

def ts_binop_wrapper(op, arr, other):  # pragma: no cover
    return

@infer_global(ts_binop_wrapper)
class TsBinopWrapperType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [op, ts_arr, other] = args
        assert isinstance(op, types.Const) and isinstance(op.value, str)
        assert ts_arr == types.Array(types.NPDatetime('ns'), 1, 'C')
        # TODO: extend to other types like string array
        assert other == string_type
        # TODO: examine all possible ops
        out = types.Array(types.NPDatetime('ns'), 1, 'C')
        if op.value in ['==', '!=', '>=', '>', '<=', '<']:
            out = types.Array(types.boolean, 1, 'C')
        return signature(out, *args)

TsBinopWrapperType.support_literals = True


# register series types for import
@typeof_impl.register(pd.Series)
def typeof_pd_str_series(val, c):
    # TODO: replace timestamp type
    if len(val) > 0 and isinstance(val[0], pd.Timestamp):
        return timestamp_series_type

    if len(val) > 0 and isinstance(val[0], str):  # and isinstance(val[-1], str):
        arr_typ = string_array_type
    else:
        arr_typ = numba.typing.typeof._typeof_ndarray(val.values, c)

    return arr_to_boxed_series_type(arr_typ)


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

class DataFrameTupleIterator(types.SimpleIteratorType):
    """
    Type class for itertuples of dataframes.
    """

    def __init__(self, col_names, arr_typs):
        self.array_types = arr_typs
        self.col_names = col_names
        name_args = ["{}={}".format(col_names[i], arr_typs[i])
                                                for i in range(len(col_names))]
        name = "itertuples({})".format(",".join(name_args))
        py_ntup = namedtuple('Pandas', col_names)
        yield_type = types.NamedTuple([_get_series_dtype(a) for a in arr_typs], py_ntup)
        super(DataFrameTupleIterator, self).__init__(name, yield_type)

def _get_series_dtype(arr_typ):
    # values of datetimeindex are extracted as Timestamp
    if arr_typ == types.Array(types.NPDatetime('ns'), 1, 'C'):
        return pandas_timestamp_type
    return arr_typ.dtype

def get_itertuples():  # pragma: no cover
    pass

@infer_global(get_itertuples)
class TypeIterTuples(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) % 2 == 0, "name and column pairs expected"
        col_names = [a.value for a in args[:len(args)//2]]
        arr_types = args[len(args)//2:]
        # XXX index handling, assuming implicit index
        assert "Index" not in col_names[0]
        col_names = ['Index'] + col_names
        arr_types = [types.Array(types.int64, 1, 'C')] + list(arr_types)
        iter_typ = DataFrameTupleIterator(col_names, arr_types)
        return signature(iter_typ, *args)

TypeIterTuples.support_literals = True

@register_model(DataFrameTupleIterator)
class DataFrameTupleIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        # XXX array_types[0] is implicit index
        members = ([('index', types.EphemeralPointer(types.uintp))]
            + [('array{}'.format(i), arr) for i, arr in enumerate(fe_type.array_types[1:])])
        super(DataFrameTupleIteratorModel, self).__init__(dmm, fe_type, members)

@lower_builtin(get_itertuples, types.VarArg(types.Any))
def get_itertuples_impl(context, builder, sig, args):
    arrays = args[len(args)//2:]
    array_types = sig.args[len(sig.args)//2:]

    iterobj = context.make_helper(builder, sig.return_type)

    zero = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero)

    iterobj.index = indexptr

    for i, arr in enumerate(arrays):
        setattr(iterobj, "array{}".format(i), arr)

    # Incref arrays
    if context.enable_nrt:
        for arr, arr_typ in zip(arrays, array_types):
            context.nrt.incref(builder, arr_typ, arr)

    res = iterobj._getvalue()

    # Note: a decref on the iterator will dereference all internal MemInfo*
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@lower_builtin('getiter', DataFrameTupleIterator)
def getiter_itertuples(context, builder, sig, args):
    # simply return the iterator
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])

# similar to iternext of ArrayIterator
@lower_builtin('iternext', DataFrameTupleIterator)
@iternext_impl
def iternext_itertuples(context, builder, sig, args, result):
    iterty, = sig.args
    it, = args

    # TODO: support string arrays
    iterobj = context.make_helper(builder, iterty, value=it)
    # first array type is implicit int index
    # use len() to support string arrays
    len_sig = signature(types.intp, iterty.array_types[1])
    nitems = context.compile_internal(builder, lambda a: len(a), len_sig, [iterobj.array0])
    # ary = make_array(iterty.array_types[1])(context, builder, value=iterobj.array0)
    # nitems, = cgutils.unpack_tuple(builder, ary.shape, count=1)

    index = builder.load(iterobj.index)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        values = [index]  # XXX implicit int index
        for i, arr_typ in enumerate(iterty.array_types[1:]):
            arr_ptr = getattr(iterobj, "array{}".format(i))

            if arr_typ == types.Array(types.NPDatetime('ns'), 1, 'C'):
                getitem_sig = signature(pandas_timestamp_type, arr_typ, types.intp)
                val = context.compile_internal(builder,
                    lambda a,i: hpat.pd_timestamp_ext.convert_datetime64_to_timestamp(np.int64(a[i])),
                        getitem_sig, [arr_ptr, index])
            else:
                getitem_sig = signature(arr_typ.dtype, arr_typ, types.intp)
                val = context.compile_internal(builder, lambda a,i: a[i], getitem_sig, [arr_ptr, index])
            # arr = make_array(arr_typ)(context, builder, value=arr_ptr)
            # val = _getitem_array1d(context, builder, arr_typ, arr, index,
            #                      wraparound=False)
            values.append(val)

        value = context.make_tuple(builder, iterty.yield_type, values)
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


# TODO: move this to array analysis
# the namedtuples created by get_itertuples-iternext-pair_first don't have
# shapes created in array analysis
# def _analyze_op_static_getitem(self, scope, equiv_set, expr):
#     var = expr.value
#     typ = self.typemap[var.name]
#     if not isinstance(typ, types.BaseTuple):
#         return self._index_to_shape(scope, equiv_set, expr.value, expr.index_var)
#     try:
#         shape = equiv_set._get_shape(var)
#         require(isinstance(expr.index, int) and expr.index < len(shape))
#         return shape[expr.index], []
#     except:
#         pass

#     return None

# numba.array_analysis.ArrayAnalysis._analyze_op_static_getitem = _analyze_op_static_getitem

# FIXME: fix array analysis for tuples in general
def _analyze_op_pair_first(self, scope, equiv_set, expr):
    # make dummy lhs since we don't have access to lhs
    typ = self.typemap[expr.value.name].first_type
    if not isinstance(typ, types.NamedTuple):
        return None
    lhs = ir.Var(scope, mk_unique_var("tuple_var"), expr.loc)
    self.typemap[lhs.name] = typ
    rhs = ir.Expr.pair_first(expr.value, expr.loc)
    lhs_assign = ir.Assign(rhs, lhs, expr.loc)
    #(shape, post) = self._gen_shape_call(equiv_set, lhs, typ.count, )
    var = lhs
    out = []
    size_vars = []
    ndims = typ.count
    for i in range(ndims):
        # get size: Asize0 = A_sh_attr[0]
        size_var = ir.Var(var.scope, mk_unique_var(
                            "{}_size{}".format(var.name, i)), var.loc)
        getitem = ir.Expr.static_getitem(lhs, i, None, var.loc)
        self.calltypes[getitem] = None
        out.append(ir.Assign(getitem, size_var, var.loc))
        self._define(equiv_set, size_var, types.intp, getitem)
        size_vars.append(size_var)
    shape = tuple(size_vars)
    return shape, [lhs_assign] + out

numba.array_analysis.ArrayAnalysis._analyze_op_pair_first = _analyze_op_pair_first

from __future__ import print_function, division, absolute_import

from collections import namedtuple
import pandas as pd
import datetime

import numba
from numba import ir, ir_utils
from numba.ir_utils import require, mk_unique_var
from numba import types
import numba.array_analysis
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing.arraydecl import _expand_integer
from numba.extending import overload, intrinsic
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed, iternext_impl
from numba.targets.arrayobj import _getitem_array1d
from numba.targets.boxing import box_array, unbox_array
import llvmlite.llvmpy.core as lc

from hpat.str_ext import StringType, string_type
from hpat.str_arr_ext import (StringArray, StringArrayType, string_array_type,
    unbox_str_series, is_str_arr_typ, box_str_arr)

from numba.typing.arraydecl import get_array_index_type
from numba.targets.imputils import lower_builtin, impl_ret_untracked, impl_ret_borrowed
import numpy as np
from hpat.pd_timestamp_ext import (pandas_timestamp_type, datetime_date_type,
    set_df_datetime_date_lower, unbox_datetime_date_array)
import hpat
from hpat.pd_series_ext import (SeriesType, BoxedSeriesType,
    string_series_type, if_arr_to_series_type, arr_to_boxed_series_type,
    series_to_array_type, if_series_to_array_type, dt_index_series_type,
    date_series_type, UnBoxedSeriesType)

from hpat.hiframes_sort import (
    alloc_shuffle_metadata, data_alloc_shuffle_metadata, alltoallv,
    alltoallv_tup, finalize_shuffle_meta, finalize_data_shuffle_meta,
    update_shuffle_meta, update_data_shuffle_meta, finalize_data_shuffle_meta,
    )
from hpat.hiframes_join import write_send_buff

# quantile imports?
from llvmlite import ir as lir
import quantile_alg
import llvmlite.binding as ll
ll.add_symbol('quantile_parallel', quantile_alg.quantile_parallel)
ll.add_symbol('nth_sequential', quantile_alg.nth_sequential)
ll.add_symbol('nth_parallel', quantile_alg.nth_parallel)
from numba.targets.arrayobj import make_array
from numba import cgutils
from hpat.utils import _numba_to_c_type_map

# boxing/unboxing
from numba.extending import typeof_impl, unbox, register_model, models, NativeValue, box
from numba import numpy_support

nth_sequential = types.ExternalFunction("nth_sequential",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.int32))

nth_parallel = types.ExternalFunction("nth_parallel",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.int32))

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

def fillna_str_alloc(A, fill):  # pragma: no cover
    return 0

def dropna(A):  # pragma: no cover
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


@numba.njit
def nth_element(arr, k, parallel=False):
    res = np.empty(1, arr.dtype)
    type_enum = hpat.distributed_api.get_type_enum(arr)
    if parallel:
        nth_parallel(res.ctypes, arr.ctypes, len(arr), k, type_enum)
    else:
        nth_sequential(res.ctypes, arr.ctypes, len(arr), k, type_enum)
    return res[0]

@numba.njit
def median(arr, parallel=False):
    # similar to numpy/lib/function_base.py:_median
    # TODO: check return types, e.g. float32 -> float32
    n = len(arr)
    k = len(arr) // 2

    # odd length case
    if n % 2 == 1:
        return nth_element(arr, k, parallel)

    v1 = nth_element(arr, k-1, parallel)
    v2 = nth_element(arr, k, parallel)
    return (v1 + v2) / 2


@infer_global(concat)
class ConcatType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr_list = args[0]
        if (isinstance(arr_list, types.UniTuple)
                and is_str_arr_typ(arr_list.dtype)):
            ret_typ = string_array_type
        else:
            # use typer of np.concatenate
            arr_list_to_arr = if_series_to_array_type(arr_list)
            ret_typ = numba.typing.npydecl.NdConcatenate(self.context).generic()(arr_list_to_arr)

        return signature(ret_typ, arr_list)

@lower_builtin(concat, types.Any)  # TODO: replace Any with types
def lower_concat(context, builder, sig, args):
    func = concat_overload(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)

# @overload(concat)
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
    return lambda a: np.concatenate(dummy_unbox_series(a))

def nunique(A):  # pragma: no cover
    return len(set(A))

def nunique_parallel(A):  # pragma: no cover
    return len(set(A))

@infer_global(nunique)
@infer_global(nunique_parallel)
class NuniqueType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        # if arr == string_series_type:
        #     arr = string_array_type
        return signature(types.intp, arr)

@lower_builtin(nunique, types.Any)  # TODO: replace Any with types
def lower_nunique(context, builder, sig, args):
    func = nunique_overload(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

# @overload(nunique)
def nunique_overload(arr_typ):
    # TODO: extend to other types like datetime?
    def nunique_seq(A):
        return len(set(A))
    return nunique_seq

@lower_builtin(nunique_parallel, types.Any)  # TODO: replace Any with types
def lower_nunique_parallel(context, builder, sig, args):
    func = nunique_overload_parallel(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

# @overload(nunique_parallel)
def nunique_overload_parallel(arr_typ):
    sum_op = hpat.distributed_api.Reduce_Type.Sum.value

    def nunique_par(A):
        uniq_A = hpat.hiframes_api.unique_parallel(A)
        loc_nuniq = len(set(uniq_A))
        return hpat.distributed_api.dist_reduce(loc_nuniq, np.int32(sum_op))

    return nunique_par


def unique(A):  # pragma: no cover
    return np.array([a for a in set(A)]).astype(A.dtype)

def unique_parallel(A):  # pragma: no cover
    return np.array([a for a in set(A)]).astype(A.dtype)

@infer_global(unique)
@infer_global(unique_parallel)
class uniqueType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        return signature(arr, arr)

@lower_builtin(unique, types.Any)  # TODO: replace Any with types
def lower_unique(context, builder, sig, args):
    func = unique_overload(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

# @overload(unique)
def unique_overload(arr_typ):
    # TODO: extend to other types like datetime?
    def unique_seq(A):
        return hpat.utils.to_array(set(A))
    return unique_seq

@lower_builtin(unique_parallel, types.Any)  # TODO: replace Any with types
def lower_unique_parallel(context, builder, sig, args):
    func = unique_overload_parallel(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

# @overload(unique_parallel)
def unique_overload_parallel(arr_typ):

    def unique_par(A):
        uniq_A = hpat.utils.to_array(set(A))

        n_pes = hpat.distributed_api.get_size()
        shuffle_meta = alloc_shuffle_metadata(uniq_A, n_pes, False)
        # calc send/recv counts
        for i in range(len(uniq_A)):
            val = uniq_A[i]
            node_id = hash(val) % n_pes
            update_shuffle_meta(shuffle_meta, node_id, i, val, False)

        finalize_shuffle_meta(uniq_A, shuffle_meta, False)

        # write send buffers
        for i in range(len(uniq_A)):
            val = uniq_A[i]
            node_id = hash(val) % n_pes
            write_send_buff(shuffle_meta, node_id, val)
            # update last since it is reused in data
            shuffle_meta.tmp_offset[node_id] += 1

        # shuffle
        alltoallv(uniq_A, shuffle_meta)

        return shuffle_meta.out_arr

    return unique_par



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

@infer_global(fillna_str_alloc)
class FillNaStrType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        # args: in_arr, value
        return signature(string_array_type, *args)

@infer_global(dropna)
class DropNAType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        # args: in_arr
        return signature(args[0], *args)

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

def alloc_shift(A):
    return np.empty_like(A)

@overload(alloc_shift)
def alloc_shift_overload(A_typ):
    if isinstance(A_typ.dtype, types.Integer):
        return lambda A: np.empty(len(A), np.float64)
    return lambda A: np.empty(len(A), A.dtype)

def shift_dtype(d):
    return d

@overload(shift_dtype)
def shift_dtype_overload(d_typ):
    if isinstance(d_typ.dtype, types.Integer):
        return lambda a: np.float64
    else:
        return lambda a: a

def isna(arr, i):
    return False

@overload(isna)
def isna_overload(arr_typ, ind_typ):
    if arr_typ == string_array_type:
        return lambda arr,i: hpat.str_arr_ext.str_arr_is_na(arr, i)
    # TODO: extend to other types
    assert isinstance(arr_typ, types.Array)
    dtype = arr_typ.dtype
    if isinstance(dtype, types.Float):
        return lambda arr,i: np.isnan(arr[i])
    # XXX integers don't have nans, extend to boolean
    return lambda arr,i: False


@numba.njit
def min_heapify(arr, n, start):
    min_ind = start
    left = 2 * start + 1
    right = 2 * start + 2

    if left < n and arr[left] < arr[min_ind]:
        min_ind = left

    if right < n and arr[right] < arr[min_ind]:
        min_ind = right

    if min_ind != start:
        arr[start], arr[min_ind] = arr[min_ind], arr[start]  # swap
        min_heapify(arr, n, min_ind)


def select_k_nonan(A, m, k):  # pragma: no cover
    return A

@overload(select_k_nonan)
def select_k_nonan_overload(A_t, m_t, k_t):
    dtype = A_t.dtype
    if isinstance(dtype, types.Integer):
        # ints don't have nans
        return lambda A,m,k: (A[:k].copy(), k)

    assert isinstance(dtype, types.Float)

    def select_k_nonan_float(A, m, k):
        # select the first k elements but ignore NANs
        min_heap_vals = np.empty(k, A.dtype)
        i = 0
        ind = 0
        while i < m and ind < k:
            val = A[i]
            i += 1
            if not np.isnan(val):
                min_heap_vals[ind] = val
                ind += 1

        # if couldn't fill with k values
        if ind < k:
            min_heap_vals = min_heap_vals[:ind]

        return min_heap_vals, i

    return select_k_nonan_float

@numba.njit
def nlargest(A, k):
    # algorithm: keep a min heap of k largest values, if a value is greater
    # than the minimum (root) in heap, replace the minimum and rebuild the heap
    m = len(A)

    # if all of A, just sort and reverse
    if k >= m:
        B = np.sort(A)
        B = B[~np.isnan(B)]
        return np.ascontiguousarray(B[::-1])

    # create min heap but
    min_heap_vals, start = select_k_nonan(A, m, k)
    # heapify k/2-1 to 0 instead of sort?
    min_heap_vals.sort()

    for i in range(start, m):
        if A[i] > min_heap_vals[0]:
            min_heap_vals[0] = A[i]
            min_heapify(min_heap_vals, k, 0)

    # sort and return the heap values
    min_heap_vals.sort()
    return np.ascontiguousarray(min_heap_vals[::-1])

MPI_ROOT = 0

@numba.njit
def nlargest_parallel(A, k):
    # parallel algorithm: assuming k << len(A), just call nlargest on chunks
    # of A, gather the result and return the largest k
    # TODO: support cases where k is not too small
    my_rank = hpat.distributed_api.get_rank()
    local_res = nlargest(A, k)
    all_largest = hpat.distributed_api.gatherv(local_res)

    # TODO: handle len(res) < k case
    if my_rank == MPI_ROOT:
        res = nlargest(all_largest, k)
    else:
        res = np.empty(k, A.dtype)
    hpat.distributed_api.bcast(res)
    return res


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


@lower_builtin(quantile, types.npytypes.Array, types.float64)
@lower_builtin(quantile_parallel, types.npytypes.Array, types.float64, types.intp)
def lower_dist_quantile(context, builder, sig, args):

    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[0].dtype]
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

@lower_builtin(set_df_col, PandasDataFrameType, types.Const, types.Array)
def set_df_col_lower(context, builder, sig, args):
    #
    col_name = sig.args[1].value
    arr_typ = sig.args[2]
    if arr_typ.dtype == datetime_date_type:
        return set_df_datetime_date_lower(context, builder, sig, args)

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

def box_df(names, arrs):
    return pd.DataFrame()

@infer_global(box_df)
class BoxDfTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) % 2 == 0, "name and column pairs expected"
        col_names = [a.value for a in args[:len(args)//2]]
        col_types =  [a.dtype for a in args[len(args)//2:]]
        df_typ = PandasDataFrameType(col_names, col_types)
        return signature(df_typ, *args)

BoxDfTyper.support_literals = True

@lower_builtin(box_df, types.Const, types.VarArg(types.Any))
def lower_box_df(context, builder, sig, args):
    assert len(sig.args) % 2 == 0, "name and column pairs expected"
    n_cols = len(sig.args)//2
    col_names = [a.value for a in sig.args[:n_cols]]
    col_arrs = [a for a in args[n_cols:]]
    arr_typs = [a for a in sig.args[n_cols:]]

    pyapi = context.get_python_api(builder)
    env_manager = context.get_env_manager(builder)
    c = numba.pythonapi._BoxContext(context, builder, pyapi, env_manager)
    gil_state = pyapi.gil_ensure()  # acquire GIL

    mod_name = context.insert_const_string(c.builder.module, "pandas")
    class_obj = pyapi.import_module_noblock(mod_name)
    res = pyapi.call_method(class_obj, "DataFrame", ())
    for cname, arr, arr_typ in zip(col_names, col_arrs, arr_typs):
        # df['cname'] = boxed_arr
        # TODO: datetime.date, DatetimeIndex?
        if arr_typ == string_array_type:
            arr_obj = box_str_arr(arr_typ, arr, c)
        else:
            arr_obj = box_array(arr_typ, arr, c)
        name_str = context.insert_const_string(c.builder.module, cname)
        cname_obj = pyapi.string_from_string(name_str)
        pyapi.object_setitem(res, cname_obj, arr_obj)
        # pyapi.decref(arr_obj)
        pyapi.decref(cname_obj)

    pyapi.decref(class_obj)
    pyapi.gil_release(gil_state)    # release GIL
    return res

@box(PandasDataFrameType)
def box_df_dummy(typ, val, c):
    return val


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
    elif typ.dtype == datetime_date_type:
        native_val = unbox_datetime_date_array(typ, val, c)
    else:
        # TODO: error handling like Numba callwrappers.py
        native_val = unbox_array(types.Array(dtype=typ.dtype, ndim=1, layout='C'), arr_obj, c)

    c.pyapi.decref(arr_obj)
    return native_val

@box(UnBoxedSeriesType)
def box_series(typ, val, c):
    """
    """
    if typ.dtype == string_type:
        arr = box_str_arr(typ, val, c)
    else:
        arr = box_array(types.Array(typ.dtype, 1, 'C'), val, c)
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    res = c.pyapi.call_method(class_obj, "Series", (arr,))
    # class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(pd.Series))
    # res = c.pyapi.call_function_objargs(class_obj, (arr,))
    c.pyapi.decref(class_obj)
    return res


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
            series_type = if_arr_to_series_type(arr)
        assert series_type is not None, "unknown type for pd.Series: {}".format(arr)
        return signature(series_type, arr)

@lower_builtin(to_series_type, types.Any)
def to_series_dummy_impl(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


def to_arr_from_series(arr):
    return arr

@infer_global(to_arr_from_series)
class ToArrFromSeriesType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        return signature(if_series_to_array_type(arr), arr)

@lower_builtin(to_arr_from_series, types.Any)
def to_arr_from_series_dummy_impl(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])

# dummy func to convert input series to array type
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


def to_date_series_type(arr):
    return arr

@infer_global(to_date_series_type)
class ToDateSeriesType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        assert (arr == types.Array(types.intp, 1, 'C')
            or arr == types.Array(datetime_date_type, 1, 'C'))
        return signature(date_series_type, arr)

@lower_builtin(to_date_series_type, types.Any)
def to_date_series_type_impl(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# convert const tuple expressions or const list to tuple statically
def to_const_tuple(arrs):  # pragma: no cover
    return tuple(arrs)

@infer_global(to_const_tuple)
class ToConstTupleTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        ret_typ = arr
        # XXX: returns a dummy type that should be fixed in hiframes_typed
        if isinstance(arr, types.List):
            ret_typ = types.Tuple((arr.dtype,))
        return signature(ret_typ, arr)


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)

if hasattr(numba.ir_utils, 'alias_func_extensions'):
    numba.ir_utils.alias_func_extensions[('dummy_unbox_series', 'hpat.hiframes_api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('to_series_type', 'hpat.hiframes_api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('to_arr_from_series', 'hpat.hiframes_api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('ts_series_to_arr_typ', 'hpat.hiframes_api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('to_date_series_type', 'hpat.hiframes_api')] = alias_ext_dummy_func


# XXX: use infer_global instead of overload, since overload fails if the same
# user function is compiled twice
@infer_global(fix_df_array)
class FixDfArrayType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        column = args[0]
        ret_typ = column
        if (isinstance(column, types.List)
            and (isinstance(column.dtype, types.Number)
                 or column.dtype == types.boolean)):
            ret_typ = types.Array(column.dtype, 1, 'C')
        if isinstance(column, types.List) and isinstance(column.dtype, StringType):
            ret_typ = string_array_type
        # TODO: add other types
        return signature(ret_typ, column)

@lower_builtin(fix_df_array, types.Any)  # TODO: replace Any with types
def lower_fix_df_array(context, builder, sig, args):
    func = fix_df_array_overload(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)

#@overload(fix_df_array)
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

# @overload(fix_rolling_array)
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
        assert args[0] == dt_index_series_type or args[0] == types.Array(types.int64, 1, 'C')
        return signature(types.Array(types.NPDatetime('ns'), 1, 'C'), *args)

@lower_builtin(ts_series_to_arr_typ, dt_index_series_type)
@lower_builtin(ts_series_to_arr_typ, types.Array(types.int64, 1, 'C'))
def lower_ts_series_to_arr_typ(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# register series types for import
@typeof_impl.register(pd.Series)
def typeof_pd_str_series(val, c):
    # TODO: handle NA as 1st value
    if len(val) > 0 and isinstance(val[0], str):  # and isinstance(val[-1], str):
        arr_typ = string_array_type
    elif len(val) > 0 and isinstance(val.values[0], datetime.date):
        # XXX: using .values to check date type since DatetimeIndex returns
        # Timestamp which is subtype of datetime.date
        return BoxedSeriesType(datetime_date_type)
    else:
        arr_typ = numba.typing.typeof._typeof_ndarray(val.values, c)

    return arr_to_boxed_series_type(arr_typ)

@typeof_impl.register(pd.Index)
def typeof_pd_index(val, c):
    if len(val) > 0 and isinstance(val[0], datetime.date):
        return BoxedSeriesType(datetime_date_type)
    else:
        raise NotImplementedError("unsupported pd.Index type")


# TODO: separate pd.DatetimeIndex type
#@typeof_impl.register(pd.DatetimeIndex)

def pd_dt_index_stub(data):  # pragma: no cover
    return data

@infer_global(pd.DatetimeIndex)
class DatetimeIndexTyper(AbstractTemplate):
    def generic(self, args, kws):
        pysig = numba.utils.pysignature(pd_dt_index_stub)
        try:
            bound = pysig.bind(*args, **kws)
        except TypeError:  # pragma: no cover
            msg = "Unsupported arguments for pd.DatetimeIndex()"
            raise ValueError(msg)

        sig = signature(
            SeriesType(types.NPDatetime('ns'), 1, 'C'), bound.args).replace(
                pysig=pysig)
        return sig

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
        arr_types =  [if_series_to_array_type(a) for a in args[len(args)//2:]]
        # XXX index handling, assuming implicit index
        assert "Index" not in col_names[0]
        col_names = ['Index'] + col_names
        arr_types = [types.Array(types.int64, 1, 'C')] + arr_types
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

import operator
from collections import namedtuple
import pandas as pd
import numpy as np

import numba
from numba import ir, ir_utils
from numba.ir_utils import require, mk_unique_var
from numba import types, cgutils
import numba.array_analysis
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate, CallableTemplate
from numba.typing.arraydecl import _expand_integer
from numba.extending import overload, intrinsic
from numba.targets.imputils import (impl_ret_new_ref, impl_ret_borrowed,
    iternext_impl, RefType)
from numba.targets.arrayobj import _getitem_array1d
from numba.extending import register_model, models

import hpat
from hpat.str_ext import string_type, list_string_array_type
from hpat.str_arr_ext import (StringArrayType, string_array_type,
    is_str_arr_typ)

from hpat.set_ext import build_set
from numba.targets.imputils import lower_builtin, impl_ret_untracked
from hpat.hiframes.pd_timestamp_ext import (pandas_timestamp_type,
    datetime_date_type, set_df_datetime_date_lower)
from hpat.hiframes.pd_series_ext import (SeriesType,
    is_str_series_typ, if_arr_to_series_type,
    series_to_array_type, if_series_to_array_type, is_dt64_series_typ)
from hpat.hiframes.pd_index_ext import DatetimeIndexType, TimedeltaIndexType
from hpat.hiframes.sort import (
      alltoallv,
    alltoallv_tup, finalize_shuffle_meta,
    update_shuffle_meta,  alloc_pre_shuffle_metadata,
    )
from hpat.hiframes.join import write_send_buff
from hpat.hiframes.split_impl import string_array_split_view_type

# XXX: used in agg func output to avoid mutating filter, agg, join, etc.
# TODO: fix type inferrer and remove this
enable_hiframes_remove_dead = True

# quantile imports?
import llvmlite.llvmpy.core as lc
from llvmlite import ir as lir
from .. import quantile_alg
import llvmlite.binding as ll
ll.add_symbol('quantile_parallel', quantile_alg.quantile_parallel)
ll.add_symbol('nth_sequential', quantile_alg.nth_sequential)
ll.add_symbol('nth_parallel', quantile_alg.nth_parallel)
from numba.targets.arrayobj import make_array
from hpat.utils import _numba_to_c_type_map, unliteral_all


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

sum_op = hpat.distributed_api.Reduce_Type.Sum.value

@numba.njit
def median(arr, parallel=False):
    # similar to numpy/lib/function_base.py:_median
    # TODO: check return types, e.g. float32 -> float32
    n = len(arr)
    if parallel:
        n = hpat.distributed_api.dist_reduce(n, np.int32(sum_op))
    k = n // 2

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
        return len(build_set(A))
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
        uniq_A = hpat.hiframes.api.unique_parallel(A)
        loc_nuniq = len(uniq_A)
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
        return hpat.utils.to_array(build_set(A))
    return unique_seq

@lower_builtin(unique_parallel, types.Any)  # TODO: replace Any with types
def lower_unique_parallel(context, builder, sig, args):
    func = unique_overload_parallel(sig.args[0])
    res = context.compile_internal(builder, func, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

# @overload(unique_parallel)
def unique_overload_parallel(arr_typ):

    def unique_par(A):
        uniq_A = hpat.utils.to_array(build_set(A))
        key_arrs = (uniq_A,)

        n_pes = hpat.distributed_api.get_size()
        pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, (), n_pes, False)

        # calc send/recv counts
        for i in range(len(uniq_A)):
            val = uniq_A[i]
            node_id = hash(val) % n_pes
            update_shuffle_meta(pre_shuffle_meta, node_id, i, (val,), (), False)

        shuffle_meta = finalize_shuffle_meta(key_arrs, (), pre_shuffle_meta, n_pes, False)

        # write send buffers
        for i in range(len(uniq_A)):
            val = uniq_A[i]
            node_id = hash(val) % n_pes
            write_send_buff(shuffle_meta, node_id, i, (val,), ())
            # update last since it is reused in data
            shuffle_meta.tmp_offset[node_id] += 1

        # shuffle
        out_arr, = alltoallv_tup(key_arrs, shuffle_meta)

        return hpat.utils.to_array(build_set(out_arr))

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
        return signature(SeriesType(string_type), *args)

@infer_global(dropna)
class DropNAType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) in (1, 2)
        # args: in_arr
        ret = args[0]
        if not isinstance(ret, types.BaseTuple):
            # series case
            ret = if_arr_to_series_type(ret)
        return signature(ret, *args)

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
        return signature(types.float64, *unliteral_all(args))


@infer_global(str_contains_regex)
@infer_global(str_contains_noregex)
class ContainsType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        # args: str_arr, pat
        return signature(types.Array(types.boolean, 1, 'C'), *unliteral_all(args))

def alloc_shift(A):
    return np.empty_like(A)

@overload(alloc_shift)
def alloc_shift_overload(A):
    if isinstance(A.dtype, types.Integer):
        return lambda A: np.empty(len(A), np.float64)
    return lambda A: np.empty(len(A), A.dtype)

def shift_dtype(d):
    return d

@overload(shift_dtype)
def shift_dtype_overload(a):
    if isinstance(a.dtype, types.Integer):
        return lambda a: np.float64
    else:
        return lambda a: a

def isna(arr, i):
    return False

@overload(isna)
def isna_overload(arr, i):
    if arr == string_array_type:
        return lambda arr, i: hpat.str_arr_ext.str_arr_is_na(arr, i)
    # TODO: support NaN in list(list(str))
    if arr == list_string_array_type:
        return lambda arr, i: False
    if arr == string_array_split_view_type:
        return lambda arr, i: False
    # TODO: extend to other types
    assert isinstance(arr, types.Array)
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


@numba.njit
def min_heapify(arr, n, start, cmp_f):
    min_ind = start
    left = 2 * start + 1
    right = 2 * start + 2

    if left < n and not cmp_f(arr[left], arr[min_ind]):  # < for nlargest
        min_ind = left

    if right < n and not cmp_f(arr[right], arr[min_ind]):
        min_ind = right

    if min_ind != start:
        arr[start], arr[min_ind] = arr[min_ind], arr[start]  # swap
        min_heapify(arr, n, min_ind, cmp_f)


def select_k_nonan(A, m, k):  # pragma: no cover
    return A

@overload(select_k_nonan)
def select_k_nonan_overload(A, m, k):
    dtype = A.dtype
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
def nlargest(A, k, is_largest, cmp_f):
    # algorithm: keep a min heap of k largest values, if a value is greater
    # than the minimum (root) in heap, replace the minimum and rebuild the heap
    m = len(A)

    # if all of A, just sort and reverse
    if k >= m:
        B = np.sort(A)
        B = B[~np.isnan(B)]
        if is_largest:
            B = B[::-1]
        return np.ascontiguousarray(B)

    # create min heap but
    min_heap_vals, start = select_k_nonan(A, m, k)
    # heapify k/2-1 to 0 instead of sort?
    min_heap_vals.sort()
    if not is_largest:
        min_heap_vals = np.ascontiguousarray(min_heap_vals[::-1])

    for i in range(start, m):
        if cmp_f(A[i], min_heap_vals[0]):  # > for nlargest
            min_heap_vals[0] = A[i]
            min_heapify(min_heap_vals, k, 0, cmp_f)

    # sort and return the heap values
    min_heap_vals.sort()
    if is_largest:
        min_heap_vals = min_heap_vals[::-1]
    return np.ascontiguousarray(min_heap_vals)

MPI_ROOT = 0

@numba.njit
def nlargest_parallel(A, k, is_largest, cmp_f):
    # parallel algorithm: assuming k << len(A), just call nlargest on chunks
    # of A, gather the result and return the largest k
    # TODO: support cases where k is not too small
    my_rank = hpat.distributed_api.get_rank()
    local_res = nlargest(A, k, is_largest, cmp_f)
    all_largest = hpat.distributed_api.gatherv(local_res)

    # TODO: handle len(res) < k case
    if my_rank == MPI_ROOT:
        res = nlargest(all_largest, k, is_largest, cmp_f)
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

# the same as fix_df_array but can be parallel
@numba.generated_jit(nopython=True)
def parallel_fix_df_array(c):  # pragma: no cover
    return lambda c: fix_df_array(c)

def fix_rolling_array(c):  # pragma: no cover
    return c

def sort_values(key_arr):  # pragma: no cover
    return

@infer_global(sort_values)
class SortTyping(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *args)

def df_isin(A, B):  # pragma: no cover
    return A

def df_isin_vals(A, B):  # pragma: no cover
    return A

@infer_global(df_isin)
@infer_global(df_isin_vals)
class DfIsinCol(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.Array(types.bool_, 1, 'C'), *unliteral_all(args))


def flatten_to_series(A):  # pragma: no cover
    return A

@infer_global(flatten_to_series)
class FlattenTyp(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        # only list of lists supported
        assert isinstance(args[0], (types.List, SeriesType))
        l_dtype = args[0].dtype
        assert isinstance(l_dtype, types.List)
        dtype = l_dtype.dtype
        return signature(SeriesType(dtype), *unliteral_all(args))

def to_numeric(A, dtype):
    return A

@infer_global(to_numeric)
class ToNumeric(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        dtype = args[1].dtype
        return signature(SeriesType(dtype), *unliteral_all(args))

def series_filter_bool(arr, bool_arr):
    return arr[bool_arr]

@infer_global(series_filter_bool)
class SeriesFilterBoolInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        ret = args[0]
        if isinstance(ret.dtype, types.Integer):
            ret = SeriesType(types.float64)
        return signature(ret, *args)


def set_df_col(df, cname, arr):
    df[cname] = arr

@infer_global(set_df_col)
class SetDfColInfer(AbstractTemplate):
    def generic(self, args, kws):
        from hpat.hiframes.pd_dataframe_ext import DataFrameType
        assert not kws
        assert len(args) == 3
        assert isinstance(args[1], types.Literal)
        target = args[0]
        ind = args[1].literal_value
        val = args[2]
        ret = target

        if isinstance(target, DataFrameType):
            if isinstance(val, SeriesType):
                val = val.data
            if ind in target.columns:
                # set existing column, with possibly a new array type
                new_cols = target.columns
                col_id = target.columns.index(ind)
                new_typs = list(target.data)
                new_typs[col_id] = val
                new_typs = tuple(new_typs)
            else:
                # set a new column
                new_cols = target.columns + (ind,)
                new_typs = target.data + (val,)
            ret = DataFrameType(
                new_typs, target.index, new_cols, target.has_parent)

        return signature(ret, *args)


#@lower_builtin(set_df_col, DataFrameType, types.Literal, types.Array)
def set_df_col_lower(context, builder, sig, args):
    #
    col_name = sig.args[1].literal_value
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


# convert tuple of Series to tuple of arrays statically (for append)
def series_tup_to_arr_tup(arrs):  # pragma: no cover
    return arrs

@infer_global(series_tup_to_arr_tup)
class SeriesTupleToArrTupleTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        ret_typ = if_series_to_array_type(arr)
        return signature(ret_typ, arr)


# this function should be used for getting S._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_data(S):
    return lambda S: S._data

# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_index(S):
    return lambda S: S._index

@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_name(S):
    return lambda S: S._name

@numba.generated_jit(nopython=True)
def get_index_data(S):
    return lambda S: S._data

def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


def alias_ext_init_series(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    if len(args) > 1:  # has index
        numba.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


if hasattr(numba.ir_utils, 'alias_func_extensions'):
    numba.ir_utils.alias_func_extensions[('init_series', 'hpat.hiframes.api')] = alias_ext_init_series
    numba.ir_utils.alias_func_extensions[('get_series_data', 'hpat.hiframes.api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('get_series_index', 'hpat.hiframes.api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('init_datetime_index', 'hpat.hiframes.api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('get_index_data', 'hpat.hiframes.api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('dummy_unbox_series', 'hpat.hiframes.api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('get_dataframe_data', 'hpat.hiframes.pd_dataframe_ext')] = alias_ext_dummy_func
    # TODO: init_dataframe
    numba.ir_utils.alias_func_extensions[('to_arr_from_series', 'hpat.hiframes.api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('ts_series_to_arr_typ', 'hpat.hiframes.api')] = alias_ext_dummy_func
    numba.ir_utils.alias_func_extensions[('to_date_series_type', 'hpat.hiframes.api')] = alias_ext_dummy_func

@numba.njit
def agg_typer(a, _agg_f):
    return np.full(1, _agg_f(a))


def convert_tup_to_rec(val):
    return val

@infer_global(convert_tup_to_rec)
class ConvertTupRecType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        in_dtype = args[0]
        out_dtype = in_dtype

        if isinstance(in_dtype, types.BaseTuple):
            np_dtype = np.dtype(
                ','.join(str(t) for t in in_dtype.types), align=True)
            out_dtype = numba.numpy_support.from_dtype(np_dtype)

        return signature(out_dtype, in_dtype)

@lower_builtin(convert_tup_to_rec, types.Any)
def lower_convert_impl(context, builder, sig, args):
    val, = args
    in_typ = sig.args[0]
    rec_typ = sig.return_type

    if not isinstance(in_typ, types.BaseTuple):
        return impl_ret_borrowed(context, builder, sig.return_type, val)

    res = cgutils.alloca_once(builder, context.get_data_type(rec_typ))

    func_text = "def _set_rec(r, val):\n"
    for i in range(len(rec_typ.members)):
        func_text += "  r.f{} = val[{}]\n".format(i, i)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    set_rec = loc_vars['_set_rec']

    context.compile_internal(builder, set_rec, types.void(rec_typ, in_typ), [res, val])
    return impl_ret_new_ref(context, builder, sig.return_type, res)


def convert_rec_to_tup(val):
    return val

@infer_global(convert_rec_to_tup)
class ConvertRecTupType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        in_dtype = args[0]
        out_dtype = in_dtype

        if isinstance(in_dtype, types.Record):
            out_dtype = types.Tuple([m[1] for m in in_dtype.members])

        return signature(out_dtype, in_dtype)

@lower_builtin(convert_rec_to_tup, types.Any)
def lower_convert_rec_tup_impl(context, builder, sig, args):
    val, = args
    rec_typ = sig.args[0]
    tup_typ = sig.return_type

    if not isinstance(rec_typ, types.Record):
        return impl_ret_borrowed(context, builder, sig.return_type, val)

    n_fields = len(rec_typ.members)

    func_text = "def _rec_to_tup(r):\n"
    func_text += "  return ({},)\n".format(
        ", ".join("r.f{}".format(i) for i in range(n_fields)))

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    _rec_to_tup = loc_vars['_rec_to_tup']

    res = context.compile_internal(builder, _rec_to_tup, tup_typ(rec_typ), [val])
    return impl_ret_borrowed(context, builder, sig.return_type, res)


# XXX: use infer_global instead of overload, since overload fails if the same
# user function is compiled twice
@infer_global(fix_df_array)
class FixDfArrayType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        column = types.unliteral(args[0])
        ret_typ = column
        if (isinstance(column, types.List)
            and (isinstance(column.dtype, types.Number)
                 or column.dtype == types.boolean)):
            ret_typ = types.Array(column.dtype, 1, 'C')
        if isinstance(column, types.List) and column.dtype == string_type:
            ret_typ = string_array_type
        if isinstance(column, DatetimeIndexType):
            ret_typ = hpat.hiframes.pd_index_ext._dt_index_data_typ
        if isinstance(column, SeriesType):
            ret_typ = column.data
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
    if isinstance(column, types.List) and column.dtype == string_type:
        def fix_df_array_str_impl(column):  # pragma: no cover
            return hpat.str_arr_ext.StringArray(column)
        return fix_df_array_str_impl

    if isinstance(column, DatetimeIndexType):
        return lambda column: hpat.hiframes.api.get_index_data(column)

    if isinstance(column, SeriesType):
        return lambda column: hpat.hiframes.api.get_series_data(column)

    # column is array if not list
    assert isinstance(column, (types.Array, StringArrayType, SeriesType))
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
        assert is_dt64_series_typ(args[0]) or args[0] == types.Array(types.int64, 1, 'C')
        return signature(types.Array(types.NPDatetime('ns'), 1, 'C'), *args)

@lower_builtin(ts_series_to_arr_typ, SeriesType)
@lower_builtin(ts_series_to_arr_typ, types.Array(types.int64, 1, 'C'))
def lower_ts_series_to_arr_typ(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


def parse_datetimes_from_strings(A):
    return A

@infer_global(parse_datetimes_from_strings)
class ParseDTArrType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        assert args[0] == string_array_type or is_str_series_typ(args[0])
        return signature(types.Array(types.NPDatetime('ns'), 1, 'C'), *args)

@lower_builtin(parse_datetimes_from_strings, types.Any)
def lower_parse_datetimes_from_strings(context, builder, sig, args):
    # dummy implementation to avoid @overload errors
    # replaced in hiframes_typed pass
    res = make_array(sig.return_type)(context, builder)
    return impl_ret_borrowed(context, builder, sig.return_type, res._getvalue())


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
                series.name = numba.unicode.make_string_from_constant(
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


@intrinsic
def init_datetime_index(typingctx, data, name=None):
    """Create a DatetimeIndex with provided data and name values.
    """
    name = types.none if name is None else name
    is_named = False if name is types.none else True

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create dt_index struct and store values
        dt_index = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        dt_index.data = data_val
        if is_named:
            if isinstance(name, types.StringLiteral):
                dt_index.name = numba.unicode.make_string_from_constant(
                    context, builder, string_type, name.literal_value)
            else:
                dt_index.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            if is_named:
                context.nrt.incref(builder, signature.args[1], name_val)

        return dt_index._getvalue()

    ret_typ = DatetimeIndexType(is_named)
    sig = signature(ret_typ, data, name)
    return sig, codegen


@intrinsic
def init_timedelta_index(typingctx, data, name=None):
    """Create a TimedeltaIndex with provided data and name values.
    """
    name = types.none if name is None else name
    is_named = False if name is types.none else True

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create timedelta_index struct and store values
        timedelta_index = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        timedelta_index.data = data_val
        if is_named:
            if isinstance(name, types.StringLiteral):
                timedelta_index.name = numba.unicode.make_string_from_constant(
                    context, builder, string_type, name.literal_value)
            else:
                timedelta_index.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            if is_named:
                context.nrt.incref(builder, signature.args[1], name_val)

        return timedelta_index._getvalue()

    ret_typ = TimedeltaIndexType(is_named)
    sig = signature(ret_typ, data, name)
    return sig, codegen


@overload(np.array)
def np_array_array_overload(A):
    if isinstance(A, types.Array):
        return lambda A: A

    if isinstance(A, types.containers.Set):
        # TODO: naive implementation, data from set can probably
        # be copied to array more efficienty
        dtype = A.dtype
        def f(A):
            n = len(A)
            arr = np.empty(n, dtype)
            i = 0
            for a in A:
                arr[i] = a
                i += 1
            return arr
        return f


class ConstList(types.List):
    def __init__(self, dtype, consts):
        dtype = types.unliteral(dtype)
        self.dtype = dtype
        self.reflected = False
        self.consts = consts
        cls_name = "list[{}]".format(consts)
        name = "%s(%s)" % (cls_name, self.dtype)
        super(types.List, self).__init__(name=name)

    def copy(self, dtype=None, reflected=None):
        if dtype is None:
            dtype = self.dtype
        return ConstList(dtype, self.consts)

    def unify(self, typingctx, other):
        if isinstance(other, ConstList) and self.consts == other.consts:
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            reflected = self.reflected or other.reflected
            if dtype is not None:
                return ConstList(dtype, reflected)

    @property
    def key(self):
        return self.dtype, self.reflected, self.consts


@register_model(ConstList)
class ConstListModel(models.ListModel):
    def __init__(self, dmm, fe_type):
        l_type = types.List(fe_type.dtype)
        super(ConstListModel, self).__init__(dmm, l_type)


# add constant metadata to list or tuple type, see hiframes.py
def add_consts_to_type(a, *args):
    return a


@infer_global(add_consts_to_type)
class AddConstsTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ret_typ = args[0]
        assert isinstance(ret_typ, types.List)  # TODO: other types
        # TODO: FloatLiteral e.g. test_fillna
        if all(isinstance(v, types.Literal) for v in args[1:]):
            consts = tuple(v.literal_value for v in args[1:])
            ret_typ = ConstList(ret_typ.dtype, consts)
        return signature(ret_typ, *args)

@lower_builtin(add_consts_to_type, types.VarArg(types.Any))
def lower_add_consts_to_type(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])

# dummy empty itertools implementation to avoid typing errors for series str
# flatten case
import itertools
@overload(itertools.chain)
def chain_overload():
    return lambda: [0]

# a dummy join function that will be replace in dataframe_pass
def join_dummy(left_df, right_df, left_on, right_on, how):
    return left_df


@infer_global(join_dummy)
class JoinTyper(AbstractTemplate):
    def generic(self, args, kws):
        from hpat.hiframes.pd_dataframe_ext import DataFrameType
        assert not kws
        left_df, right_df, left_on, right_on, how = args

        columns = list(left_df.columns)
        data = list(left_df.data)
        for i, c in enumerate(right_df.columns):
            if c not in left_df.columns:
                columns.append(c)
                data.append(right_df.data[i])

        out_df = DataFrameType(tuple(data), None, tuple(columns))
        return signature(out_df, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(join_dummy, types.VarArg(types.Any))
def lower_join_dummy(context, builder, sig, args):
    dataframe = cgutils.create_struct_proxy(
            sig.return_type)(context, builder)
    return dataframe._getvalue()


# type used to pass metadata to type inference functions
# see hiframes.py and df.pivot_table()
class MetaType(types.Type):
    def __init__(self, meta):
        self.meta = meta
        super(MetaType, self).__init__("MetaType({})".format(meta))

    def can_convert_from(self, typingctx, other):
        return True

    @property
    def key(self):
        # XXX this is needed for _TypeMetaclass._intern to return the proper
        # cached instance in case meta is changed
        # (e.g. TestGroupBy -k pivot -k cross)
        return tuple(self.meta)

register_model(MetaType)(models.OpaqueModel)


def drop_inplace(df):
    res = None
    return df, res

@overload(drop_inplace)
def drop_inplace_overload(df, labels=None, axis=0, index=None, columns=None,
        level=None, inplace=False, errors='raise'):

    from hpat.hiframes.pd_dataframe_ext import DataFrameType
    assert isinstance(df, DataFrameType)
    # TODO: support recovery when object is not df
    def _impl(df, labels=None, axis=0, index=None, columns=None,
            level=None, inplace=False, errors='raise'):
        new_df = hpat.hiframes.pd_dataframe_ext.drop_dummy(
            df, labels, axis, columns, inplace)
        return new_df, None

    return _impl

# taken from numba/typing/listdecl.py
@infer_global(sorted)
class SortedBuiltinLambda(CallableTemplate):

    def generic(self):
        # TODO: reverse=None
        def typer(iterable, key=None):
            if not isinstance(iterable, types.IterableType):
                return
            return types.List(iterable.iterator_type.yield_type)

        return typer


@overload(operator.getitem)
def list_str_arr_getitem_array(arr, ind):
    if (arr == list_string_array_type and isinstance(ind, types.Array)
            and ind.ndim == 1 and isinstance(
            ind.dtype, (types.Integer, types.Boolean))):
        # TODO: convert to parfor in typed pass
        def list_str_arr_getitem_impl(arr, ind):
            n = ind.sum()
            out_arr = hpat.str_ext.alloc_list_list_str(n)
            j = 0
            for i in range(len(ind)):
                if ind[i]:
                    out_arr[j] = arr[i]
                    j += 1
            return out_arr

        return list_str_arr_getitem_impl


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
        col_names = [a.literal_value for a in args[:len(args)//2]]
        arr_types =  [if_series_to_array_type(a) for a in args[len(args)//2:]]
        # XXX index handling, assuming implicit index
        assert "Index" not in col_names[0]
        col_names = ['Index'] + col_names
        arr_types = [types.Array(types.int64, 1, 'C')] + arr_types
        iter_typ = DataFrameTupleIterator(col_names, arr_types)
        return signature(iter_typ, *args)


@register_model(DataFrameTupleIterator)
class DataFrameTupleIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        # XXX array_types[0] is implicit index
        members = ([('index', types.EphemeralPointer(types.uintp))]
            + [('array{}'.format(i), arr) for i, arr in enumerate(fe_type.array_types[1:])])
        super(DataFrameTupleIteratorModel, self).__init__(dmm, fe_type, members)

    def from_return(self, builder, value):
        # dummy to avoid lowering error for itertuples_overload
        # TODO: remove when overload_method can avoid lowering or avoid cpython
        # wrapper
        return value


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
@iternext_impl(RefType.UNTRACKED)
def iternext_itertuples(context, builder, sig, args, result):
    # TODO: refcount issues?
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
                    lambda a,i: hpat.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(np.int64(a[i])),
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

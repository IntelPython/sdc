from __future__ import print_function, division, absolute_import

import numba
from numba import typeinfer, ir, ir_utils, config
from numba.ir_utils import visit_vars_inner
from numba import types
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution
from hpat.str_ext import StringType, string_type
from hpat.str_arr_ext import StringArray, StringArrayType, string_array_type

class Filter(ir.Stmt):
    def __init__(self, df_out, df_in, bool_arr, df_vars, loc):
        self.df_out = df_out
        self.df_in = df_in
        self.df_out_vars = df_vars[self.df_out]
        self.df_in_vars = df_vars[self.df_in]
        self.bool_arr = bool_arr
        # needs df columns for type inference stage
        self.df_vars = df_vars
        self.loc = loc

    def __repr__(self):
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)
        in_cols = ""
        for (c, v) in self.df_in_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        return "filter: {} = {} [cond: {}] ".format(df_out_str, df_in_str,
                                                                self.bool_arr)

def filter_array_analysis(filter_node, equiv_set, typemap, array_analysis):
    post = []
    # empty filter nodes should be deleted in remove dead
    assert len(filter_node.df_in_vars) > 0, "empty filter in array analysis"

    # arrays of input df have same size in first dimension as bool array
    col_shape = equiv_set.get_shape(filter_node.bool_arr)
    all_shapes = [col_shape[0]]
    for _, col_var in filter_node.df_in_vars.items():
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        col_shape = equiv_set.get_shape(col_var)
        all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for _, col_var in filter_node.df_out_vars.items():
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        (shape, c_post) = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None)
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var)

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post

numba.array_analysis.array_analysis_extensions[Filter] = filter_array_analysis

def filter_distributed_analysis(filter_node, array_dists):

    # input columns have same distribution
    in_dist = Distribution.OneD
    for _, col_var in filter_node.df_in_vars.items():
        in_dist = Distribution(min(in_dist.value, array_dists[col_var.name].value))

    # bool arr
    in_dist = Distribution(min(in_dist.value, array_dists[filter_node.bool_arr.name].value))
    for _, col_var in filter_node.df_in_vars.items():
        array_dists[col_var.name] = in_dist
    array_dists[filter_node.bool_arr.name] = in_dist

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in filter_node.df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for _, col_var in filter_node.df_out_vars.items():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        array_dists[filter_node.bool_arr.name] = out_dist
        for _, col_var in filter_node.df_in_vars.items():
            array_dists[col_var.name] = out_dist

    return

distributed_analysis.distributed_analysis_extensions[Filter] = filter_distributed_analysis

def filter_distributed_run(filter_node, typemap, calltypes):
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    loc = filter_node.loc
    bool_arr = filter_node.bool_arr

    out = []
    for col_name, col_in_var in filter_node.df_in_vars.items():
        col_out_var = filter_node.df_out_vars[col_name]
        # using getitem like Numba for filtering arrays
        # TODO: generate parfor
        getitem_call = ir.Expr.getitem(col_in_var, bool_arr, loc)
        calltypes[getitem_call] = signature(
                typemap[col_out_var.name],  # output type
                typemap[col_in_var.name],  # input type
                typemap[bool_arr.name])  # index type
        out.append(ir.Assign(getitem_call, col_out_var, loc))

    return out

distributed.distributed_run_extensions[Filter] = filter_distributed_run


def filter_typeinfer(filter_node, typeinferer):
    for col_name, col_var in filter_node.df_in_vars.items():
        out_col_var = filter_node.df_out_vars[col_name]
        typeinferer.constraints.append(typeinfer.Propagate(dst=out_col_var.name,
                                              src=col_var.name, loc=filter_node.loc))
    return

typeinfer.typeinfer_extensions[Filter] = filter_typeinfer


def visit_vars_filter(filter_node, callback, cbdata):
    if config.DEBUG_ARRAY_OPT == 1:
        print("visiting filter vars for:", filter_node)
        print("cbdata: ", sorted(cbdata.items()))

    filter_node.bool_arr = visit_vars_inner(filter_node.bool_arr, callback, cbdata)

    for col_name in list(filter_node.df_in_vars.keys()):
        filter_node.df_in_vars[col_name] = visit_vars_inner(filter_node.df_in_vars[col_name], callback, cbdata)
    for col_name in list(filter_node.df_out_vars.keys()):
        filter_node.df_out_vars[col_name] = visit_vars_inner(filter_node.df_out_vars[col_name], callback, cbdata)


# add call to visit parfor variable
ir_utils.visit_vars_extensions[Filter] = visit_vars_filter

def remove_dead_filter(filter_node, lives, arg_aliases, alias_map, typemap):
    #
    dead_cols = []

    for col_name, col_var in filter_node.df_out_vars.items():
        if col_var.name not in lives:
            dead_cols.append(col_name)

    for cname in dead_cols:
        filter_node.df_in_vars.pop(cname)
        filter_node.df_out_vars.pop(cname)

    # remove empty filter node
    if len(filter_node.df_in_vars) == 0:
        return None

    return filter_node

ir_utils.remove_dead_extensions[Filter] = remove_dead_filter

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

def count(A):
    return 0

def fillna(A):
    return 0

def column_sum(A):
    return 0

def var(A):
    return 0

def std(A):
    return 0

def mean(A):
    return 0

def quantile(A, q):
    return 0

def quantile_parallel(A, q):
    return 0

def str_contains_regex(str_arr, pat):
    return 0;

def str_contains_noregex(str_arr, pat):
    return 0;

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

from numba.targets.imputils import lower_builtin, impl_ret_untracked
import numpy as np

@lower_builtin(mean, types.Array)
def lower_column_mean_impl(context, builder, sig, args):
    zero = sig.return_type(0)
    def array_mean_impl(arr):
        count = 0
        s = zero
        for val in arr:
            if not np.isnan(val):
                s += val
                count += 1
        if not count:
            s = np.nan
        else:
            s = s/count
        return s

    res = context.compile_internal(builder, array_mean_impl, sig, args,
                                   locals=dict(s=sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)


# copied from numba/numba/targets/arraymath.py:119
@lower_builtin(var, types.Array)
def array_var(context, builder, sig, args):
    def array_var_impl(arr):
        # TODO: ignore NA
        # Compute the mean
        m = arr.mean()

        # Compute the sum of square diffs
        ssd = 0
        for v in np.nditer(arr):
            ssd += (v.item() - m) ** 2
        return ssd / (arr.size-1)  # ddof=1 in pandas

    res = context.compile_internal(builder, array_var_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(std, types.Array)
def array_std(context, builder, sig, args):
    def array_std_impl(arry):
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

@lower_builtin(quantile_parallel, types.npytypes.Array, types.float64, types.intp)
def lower_dist_quantile(context, builder, sig, args):

    # store an int to specify data type
    typ_enum = _h5_typ_table[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))
    assert sig.args[0].ndim == 1

    arr = make_array(sig.args[0])(context, builder, args[0])
    local_size = builder.extract_value(arr.shape, 0)

    call_args = [builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
                local_size, args[2], args[1], builder.load(typ_arg)]

    # array, size, total_size, quantile, type enum
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(64),
                                            lir.DoubleType(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.DoubleType(), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="quantile_parallel")
    return builder.call(fn, call_args)


def fix_df_array(c):
    return c

def fix_rolling_array(c):
    return c

from numba.extending import overload

@overload(fix_df_array)
def fix_df_array_overload(column):
    # convert list of numbers/bools to numpy array
    if (isinstance(column, types.List)
            and (isinstance(column.dtype, types.Number)
            or column.dtype==types.boolean)):
        def fix_df_array_impl(column):
            return np.array(column)
        return fix_df_array_impl
    # convert list of strings to string array
    if isinstance(column, types.List) and isinstance(column.dtype, StringType):
        def fix_df_array_impl(column):
            return StringArray(column)
        return fix_df_array_impl
    # column is array if not list
    assert isinstance(column, (types.Array, StringArrayType))
    def fix_df_array_impl(column):
        return column
    return fix_df_array_impl

@overload(fix_rolling_array)
def fix_rolling_array_overload(column):
    assert isinstance(column, types.Array)
    dtype = column.dtype
    # convert bool and integer to float64
    if dtype == types.boolean or isinstance(dtype, types.Integer):
        def fix_rolling_array_impl(column):
            return column.astype(np.float64)
    else:
        def fix_rolling_array_impl(column):
            return column
    return fix_rolling_array_impl

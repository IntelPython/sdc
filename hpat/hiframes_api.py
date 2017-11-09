from __future__ import print_function, division, absolute_import

import numba
from numba import typeinfer, ir
from numba import types
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution

class Filter(ir.Stmt):
    def __init__(self, df_out, df_in, bool_arr, df_vars, loc):
        self.df_out = df_out
        self.df_in = df_in
        self.bool_arr = bool_arr
        # needs df columns for type inference stage
        self.df_vars = df_vars
        self.loc = loc

    def __repr__(self):
        out_cols = ""
        for (c, v) in self.df_vars[self.df_out].items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)
        in_cols = ""
        for (c, v) in self.df_vars[self.df_in].items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        return "filter: {} = {} [cond: {}] ".format(df_out_str, df_in_str,
                                                                self.bool_arr)

def filter_array_analysis(filter_node, equiv_set, typemap, array_analysis):
    post = []
    df_vars = filter_node.df_vars
    df_in_vars = df_vars[filter_node.df_in]
    df_out_vars = df_vars[filter_node.df_out]

    # arrays of input df have same size in first dimension
    all_shapes = []
    for _, col_var in df_in_vars.items():
        col_shape = equiv_set.get_shape(col_var)
        all_shapes.append(col_shape[0])
    equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for _, col_var in df_out_vars.items():
        typ = typemap[col_var.name]
        (shape, c_post) = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None)
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var)
    equiv_set.insert_equiv(*all_shapes)

    return [], post

numba.array_analysis.array_analysis_extensions[Filter] = filter_array_analysis

def filter_distributed_analysis(filter_node, array_dists):
    df_vars = filter_node.df_vars
    df_in_vars = df_vars[filter_node.df_in]
    df_out_vars = df_vars[filter_node.df_out]

    # input columns have same distribution
    in_dist = Distribution.OneD
    for _, col_var in df_in_vars.items():
        in_dist = Distribution(min(in_dist.value, array_dists[col_var.name].value))
    for _, col_var in df_in_vars.items():
        array_dists[col_var.name] = in_dist

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[col_var.name].value))
    for _, col_var in df_out_vars.items():
        array_dists[col_var.name] = out_dist

    return

distributed_analysis.distributed_analysis_extensions[Filter] = filter_distributed_analysis

def filter_distributed_run(filter_node, typemap, calltypes):
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    df_vars = filter_node.df_vars
    df_in_vars = df_vars[filter_node.df_in]
    df_out_vars = df_vars[filter_node.df_out]
    loc = filter_node.loc
    bool_arr = filter_node.bool_arr

    out = []
    for col_name, col_in_var in df_in_vars.items():
        col_out_var = df_out_vars[col_name]
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
    df_vars = filter_node.df_vars
    df_in_vars = df_vars[filter_node.df_in]
    df_out_vars = df_vars[filter_node.df_out]
    for col_name, col_var in df_in_vars.items():
        out_col_var = df_out_vars[col_name]
        typeinferer.constraints.append(typeinfer.Propagate(dst=out_col_var.name,
                                              src=col_var.name, loc=filter_node.loc))
    return

typeinfer.typeinfer_extensions[Filter] = filter_typeinfer


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

from numba.typing.arraydecl import _expand_integer

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

def fix_df_array(c):
    return c

from numba.extending import overload
from hpat.str_ext import StringType
from hpat.str_arr_ext import StringArray, StringArrayType

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

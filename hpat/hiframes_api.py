from __future__ import print_function, division, absolute_import

import numba
from numba import typeinfer, ir
from numba.typing import signature
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
        return "filter: {} = {}[{}]".format(self.df_out, self.df_in,
                                                                self.bool_arr)

def filter_array_analysis(filter_node, array_analysis):
    df_vars = filter_node.df_vars
    df_in_vars = df_vars[filter_node.df_in]
    df_out_vars = df_vars[filter_node.df_out]

    # arrays of input df have same size in last dimension
    c_in = array_analysis._get_next_class()
    for _, col_var in df_in_vars.items():
        c_in = array_analysis._merge_classes(c_in,
                            array_analysis.array_shape_classes[col_var.name][0])

    # create correlations for output arrays
    for _, col_var in df_out_vars.items():
        array_analysis._add_array_corr(col_var.name)

    # arrays of output df have same size in last dimension
    c_out = array_analysis._get_next_class()
    for _, col_var in df_out_vars.items():
        c_out = array_analysis._merge_classes(c_out,
                            array_analysis.array_shape_classes[col_var.name][0])

    # gen size variable for an output column
    out_col = list(df_out_vars.items())[0][1]
    size_nodes = array_analysis._gen_size_call(out_col, 0)
    size_var = size_nodes[-1].target
    array_analysis.class_sizes[c_out] = [size_var]
    return size_nodes

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

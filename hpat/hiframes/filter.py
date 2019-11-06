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


from __future__ import print_function, division, absolute_import
from collections import defaultdict
import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.ir_utils import visit_vars_inner, replace_vars_inner
from numba.typing import signature
import hpat
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution
from hpat.utils import debug_prints
from hpat.str_arr_ext import string_array_type
from hpat.hiframes.split_impl import string_array_split_view_type


class Filter(ir.Stmt):
    def __init__(self, df_out, df_in, bool_arr, out_vars, in_vars, loc):
        self.df_out = df_out
        self.df_in = df_in
        self.df_out_vars = out_vars
        self.df_in_vars = in_vars
        self.bool_arr = bool_arr
        self.loc = loc

    def __repr__(self):  # pragma: no cover
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
    from hpat.str_ext import list_string_array_type

    # arrays of input df have same size in first dimension
    all_shapes = []
    index_typ = typemap[filter_node.bool_arr.name]
    # add shape for bool array indices
    if isinstance(index_typ, types.Array) and index_typ.dtype == types.bool_:
        col_shape = equiv_set.get_shape(filter_node.bool_arr)
        all_shapes.append(col_shape[0])
    for _, col_var in filter_node.df_in_vars.items():
        typ = typemap[col_var.name]
        # TODO handle list_string_array_type in other nodes
        if typ in (string_array_type, list_string_array_type,
                   string_array_split_view_type):
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
        if typ in (string_array_type, list_string_array_type,
                   string_array_split_view_type):
            continue
        (shape, c_post) = array_analysis._gen_shape_call(
            equiv_set, col_var, typ.ndim, None)
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
        in_dist = Distribution(
            min(in_dist.value, array_dists[col_var.name].value))

    # bool arr
    if filter_node.bool_arr.name in array_dists:
        in_dist = Distribution(
            min(in_dist.value, array_dists[filter_node.bool_arr.name].value))
    for _, col_var in filter_node.df_in_vars.items():
        array_dists[col_var.name] = in_dist
    if filter_node.bool_arr.name in array_dists:
        array_dists[filter_node.bool_arr.name] = in_dist

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in filter_node.df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for _, col_var in filter_node.df_out_vars.items():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        if filter_node.bool_arr.name in array_dists:
            array_dists[filter_node.bool_arr.name] = out_dist
        for _, col_var in filter_node.df_in_vars.items():
            array_dists[col_var.name] = out_dist

    return


distributed_analysis.distributed_analysis_extensions[Filter] = filter_distributed_analysis


def build_filter_definitions(filter_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in filter_node.df_out_vars.values():
        definitions[col_var.name].append(filter_node)

    return definitions


ir_utils.build_defs_extensions[Filter] = build_filter_definitions


def filter_distributed_run(filter_node, array_dists, typemap, calltypes, typingctx, targetctx, dist_pass):
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
    if debug_prints():  # pragma: no cover
        print("visiting filter vars for:", filter_node)
        print("cbdata: ", sorted(cbdata.items()))

    filter_node.bool_arr = visit_vars_inner(
        filter_node.bool_arr, callback, cbdata)

    for col_name in list(filter_node.df_in_vars.keys()):
        filter_node.df_in_vars[col_name] = visit_vars_inner(
            filter_node.df_in_vars[col_name], callback, cbdata)
    for col_name in list(filter_node.df_out_vars.keys()):
        filter_node.df_out_vars[col_name] = visit_vars_inner(
            filter_node.df_out_vars[col_name], callback, cbdata)


# add call to visit filter variable
ir_utils.visit_vars_extensions[Filter] = visit_vars_filter


def remove_dead_filter(filter_node, lives, arg_aliases, alias_map, func_ir, typemap):
    if not hpat.hiframes.api.enable_hiframes_remove_dead:
        return filter_node

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


def filter_usedefs(filter_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # bool array and input columns are used
    use_set.add(filter_node.bool_arr.name)
    use_set.update({v.name for v in filter_node.df_in_vars.values()})

    # output columns are defined
    def_set.update({v.name for v in filter_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Filter] = filter_usedefs


def get_copies_filter(filter_node, typemap):
    # filter doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in filter_node.df_out_vars.values())
    return set(), kill_set


ir_utils.copy_propagate_extensions[Filter] = get_copies_filter


def apply_copies_filter(filter_node, var_dict, name_var_table,
                        typemap, calltypes, save_copies):
    """apply copy propagate in filter node"""
    filter_node.bool_arr = replace_vars_inner(filter_node.bool_arr, var_dict)

    for col_name in list(filter_node.df_in_vars.keys()):
        filter_node.df_in_vars[col_name] = replace_vars_inner(
            filter_node.df_in_vars[col_name], var_dict)
    for col_name in list(filter_node.df_out_vars.keys()):
        filter_node.df_out_vars[col_name] = replace_vars_inner(
            filter_node.df_out_vars[col_name], var_dict)

    return


ir_utils.apply_copy_propagate_extensions[Filter] = apply_copies_filter

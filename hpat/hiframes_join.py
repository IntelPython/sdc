from __future__ import print_function, division, absolute_import

import numba
from numba import typeinfer, ir, ir_utils, config
from numba.ir_utils import visit_vars_inner, replace_vars_inner
from numba.typing import signature
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution
from hpat.str_arr_ext import string_array_type

class Join(ir.Stmt):
    def __init__(self, df_out, left_df, right_df, left_key, right_key, df_vars, loc):
        self.df_out = df_out
        self.left_df = left_df
        self.right_df = right_df
        self.left_key = left_key
        self.right_key = right_key
        self.df_out_vars = df_vars[self.df_out]
        self.left_vars = df_vars[left_df]
        self.right_vars = df_vars[right_df]
        # needs df columns for type inference stage
        self.df_vars = df_vars
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)

        in_cols = ""
        for (c, v) in self.left_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_left_str = "{}{{{}}}".format(self.left_df, in_cols)

        in_cols = ""
        for (c, v) in self.right_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_right_str = "{}{{{}}}".format(self.right_df, in_cols)
        return "join [{}={}]: {} , {}, {}".format(self.left_key,
            self.right_key, df_out_str, df_left_str, df_right_str)

def join_array_analysis(join_node, equiv_set, typemap, array_analysis):
    post = []
    # empty join nodes should be deleted in remove dead
    assert len(join_node.df_out_vars) > 0, "empty join in array analysis"

    # arrays of left_df and right_df have same size in first dimension
    all_shapes = []
    for _, col_var in (list(join_node.left_vars.items())
                        +list(join_node.right_vars.items())):
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
    for _, col_var in join_node.df_out_vars.items():
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

numba.array_analysis.array_analysis_extensions[Join] = join_array_analysis

def join_distributed_analysis(join_node, array_dists):

    # input columns have same distribution
    in_dist = Distribution.OneD
    for _, col_var in (list(join_node.left_vars.items())
                        +list(join_node.right_vars.items())):
        in_dist = Distribution(min(in_dist.value, array_dists[col_var.name].value))


    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in join_node.df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for _, col_var in join_node.df_out_vars.items():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        array_dists[join_node.bool_arr.name] = out_dist
        for _, col_var in join_node.df_in_vars.items():
            array_dists[col_var.name] = out_dist

    return

distributed_analysis.distributed_analysis_extensions[Join] = join_distributed_analysis

def join_distributed_run(join_node, typemap, calltypes):
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    loc = join_node.loc

    out = []
    # TODO: implement

    return out

distributed.distributed_run_extensions[Join] = join_distributed_run


def join_typeinfer(join_node, typeinferer):
    # TODO: consider keys with same name, cols with suffix
    for col_name, col_var in (list(join_node.left_vars.items())
                        +list(join_node.right_vars.items())):
        out_col_var = join_node.df_out_vars[col_name]
        typeinferer.constraints.append(typeinfer.Propagate(dst=out_col_var.name,
                                              src=col_var.name, loc=join_node.loc))
    return

typeinfer.typeinfer_extensions[Join] = join_typeinfer


def visit_vars_join(join_node, callback, cbdata):
    if config.DEBUG_ARRAY_OPT == 1:  # pragma: no cover
        print("visiting join vars for:", join_node)
        print("cbdata: ", sorted(cbdata.items()))

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = visit_vars_inner(join_node.left_vars[col_name], callback, cbdata)
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = visit_vars_inner(join_node.right_vars[col_name], callback, cbdata)
    # output
    for col_name in list(join_node.df_out_vars.keys()):
        join_node.df_out_vars[col_name] = visit_vars_inner(join_node.df_out_vars[col_name], callback, cbdata)

# add call to visit Join variable
ir_utils.visit_vars_extensions[Join] = visit_vars_join

def remove_dead_join(join_node, lives, arg_aliases, alias_map, typemap):
    # if an output column is dead, the related input column is not needed
    # anymore in the join
    dead_cols = []

    for col_name, col_var in join_node.df_out_vars.items():
        if col_var.name not in lives:
            dead_cols.append(col_name)

    for cname in dead_cols:
        assert cname in join_node.left_vars or cname in join_node.right_vars
        join_node.left_vars.pop(cname, None)
        join_node.right_vars.pop(cname, None)
        join_node.df_out_vars.pop(cname)

    # remove empty join node
    if len(join_node.df_out_vars) == 0:
        return None

    return join_node

ir_utils.remove_dead_extensions[Join] = remove_dead_join

def join_usedefs(join_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # input columns are used
    use_set.update({v.name for v in join_node.left_vars.values()})
    use_set.update({v.name for v in join_node.right_vars.values()})

    # output columns are defined
    def_set.update({v.name for v in join_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)

numba.analysis.ir_extension_usedefs[Join] = join_usedefs

def get_copies_join(join_node, typemap):
    # join doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in join_node.df_out_vars.values())
    return set(), kill_set

ir_utils.copy_propagate_extensions[Join] = get_copies_join

def apply_copies_join(join_node, var_dict, name_var_table, ext_func, ext_data,
                        typemap, calltypes, save_copies):
    """apply copy propagate in join node"""

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = replace_vars_inner(join_node.left_vars[col_name], var_dict)
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = replace_vars_inner(join_node.right_vars[col_name], var_dict)
    # output
    for col_name in list(join_node.df_out_vars.keys()):
        join_node.df_out_vars[col_name] = replace_vars_inner(join_node.df_out_vars[col_name], var_dict)

    return

ir_utils.apply_copy_propagate_extensions[Join] = apply_copies_join

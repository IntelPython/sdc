from __future__ import print_function, division, absolute_import

import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.ir_utils import visit_vars_inner, replace_vars_inner
from numba.typing import signature
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type


class Aggregate(ir.Stmt):
    def __init__(self, df_out, df_in, key_name, df_out_vars, df_in_vars,
                                             key_arr, agg_func, out_typs, loc):
        # name of output dataframe (just for printing purposes)
        self.df_out = df_out
        # name of input dataframe (just for printing purposes)
        self.df_in = df_in
        # key name (for printing)
        self.key_name = key_name

        self.df_out_vars = df_out_vars
        self.df_in_vars = df_in_vars
        self.key_arr = key_arr

        self.agg_func = agg_func
        self.out_typs = out_typs

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
        return "aggregate: {} = {} [key: {}] ".format(df_out_str, df_in_str,
                                                    self.key_name)


def aggregate_typeinfer(aggregate_node, typeinferer):
    for out_name, out_var in aggregate_node.df_out_vars.items():
        typ = aggregate_node.out_typs[out_name]
        # TODO: are there other non-numpy array types?
        if typ == string_type:
            arr_type = string_array_type
        else:
            arr_type = types.Array(typ, 1, 'C')

        typeinferer.lock_type(out_var.name, arr_type, loc=aggregate_node.loc)

    return

typeinfer.typeinfer_extensions[Aggregate] = aggregate_typeinfer


def aggregate_usedefs(aggregate_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # key array and input columns are used
    use_set.add(aggregate_node.key_arr.name)
    use_set.update({v.name for v in aggregate_node.df_in_vars.values()})

    # output columns are defined
    def_set.update({v.name for v in aggregate_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Aggregate] = aggregate_usedefs


def remove_dead_aggregate(aggregate_node, lives, arg_aliases, alias_map, typemap):
    #
    dead_cols = []

    for col_name, col_var in aggregate_node.df_out_vars.items():
        if col_var.name not in lives:
            dead_cols.append(col_name)

    for cname in dead_cols:
        aggregate_node.df_in_vars.pop(cname)
        aggregate_node.df_out_vars.pop(cname)

    # TODO: test agg remove
    # remove empty aggregate node
    if len(aggregate_node.df_in_vars) == 0:
        return None

    return aggregate_node


ir_utils.remove_dead_extensions[Aggregate] = remove_dead_aggregate

def get_copies_aggregate(aggregate_node, typemap):
    # aggregate doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in aggregate_node.df_out_vars.values())
    return set(), kill_set


ir_utils.copy_propagate_extensions[Aggregate] = get_copies_aggregate


def apply_copies_aggregate(aggregate_node, var_dict, name_var_table,
                        typemap, calltypes, save_copies):
    """apply copy propagate in aggregate node"""
    aggregate_node.key_arr = replace_vars_inner(aggregate_node.key_arr,
                                                                     var_dict)

    for col_name in list(aggregate_node.df_in_vars.keys()):
        aggregate_node.df_in_vars[col_name] = replace_vars_inner(
            aggregate_node.df_in_vars[col_name], var_dict)
    for col_name in list(aggregate_node.df_out_vars.keys()):
        aggregate_node.df_out_vars[col_name] = replace_vars_inner(
            aggregate_node.df_out_vars[col_name], var_dict)

    return


ir_utils.apply_copy_propagate_extensions[Aggregate] = apply_copies_aggregate


def visit_vars_aggregate(aggregate_node, callback, cbdata):
    if config.DEBUG_ARRAY_OPT == 1:  # pragma: no cover
        print("visiting aggregate vars for:", aggregate_node)
        print("cbdata: ", sorted(cbdata.items()))

    aggregate_node.key_arr = visit_vars_inner(
        aggregate_node.key_arr, callback, cbdata)

    for col_name in list(aggregate_node.df_in_vars.keys()):
        aggregate_node.df_in_vars[col_name] = visit_vars_inner(
            aggregate_node.df_in_vars[col_name], callback, cbdata)
    for col_name in list(aggregate_node.df_out_vars.keys()):
        aggregate_node.df_out_vars[col_name] = visit_vars_inner(
            aggregate_node.df_out_vars[col_name], callback, cbdata)


# add call to visit aggregate variable
ir_utils.visit_vars_extensions[Aggregate] = visit_vars_aggregate


def aggregate_array_analysis(aggregate_node, equiv_set, typemap,
                                                            array_analysis):
    # empty aggregate nodes should be deleted in remove dead
    assert len(aggregate_node.df_in_vars) > 0, ("empty aggregate in array"
                                                                   "analysis")

    # arrays of input df have same size in first dimension as key array
    col_shape = equiv_set.get_shape(aggregate_node.key_arr)
    all_shapes = [col_shape[0]]
    for _, col_var in aggregate_node.df_in_vars.items():
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
    post = []
    all_shapes = []
    for _, col_var in aggregate_node.df_out_vars.items():
        typ = typemap[col_var.name]
        if typ == string_array_type:
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


numba.array_analysis.array_analysis_extensions[Aggregate] = aggregate_array_analysis


def aggregate_distributed_analysis(aggregate_node, array_dists):

    # input columns have same distribution
    in_dist = Distribution.OneD
    for _, col_var in aggregate_node.df_in_vars.items():
        in_dist = Distribution(
            min(in_dist.value, array_dists[col_var.name].value))

    # key arr
    in_dist = Distribution(
        min(in_dist.value, array_dists[aggregate_node.key_arr.name].value))
    for _, col_var in aggregate_node.df_in_vars.items():
        array_dists[col_var.name] = in_dist
    array_dists[aggregate_node.key_arr.name] = in_dist

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in aggregate_node.df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for _, col_var in aggregate_node.df_out_vars.items():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        array_dists[aggregate_node.key_arr.name] = out_dist
        for _, col_var in aggregate_node.df_in_vars.items():
            array_dists[col_var.name] = out_dist

    return


distributed_analysis.distributed_analysis_extensions[Aggregate] = aggregate_distributed_analysis
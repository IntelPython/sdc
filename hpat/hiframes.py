from __future__ import print_function, division, absolute_import
import types as pytypes # avoid confusion with numba.types

import numba
from numba import ir, analysis, types, config, numpy_support, typeinfer
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc)
from hpat import distributed
from hpat.distributed import Distribution
import pandas

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
    out_dist = Distribution.OneD
    for _, col_var in df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(min(out_dist.value, array_dists[col_var.name].value))
    for _, col_var in df_out_vars.items():
        array_dists[col_var.name] = out_dist

    return

distributed.distributed_analysis_extensions[Filter] = filter_distributed_analysis

class HiFrames(object):
    """analyze and transform hiframes calls"""
    def __init__(self, func_ir):
        self.func_ir = func_ir
        # varname -> 'str'
        self.str_const_table = {}
        # var -> list
        self.map_calls = {}
        self.pd_globals = []
        self.pd_df_calls = []
        # df_var -> {col1:col1_var ...}
        self.df_vars = {}

    def run(self):
        dprint_func_ir(self.func_ir, "starting hiframes")
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            new_body = []
            for inst in self.func_ir.blocks[label].body:
                if isinstance(inst, ir.Assign):
                    inst_list = self._run_assign(inst)
                    if inst_list is not None:
                        new_body.extend(inst_list)
                else:
                    new_body.append(inst)
            self.func_ir.blocks[label].body = new_body
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        dprint_func_ir(self.func_ir, "after hiframes")
        if config.DEBUG_ARRAY_OPT==1:
            print("df_vars: ", self.df_vars)
        return

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value
        # lhs = pandas
        if (isinstance(rhs, ir.Global) and isinstance(rhs.value, pytypes.ModuleType)
                    and rhs.value==pandas):
            self.pd_globals.append(lhs)

        if isinstance(rhs, ir.Expr):
            # df_call = pd.DataFrame
            if rhs.op=='getattr' and rhs.value.name in self.pd_globals and rhs.attr=='DataFrame':
                self.pd_df_calls.append(lhs)

            # df = pd.DataFrame(map_var)
            if rhs.op=='call' and rhs.func.name in self.pd_df_calls:
                # only map input allowed now
                assert len(rhs.args) is 1 and rhs.args[0].name in self.map_calls

                self.df_vars[lhs] = self._process_df_build_map(
                                            self.map_calls[rhs.args[0].name])
                # remove DataFrame call
                return []

            # d = df['column']
            if (rhs.op == 'static_getitem' and rhs.value.name in self.df_vars
                                            and isinstance(rhs.index, str)):
                df = rhs.value.name
                assign.value = self.df_vars[df][rhs.index]

            # df1 = df[df.A > .5]
            if (rhs.op == 'getitem' and rhs.value.name in self.df_vars):
                # output df1 has same columns as df, create new vars
                scope = assign.target.scope
                loc = assign.target.loc
                self.df_vars[lhs] = {}
                for col, _ in self.df_vars[rhs.value.name].items():
                    self.df_vars[lhs][col] = ir.Var(scope, mk_unique_var(col),
                                                                            loc)
                return [Filter(lhs, rhs.value.name, rhs.index, self.df_vars, rhs.loc)]

            # d = df.column
            if rhs.op=='getattr' and rhs.value.name in self.df_vars:
                df = rhs.value.name
                df_cols = self.df_vars[df]
                assert rhs.attr in df_cols
                assign.value = df_cols[rhs.attr]

            if rhs.op == 'build_map':
                self.map_calls[lhs] = rhs.items

        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.df_vars:
            self.df_vars[lhs] = self.df_vars[rhs.name]
        if isinstance(rhs, ir.Const) and isinstance(rhs.value, str):
            self.str_const_table[lhs] = rhs.value
        return [assign]

    def _process_df_build_map(self, items_list):
        df_cols = {}
        for item in items_list:
            col_var = item[0].name
            assert col_var in self.str_const_table
            col_name = self.str_const_table[col_var]
            df_cols[col_name] = item[1]
        return df_cols


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

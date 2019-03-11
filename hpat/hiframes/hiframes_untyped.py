from __future__ import print_function, division, absolute_import
import warnings
from collections import namedtuple
import itertools

import numba
from numba import ir, ir_utils, types
from numba import compiler as numba_compiler
from numba.targets.registry import CPUDispatcher

from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc, remove_dels,
                            get_name_var_table, replace_var_names,
                            add_offset_to_labels, get_ir_of_code, find_const,
                            compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard, require, get_definition,
                            build_definitions, replace_vars_stmt,
                            replace_vars_inner, find_build_sequence)

from numba.inline_closurecall import inline_closure_call
from numba.analysis import compute_cfg_from_blocks

import hpat
from hpat import utils, pio, parquet_pio, config
from hpat.hiframes import filter, join, aggregate, sort
from hpat.utils import (get_constant, NOT_CONSTANT, debug_prints,
    inline_new_blocks, ReplaceFunc, is_call)
import hpat.hiframes.api
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type
from hpat import csv_ext

import pandas as pd
import numpy as np
import math
from hpat.parquet_pio import ParquetHandler
from hpat.hiframes.pd_timestamp_ext import (datetime_date_type,
                                    datetime_date_to_int, int_to_datetime_date)
from hpat.hiframes.pd_series_ext import SeriesType
from hpat.hiframes.pd_categorical_ext import PDCategoricalDtype, CategoricalArray
from hpat.hiframes.rolling import get_rolling_setup_args, supported_rolling_funcs
from hpat.hiframes.aggregate import get_agg_func, supported_agg_funcs
import hpat.hiframes.pd_dataframe_ext


def remove_hiframes(rhs, lives, call_list):
    # used in stencil generation of rolling
    if len(call_list) == 1 and call_list[0] in [int, min, max, abs]:
        return True
    # used in stencil generation of rolling
    if (len(call_list) == 1 and isinstance(call_list[0], CPUDispatcher)
            and call_list[0].py_func == numba.stencilparfor._compute_last_ind):
        return True
    # used in stencil generation of rolling
    if call_list == ['ceil', math]:
        return True
    if (len(call_list) == 4 and call_list[1:] == ['api', 'hiframes', hpat] and
            call_list[0] in ['fix_df_array', 'fix_rolling_array',
            'concat', 'count', 'mean', 'quantile', 'var',
            'str_contains_regex', 'str_contains_noregex', 'column_sum',
            'nunique', 'init_series', 'init_datetime_index']):
        return True
    if (len(call_list) == 4 and call_list[1:] == ['series_kernels', 'hiframes', hpat] and
            call_list[0]
            in ['_sum_handle_nan', '_mean_handle_nan', '_var_handle_nan']):
        return True
    if call_list == ['dist_return', 'distributed_api', hpat]:
        return True
    if call_list == ['init_dataframe', 'pd_dataframe_ext', 'hiframes', hpat]:
        return True
    # if call_list == ['set_parent_dummy', 'pd_dataframe_ext', 'hiframes', hpat]:
    #     return True
    if call_list == ['rolling_dummy', 'pd_rolling_ext', 'hiframes', hpat]:
        return True
    if call_list == ['agg_typer', 'api', 'hiframes', hpat]:
        return True
    if call_list == [list]:
        return True
    if call_list == ['groupby']:
        return True
    if call_list == ['rolling']:
        return True
    if call_list == [pd.api.types.CategoricalDtype]:
        return True
    # TODO: move to Numba
    if call_list == ['empty_inferred', 'ndarray', 'unsafe', numba]:
        return True
    if call_list == ['chain', itertools]:
        return True
    return False


numba.ir_utils.remove_call_handlers.append(remove_hiframes)

class HiFrames(object):
    """analyze and transform hiframes calls"""

    def __init__(self, func_ir, typingctx, args, _locals, metadata):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals
        self.metadata = metadata
        ir_utils._max_label = max(func_ir.blocks.keys())
        # replace inst variables as determined previously during the pass
        # currently use to keep lhs of Arg nodes intact
        self.replace_var_dict = {}

        # df_var -> {col1:col1_var ...}
        self.df_vars = {}
        # df_var -> label where it is defined
        self.df_labels = {}

        self.arrow_tables = {}
        self.reverse_copies = {}
        self.pq_handler = ParquetHandler(
            func_ir, typingctx, args, _locals, self.reverse_copies)
        self.h5_handler = pio.PIO(self.func_ir, _locals, self.reverse_copies)


    def run(self):
        # FIXME: see why this breaks test_kmeans
        # remove_dels(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "starting hiframes")
        self._handle_metadata()
        blocks = self.func_ir.blocks
        # topo_order necessary since df vars need to be found before use
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            self._get_reverse_copies(blocks[label].body)
            new_body = []
            replaced = False
            self._working_body = new_body
            for i, inst in enumerate(block.body):
                ir_utils.replace_vars_stmt(inst, self.replace_var_dict)
                out_nodes = [inst]

                # handle potential dataframe set column here
                # df['col'] = arr
                if (isinstance(inst, ir.StaticSetItem)
                        and isinstance(inst.index, str)):
                    # cfg needed for set df column
                    cfg = compute_cfg_from_blocks(blocks)
                    out_nodes = self._run_df_set_column(inst, label, cfg)
                elif isinstance(inst, ir.Assign):
                    out_nodes = self._run_assign(inst, label)
                elif isinstance(inst, ir.Return):
                    out_nodes = self._run_return(inst)

                if isinstance(out_nodes, list):
                    # TODO: fix scope/loc
                    new_body.extend(out_nodes)
                if isinstance(out_nodes, ReplaceFunc):
                    rp_func = out_nodes
                    if rp_func.pre_nodes is not None:
                        new_body.extend(rp_func.pre_nodes)
                    # replace inst.value to a call with target args
                    # as expected by inline_closure_call
                    inst.value = ir.Expr.call(None, rp_func.args, (), inst.loc)
                    block.body = new_body + block.body[i:]
                    inline_closure_call(self.func_ir, rp_func.glbls,
                        block, len(new_body), rp_func.func, work_list=work_list)
                    replaced = True
                    break
                if isinstance(out_nodes, dict):
                    block.body = new_body + block.body[i:]
                    # TODO: insert new blocks in current spot of work_list
                    # instead of append?
                    # TODO: rename variables, fix scope/loc
                    inline_new_blocks(self.func_ir, block, len(new_body), out_nodes, work_list)
                    replaced = True
                    break
            if not replaced:
                blocks[label].body = new_body

        self.func_ir._definitions = build_definitions(blocks)
        # XXX: remove dead here fixes h5 slice issue
        # iterative remove dead to make sure all extra code (e.g. df vars) is removed
        # while remove_dead(blocks, self.func_ir.arg_names, self.func_ir):
        #     pass
        self.func_ir._definitions = build_definitions(blocks)
        dprint_func_ir(self.func_ir, "after hiframes")
        if debug_prints():  # pragma: no cover
            print("df_vars: ", self.df_vars)
        return

    def _run_assign(self, assign, label):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            if rhs.op == 'call':
                return self._run_call(assign, label)

            # fix type for f['A'][:] dset reads
            if rhs.op in ('getitem', 'static_getitem'):
                h5_nodes = self.h5_handler.handle_possible_h5_read(
                    assign, lhs, rhs)
                if h5_nodes is not None:
                    return h5_nodes

            # HACK: delete pd.DataFrame({}) nodes to avoid typing errors
            # TODO: remove when dictionaries are implemented and typing works
            if rhs.op == 'getattr':
                val_def = guard(get_definition, self.func_ir, rhs.value)
                if (isinstance(val_def, ir.Global) and val_def.value == pd
                        and rhs.attr in ('DataFrame', 'read_csv',
                                        'read_parquet', 'to_numeric')):
                    # TODO: implement to_numeric in typed pass?
                    return []

            if rhs.op == 'build_map':
                return []

            # HACK: delete pyarrow.parquet.read_table() to avoid typing errors
            if rhs.op == 'getattr' and rhs.attr == 'read_table':
                import pyarrow.parquet as pq
                val_def = guard(get_definition, self.func_ir, rhs.value)
                if isinstance(val_def, ir.Global) and val_def.value == pq:
                    return []

            if (rhs.op == 'getattr' and rhs.value.name in self.arrow_tables
                    and rhs.attr == 'to_pandas'):
                return []

            if rhs.op in ('build_list', 'build_tuple'):
                # if build_list items are constant, add the constant values
                # to the returned list type as metadata. This enables type
                # inference for calls like pd.merge() where the values
                # determine output dataframe type
                # TODO: add proper metadata to Numba types
                # XXX: when constants are used, all the uses of the list object
                # have to be checked since lists are mutable
                try:
                    vals = tuple(find_const(self.func_ir, v) for v in rhs.items)
                    # a = ['A', 'B'] ->
                    # tmp = ['A', 'B']
                    # a = add_consts_to_type(tmp, 'A', 'B')
                    vals_expr = ", ".join("'{}'".format(c) if isinstance(c, str) else "{}".format(c) for c in vals)
                    func_text = "def _build_f(a):\n"
                    func_text += "  return hpat.hiframes.api.add_consts_to_type(a, {})\n".format(vals_expr)
                    loc_vars = {}
                    exec(func_text, {'hpat': hpat}, loc_vars)
                    _build_f = loc_vars['_build_f']
                    target = assign.target
                    tmp_target = ir.Var(
                        target.scope, mk_unique_var(target.name), rhs.loc)
                    tmp_assign = ir.Assign(rhs, tmp_target, rhs.loc)
                    return self._replace_func(
                        _build_f, (tmp_target,), pre_nodes=[tmp_assign])
                except numba.ir_utils.GuardException:
                    pass

        # pass pivot values to df.pivot_table() calls using a meta
        # variable passed as argument. The meta variable's type
        # is set to MetaType with pivot values baked in.
        pivot_key = lhs + ":pivot"
        if pivot_key in self.locals:
            pivot_values = self.locals.pop(pivot_key)
            pivot_call = guard(get_definition, self.func_ir, lhs)
            assert pivot_call is not None
            meta_var = ir.Var(
                assign.target.scope, mk_unique_var('pivot_meta'), rhs.loc)
            meta_assign = ir.Assign(
                ir.Const(0, rhs.loc), meta_var, rhs.loc)
            self._working_body.insert(0, meta_assign)
            pivot_call.kws = list(pivot_call.kws)
            pivot_call.kws.append(('_pivot_values', meta_var))
            self.locals[meta_var.name] = hpat.hiframes.api.MetaType(pivot_values)


        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.df_vars:
            self.df_vars[lhs] = self.df_vars[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.df_labels:
            self.df_labels[lhs] = self.df_labels[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.arrow_tables:
            self.arrow_tables[lhs] = self.arrow_tables[rhs.name]
            return []
        return [assign]

    def _run_call(self, assign, label):
        """handle calls and return new nodes if needed
        """
        lhs = assign.target
        rhs = assign.value

        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs)
        if fdef is None:
            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == 'make_function':
                return [assign]
            warnings.warn(
                "function call couldn't be found for initial analysis")
            return [assign]
        else:
            func_name, func_mod = fdef

        # handling pd.DataFrame() here since input can be constant dictionary
        if fdef == ('DataFrame', 'pandas'):
            return self._handle_pd_DataFrame(assign, lhs, rhs, label)

        # handling pd.read_csv() here since input can have constants
        # like dictionaries for typing
        if fdef == ('read_csv', 'pandas'):
            return self._handle_pd_read_csv(assign, lhs, rhs, label)

        # match flatmap pd.Series(list(itertools.chain(*A))) and flatten
        if fdef == ('Series', 'pandas'):
            return self._handle_pd_Series(assign, lhs, rhs)

        if fdef == ('read_table', 'pyarrow.parquet'):
            return self._handle_pq_read_table(assign, lhs, rhs)

        if (func_name == 'to_pandas' and isinstance(func_mod, ir.Var)
                and func_mod.name in self.arrow_tables):
            return self._handle_pq_to_pandas(assign, lhs, rhs, func_mod, label)

        if fdef == ('read_parquet', 'pandas'):
            return self._handle_pd_read_parquet(assign, lhs, rhs, label)

        if fdef == ('concat', 'pandas'):
            return self._handle_concat(assign, lhs, rhs, label)

        # if fdef == ('crosstab', 'pandas'):
        #     return self._handle_crosstab(lhs, rhs, label)

        if fdef == ('to_numeric', 'pandas'):
            return self._handle_pd_to_numeric(assign, lhs, rhs)

        if fdef == ('read_ros_images', 'hpat.ros'):
            return self._handle_ros(assign, lhs, rhs)

        if isinstance(func_mod, ir.Var) and self._is_df_var(func_mod):
            return self._run_call_df(
                assign, lhs, rhs, func_mod, func_name, label)

        # groupby aggregate
        # e.g. df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
        if isinstance(func_mod, ir.Var) and self._is_df_obj_call(func_mod, 'groupby'):
            return self._handle_aggregate(lhs, rhs, func_mod, func_name, label)

        # rolling window
        # e.g. df.rolling(2).sum
        if isinstance(func_mod, ir.Var) and self._is_df_obj_call(func_mod, 'rolling'):
            return self._handle_rolling(lhs, rhs, func_mod, func_name, label)


        if fdef == ('File', 'h5py'):
            return self.h5_handler._handle_h5_File_call(assign, lhs, rhs)

        if fdef == ('fromfile', 'numpy'):
            return hpat.io._handle_np_fromfile(assign, lhs, rhs)

        if fdef == ('read_xenon', 'hpat.xenon_ext'):
            col_items, nodes = hpat.xenon_ext._handle_read(assign, lhs, rhs, self.func_ir)
            df_nodes, col_map = self._process_df_build_map(col_items)
            self._create_df(lhs.name, col_map, label)
            nodes += df_nodes
            return nodes

        return [assign]

    def _run_call_df(self, assign, lhs, rhs, df_var, func_name, label):

        # df.pivot_table()
        if func_name == 'pivot_table':
            return self._handle_df_pivot_table(lhs, rhs, df_var, label)

        # df.isin()
        if func_name == 'isin':
            return self._handle_df_isin(lhs, rhs, df_var, label)

        # df.append()
        if func_name == 'append':
            return self._handle_df_append(lhs, rhs, df_var, label)

        # df.fillna()
        if func_name == 'fillna':
            return self._handle_df_fillna(lhs, rhs, df_var, label)

        # df.dropna()
        if func_name == 'dropna':
            return self._handle_df_dropna(lhs, rhs, df_var, label)

        # df.drop()
        if func_name == 'drop':
            return self._handle_df_drop(lhs, rhs, df_var, label)

        # df.reset_index(drop=True)
        if func_name == 'reset_index':
            return self._handle_df_reset_index(lhs, rhs, df_var, label)

        if func_name not in ('groupby', 'rolling'):
            raise NotImplementedError(
                "data frame function {} not implemented yet".format(func_name))

        return [assign]

    def _handle_df_isin(self, lhs, rhs, df_var, label):
        other = self._get_arg('isin', rhs.args, dict(rhs.kws), 0, 'values')
        other_colmap = {}
        df_col_map = self._get_df_cols(df_var)
        nodes = []
        df_case = False

        # dataframe case
        if self._is_df_var(other):
            df_case = True
            arg_df_map = self._get_df_cols(other)
            for cname in df_col_map:
                if cname in arg_df_map:
                    other_colmap[cname] = arg_df_map[cname]
        else:
            other_def = guard(get_definition, self.func_ir, other)
            # dict case
            if isinstance(other_def, ir.Expr) and other_def.op == 'build_map':
                for c, v in other_def.items:
                    cname = guard(find_const, self.func_ir, c)
                    if not isinstance(cname, str):
                        raise ValueError("dictionary argument to isin() should have constant keys")
                    other_colmap[cname] = v
            else:
                # general iterable (e.g. list, set) case
                # TODO: handle passed in dict case (pass colname to func?)
                other_colmap = {c: other for c in df_col_map.keys()}

        out_df_map = {}
        isin_func = lambda A, B: hpat.hiframes.api.df_isin(A, B)
        isin_vals_func = lambda A, B: hpat.hiframes.api.df_isin_vals(A, B)
        # create array of False values used when other col not available
        bool_arr_func = lambda A: hpat.hiframes.api.init_series(np.zeros(len(A), np.bool_))
        # use the first array of df to get len. TODO: check for empty df
        false_arr_args = [list(df_col_map.values())[0]]

        for cname, in_var in self.df_vars[df_var.name].items():
            if cname in other_colmap:
                if df_case:
                    func = isin_func
                else:
                    func = isin_vals_func
                other_col_var = other_colmap[cname]
                args = [in_var, other_col_var]
            else:
                func = bool_arr_func
                args = false_arr_args
            f_block = compile_to_numba_ir(func, {'hpat': hpat, 'np': np}).blocks.popitem()[1]
            replace_arg_nodes(f_block, args)
            nodes += f_block.body[:-2]
            out_df_map[cname] = nodes[-1].target

        self._create_df(lhs.name, out_df_map, label)
        return nodes

    def _handle_df_append(self, lhs, rhs, df_var, label):
        other = self._get_arg('append', rhs.args, dict(rhs.kws), 0, 'other')
        # only handles df or list of df input
        # TODO: check for series/dict/list input
        # TODO: enforce ignore_index=True?
        # single df case
        if self._is_df_var(other):
            return self._handle_concat_df(lhs, [df_var, other], label)
        # list of dfs
        df_list = guard(get_definition, self.func_ir, other)
        if len(df_list.items) > 0 and self._is_df_var(df_list.items[0]):
            return self._handle_concat_df(lhs, [df_var] + df_list.items, label)
        raise ValueError("invalid df.append() input. Only dataframe and list"
                         " of dataframes supported")

    def _handle_df_fillna(self, lhs, rhs, df_var, label):
        nodes = []
        inplace_default = ir.Var(lhs.scope, mk_unique_var("fillna_default"), lhs.loc)
        nodes.append(ir.Assign(ir.Const(False, lhs.loc), inplace_default, lhs.loc))
        val_var = self._get_arg('fillna', rhs.args, dict(rhs.kws), 0, 'value')
        inplace_var = self._get_arg('fillna', rhs.args, dict(rhs.kws), 3, 'inplace', default=inplace_default)

        _fillna_func = lambda A, val, inplace: A.fillna(val, inplace=inplace)
        out_col_map = {}
        for cname, in_var in self._get_df_cols(df_var).items():
            f_block = compile_to_numba_ir(_fillna_func, {}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [in_var, val_var, inplace_var])
            nodes += f_block.body[:-2]
            out_col_map[cname] = nodes[-1].target

        # create output df if not inplace
        if (inplace_var.name == inplace_default.name
                or guard(find_const, self.func_ir, inplace_var) == False):
            self._create_df(lhs.name, out_col_map, label)
        return nodes

    def _handle_df_reset_index(self, lhs, rhs, df_var, label):
        nodes = []
        drop_var = self._get_arg('reset_index', rhs.args, dict(rhs.kws), 1, 'drop')
        assert guard(find_const, self.func_ir, drop_var) == True

        inplace_default = False
        inplace_var = self._get_arg('reset_index', rhs.args, dict(rhs.kws), 3, 'inplace', default=inplace_default)

        out_col_map = self._get_df_cols(df_var).copy()

        # create output df if not inplace
        if (inplace_var == False
                or guard(find_const, self.func_ir, inplace_var) == False):
            self._create_df(lhs.name, out_col_map, label)
        return nodes

    def _handle_df_dropna(self, lhs, rhs, df_var, label):
        nodes = []
        inplace_default = ir.Var(lhs.scope, mk_unique_var("dropna_default"), lhs.loc)
        nodes.append(ir.Assign(ir.Const(False, lhs.loc), inplace_default, lhs.loc))
        inplace_var = self._get_arg('dropna', rhs.args, dict(rhs.kws), 4, 'inplace', default=inplace_default)

        col_names = self._get_df_col_names(df_var)
        col_vars = self._get_df_col_vars(df_var)
        arg_names = ", ".join([mk_unique_var(cname).replace('.', '_') for cname in col_names])
        out_names = ", ".join([mk_unique_var(cname).replace('.', '_') for cname in col_names])

        func_text = "def _dropna_imp({}, inplace):\n".format(arg_names)
        func_text += "  ({},) = hpat.hiframes.api.dropna(({},), inplace)\n".format(
            out_names, arg_names)
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _dropna_imp = loc_vars['_dropna_imp']

        f_block = compile_to_numba_ir(_dropna_imp, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, col_vars + [inplace_var])
        nodes += f_block.body[:-3]

        # extract column vars from output
        out_col_map = {}
        for i, cname in enumerate(col_names):
            out_col_map[cname] = nodes[-len(col_names) + i].target

        # create output df if not inplace
        if (inplace_var.name == inplace_default.name
                or guard(find_const, self.func_ir, inplace_var) == False):
            self._create_df(lhs.name, out_col_map, label)
        else:
            # assign back to column vars for inplace case
            for i in range(len(col_vars)):
                c_var = col_vars[i]
                dropped_var = list(out_col_map.values())[i]
                nodes.append(ir.Assign(dropped_var, c_var, lhs.loc))
        return nodes

    def _handle_df_drop(self, lhs, rhs, df_var, label):
        # df.drop(labels=None, axis=0, index=None, columns=None, level=None,
        #         inplace=False, errors='raise')
        kws = dict(rhs.kws)
        labels_var = self._get_arg('drop', rhs.args, kws, 0, 'labels', '')
        axis_var = self._get_arg('drop', rhs.args, kws, 1, 'axis', '')
        labels = self._get_str_or_list(labels_var, default='')
        axis = guard(find_const, self.func_ir, axis_var)

        if labels != '' and axis is not None:
            if axis != 1:
                raise ValueError("only dropping columns (axis=1) supported")
            columns = labels
        else:
            columns_var = self._get_arg('drop', rhs.args, kws, 3, 'columns', '')
            err_msg = ("columns argument (constant string list) "
                       "or labels and axis required")
            columns = self._get_str_or_list(columns_var, err_msg=err_msg)

        inplace_var = self._get_arg('drop', rhs.args, kws, 5, 'inplace', '')
        inplace = guard(find_const, self.func_ir, inplace_var)

        if inplace is not None and inplace == True:
            df_label = self.df_labels[df_var.name]
            cfg = compute_cfg_from_blocks(self.func_ir.blocks)
            # dropping columns inplace possible only when it dominates the df
            # creation to keep schema consistent
            if label not in cfg.backbone() and label not in cfg.post_dominators()[df_label]:
                raise ValueError("dropping dataframe columns inplace inside "
                             "conditionals and loops not supported yet")
            # TODO: rename df name
            # TODO: support dropping columns of input dfs (reflection)
            for cname in columns:
                self.df_vars[df_var.name].pop(cname)
            return []

        in_df_map = self._get_df_cols(df_var)
        nodes = []
        out_df_map = {c:_gen_arr_copy(in_df_map[c], nodes)
                      for c in in_df_map.keys() if c not in columns}
        self._create_df(lhs.name, out_df_map, label)
        return nodes

    def _get_reverse_copies(self, body):
        for inst in body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var):
                self.reverse_copies[inst.value.name] = inst.target.name
        return

    def _handle_pd_DataFrame(self, assign, lhs, rhs, label):
        """transform pd.DataFrame({'A': A}) call
        """
        kws = dict(rhs.kws)
        if 'data' in kws:
            data = kws['data']
            if len(rhs.args) != 0:  # pragma: no cover
                raise ValueError(
                    "only data argument suppoted in pd.DataFrame()")
        else:
            if len(rhs.args) != 1:  # pragma: no cover
                raise ValueError(
                    "data argument in pd.DataFrame() expected")
            data = rhs.args[0]

        arg_def = guard(get_definition, self.func_ir, data)
        if (not isinstance(arg_def, ir.Expr)
                or arg_def.op != 'build_map'):  # pragma: no cover
            raise ValueError(
                "Invalid DataFrame() arguments (constant dict of columns expected)")
        nodes, items = self._fix_df_arrays(arg_def.items)

        n_cols = len(items)
        data_args = ", ".join('data{}'.format(i) for i in range(n_cols))
        col_args = ", ".join('col{}'.format(i) for i in range(n_cols))

        func_text = "def _init_df({}, index, {}):\n".format(data_args, col_args)
        func_text += "  return hpat.hiframes.pd_dataframe_ext.init_dataframe({}, index, {})\n".format(
            data_args, col_args)
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars['_init_df']

        # TODO: support index var
        index = ir.Var(lhs.scope, mk_unique_var('df_index_none'), lhs.loc)
        nodes.append(ir.Assign(ir.Const(None, lhs.loc), index, lhs.loc))
        data_vars = [a[1] for a in items]
        col_vars = [a[0] for a in items]
        args = data_vars + [index] + col_vars

        return self._replace_func(_init_df, args,
                    pre_nodes=nodes
                )

        # df_nodes, col_map = self._process_df_build_map(items)
        # nodes += df_nodes
        # self._create_df(lhs.name, col_map, label)
        # # remove DataFrame call
        # return nodes

    def _handle_pd_read_csv(self, assign, lhs, rhs, label):
        """transform pd.read_csv(names=[A], dtype={'A': np.int32}) call
        """
        # schema: pd.read_csv(filepath_or_buffer, sep=',', delimiter=None,
        # header='infer', names=None, index_col=None, usecols=None,
        # squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None,
        # engine=None, converters=None, true_values=None, false_values=None,
        # skipinitialspace=False, skiprows=None, nrows=None, na_values=None,
        # keep_default_na=True, na_filter=True, verbose=False,
        # skip_blank_lines=True, parse_dates=False,
        # infer_datetime_format=False, keep_date_col=False, date_parser=None,
        # dayfirst=False, iterator=False, chunksize=None, compression='infer',
        # thousands=None, decimal=b'.', lineterminator=None, quotechar='"',
        # quoting=0, escapechar=None, comment=None, encoding=None,
        # dialect=None, tupleize_cols=None, error_bad_lines=True,
        # warn_bad_lines=True, skipfooter=0, doublequote=True,
        # delim_whitespace=False, low_memory=True, memory_map=False,
        # float_precision=None)

        kws = dict(rhs.kws)
        fname = self._get_arg('read_csv', rhs.args, kws, 0, 'filepath_or_buffer')
        sep = self._get_str_arg('read_csv', rhs.args, kws, 1, 'sep', ',')
        sep = self._get_str_arg('read_csv', rhs.args, kws, 2, 'delimiter', sep)
        # TODO: header arg
        names_var = self._get_arg('read_csv', rhs.args, kws, 4, 'names')
        err_msg = "pd.read_csv() names should be constant list"
        col_names = self._get_str_or_list(names_var, err_msg=err_msg)
        usecols_var = self._get_arg('read_csv', rhs.args, kws, 6, 'usecols', '')
        usecols = list(range(len(col_names)))
        if usecols_var != '':
            err_msg = "pd.read_csv() usecols should be constant list of ints"
            usecols = self._get_str_or_list(usecols_var, err_msg=err_msg, typ=int)
        # TODO: support other args
        dtype_var = self._get_arg('read_csv', rhs.args, kws, 10, 'dtype')

        dtype_map = guard(get_definition, self.func_ir, dtype_var)
        if (not isinstance(dtype_map, ir.Expr)
                 or dtype_map.op != 'build_map'):  # pragma: no cover
            raise ValueError("pd.read_csv() dtype should be constant dictionary")

        date_cols = []
        if 'parse_dates' in kws:
            date_list = guard(get_definition, self.func_ir, kws['parse_dates'])
            if not isinstance(date_list, ir.Expr) or date_list.op != 'build_list':
                raise ValueError("pd.read_csv() parse_dates should be constant list")
            for v in date_list.items:
                col_val = guard(find_const, self.func_ir, v)
                if col_val is None:
                    raise ValueError("pd.read_csv() parse_dates expects constant column numbers")
                date_cols.append(col_val)

        columns = []
        data_arrs = []
        out_types = []
        for i, (name_var, dtype_var) in enumerate(dtype_map.items):
            # find constant column name
            col_name = guard(find_const, self.func_ir, name_var)
            if col_name is None:  # pragma: no cover
                raise ValueError("dtype column names should be constant")
            columns.append(col_name)
            # get array dtype
            typ = self._get_const_dtype(dtype_var)
            if i in date_cols:
                typ = types.Array(types.NPDatetime('ns'), 1, 'C')
            out_types.append(typ)
            # output array variable
            data_arrs.append(
                ir.Var(lhs.scope, mk_unique_var(col_name), lhs.loc))


        nodes = [csv_ext.CsvReader(
            fname, lhs.name, sep, columns, data_arrs, out_types, usecols, lhs.loc)]

        n_cols = len(dtype_map.items)
        data_args = ", ".join('data{}'.format(i) for i in range(n_cols))

        func_text = "def _init_df({}):\n".format(data_args)
        func_text += "  return hpat.hiframes.pd_dataframe_ext.init_dataframe({}, None, {})\n".format(
            data_args, ", ".join("'{}'".format(c) for c in columns))
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars['_init_df']

        f_block = compile_to_numba_ir(
            _init_df, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, data_arrs)
        nodes += f_block.body[:-2]
        nodes[-1].target = lhs
        return nodes


    def _get_const_dtype(self, dtype_var):
        dtype_def = guard(get_definition, self.func_ir, dtype_var)
        # str case
        if isinstance(dtype_def, ir.Global) and dtype_def.value == str:
            return string_array_type
        # categorical case
        if isinstance(dtype_def, ir.Expr) and dtype_def.op == 'call':
            if (not guard(find_callname, self.func_ir, dtype_def)
                    == ('category', 'pandas.core.dtypes.dtypes')):
                raise ValueError("pd.read_csv() invalid dtype "
                    "(built using a call but not Categorical)")
            cats_var = self._get_arg('CategoricalDtype', dtype_def.args,
                dict(dtype_def.kws), 0, 'categories')
            err_msg = "categories should be constant list"
            cats = self._get_str_or_list(cats_var, list_only=True, err_msg=err_msg)
            typ = PDCategoricalDtype(cats)
            return CategoricalArray(typ)
        if not isinstance(dtype_def, ir.Expr) or dtype_def.op != 'getattr':
            raise ValueError("pd.read_csv() invalid dtype")
        glob_def = guard(get_definition, self.func_ir, dtype_def.value)
        if not isinstance(glob_def, ir.Global) or glob_def.value != np:
            raise ValueError("pd.read_csv() invalid dtype")
        # TODO: extend to other types like string and date, check error
        typ_name = dtype_def.attr
        typ_name = 'int64' if typ_name == 'int' else typ_name
        typ_name = 'float64' if typ_name == 'float' else typ_name
        typ = getattr(types, typ_name)
        typ = types.Array(typ, 1, 'C')
        return typ

    def _handle_pd_Series(self, assign, lhs, rhs):
        """transform pd.Series(A) call
        """
        kws = dict(rhs.kws)
        data = self._get_arg('pd.Series', rhs.args, kws, 0, 'data')

        # match flatmap pd.Series(list(itertools.chain(*A))) and flatten
        data_def = guard(get_definition, self.func_ir, data)
        if (is_call(data_def) and guard(find_callname, self.func_ir, data_def)
                == ('list', 'builtins') and len(data_def.args) == 1):
            arg_def = guard(get_definition, self.func_ir, data_def.args[0])
            if (is_call(arg_def) and guard(find_callname, self.func_ir,
                    arg_def) == ('chain', 'itertools')):
                in_data = arg_def.vararg
                return self._replace_func(
                    lambda l: hpat.hiframes.api.flatten_to_series(l),
                    [in_data]
                )

        # pd.Series() is handled in typed pass now
        # return self._replace_func(lambda arr: hpat.hiframes.api.init_series(
        #         hpat.hiframes.api.fix_df_array(arr)),
        #     [data])
        return [assign]

    def _handle_pd_to_numeric(self, assign, lhs, rhs):
        """transform pd.to_numeric(A, errors='coerce') call here since dtype
        has to be specified in locals and applied
        """
        kws = dict(rhs.kws)
        if 'errors' not in kws and guard(find_const, self.func_ir, kws['errors']) != 'coerce':
            raise ValueError("pd.to_numeric() only supports errors='coerce'")

        if lhs.name not in self.reverse_copies or (self.reverse_copies[lhs.name]) not in self.locals:
            raise ValueError("pd.to_numeric() requires annotation of output type")

        typ = self.locals.pop(self.reverse_copies[lhs.name])
        dtype = numba.numpy_support.as_dtype(typ.dtype)
        arg = rhs.args[0]

        return self._replace_func(
            lambda arr: hpat.hiframes.api.to_numeric(arr, dtype),
            [arg], extra_globals={'dtype': dtype})

    def _handle_pq_read_table(self, assign, lhs, rhs):
        if len(rhs.args) != 1:  # pragma: no cover
            raise ValueError("Invalid read_table() arguments")
        self.arrow_tables[lhs.name] = rhs.args[0]
        return []

    def _handle_pq_to_pandas(self, assign, lhs, rhs, t_var, label):
        return self._gen_parquet_read(self.arrow_tables[t_var.name], lhs, label)

    def _gen_parquet_read(self, fname, lhs, label):
        columns, data_arrs, nodes = self.pq_handler.gen_parquet_read(
            fname, lhs)
        n_cols = len(columns)
        data_args = ", ".join('data{}'.format(i) for i in range(n_cols))

        func_text = "def _init_df({}):\n".format(data_args)
        func_text += "  return hpat.hiframes.pd_dataframe_ext.init_dataframe({}, None, {})\n".format(
            data_args, ", ".join("'{}'".format(c) for c in columns))
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars['_init_df']

        return self._replace_func(_init_df, data_arrs, pre_nodes=nodes)

    def _handle_pd_read_parquet(self, assign, lhs, rhs, label):
        fname = rhs.args[0]
        return self._gen_parquet_read(fname, lhs, label)

    def _handle_concat(self, assign, lhs, rhs, label):
        # converting build_list to build_tuple before type inference to avoid
        # errors
        if len(rhs.args) != 1 or len(rhs.kws) != 0:
            raise ValueError(
                "only a list/tuple argument is supported in concat")
        df_list = guard(get_definition, self.func_ir, rhs.args[0])
        if not isinstance(df_list, ir.Expr) or not (df_list.op
                                            in ['build_tuple', 'build_list']):
            raise ValueError("pd.concat input should be constant list or tuple")

        # XXX convert build_list to build_tuple since Numba doesn't handle list of
        # arrays for np.concatenate()
        if df_list.op == 'build_list':
            df_list.op = 'build_tuple'

        if len(df_list.items) == 0:
            # copied error from pandas
            raise ValueError("No objects to concatenate")

        return [assign]

    def _handle_concat_df(self, lhs, df_list, label):
        # TODO: handle non-numerical (e.g. string, datetime) columns
        nodes = []

        # get output column names
        all_colnames = []
        for df in df_list:
            all_colnames.extend(self._get_df_col_names(df))
        # TODO: verify how Pandas sorts column names
        all_colnames = sorted(set(all_colnames))

        # generate a concat call for each output column
        # TODO: support non-numericals like string
        gen_nan_func = lambda A: np.full(len(A), np.nan)
        # gen concat function
        arg_names = ", ".join(['in{}'.format(i) for i in range(len(df_list))])
        func_text = "def _concat_imp({}):\n".format(arg_names)
        func_text += "    return hpat.hiframes.api.init_series(hpat.hiframes.api.concat(({})))\n".format(
            arg_names)
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _concat_imp = loc_vars['_concat_imp']

        done_cols = {}
        for cname in all_colnames:
            # arguments to the generated function
            args = []
            # get input columns
            for df in df_list:
                df_col_map = self._get_df_cols(df)
                # generate full NaN column
                if cname not in df_col_map:
                    # use a df column just for len()
                    len_arr = list(df_col_map.values())[0]
                    f_block = compile_to_numba_ir(gen_nan_func,
                            {'hpat': hpat, 'np': np}).blocks.popitem()[1]
                    replace_arg_nodes(f_block, [len_arr])
                    nodes += f_block.body[:-2]
                    args.append(nodes[-1].target)
                else:
                    args.append(df_col_map[cname])

            f_block = compile_to_numba_ir(_concat_imp,
                        {'hpat': hpat, 'np': np}).blocks.popitem()[1]
            replace_arg_nodes(f_block, args)
            nodes += f_block.body[:-2]
            done_cols[cname] = nodes[-1].target

        self._create_df(lhs.name, done_cols, label)
        return nodes

    def _handle_concat_series(self, lhs, rhs):
        # defer to typed pass since the type might be non-numerical
        def f(arr_list):  # pragma: no cover
            return hpat.hiframes.api.init_series(hpat.hiframes.api.concat(arr_list))
        return self._replace_func(f, rhs.args)

    def _handle_ros(self, assign, lhs, rhs):
        if len(rhs.args) != 1:  # pragma: no cover
            raise ValueError("Invalid read_ros_images() arguments")
        import hpat.ros
        return hpat.ros._handle_read_images(lhs, rhs)

    def _fix_df_arrays(self, items_list):
        nodes = []
        new_list = []
        for item in items_list:
            col_varname = item[0]
            col_arr = item[1]
            # fix list(multi-dim arrays) (packing images)
            # FIXME: does this break for list(other things)?
            col_arr = self._fix_df_list_of_array(col_arr)

            def f(arr):  # pragma: no cover
                df_arr = hpat.hiframes.api.fix_df_array(arr)
            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [col_arr])
            nodes += f_block.body[:-3]  # remove none return
            new_col_arr = nodes[-1].target
            new_list.append((col_varname, new_col_arr))
        return nodes, new_list

    def _fix_df_list_of_array(self, col_arr):
        list_call = guard(get_definition, self.func_ir, col_arr)
        if guard(find_callname, self.func_ir, list_call) == ('list', 'builtins'):
            return list_call.args[0]
        return col_arr

    def _process_df_build_map(self, items_list):
        df_cols = {}
        nodes = []
        for item in items_list:
            col_var = item[0]
            if isinstance(col_var, str):
                col_name = col_var
            else:
                col_name = get_constant(self.func_ir, col_var)
                if col_name is NOT_CONSTANT:  # pragma: no cover
                    raise ValueError(
                        "data frame column names should be constant")
            # cast to series type
            def f(arr):  # pragma: no cover
                df_arr = hpat.hiframes.api.init_series(arr)
            f_block = compile_to_numba_ir(
                f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [item[1]])
            nodes += f_block.body[:-3]  # remove none return
            new_col_arr = nodes[-1].target
            df_cols[col_name] = new_col_arr
        return nodes, df_cols

    def _get_func_output_typ(self, col_var, func, wrapper_func, label):
        # stich together all blocks before the current block for type inference
        # XXX: does control flow affect type inference in Numba?
        dummy_ir = self.func_ir.copy()
        dummy_ir.blocks[label].body.append(ir.Return(0, col_var.loc))
        topo_order = find_topo_order(dummy_ir.blocks)
        all_body = []
        for l in topo_order:
            if l == label:
                break
            all_body += dummy_ir.blocks[l].body

        # add nodes created for current block so far
        all_body += self._working_body
        dummy_ir.blocks = {0: ir.Block(col_var.scope, col_var.loc)}
        dummy_ir.blocks[0].body = all_body

        _globals = self.func_ir.func_id.func.__globals__
        _globals.update({'hpat': hpat, 'numba': numba, 'np': np})
        f_ir = compile_to_numba_ir(wrapper_func, {'hpat': hpat})
        # fix definitions to enable finding sentinel
        f_ir._definitions = build_definitions(f_ir.blocks)
        first_label = min(f_ir.blocks)
        for i, stmt in enumerate(f_ir.blocks[first_label].body):
            if (isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == 'call'):
                fdef = guard(get_definition, f_ir, stmt.value.func)
                if isinstance(fdef, ir.Global) and fdef.name == 'map_func':
                    inline_closure_call(f_ir, _globals, f_ir.blocks[first_label], i, func)
                    break

        f_ir.blocks = ir_utils.simplify_CFG(f_ir.blocks)
        f_topo_order = find_topo_order(f_ir.blocks)
        assert isinstance(f_ir.blocks[f_topo_order[-1]].body[-1], ir.Return)
        output_var = f_ir.blocks[f_topo_order[-1]].body[-1].value
        first_label = f_topo_order[0]
        replace_arg_nodes(f_ir.blocks[first_label], [col_var])
        assert first_label != topo_order[0]  #  TODO: check for 0 and handle
        dummy_ir.blocks.update(f_ir.blocks)
        dummy_ir.blocks[0].body.append(ir.Jump(first_label, col_var.loc))
        # dead df code can cause type inference issues
        # TODO: remove this
        hiframes.api.enable_hiframes_remove_dead = False
        while remove_dead(dummy_ir.blocks, dummy_ir.arg_names, dummy_ir):
            pass
        hiframes.api.enable_hiframes_remove_dead = True

        # run type inference on the dummy IR
        warnings = numba.errors.WarningsFixer(numba.errors.NumbaWarning)
        infer = numba.typeinfer.TypeInferer(self.typingctx, dummy_ir, warnings)
        for index, (name, ty) in enumerate(zip(dummy_ir.arg_names, self.args)):
            infer.seed_argument(name, index, ty)
        infer.build_constraint()
        infer.propagate()
        out_tp = infer.typevars[output_var.name].getone()
        # typemap, restype, calltypes = numba.compiler.type_inference_stage(self.typingctx, dummy_ir, self.args, None)
        return out_tp


    def _is_df_obj_call(self, call_var, obj_name):
        """determines whether variable is coming from groupby() or groupby()[],
        rolling(), rolling()[]
        """
        var_def = guard(get_definition, self.func_ir, call_var)
        # groupby()['B'] case
        if (isinstance(var_def, ir.Expr)
                and var_def.op in ['getitem', 'static_getitem']):
            return self._is_df_obj_call(var_def.value, obj_name)
        # groupby() called on column or df
        call_def = guard(find_callname, self.func_ir, var_def)
        if (call_def is not None and call_def[0] == obj_name
                and isinstance(call_def[1], ir.Var)
                and self._is_df_var(call_def[1])):
            return True
        return False

    def _handle_df_pivot_table(self, lhs, rhs, df_var, label):
        # TODO: multiple keys (index columns)
        kws = dict(rhs.kws)
        values_arg = self._get_str_arg('pivot_table', rhs.args, kws, 0, 'values')
        index_arg = self._get_str_arg('pivot_table', rhs.args, kws, 1, 'index')
        columns_arg = self._get_str_arg('pivot_table', rhs.args, kws, 2, 'columns')
        agg_func_arg = self._get_str_arg('pivot_table', rhs.args, kws, 3, 'aggfunc', 'mean')

        agg_func = get_agg_func(self.func_ir, agg_func_arg, rhs)

        in_vars = {values_arg: self.df_vars[df_var.name][values_arg]}
        # get output type
        agg_func_dis = numba.njit(agg_func)
        agg_gb_var = ir.Var(lhs.scope, mk_unique_var("agg_gb"), lhs.loc)
        nodes = [ir.Assign(ir.Global("agg_gb", agg_func_dis, lhs.loc), agg_gb_var, lhs.loc)]
        def to_arr(a, _agg_f):
            b = hpat.hiframes.api.to_arr_from_series(a)
            res = hpat.hiframes.api.init_series(hpat.hiframes.api.agg_typer(b, _agg_f))
        f_block = compile_to_numba_ir(to_arr, {'hpat': hpat, 'np': np}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [in_vars[values_arg], agg_gb_var])
        nodes += f_block.body[:-3]  # remove none return
        out_types = {values_arg: nodes[-1].target}

        pivot_values = self._get_pivot_values(lhs.name)
        df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
                                for col in pivot_values})
        # df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
        #                         for col in [values_arg]})
        out_df = df_col_map.copy()
        self._create_df(lhs.name, out_df, label)
        pivot_arr = self.df_vars[df_var.name][columns_arg]
        agg_node = aggregate.Aggregate(
            lhs.name, df_var.name, [index_arg], None, df_col_map,
            in_vars, [self.df_vars[df_var.name][index_arg]],
            agg_func, out_types, lhs.loc, pivot_arr, pivot_values)
        nodes.append(agg_node)
        return nodes


    def _get_pivot_values(self, varname):
        if varname not in self.reverse_copies or (self.reverse_copies[varname] + ':pivot') not in self.locals:
            raise ValueError("pivot_table() requires annotation of pivot values")
        new_name = self.reverse_copies[varname]
        values = self.locals.pop(new_name + ":pivot")
        return values

    def _get_str_arg(self, f_name, args, kws, arg_no, arg_name, default=None,
                                                                 err_msg=None):
        arg = None
        if len(args) > arg_no:
            arg = guard(find_const, self.func_ir, args[arg_no])
        elif arg_name in kws:
            arg = guard(find_const, self.func_ir, kws[arg_name])

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = ("{} requires '{}' argument as a "
                             "constant string").format(f_name, arg_name)
            raise ValueError(err_msg)
        return arg

    def _get_arg(self, f_name, args, kws, arg_no, arg_name, default=None,
                                                                 err_msg=None):
        arg = None
        if len(args) > arg_no:
            arg = args[arg_no]
        elif arg_name in kws:
            arg = kws[arg_name]

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = "{} requires '{}' argument".format(f_name, arg_name)
            raise ValueError(err_msg)
        return arg

    def _handle_crosstab(self, lhs, rhs, label):
        kws = dict(rhs.kws)
        # TODO: hanlde multiple keys (index args)
        index_arg = self._get_arg('crosstab', rhs.args, kws, 0, 'index')
        columns_arg = self._get_arg('crosstab', rhs.args, kws, 1, 'columns')
        # TODO: handle values and aggfunc options

        in_vars = {}
        # output of crosstab is array[int64]
        def to_arr():
            res = hpat.hiframes.api.init_series(np.empty(1, np.int64))
        f_block = compile_to_numba_ir(to_arr, {'hpat': hpat, 'np': np}).blocks.popitem()[1]
        nodes = f_block.body[:-3]  # remove none return
        out_tp_var = nodes[-1].target
        out_types = {'__dummy__': out_tp_var}

        pivot_values = self._get_pivot_values(lhs.name)
        df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
                                for col in pivot_values})
        out_df = df_col_map.copy()
        self._create_df(lhs.name, out_df, label)
        pivot_arr = columns_arg

        def _agg_len_impl(in_arr):  # pragma: no cover
            numba.parfor.init_prange()
            count = 0
            for i in numba.parfor.internal_prange(len(in_arr)):
                count += 1
            return count

        # TODO: make out_key_var an index column

        agg_node = aggregate.Aggregate(
            lhs.name, 'crosstab', [index_arg.name], None, df_col_map,
            in_vars, [index_arg],
            _agg_len_impl, out_types, lhs.loc, pivot_arr, pivot_values, True)
        nodes.append(agg_node)
        return nodes

    def _handle_aggregate(self, lhs, rhs, obj_var, func_name, label):
        # format df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
        # TODO: support aggregation functions sum, count, etc.
        if func_name not in supported_agg_funcs:
            raise ValueError("only {} supported in groupby".format(
                                             ", ".join(supported_agg_funcs)))


        # find selected output columns
        df_var, out_colnames, explicit_select, obj_var = self._get_df_obj_select(obj_var, 'groupby')
        key_colnames, as_index = self._get_agg_obj_args(obj_var)
        if out_colnames is None:
            out_colnames = list(self.df_vars[df_var.name].keys())
            # key arr is not output by default
            # as_index should be handled separately since it just returns keys
            for k in key_colnames:
                out_colnames.remove(k)

        # find input vars
        in_vars = {out_cname: self.df_vars[df_var.name][out_cname]
                    for out_cname in out_colnames}

        nodes, agg_func, out_tp_vars = self._handle_agg_func(
            in_vars, out_colnames, func_name, lhs, rhs)

        # output column map, create dataframe if multiple outputs
        out_key_vars = None
        # XXX output becomes series if single output and explicitly selected
        if len(out_colnames) == 1 and explicit_select and as_index:
            df_col_map = {out_colnames[0]: lhs}
        else:
            out_df = {}
            # keys come first in column list
            if as_index is False:
                out_key_vars = []
                for k in key_colnames:
                    out_key_var = ir.Var(lhs.scope, mk_unique_var(k), lhs.loc)
                    out_df[k] = out_key_var
                    out_key_vars.append(out_key_var)
            df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
                                for col in out_colnames})
            out_df.update(df_col_map)

            self._create_df(lhs.name, out_df, label)

        in_key_vars = [self.df_vars[df_var.name][k] for k in key_colnames]

        agg_node = aggregate.Aggregate(
            lhs.name, df_var.name, key_colnames, out_key_vars, df_col_map,
            in_vars, in_key_vars,
            agg_func, out_tp_vars, lhs.loc)
        nodes.append(agg_node)
        return nodes

    def _handle_agg_func(self, in_vars, out_colnames, func_name, lhs, rhs):
        agg_func = get_agg_func(self.func_ir, func_name, rhs)
        out_tp_vars = {}

        # hpat.jit() instead of numba.njit() to handle str arrs etc
        agg_func_dis = hpat.jit(agg_func)
        #agg_func_dis = numba.njit(agg_func)
        agg_gb_var = ir.Var(lhs.scope, mk_unique_var("agg_gb"), lhs.loc)
        nodes = [ir.Assign(ir.Global("agg_gb", agg_func_dis, lhs.loc), agg_gb_var, lhs.loc)]
        for out_cname in out_colnames:
            in_var = in_vars[out_cname]
            def to_arr(a, _agg_f):
                b = hpat.hiframes.api.to_arr_from_series(a)
                res = hpat.hiframes.api.init_series(hpat.hiframes.api.agg_typer(b, _agg_f))
            f_block = compile_to_numba_ir(to_arr, {'hpat': hpat, 'np': np}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [in_var, agg_gb_var])
            nodes += f_block.body[:-3]  # remove none return
            out_tp_vars[out_cname] = nodes[-1].target
        return nodes, agg_func, out_tp_vars

    def _get_agg_obj_args(self, agg_var):
        # find groupby key and as_index
        groubpy_call = guard(get_definition, self.func_ir, agg_var)
        assert isinstance(groubpy_call, ir.Expr) and groubpy_call.op == 'call'
        kws = dict(groubpy_call.kws)
        as_index = True
        if 'as_index' in kws:
            as_index = guard(find_const, self.func_ir, kws['as_index'])
            if as_index is None:
                raise ValueError(
                    "groupby as_index argument should be constant")
        if len(groubpy_call.args) == 1:
            by_arg = groubpy_call.args[0]
        elif 'by' in kws:
            by_arg = kws['by']
        else:  # pragma: no cover
            raise ValueError("by argument for groupby() required")

        err_msg = ("groupby() by argument should be "
                   "list of column names or a column name")
        key_colnames = self._get_str_or_list(by_arg, True, err_msg=err_msg)

        return key_colnames, as_index

    def _get_str_or_list(self, by_arg, list_only=False, default=None, err_msg=None, typ=None):
        typ = str if typ is None else typ
        by_arg_def = guard(find_build_sequence, self.func_ir, by_arg)
        if by_arg_def is None:
            # try single key column
            by_arg_def = guard(find_const, self.func_ir, by_arg)
            if by_arg_def is None:
                if default is not None:
                    return default
                raise ValueError(err_msg)
            key_colnames = [by_arg_def]
        else:
            if list_only and by_arg_def[1] != 'build_list':
                if default is not None:
                    return default
                raise ValueError(err_msg)
            key_colnames = [guard(find_const, self.func_ir, v) for v in by_arg_def[0]]
            if any(not isinstance(v, typ) for v in key_colnames):
                if default is not None:
                    return default
                raise ValueError(err_msg)
        return key_colnames


    def _get_df_obj_select(self, obj_var, obj_name):
        """analyze selection of columns in after groupby() or rolling()
        e.g. groupby('A')['B'], groupby('A')['B', 'C'], groupby('A')
        """
        select_def = guard(get_definition, self.func_ir, obj_var)
        out_colnames = None
        explicit_select = False
        if isinstance(select_def, ir.Expr) and select_def.op in ('getitem', 'static_getitem'):
            obj_var = select_def.value
            out_colnames = (select_def.index
                if select_def.op == 'static_getitem'
                else guard(find_const, self.func_ir, select_def.index))
            if not isinstance(out_colnames, (str, tuple)):
                raise ValueError("{} output column names should be constant".format(obj_name))
            if isinstance(out_colnames, str):
                out_colnames = [out_colnames]
            explicit_select = True

        obj_call = guard(get_definition, self.func_ir, obj_var)
        # find dataframe
        call_def = guard(find_callname, self.func_ir, obj_call)
        assert (call_def is not None and call_def[0] == obj_name
                and isinstance(call_def[1], ir.Var)
                and self._is_df_var(call_def[1]))
        df_var = call_def[1]

        return df_var, out_colnames, explicit_select, obj_var


    def _handle_rolling(self, lhs, rhs, obj_var, func_name, label):
        # format df.rolling(w)['B'].sum()
        # TODO: support aggregation functions sum, count, etc.
        if func_name not in supported_rolling_funcs:
            raise ValueError("only ({}) supported in rolling".format(
                                             ", ".join(supported_rolling_funcs)))

        nodes = []
        # find selected output columns
        df_var, out_colnames, explicit_select, obj_var = self._get_df_obj_select(obj_var, 'rolling')
        rolling_call = guard(get_definition, self.func_ir, obj_var)
        window, center, on = get_rolling_setup_args(self.func_ir, rolling_call, False)
        on_arr = self.df_vars[df_var.name][on] if on is not None else None
        if not isinstance(center, ir.Var):
            center_var = ir.Var(lhs.scope, mk_unique_var("center"), lhs.loc)
            nodes.append(ir.Assign(ir.Const(center, lhs.loc), center_var, lhs.loc))
            center = center_var
        if not isinstance (window, ir.Var):
            window_var = ir.Var(lhs.scope, mk_unique_var("window"), lhs.loc)
            nodes.append(ir.Assign(ir.Const(window, lhs.loc), window_var, lhs.loc))
            window = window_var
        # TODO: get 'on' arg for offset case
        if out_colnames is None:
            out_colnames = list(self.df_vars[df_var.name].keys())
            # TODO: remove index col for offset case

        nan_cols = []
        if func_name in ('cov', 'corr'):
            if len(rhs.args) != 1:
                raise ValueError("rolling {} requires one argument (other)".format(func_name))
            # XXX pandas only accepts variable window cov/corr
            # when both inputs have time index
            if on_arr is not None:
                raise ValueError("variable window rolling {} not supported yet.".format(func_name))
            # TODO: support variable window rolling cov/corr which is only
            # possible in pandas with time index
            other = rhs.args[0]
            if self._is_df_var(other):
                # df on df cov/corr returns common columns only (without
                # pairwise flag)
                # TODO: support pairwise arg
                col_set1 = set(out_colnames)
                col_set2 = set(self._get_df_col_names(other))
                out_colnames = list(col_set1 & col_set2)
                # Pandas makes non-common columns NaNs
                nan_cols = list(col_set1 ^ col_set2)

        # output column map, create dataframe if multiple outputs
        out_df = None
        if len(out_colnames) == 1 and explicit_select:
            df_col_map = {out_colnames[0]: lhs}
        else:
            df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
                                for col in out_colnames})
            if on is not None:
                df_col_map[on] = on_arr
            out_df = df_col_map.copy()
            # TODO: add datetime index for offset case

        args = rhs.args
        for cname, out_col_var in df_col_map.items():
            if cname == on:
                continue
            in_col_var = self.df_vars[df_var.name][cname]
            if func_name in ('cov', 'corr'):
                args[0] = self.df_vars[other.name][cname]
            nodes += self._gen_rolling_call(in_col_var, out_col_var, window, center, args, func_name, on_arr)

        # create NaN columns for cov/corr case
        len_arr = self.df_vars[df_var.name][out_colnames[0]]
        for cname in nan_cols:
            def f(arr):
                nan_arr = np.full(len(arr), np.nan)
            f_block = compile_to_numba_ir(f, {'np': np}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [len_arr])
            nodes += f_block.body[:-3]  # remove none return
            out_df[cname] = nodes[-1].target
        if out_df is not None:
            # Pandas sorts the output column names _flex_binary_moment
            # line: res_columns = arg1.columns.union(arg2.columns)
            self._create_df(lhs.name, dict(sorted(out_df.items())), label)

        return nodes

    def _gen_rolling_call(self, in_col_var, out_col_var, window, center, args, func_name, on_arr):
        nodes = []
        if func_name in ('cov', 'corr'):
            other = args[0]
            if on_arr is not None:
                if func_name == 'cov':
                    def f(arr, other, on_arr, w, center):  # pragma: no cover
                        df_arr = hpat.hiframes.api.init_series(
                            hpat.hiframes.rolling.rolling_cov(
                                arr, other, on_arr, w, center))
                if func_name == 'corr':
                    def f(arr, other, on_arr, w, center):  # pragma: no cover
                        df_arr = hpat.hiframes.api.init_series(
                            hpat.hiframes.rolling.rolling_corr(
                                arr, other, on_arr, w, center))
                args = [in_col_var, other, on_arr, window, center]
            else:
                if func_name == 'cov':
                    def f(arr, other, w, center):  # pragma: no cover
                        df_arr = hpat.hiframes.api.init_series(
                            hpat.hiframes.rolling.rolling_cov(
                                arr, other, w, center))
                if func_name == 'corr':
                    def f(arr, other, w, center):  # pragma: no cover
                        df_arr = hpat.hiframes.api.init_series(
                            hpat.hiframes.rolling.rolling_corr(
                                arr, other, w, center))
                args = [in_col_var, other, window, center]
        # variable window case
        elif on_arr is not None:
            if func_name == 'apply':
                def f(arr, on_arr, w, center, func):  # pragma: no cover
                    df_arr = hpat.hiframes.api.init_series(
                        hpat.hiframes.rolling.rolling_variable(
                            arr, on_arr, w, center, False, func))
                args = [in_col_var, on_arr, window, center, args[0]]
            else:
                def f(arr, on_arr, w, center):  # pragma: no cover
                    df_arr = hpat.hiframes.api.init_series(
                        hpat.hiframes.rolling.rolling_variable(
                            arr, on_arr, w, center, False, _func_name))
                args = [in_col_var, on_arr, window, center]
        else:  # fixed window
            # apply case takes the passed function instead of just name
            if func_name == 'apply':
                def f(arr, w, center, func):  # pragma: no cover
                    df_arr = hpat.hiframes.api.init_series(
                        hpat.hiframes.rolling.rolling_fixed(
                            arr, w, center, False, func))
                args = [in_col_var, window, center, args[0]]
            else:
                def f(arr, w, center):  # pragma: no cover
                    df_arr = hpat.hiframes.api.init_series(
                        hpat.hiframes.rolling.rolling_fixed(
                            arr, w, center, False, _func_name))
                args = [in_col_var, window, center]
        f_block = compile_to_numba_ir(f, {'hpat': hpat, '_func_name': func_name}).blocks.popitem()[1]
        replace_arg_nodes(f_block, args)
        nodes += f_block.body[:-3]  # remove none return
        nodes[-1].target = out_col_var
        return nodes

    def _fix_rolling_array(self, col_var, func):
        """
        for integers and bools, the output should be converted to float64
        """
        # TODO: check all possible funcs
        def f(arr):  # pragma: no cover
            df_arr = hpat.hiframes.api.fix_rolling_array(arr)
        f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [col_var])
        nodes = f_block.body[:-3]  # remove none return
        new_col_var = nodes[-1].target
        return new_col_var, nodes

    def _handle_metadata(self):
        """remove distributed input annotation from locals and add to metadata
        """
        if 'distributed' not in self.metadata:
            # TODO: keep updated in variable renaming?
            self.metadata['distributed'] = self.locals.pop(
                '##distributed', set())

        if 'threaded' not in self.metadata:
            self.metadata['threaded'] = self.locals.pop('##threaded', set())

        # handle old input flags
        # e.g. {"A:input": "distributed"} -> "A"
        dist_inputs = { var_name.split(":")[0]
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":input") and flag == 'distributed'}

        thread_inputs = { var_name.split(":")[0]
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":input") and flag == 'threaded'}

        # check inputs to be in actuall args
        for arg_name in dist_inputs | thread_inputs:
            if arg_name not in self.func_ir.arg_names:
                raise ValueError(
                    "distributed input {} not found in arguments".format(
                        arg_name))
            self.locals.pop(arg_name + ":input")

        self.metadata['distributed'] |= dist_inputs
        self.metadata['threaded'] |= thread_inputs


        # handle old return flags
        # e.g. {"A:return":"distributed"} -> "A"
        flagged_returns = { var_name.split(":")[0]: flag
                    for (var_name, flag) in self.locals.items()
                    if var_name.endswith(":return") }

        for v, flag in flagged_returns.items():
            if flag == 'distributed':
                self.metadata['distributed'].add(v)
            elif flag == 'threaded':
                self.metadata['threaded'].add(v)

            self.locals.pop(v + ":return")

        return

    def _add_node_defs(self, nodes):
        # TODO: add node defs for all new nodes
        loc = ir.Loc("", -1)
        dummy_block = ir.Block(ir.Scope(None, loc), loc)
        dummy_block.body = nodes
        build_definitions({0: dummy_block}, self.func_ir._definitions)

    def _run_return(self, ret_node):
        # TODO: handle distributed analysis, requires handling variable name
        # change in simplify() and replace_var_names()
        flagged_vars = self.metadata['distributed'] | self.metadata['threaded']
        nodes = [ret_node]
        cast = guard(get_definition, self.func_ir, ret_node.value)
        assert cast is not None, "return cast not found"
        assert isinstance(cast, ir.Expr) and cast.op == 'cast'
        scope = cast.value.scope
        loc = cast.loc
        # XXX: using split('.') since the variable might be renamed (e.g. A.2)
        ret_name = cast.value.name.split('.')[0]

        if ret_name in flagged_vars:
            flag = ('distributed' if ret_name in self.metadata['distributed']
                        else 'threaded')
            nodes = self._gen_replace_dist_return(cast.value, flag)
            new_arr = nodes[-1].target
            new_cast = ir.Expr.cast(new_arr, loc)
            new_out = ir.Var(scope, mk_unique_var(flag + "_return"), loc)
            nodes.append(ir.Assign(new_cast, new_out, loc))
            ret_node.value = new_out
            nodes.append(ret_node)
            return nodes

        # shortcut if no dist return
        if len(flagged_vars) == 0:
            return nodes

        cast_def = guard(get_definition, self.func_ir, cast.value)
        if (cast_def is not None and isinstance(cast_def, ir.Expr)
                and cast_def.op == 'build_tuple'):
            nodes = []
            new_var_list = []
            for v in cast_def.items:
                vname = v.name.split('.')[0]
                if vname in flagged_vars:
                    flag = ('distributed' if vname in self.metadata['distributed']
                        else 'threaded')
                    nodes += self._gen_replace_dist_return(v, flag)
                    new_var_list.append(nodes[-1].target)
                else:
                    new_var_list.append(v)
            new_tuple_node = ir.Expr.build_tuple(new_var_list, loc)
            new_tuple_var = ir.Var(scope, mk_unique_var("dist_return_tp"), loc)
            nodes.append(ir.Assign(new_tuple_node, new_tuple_var, loc))
            new_cast = ir.Expr.cast(new_tuple_var, loc)
            new_out = ir.Var(scope, mk_unique_var("dist_return"), loc)
            nodes.append(ir.Assign(new_cast, new_out, loc))
            ret_node.value = new_out
            nodes.append(ret_node)

        return nodes

    def _gen_replace_dist_return(self, var, flag):
        if flag == 'distributed':
            def f(_dist_arr):  # pragma: no cover
                _d_arr = hpat.distributed_api.dist_return(_dist_arr)
        elif flag == 'threaded':
            def f(_threaded_arr):  # pragma: no cover
                _th_arr = hpat.distributed_api.threaded_return(_threaded_arr)
        else:
            raise ValueError("Invalid return flag {}".format(flag))
        f_block = compile_to_numba_ir(
            f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [var])
        return f_block.body[:-3]  # remove none return

    def _run_df_set_column(self, inst, label, cfg):
        """replace setitem of string index with a call to handle possible
        dataframe case where schema is changed:
        df['new_col'] = arr  ->  df2 = set_df_col(df, 'new_col', arr)
        dataframe_pass will replace set_df_col() with regular setitem if target
        is not dataframe
        """
        # setting column possible only when it dominates the df creation to
        # keep schema consistent
        # invalid case:
        # df = pd.DataFrame({'A': A})
        # if cond:
        #     df['B'] = B
        # return df
        # TODO: add this check back in
        # if label not in cfg.backbone() and label not in cfg.post_dominators()[df_label]:
        #     raise ValueError("setting dataframe columns inside conditionals and"
        #                      " loops not supported yet")

        # TODO: generalize to more cases
        # for example:
        # df = pd.DataFrame({'A': A})
        # if cond:
        #     df['B'] = B
        # else:
        #     df['B'] = C
        # return df
        # TODO: check for references to df
        # for example:
        # df = pd.DataFrame({'A': A})
        # df2 = df
        # df['B'] = C
        # return df2
        df_var = inst.target
        # create var for string index
        cname_var = ir.Var(inst.value.scope, mk_unique_var("$cname_const"), inst.loc)
        nodes = [ir.Assign(ir.Const(inst.index, inst.loc), cname_var, inst.loc)]

        func = lambda df, cname, arr: hpat.hiframes.api.set_df_col(df, cname, arr)
        f_block = compile_to_numba_ir(func, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, [df_var, cname_var, inst.value])
        nodes += f_block.body[:-2]

        # rename the dataframe variable to keep schema static
        new_df_var = ir.Var(df_var.scope, mk_unique_var(df_var.name), df_var.loc)
        nodes[-1].target = new_df_var
        self.replace_var_dict[df_var.name] = new_df_var
        self._add_node_defs(nodes)
        return nodes

    def _replace_func(self, func, args, const=False, array_typ_convert=True,
                      pre_nodes=None, extra_globals=None):
        glbls = {'numba': numba, 'np': np, 'hpat': hpat}
        if extra_globals is not None:
            glbls.update(extra_globals)
        return ReplaceFunc(func, None, args, glbls, pre_nodes)

    def _create_df(self, df_varname, df_col_map, label):
        # order is important for proper handling of itertuples, apply, etc.
        # starting pandas 0.23 and Python 3.6, regular dict order is OK
        # for <0.23 ordered_df_map = OrderedDict(sorted(df_col_map.items()))
        self.df_vars[df_varname] = df_col_map
        self.df_labels[df_varname] = label

    def _is_df_colname(self, df_var, cname):
        """ is cname a column name in df_var
        """
        df_var_renamed = self._get_renamed_df(df_var)
        return cname in self.df_vars[df_var_renamed.name]


    def _is_df_var(self, var):
        assert isinstance(var, ir.Var)
        return (var.name in self.df_vars)

    def _get_df_cols(self, df_var):
        #
        assert isinstance(df_var, ir.Var)
        df_var_renamed = self._get_renamed_df(df_var)
        return self.df_vars[df_var_renamed.name]

    def _get_df_col_names(self, df_var):
        assert isinstance(df_var, ir.Var)
        df_var_renamed = self._get_renamed_df(df_var)
        return list(self.df_vars[df_var_renamed.name].keys())

    def _get_df_col_vars(self, df_var):
        #
        assert isinstance(df_var, ir.Var)
        df_var_renamed = self._get_renamed_df(df_var)
        return list(self.df_vars[df_var_renamed.name].values())

    def _get_df_colvar(self, df_var, cname):
        assert isinstance(df_var, ir.Var)
        df_var_renamed = self._get_renamed_df(df_var)
        return self.df_vars[df_var_renamed.name][cname]

    def _get_renamed_df(self, df_var):
        # XXX placeholder for df variable renaming
        assert isinstance(df_var, ir.Var)
        return df_var


def _gen_arr_copy(in_arr, nodes):
    f_block = compile_to_numba_ir(
        lambda A: A.copy(), {}).blocks.popitem()[1]
    replace_arg_nodes(f_block, [in_arr])
    nodes += f_block.body[:-2]
    return nodes[-1].target


def simple_block_copy_propagate(block):
    """simple copy propagate for a single block before typing, without Parfor"""

    var_dict = {}
    # assignments as dict to replace with latest value
    for stmt in block.body:
        # only rhs of assignments should be replaced
        # e.g. if x=y is available, x in x=z shouldn't be replaced
        if isinstance(stmt, ir.Assign):
            stmt.value = replace_vars_inner(stmt.value, var_dict)
        else:
            replace_vars_stmt(stmt, var_dict)
        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
            lhs = stmt.target.name
            rhs = stmt.value.name
            # rhs could be replaced with lhs from previous copies
            if lhs != rhs:
                var_dict[lhs] = stmt.value
                # a=b kills previous t=a
                lhs_kill = []
                for k, v in var_dict.items():
                    if v.name == lhs:
                        lhs_kill.append(k)
                for k in lhs_kill:
                    var_dict.pop(k, None)
        if (isinstance(stmt, ir.Assign)
                                    and not isinstance(stmt.value, ir.Var)):
            lhs = stmt.target.name
            var_dict.pop(lhs, None)
            # previous t=a is killed if a is killed
            lhs_kill = []
            for k, v in var_dict.items():
                if v.name == lhs:
                    lhs_kill.append(k)
            for k in lhs_kill:
                var_dict.pop(k, None)
    return


def _sanitize_varname(varname):
    return varname.replace('$', '_').replace('.', '_')

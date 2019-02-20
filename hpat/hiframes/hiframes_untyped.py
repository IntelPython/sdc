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
from hpat.hiframes.api import PandasDataFrameType
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type
from hpat import csv_ext

import pandas as pd
import numpy as np
import math
from hpat.parquet_pio import ParquetHandler
from hpat.hiframes.pd_timestamp_ext import (datetime_date_type,
                                    datetime_date_to_int, int_to_datetime_date)
from hpat.hiframes.pd_series_ext import SeriesType, string_series_type
from hpat.hiframes.pd_categorical_ext import PDCategoricalDtype
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
    if call_list == ['unbox_df_column', 'boxing', 'hiframes', hpat]:
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
                if (isinstance(inst, (ir.StaticSetItem, ir.SetItem)) and self._is_iat(inst.target)):
                    ind = inst.index if isinstance(inst, ir.SetItem) else inst.index_var
                    out_nodes = self._handle_iat_setitem(inst.target, ind, inst.value, inst.loc)
                    new_body.extend(out_nodes)
                # df['col'] = arr
                elif (isinstance(inst, ir.StaticSetItem)
                        and self._is_df_var(inst.target)):
                    # cfg needed for set df column
                    cfg = compute_cfg_from_blocks(blocks)
                    new_body += self._run_df_set_column(inst, label, cfg)
                elif isinstance(inst, ir.Assign):
                    out_nodes = self._run_assign(inst, label)
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
                elif isinstance(inst, ir.Return):
                    nodes = self._run_return(inst)
                    new_body += nodes
                else:
                    new_body.append(inst)
            if not replaced:
                blocks[label].body = new_body

        self.func_ir._definitions = build_definitions(blocks)
        # XXX: remove dead here fixes h5 slice issue
        # iterative remove dead to make sure all extra code (e.g. df vars) is removed
        while remove_dead(blocks, self.func_ir.arg_names, self.func_ir):
            pass
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
                if self._is_iat(rhs.value):
                    return self._handle_iat_getitem(assign, lhs, rhs)
                h5_nodes = self.h5_handler.handle_possible_h5_read(
                    assign, lhs, rhs)
                if h5_nodes is not None:
                    return h5_nodes

            # d = df['column'] or df[['C1', 'C2']]
            if rhs.op == 'static_getitem' and self._is_df_var(rhs.value):
                # d = df['column']
                if isinstance(rhs.index, str):
                    assign.value = self._get_df_cols(rhs.value)[rhs.index]
                    return [assign]
                # df[['C1', 'C2']]
                if isinstance(rhs.index, list) and all(
                        isinstance(rhs.index[i], str)
                        for i in range(len(rhs.index))):
                    in_df_map = self._get_df_cols(rhs.value)
                    nodes = []
                    out_df_map = {c:_gen_arr_copy(in_df_map[c], nodes)
                                                            for c in rhs.index}
                    self._create_df(lhs, out_df_map, label)
                    return nodes
                # raise ValueError("unsupported dataframe access {}[{}]".format(
                #                  rhs.value.name, rhs.index))

            # df1 = df[df.A > .5], df.iloc[1:n], df.iloc[[1,2,3]], ...
            if rhs.op in ('getitem', 'static_getitem') and (
                    self._is_df_var(rhs.value)
                    or self._is_iloc_loc(rhs.value)):
                # XXX handling getitem, iloc, and loc the same way
                # TODO: support their differences
                # XXX: integer index not supported
                # TODO: check index for non-integer
                # TODO: support constant integer (return namedtuple)
                df = (rhs.value if self._is_df_var(rhs.value)
                    else guard(get_definition, self.func_ir, rhs.value).value)
                index_var = (rhs.index_var if rhs.op == 'static_getitem'
                             else rhs.index)
                # output df1 has same columns as df, create new vars
                scope = assign.target.scope
                loc = assign.target.loc
                in_df_col_names = self._get_df_col_names(df)
                df_col_map = {col: ir.Var(scope, mk_unique_var(col), loc)
                                for col in in_df_col_names}

                # column selection like df.iloc[:,0]
                if (rhs.op == 'static_getitem' and isinstance(rhs.index, tuple)
                        and len(rhs.index) == 2
                        and isinstance(rhs.index[1], int)
                        and rhs.index[0] == slice(None)):
                    col_no = rhs.index[1]
                    col_var = self._get_df_colvar(df, in_df_col_names[col_no])
                    assign.value = col_var
                    return [assign]

                self._create_df(lhs, df_col_map, label)
                in_df = self._get_renamed_df(df)
                return [filter.Filter(lhs, in_df.name, index_var,
                                               self.df_vars, rhs.loc)]

            # d = df.column
            if (rhs.op == 'getattr' and self._is_df_var(rhs.value)
                    and self._is_df_colname(rhs.value, rhs.attr)):
                df = rhs.value.name
                col_var = self._get_df_colvar(rhs.value, rhs.attr)
                assign.value = col_var
                # need to remove the lhs definition so that find_callname can
                # match column function calls (i.e. A.f instead of df.A.f)
                assert self.func_ir._definitions[lhs] == [rhs], "invalid def"
                self.func_ir._definitions[lhs] = [None]

            # A = df.values
            if (rhs.op == 'getattr' and self._is_df_var(rhs.value)
                    and rhs.attr == 'values'):
                return self._handle_df_values(assign.target, rhs.value)

        if isinstance(rhs, ir.Arg):
            return self._run_arg(assign, label)

        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.df_vars:
            self.df_vars[lhs] = self.df_vars[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.df_labels:
            self.df_labels[lhs] = self.df_labels[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.arrow_tables:
            self.arrow_tables[lhs] = self.arrow_tables[rhs.name]
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

        if fdef == ('DataFrame', 'pandas'):
            return self._handle_pd_DataFrame(assign, lhs, rhs, label)

        if fdef == ('read_csv', 'pandas'):
            return self._handle_pd_read_csv(assign, lhs, rhs, label)

        if fdef == ('Series', 'pandas'):
            return self._handle_pd_Series(assign, lhs, rhs)

        if fdef == ('len', 'builtins') and self._is_df_var(rhs.args[0]):
            return self._df_len(lhs, rhs.args[0])

        if fdef == ('read_table', 'pyarrow.parquet'):
            return self._handle_pq_read_table(assign, lhs, rhs)

        if (func_name == 'to_pandas' and isinstance(func_mod, ir.Var)
                and func_mod.name in self.arrow_tables):
            return self._handle_pq_to_pandas(assign, lhs, rhs, func_mod, label)

        if fdef == ('read_parquet', 'pandas'):
            return self._handle_pd_read_parquet(assign, lhs, rhs, label)

        if fdef == ('merge', 'pandas'):
            return self._handle_merge(assign, lhs, rhs, False, label)

        if fdef == ('merge_asof', 'pandas'):
            return self._handle_merge(assign, lhs, rhs, True, label)

        if fdef == ('concat', 'pandas'):
            return self._handle_concat(assign, lhs, rhs, label)

        if fdef == ('crosstab', 'pandas'):
            return self._handle_crosstab(lhs, rhs, label)

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
        # df.apply(lambda a:..., axis=1)
        if func_name == 'apply':
            return self._handle_df_apply(assign, lhs, rhs, df_var)

        # df.describe()
        if func_name == 'describe':
            return self._handle_df_describe(assign, lhs, rhs, df_var)

        # df.sort_values()
        if func_name == 'sort_values':
            return self._handle_df_sort_values(assign, lhs, rhs, df_var, label)

        # df.itertuples()
        if func_name == 'itertuples':
            return self._handle_df_itertuples(assign, lhs, rhs, df_var)

        # df.pivot_table()
        if func_name == 'pivot_table':
            return self._handle_df_pivot_table(lhs, rhs, df_var, label)

        # df.head()
        if func_name == 'head':
            return self._handle_df_head(lhs, rhs, df_var, label)

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

        # df.merge()
        if func_name == 'merge':
            rhs.args.insert(0, df_var)
            return self._handle_merge(assign, lhs, rhs, False, label)

        # df.reset_index(drop=True)
        if func_name == 'reset_index':
            return self._handle_df_reset_index(lhs, rhs, df_var, label)

        if func_name not in ('groupby', 'rolling'):
            raise NotImplementedError(
                "data frame function {} not implemented yet".format(func_name))

        return [assign]

    def _handle_df_head(self, lhs, rhs, df_var, label):
        nodes = []
        out_df_map = {}

        if len(rhs.args) == 0:
            def series_head(a):
                res = a.head()
        else:
            if len(rhs.args) != 1:
                raise ValueError("invalid df.head args")
            def series_head(a, n):
                res = a.head(n)

        for cname, in_var in self.df_vars[df_var.name].items():
            f_block = compile_to_numba_ir(series_head, {}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [in_var] + rhs.args)
            nodes += f_block.body[:-3]  # remove none return
            out_df_map[cname] = nodes[-1].target

        self._create_df(lhs.name, out_df_map, label)
        return nodes


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

    def _is_iloc_loc(self, var):
        val_def = guard(get_definition, self.func_ir, var)
        # check for df.at[] pattern
        return (isinstance(val_def, ir.Expr) and val_def.op == 'getattr'
                and val_def.attr in ('iloc', 'loc')
                and self._is_df_var(val_def.value))

    def _is_iat(self, var):
        val_def = guard(get_definition, self.func_ir, var)
        # check for df.at[] pattern
        return (isinstance(val_def, ir.Expr) and val_def.op == 'getattr'
                and val_def.attr == 'iat' and self._is_df_var(val_def.value))

    def _handle_iat_getitem(self, assign, lhs, rhs):
        if rhs.op == 'static_getitem':
            index_var = rhs.index_var
        else:
            index_var = rhs.index

        val_def = guard(get_definition, self.func_ir, rhs.value)
        df = val_def.value  # check already done in _is_iat
        col_var, row_ind, nodes = self._get_iat_col_ind(df, index_var)
        getitem_node = ir.Expr.getitem(col_var, row_ind, rhs.loc)
        assign.value = getitem_node
        nodes.append(assign)
        return nodes

    def _handle_iat_setitem(self, target, index_var, val, loc):
        val_def = guard(get_definition, self.func_ir, target)
        df = val_def.value  # check already done in _is_iat
        col_var, row_ind, nodes = self._get_iat_col_ind(df, index_var)
        setitem_node = ir.SetItem(col_var, row_ind, val, loc)
        nodes.append(setitem_node)
        return nodes

    def _get_iat_col_ind(self, df, index_var):
        nodes = []
        scope = index_var.scope
        loc = index_var.loc
        # find column/row indices
        col_ind = None
        row_ind = None
        ind_def = guard(get_definition, self.func_ir, index_var)

        # index is constant tuple
        if isinstance(ind_def, ir.Const):
            val = ind_def.value
            if not isinstance(val, tuple) or not len(val) == 2:
                raise ValueError("invalid index {} for df.iat[]".format(val))
            row_ind, col_ind = val
            r_var = ir.Var(scope, mk_unique_var(index_var.name), loc)
            nodes.append(ir.Assign(ir.Const(row_ind, loc), r_var, loc))
            row_ind = r_var

        # index is variable tuple
        elif isinstance(ind_def, ir.Expr) and ind_def.op == 'build_tuple':
            if len(ind_def.items) != 2:
                raise ValueError("invalid index length for df.iat[], "
                                 "[row, column] expected")
            row_ind = ind_def.items[0]
            col_ind = guard(find_const, self.func_ir, ind_def.items[1])
            if col_ind is None:
                raise ValueError("column index in iat[] should be constant")
        else:
            raise ValueError("invalid index in iat[]")

        # XXX assuming the order of the dictionary is the same as Pandas
        # TODO: check dictionary order
        col_var_list = list(self._get_df_cols(df).values())
        if not col_ind < len(col_var_list):
            raise ValueError("invalid column index in iat[]")
        col_var = col_var_list[col_ind]

        return col_var, row_ind, nodes

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

        col_map = {}
        out_types = []
        for i, (name_var, dtype_var) in enumerate(dtype_map.items):
            col_name = guard(find_const, self.func_ir, name_var)
            if col_name is None:  # pragma: no cover
                raise ValueError("dtype column names should be constant")
            typ = self._get_const_dtype(dtype_var)
            if i in date_cols:
                typ = SeriesType(types.NPDatetime('ns'))
            out_types.append(typ)
            col_map[col_name] = ir.Var(
                lhs.scope, mk_unique_var(col_name), lhs.loc)

        self._create_df(lhs.name, col_map, label)
        return [csv_ext.CsvReader(
            fname, lhs.name, sep, list(col_map.keys()), list(col_map.values()), out_types, usecols, lhs.loc)]

    def _get_const_dtype(self, dtype_var):
        dtype_def = guard(get_definition, self.func_ir, dtype_var)
        # str case
        if isinstance(dtype_def, ir.Global) and dtype_def.value == str:
            return string_series_type  # string_array_type
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
            return SeriesType(typ)
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
        typ = SeriesType(typ)
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

    def _df_len(self, lhs, df_var):
        # run len on one of the columns
        # FIXME: it could potentially avoid remove dead for the column if
        # array analysis doesn't replace len() with it's size
        df_arrs = list(self.df_vars[df_var.name].values())
        # empty dataframe has 0 len
        if len(df_arrs) == 0:
            return [ir.Assign(ir.Const(0, lhs.loc), lhs, lhs.loc)]
        arr = df_arrs[0]
        def f(df_arr):  # pragma: no cover
            return len(df_arr)
        return self._replace_func(f, [arr])

    def _handle_pq_read_table(self, assign, lhs, rhs):
        if len(rhs.args) != 1:  # pragma: no cover
            raise ValueError("Invalid read_table() arguments")
        self.arrow_tables[lhs.name] = rhs.args[0]
        return []

    def _handle_pq_to_pandas(self, assign, lhs, rhs, t_var, label):
        return self._gen_parquet_read(self.arrow_tables[t_var.name], lhs, label)

    def _gen_parquet_read(self, fname, lhs, label):
        col_items, col_types, nodes = self.pq_handler.gen_parquet_read(
            fname, lhs)
        df_nodes, col_map = self._process_df_build_map(col_items)
        nodes += df_nodes
        self._create_df(lhs.name, col_map, label)
        return nodes

    def _handle_pd_read_parquet(self, assign, lhs, rhs, label):
        fname = rhs.args[0]
        return self._gen_parquet_read(fname, lhs, label)

    def _handle_merge(self, assign, lhs, rhs, is_asof, label):
        """transform pd.merge() into a Join node

        signature: pd.merge(left, right, how='inner', on=None, left_on=None,
            right_on=None, left_index=False, right_index=False, sort=False,
            suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
        pd.merge_asof(left, right, on=None, left_on=None, right_on=None,
        left_index=False, right_index=False, by=None, left_by=None,
        right_by=None, suffixes=('_x', '_y'), tolerance=None,
        allow_exact_matches=True, direction='backward')
        """
        kws = dict(rhs.kws)
        left_df = self._get_arg('merge', rhs.args, kws, 0, 'left')
        right_df = self._get_arg('merge', rhs.args, kws, 1, 'right')
        on_argno = 3  # merge() has 'how' arg but merge_asof doesn't
        if is_asof:
            how = 'asof'
            on_argno = 2
        else:
            how =  self._get_str_arg('merge', rhs.args, kws, 2, 'how', 'inner')

        # find key columns
        left_on = right_on = None
        on_arg = self._get_arg('merge', rhs.args, kws, on_argno, 'on', '')
        on = self._get_str_or_list(on_arg, default=[''])

        if on != ['']:
            left_on = on
            right_on = left_on
        else:  # pragma: no cover
            err_msg = "merge 'on' or 'left_on'/'right_on' arguments required"
            left_on_var = self._get_arg('merge', rhs.args, kws, on_argno+1,
                                                    'left_on', err_msg=err_msg)
            left_on = self._get_str_or_list(left_on_var, err_msg=err_msg)
            right_on_var = self._get_arg('merge', rhs.args, kws, on_argno+2,
                                                   'right_on', err_msg=err_msg)
            right_on = self._get_str_or_list(right_on_var, err_msg=err_msg)

        # convert right join to left join
        if how == 'right':
            how = 'left'
            left_df, right_df = right_df, left_df
            left_on, right_on = right_on, left_on

        scope = lhs.scope
        loc = lhs.loc
        # add columns from left to output
        left_colnames = self._get_df_col_names(left_df)
        df_col_map = {col: ir.Var(scope, mk_unique_var(col), loc)
                                for col in left_colnames}
        # add columns from right to output
        right_colnames = self._get_df_col_names(right_df)
        df_col_map.update({col: ir.Var(scope, mk_unique_var(col), loc)
                                for col in right_colnames})
        self._create_df(lhs.name, df_col_map, label)
        return [join.Join(lhs.name, self._get_renamed_df(left_df).name,
                                   self._get_renamed_df(right_df).name,
                                   left_on, right_on, self.df_vars, how,
                                   lhs.loc)]

    def _handle_concat(self, assign, lhs, rhs, label):
        if len(rhs.args) != 1 or len(rhs.kws) != 0:
            raise ValueError(
                "only a list/tuple argument is supported in concat")
        df_list = guard(get_definition, self.func_ir, rhs.args[0])
        if not isinstance(df_list, ir.Expr) or not (df_list.op
                                            in ['build_tuple', 'build_list']):
            raise ValueError("pd.concat input should be constant list or tuple")

        if len(df_list.items) == 0:
            # copied error from pandas
            raise ValueError("No objects to concatenate")

        first_varname = df_list.items[0].name

        if first_varname in self.df_vars:
            return self._handle_concat_df(lhs, df_list.items, label)

        # XXX convert build_list to build_tuple since Numba doesn't handle list of
        # arrays
        if df_list.op == 'build_list':
            df_list.op = 'build_tuple'
        return self._handle_concat_series(lhs, rhs)

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

    def _handle_df_apply(self, assign, lhs, rhs, func_mod):
        # check for axis=1
        if not (len(rhs.kws) == 1 and rhs.kws[0][0] == 'axis'
                and get_constant(self.func_ir, rhs.kws[0][1]) == 1):
            raise ValueError("only apply() with axis=1 supported")

        if len(rhs.args) != 1:
            raise ValueError("lambda arg to apply() expected")

        # get apply function
        func = guard(get_definition, self.func_ir, rhs.args[0])
        if func is None or not (isinstance(func, ir.Expr)
                                and func.op == 'make_function'):
            raise ValueError("lambda for apply not found")

        _globals = self.func_ir.func_id.func.__globals__
        col_names = self._get_df_col_names(func_mod)

        # find columns that are actually used if possible
        used_cols = []
        lambda_ir = compile_to_numba_ir(func, _globals)
        l_topo_order = find_topo_order(lambda_ir.blocks)
        first_stmt = lambda_ir.blocks[l_topo_order[0]].body[0]
        assert isinstance(first_stmt, ir.Assign) and isinstance(first_stmt.value, ir.Arg)
        arg_var = first_stmt.target
        use_all_cols = False
        for bl in lambda_ir.blocks.values():
            for stmt in bl.body:
                vnames = [v.name for v in stmt.list_vars()]
                if arg_var.name in vnames:
                    if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                        continue
                    if (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)
                            and stmt.value.op == 'getattr'):
                        assert stmt.value.attr in col_names
                        used_cols.append(stmt.value.attr)
                    else:
                        # argument is used in some other form
                        # be conservative and use all cols
                        use_all_cols = True
                        used_cols = col_names
                        break

            if use_all_cols:
                break

        # remove duplicates with set() since a column can be used multiple times
        used_cols = set(used_cols)
        Row = namedtuple(_sanitize_varname(func_mod.name), used_cols)
        # TODO: handle non numpy alloc types
        # prange func to inline
        col_name_args = ', '.join(["c"+str(i) for i in range(len(used_cols))])
        row_args = ', '.join(["c"+str(i)+"[i]" for i in range(len(used_cols))])

        func_text = "def f({}):\n".format(col_name_args)
        func_text += "  numba.parfor.init_prange()\n"
        func_text += "  n = len(c0)\n"
        func_text += "  S = numba.unsafe.ndarray.empty_inferred((n,))\n"
        func_text += "  for i in numba.parfor.internal_prange(n):\n"
        func_text += "     row = Row({})\n".format(row_args)
        func_text += "     S[i] = map_func(row)\n"
        func_text += "  return S\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        f_ir = compile_to_numba_ir(f, {'numba': numba, 'np': np, 'Row': Row})
        # fix definitions to enable finding sentinel
        f_ir._definitions = build_definitions(f_ir.blocks)
        topo_order = find_topo_order(f_ir.blocks)

        # find sentinel function and replace with user func
        for l in topo_order:
            block = f_ir.blocks[l]
            for i, stmt in enumerate(block.body):
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'):
                    fdef = guard(get_definition, f_ir, stmt.value.func)
                    if isinstance(fdef, ir.Global) and fdef.name == 'map_func':
                        inline_closure_call(f_ir, _globals, block, i, func)
                        break

        df_col_map = self._get_df_cols(func_mod)
        col_vars = [df_col_map[c] for c in used_cols]
        replace_arg_nodes(f_ir.blocks[topo_order[0]], col_vars)
        return f_ir.blocks

    def _handle_df_describe(self, assign, lhs, rhs, func_mod):
        """translate df.describe() call with no input or just include='all'
        """
        # check for no arg or just include='all'
        if not (len(rhs.args) == 0 and (len(rhs.kws) == 0 or (len(rhs.kws) == 1
                and rhs.kws[0][0] == 'include'
                and get_constant(self.func_ir, rhs.kws[0][1]) == 'all'))):
            raise ValueError("only describe() with include='all' supported")

        col_names = self._get_df_col_names(func_mod)
        col_name_args = ["c"+str(i) for i in range(len(col_names))]
        # TODO: pandas returns dataframe, maybe return namedtuple instread of
        # string?

        func_text = "def f({}):\n".format(', '.join(col_name_args))
        # compute stat values
        for c in col_name_args:
            func_text += "  {}_count = np.float64({}.count())\n".format(c, c)
            func_text += "  {}_min = {}.min()\n".format(c, c)
            func_text += "  {}_max = {}.max()\n".format(c, c)
            func_text += "  {}_mean = {}.mean()\n".format(c, c)
            func_text += "  {}_std = {}.var()**0.5\n".format(c, c)
            func_text += "  {}_q25 = {}.quantile(.25)\n".format(c, c)
            func_text += "  {}_q50 = {}.quantile(.5)\n".format(c, c)
            func_text += "  {}_q75 = {}.quantile(.75)\n".format(c, c)


        col_header = "      ".join([c for c in col_names])
        func_text += "  return '        {}\\n' + \\\n".format(col_header)
        count_strs = "+ '   ' + ".join(["str({}_count)".format(c) for c in col_name_args])
        func_text += "   'count   ' + {} + '\\n' + \\\n".format(count_strs)
        mean_strs = "+ '   ' + ".join(["str({}_mean)".format(c) for c in col_name_args])
        func_text += "   'mean    ' + {} + '\\n' + \\\n".format(mean_strs)
        std_strs = "+ '   ' + ".join(["str({}_std)".format(c) for c in col_name_args])
        func_text += "   'std     ' + {} + '\\n' + \\\n".format(std_strs)
        min_strs = "+ '   ' + ".join(["str({}_min)".format(c) for c in col_name_args])
        func_text += "   'min     ' + {} + '\\n' + \\\n".format(min_strs)
        q25_strs = "+ '   ' + ".join(["str({}_q25)".format(c) for c in col_name_args])
        func_text += "   '25%     ' + {} + '\\n' + \\\n".format(q25_strs)
        q50_strs = "+ '   ' + ".join(["str({}_q50)".format(c) for c in col_name_args])
        func_text += "   '50%     ' + {} + '\\n' + \\\n".format(q50_strs)
        q75_strs = "+ '   ' + ".join(["str({}_q75)".format(c) for c in col_name_args])
        func_text += "   '75%     ' + {} + '\\n' + \\\n".format(q75_strs)
        max_strs = "+ '   ' + ".join(["str({}_max)".format(c) for c in col_name_args])
        func_text += "   'max     ' + {}\n".format(max_strs)

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        col_vars = self._get_df_col_vars(func_mod)
        return self._replace_func(f, col_vars)

    def _handle_df_sort_values(self, assign, lhs, rhs, df, label):
        kws = dict(rhs.kws)
        # find key array for sort ('by' arg)
        by_arg = self._get_arg('sort_values', rhs.args, kws, 0, 'by')
        err_msg = ("'by' argument is required for sort_values() "
                             "which should be a constant string")
        key_names = self._get_str_or_list(by_arg, err_msg=err_msg)

        inplace = False
        if 'inplace' in kws and guard(find_const, self.func_ir, kws['inplace']) == True:
            inplace = True

        # TODO: support ascending=False

        out = []
        in_df = self._get_df_cols(df).copy()  # copy since it'll be modified
        out_df = in_df.copy()
        if not inplace:
            out_df = {cname: ir.Var(lhs.scope, mk_unique_var(v.name), lhs.loc)
                                                for cname, v in in_df.items()}
            self._create_df(lhs.name, out_df.copy(), label)

        if any(k not in in_df for k in key_names):
            raise ValueError("invalid sort keys {}".format(key_names))

        # remove key from dfs (only data is kept)
        key_vars = [in_df.pop(k) for k in key_names]
        out_key_vars = [out_df.pop(k) for k in key_names]

        out.append(sort.Sort(df.name, lhs.name, key_vars, out_key_vars,
                                      in_df, out_df, inplace, lhs.loc))
        return out

    def _handle_df_itertuples(self, assign, lhs, rhs, df_var):
        """pass df column names and variables to get_itertuples() to be able
        to create the iterator.
        e.g. get_itertuples("A", "B", A_arr, B_arr)
        """
        col_names = self._get_df_col_names(df_var)

        col_name_args = ', '.join(["c"+str(i) for i in range(len(col_names))])
        name_consts = ', '.join(["'{}'".format(c) for c in col_names])

        func_text = "def f({}):\n".format(col_name_args)
        func_text += "  return hpat.hiframes.api.get_itertuples({}, {})\n"\
                                            .format(name_consts, col_name_args)

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        col_vars = self._get_df_col_vars(df_var)
        return self._replace_func(f, col_vars)

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

    def _run_arg(self, arg_assign, label):
        nodes = [arg_assign]
        arg_name = arg_assign.value.name
        arg_ind = arg_assign.value.index
        arg_var = arg_assign.target
        arg_typ = self.args[arg_ind]
        scope = arg_var.scope
        loc = arg_var.loc

        # TODO: handle datetime.date() series

        # input dataframe arg
        if isinstance(self.args[arg_ind], PandasDataFrameType):
            df_typ = self.args[arg_ind]
            df_items = {}
            for i, col in enumerate(df_typ.col_names):
                col_dtype = df_typ.col_types[i]
                if col_dtype == string_type:
                    alloc_dt = 11  # dummy string value
                elif col_dtype == types.List(string_type):
                    alloc_dt = 13  # dummy list(str) value
                elif col_dtype == types.boolean:
                    alloc_dt = "np.bool_"
                elif col_dtype == types.NPDatetime('ns'):
                    alloc_dt = 12  # XXX const code for dt64 since we can't init dt64 dtype
                else:
                    alloc_dt = "np.{}".format(col_dtype)

                func_text = "def f(_df):\n"
                func_text += "  _col_input_{} = hpat.hiframes.api.init_series(hpat.hiframes.boxing.unbox_df_column(_df, {}, {}))\n".format(col, i, alloc_dt)
                loc_vars = {}
                exec(func_text, {}, loc_vars)
                f = loc_vars['f']
                f_block = compile_to_numba_ir(f,
                            {'hpat': hpat, 'np': np}).blocks.popitem()[1]
                replace_arg_nodes(f_block, [arg_var])
                nodes += f_block.body[:-3]
                df_items[col] = nodes[-1].target

            self._create_df(arg_var.name, df_items, label)
            return nodes

        return nodes

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

    def _box_return_df(self, df_map):
        #
        arrs = list(df_map.values())
        names = list(df_map.keys())
        n_cols = len(arrs)

        arg_names = ", ".join(['in_{}'.format(i) for i in range(n_cols)])
        col_names = ", ".join(['"{}"'.format(cname) for cname in names])

        func_text = "def f({}):\n".format(arg_names)
        func_text += "  _dt_arr = hpat.hiframes.boxing.box_df({}, {})\n".format(col_names, arg_names)
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        f_block = compile_to_numba_ir(
            f, {'hpat': hpat}).blocks.popitem()[1]
        replace_arg_nodes(f_block, arrs)
        nodes = f_block.body[:-3]  # remove none return
        return nodes


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
        # if boxing df is required
        if self._is_df_var(cast.value):
            col_map = self.df_vars[cast.value.name]
            nodes = []
            # dist return arrays first
            if ret_name in flagged_vars:
                new_col_map = {}
                flag = ('distributed' if ret_name in self.metadata['distributed']
                        else 'threaded')
                for cname, var in col_map.items():
                    nodes += self._gen_replace_dist_return(var, flag)
                    new_col_map[cname] = nodes[-1].target
                col_map = new_col_map

            nodes += self._box_return_df(col_map)
            new_arr = nodes[-1].target
            new_cast = ir.Expr.cast(new_arr, loc)
            new_out = ir.Var(scope, mk_unique_var("df_return"), loc)
            nodes.append(ir.Assign(new_cast, new_out, loc))
            ret_node.value = new_out
            nodes.append(ret_node)
            return nodes

        elif ret_name in flagged_vars:
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
        """handle setitem: df['col_name'] = arr
        """
        # TODO: generalize to more cases
        # TODO: rename the dataframe variable to keep schema static
        df_label = self.df_labels[inst.target.name]
        # setting column possible only when it dominates the df creation to
        # keep schema consistent
        if label not in cfg.backbone() and label not in cfg.post_dominators()[df_label]:
            raise ValueError("setting dataframe columns inside conditionals and"
                             " loops not supported yet")
        if not isinstance(inst.index, str):
            raise ValueError("dataframe column name should be a string constant")

        df_name = inst.target.name
        # TODO: handle case where type has to be converted due to int64 NaNs
        self.df_vars[df_name][inst.index] = inst.value

        # set dataframe column if it is input and needs to be reflected
        df_def = guard(get_definition, self.func_ir, df_name)
        if isinstance(df_def, ir.Arg):
            # assign column name to variable
            cname_var = ir.Var(inst.value.scope, mk_unique_var("$cname_const"), inst.loc)
            nodes = [ir.Assign(ir.Const(inst.index, inst.loc), cname_var, inst.loc)]
            series_arr = inst.value

            def f(_df, _cname, _arr):  # pragma: no cover
                s = hpat.hiframes.api.set_df_col(_df, _cname, _arr)

            f_block = compile_to_numba_ir(f, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [inst.target, cname_var, series_arr])
            # copy propagate to enable string Const in typing and lowering
            simple_block_copy_propagate(f_block)
            nodes += f_block.body[:-3]  # remove none return
            return nodes

        return []

    def _handle_df_values(self, lhs, df):
        col_vars = self._get_df_col_vars(df)
        n_cols = len(col_vars)
        arg_names = ["C{}".format(i) for i in range(n_cols)]
        func_text = "def f({}):\n".format(", ".join(arg_names))
        func_text += "    return np.stack(({}), 1)\n".format(
            ",".join([s+".values" for s in arg_names]))

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, col_vars)

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

import operator
from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import warnings

import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (replace_arg_nodes, compile_to_numba_ir,
                            find_topo_order, gen_np_call, get_definition, guard,
                            find_callname, mk_alloc, find_const, is_setitem,
                            is_getitem, mk_unique_var, dprint_func_ir,
                            build_definitions, find_build_sequence,
                            GuardException, compute_cfg_from_blocks)
from numba.inline_closurecall import inline_closure_call
from numba.typing.templates import Signature, bound_function, signature
from numba.typing.arraydecl import ArrayAttribute
from numba.extending import overload
from numba.typing.templates import infer_global, AbstractTemplate, signature
import hpat
from hpat import hiframes
from hpat.utils import (debug_prints, inline_new_blocks, ReplaceFunc,
    is_whole_slice, is_array, is_assign)
from hpat.str_ext import string_type, unicode_to_std_str, std_str_to_unicode
from hpat.str_arr_ext import (string_array_type, StringArrayType,
    is_str_arr_typ, pre_alloc_string_array)
from hpat.pio_api import h5dataset_type
from hpat.hiframes.rolling import get_rolling_setup_args
from hpat.hiframes.pd_dataframe_ext import (DataFrameType, DataFrameLocType,
    DataFrameILocType, DataFrameIatType)
from hpat.hiframes.pd_series_ext import SeriesType, is_series_type
import hpat.hiframes.pd_groupby_ext
from hpat.hiframes.pd_groupby_ext import DataFrameGroupByType
import hpat.hiframes.pd_rolling_ext
from hpat.hiframes.pd_rolling_ext import RollingType
from hpat.hiframes.aggregate import get_agg_func


class DataFramePass(object):
    """Analyze and transform dataframe calls after typing"""

    def __init__(self, func_ir, typingctx, typemap, calltypes):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.typemap = typemap
        self.calltypes = calltypes

    def run(self):
        blocks = self.func_ir.blocks
        # topo_order necessary so DataFrame data replacement optimization can
        # be performed in one pass
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        dead_labels = []
        while work_list:
            label, block = work_list.pop()
            if label in dead_labels:
                continue

            # find dead blocks based on constant condition expression
            # for example, implementation of pd.merge() has comparison to None
            # TODO: add dead_branch_prune pass to inline_closure_call
            branch_or_jump = block.body[-1]
            if isinstance(branch_or_jump, ir.Branch):
                branch = branch_or_jump
                cond_val = guard(_eval_const_var, self.func_ir, branch.cond)
                if cond_val is not None:
                    # replace branch with Jump
                    dead_label = branch.falsebr if cond_val else branch.truebr
                    jmp_label = branch.truebr if cond_val else branch.falsebr
                    jmp = ir.Jump(jmp_label, branch.loc)
                    block.body[-1] = jmp
                    cfg = compute_cfg_from_blocks(self.func_ir.blocks)
                    if dead_label in cfg.dead_nodes():
                        dead_labels.append(dead_label)
                        # remove definitions in dead block so const variables can
                        # be found later (pd.merge() example)
                        # TODO: add this to dead_branch_prune pass
                        for inst in self.func_ir.blocks[dead_label].body:
                            if is_assign(inst):
                                self.func_ir._definitions[inst.target.name].remove(
                                    inst.value)

                        del self.func_ir.blocks[dead_label]
                    else:
                        # the jmp block overrides some definitions of current
                        # block so remove dead defs and update _definitions
                        # example: test_join_left_seq1
                        jmp_defs = set()
                        for inst in self.func_ir.blocks[jmp_label].body:
                            if is_assign(inst):
                                jmp_defs.add(inst.target.name)
                        used_vars = set()
                        new_body = []
                        for inst in reversed(block.body):
                            if (is_assign(inst)
                                    and inst.target.name not in used_vars
                                    and inst.target.name in jmp_defs):
                                self.func_ir._definitions[inst.target.name].remove(inst.value)
                                continue
                            used_vars.update(v.name for v in inst.list_vars())
                            new_body.append(inst)
                        new_body.reverse()
                        block.body = new_body

            new_body = []
            replaced = False
            for i, inst in enumerate(block.body):
                out_nodes = [inst]

                if isinstance(inst, ir.Assign):
                    self.func_ir._definitions[inst.target.name].remove(inst.value)
                    out_nodes = self._run_assign(inst)
                elif isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                    out_nodes = self._run_setitem(inst)

                if isinstance(out_nodes, list):
                    new_body.extend(out_nodes)
                    self._update_definitions(out_nodes)
                if isinstance(out_nodes, ReplaceFunc):
                    rp_func = out_nodes
                    if rp_func.pre_nodes is not None:
                        new_body.extend(rp_func.pre_nodes)
                        self._update_definitions(rp_func.pre_nodes)
                    # replace inst.value to a call with target args
                    # as expected by inline_closure_call
                    inst.value = ir.Expr.call(None, rp_func.args, (), inst.loc)
                    block.body = new_body + block.body[i:]
                    callee_blocks = inline_closure_call(self.func_ir, rp_func.glbls,
                        block, len(new_body), rp_func.func, self.typingctx,
                        rp_func.arg_types,
                        self.typemap, self.calltypes)
                    # add blocks in reversed topo order to enable dead branch
                    # pruning (merge example)
                    # TODO: fix inline_closure_call()
                    topo_order = find_topo_order(self.func_ir.blocks)
                    for c_label in reversed(topo_order):
                        if c_label in callee_blocks:
                            c_block = callee_blocks[c_label]
                            # include the new block created after callee used
                            # to split the original block
                            # find it using jumps out of callee (returns
                            # originally) but include only once
                            if isinstance(c_block.body[-1], ir.Jump):
                                target_label = c_block.body[-1].target
                                if (target_label not in callee_blocks
                                        and target_label not in work_list):
                                    work_list.append((target_label,
                                        self.func_ir.blocks[target_label]))
                            work_list.append((c_label, c_block))
                    replaced = True
                    break
                if isinstance(out_nodes, dict):
                    block.body = new_body + block.body[i:]
                    inline_new_blocks(self.func_ir, block, i, out_nodes, work_list)
                    replaced = True
                    break

            if not replaced:
                blocks[label].body = new_body

        while ir_utils.remove_dead(self.func_ir.blocks, self.func_ir.arg_names,
                                   self.func_ir, self.typemap):
            pass

        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "after hiframes_typed")
        return

    def _run_assign(self, assign):
        lhs = assign.target
        rhs = assign.value


        if isinstance(rhs, ir.Expr):
            if rhs.op == 'getattr':
                return self._run_getattr(assign, rhs)

            # if rhs.op == 'binop':
            #     return self._run_binop(assign, rhs)

            # # XXX handling inplace_binop similar to binop for now
            # # TODO handle inplace alignment
            # if rhs.op == 'inplace_binop':
            #     return self._run_binop(assign, rhs)

            # if rhs.op == 'unary':
            #     return self._run_unary(assign, rhs)

            # replace getitems on dataframe
            if rhs.op in ('getitem', 'static_getitem'):
                return self._run_getitem(assign, rhs)

            if rhs.op == 'call':
                return self._run_call(assign, lhs, rhs)

        return [assign]

    def _run_getitem(self, assign, rhs):
        lhs = assign.target
        nodes = []
        index_var = (rhs.index_var if rhs.op == 'static_getitem'
                                    else rhs.index)
        index_typ = self.typemap[index_var.name]

        # A = df['column'] or df[['C1', 'C2']]
        if rhs.op == 'static_getitem' and self._is_df_var(rhs.value):
            df_var = rhs.value
            df_typ = self.typemap[df_var.name]
            index = rhs.index

            # A = df['column']
            if isinstance(index, str):
                if index not in df_typ.columns:
                    raise ValueError(
                        "dataframe {} does not include column {}".format(
                            df_var.name, index))

                arr = self._get_dataframe_data(df_var, index, nodes)
                # TODO: index
                return  self._replace_func(
                    lambda A: hpat.hiframes.api.init_series(A),
                    [arr], pre_nodes=nodes)

            # df[['C1', 'C2']]
            if isinstance(index, list) and all(isinstance(c, str)
                                                               for c in index):
                nodes = []
                in_arrs = [self._get_dataframe_data(df_var, c, nodes)
                            for c in index]
                out_arrs = [self._gen_arr_copy(A, nodes) for A in in_arrs]
                #
                _init_df = _gen_init_df(index)
                return self._replace_func(_init_df, out_arrs, pre_nodes=nodes)
                # raise ValueError("unsupported dataframe access {}[{}]".format(
                #                  rhs.value.name, index))

        # df1 = df[df.A > .5]
        if self.is_bool_arr(index_var.name) and self._is_df_var(rhs.value):
            df_var = rhs.value
            return self._gen_df_filter(df_var, index_var, lhs)

        # df.loc[df.A > .5], df.iloc[df.A > .5]
        # df.iloc[1:n], df.iloc[np.array([1,2,3])], ...
        if ((self._is_df_loc_var(rhs.value) or self._is_df_iloc_var(rhs.value))
                and (self.is_bool_arr(index_var.name)
                or self.is_int_list_or_arr(index_var.name)
                or isinstance(index_typ, types.SliceType))):
            # TODO: check for errors
            df_var = guard(get_definition, self.func_ir, rhs.value).value
            return self._gen_df_filter(df_var, index_var, lhs)

        # df.iloc[1:n,0], df.loc[1:n,'A']
        if ((self._is_df_loc_var(rhs.value) or self._is_df_iloc_var(rhs.value))
                and isinstance(index_typ, types.Tuple)
                and len(index_typ) == 2):
            #
            df_var = guard(get_definition, self.func_ir, rhs.value).value
            df_typ = self.typemap[df_var.name]
            ind_def = guard(get_definition, self.func_ir, index_var)
            # TODO check and report errors
            assert isinstance(ind_def, ir.Expr) and ind_def.op == 'build_tuple'

            if self._is_df_iloc_var(rhs.value):
                col_ind = guard(find_const, self.func_ir, ind_def.items[1])
                col_name = df_typ.columns[col_ind]
            else:  # df.loc
                col_name = guard(find_const, self.func_ir, ind_def.items[1])

            col_filter_var = ind_def.items[0]
            name_var = ir.Var(lhs.scope, mk_unique_var('df_col_name'), lhs.loc)
            self.typemap[name_var.name] = types.StringLiteral(col_name)
            nodes.append(
                ir.Assign(ir.Const(col_name, lhs.loc), name_var, lhs.loc))
            in_arr = self._get_dataframe_data(df_var, col_name, nodes)

            if guard(is_whole_slice, self.typemap, self.func_ir, col_filter_var):
                func = lambda A, ind, name: hpat.hiframes.api.init_series(
                    A, None, name)
            else:
                # TODO: test this case
                func = lambda A, ind, name: hpat.hiframes.api.init_series(
                    A[ind], None, name)

            return self._replace_func(func,
                [in_arr, col_filter_var, name_var], pre_nodes=nodes)

        if self._is_df_iat_var(rhs.value):
            df_var = guard(get_definition, self.func_ir, rhs.value).value
            df_typ = self.typemap[df_var.name]
            # df.iat[3,1]
            if (rhs.op == 'static_getitem' and isinstance(rhs.index, tuple)
                    and len(rhs.index) == 2 and isinstance(rhs.index[0], int)
                    and isinstance(rhs.index[1], int)):
                col_ind = rhs.index[1]
                row_ind = rhs.index[0]
                col_name = df_typ.columns[col_ind]
                in_arr = self._get_dataframe_data(df_var, col_name, nodes)
                return self._replace_func(lambda A: A[row_ind], [in_arr],
                    extra_globals={'row_ind': row_ind}, pre_nodes=nodes)

            # df.iat[n,1]
            if isinstance(index_typ, types.Tuple) and len(index_typ) == 2:
                ind_def = guard(get_definition, self.func_ir, index_var)
                col_ind = guard(find_const, self.func_ir, ind_def.items[1])
                col_name = df_typ.columns[col_ind]
                in_arr = self._get_dataframe_data(df_var, col_name, nodes)
                row_ind = ind_def.items[0]
                return self._replace_func(lambda A, row_ind: A[row_ind],
                    [in_arr, row_ind], pre_nodes=nodes)

        nodes.append(assign)
        return nodes

    def _gen_df_filter(self, df_var, index_var, lhs):
        nodes = []
        df_typ = self.typemap[df_var.name]
        in_vars = {}
        out_vars = {}
        for col in df_typ.columns:
            in_arr = self._get_dataframe_data(df_var, col, nodes)
            out_arr = ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
            self.typemap[out_arr.name] = self.typemap[in_arr.name]
            in_vars[col] = in_arr
            out_vars[col] = out_arr

        nodes.append(hiframes.filter.Filter(
            lhs.name, df_var.name, index_var, out_vars, in_vars, lhs.loc))

        _init_df = _gen_init_df(df_typ.columns)

        return self._replace_func(
            _init_df, list(out_vars.values()), pre_nodes=nodes)

    def _run_setitem(self, inst):
        target_typ = self.typemap[inst.target.name]
        nodes = []
        index_var = (inst.index_var if isinstance(inst, ir.StaticSetItem)
                                    else inst.index)
        index_typ = self.typemap[index_var.name]

        if self._is_df_iat_var(inst.target):
            df_var = guard(get_definition, self.func_ir, inst.target).value
            df_typ = self.typemap[df_var.name]
            val = inst.value
            # df.iat[n,1] = 3
            if isinstance(index_typ, types.Tuple) and len(index_typ) == 2:
                ind_def = guard(get_definition, self.func_ir, index_var)
                col_ind = guard(find_const, self.func_ir, ind_def.items[1])
                col_name = df_typ.columns[col_ind]
                in_arr = self._get_dataframe_data(df_var, col_name, nodes)
                row_ind = ind_def.items[0]
                def _impl(A, row_ind, val):
                    A[row_ind] = val
                return self._replace_func(_impl,
                    [in_arr, row_ind, val], pre_nodes=nodes)
        return [inst]

    def _run_getattr(self, assign, rhs):
        rhs_type = self.typemap[rhs.value.name]  # get type of rhs value "df"

        # S = df.A (get dataframe column)
        # TODO: check invalid df.Attr?
        if isinstance(rhs_type, DataFrameType) and rhs.attr in rhs_type.columns:
            nodes = []
            col_name = rhs.attr
            arr = self._get_dataframe_data(rhs.value, col_name, nodes)
            index = self._get_dataframe_index(rhs.value, nodes)
            name = ir.Var(arr.scope, mk_unique_var('df_col_name'), arr.loc)
            self.typemap[name.name] = types.StringLiteral(col_name)
            nodes.append(ir.Assign(ir.Const(col_name, arr.loc), name, arr.loc))
            return self._replace_func(
                lambda arr, index, name: hpat.hiframes.api.init_series(
                    arr, index, name), [arr, index, name], pre_nodes=nodes)

        # A = df.values
        if isinstance(rhs_type, DataFrameType) and rhs.attr == 'values':
            return self._handle_df_values(assign.target, rhs.value)

        return [assign]

    def _handle_df_values(self, lhs, df_var):
        df_typ = self.typemap[df_var.name]
        n_cols = len(df_typ.columns)
        nodes = []
        data_arrs = [self._get_dataframe_data(df_var, c, nodes)
                    for c in df_typ.columns]
        data_args = ", ".join('data{}'.format(i) for i in range(n_cols))

        func_text = "def f({}):\n".format(data_args)
        func_text += "    return np.stack(({}), 1)\n".format(data_args)

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, data_arrs, pre_nodes=nodes)

    def _run_binop(self, assign, rhs):

        arg1, arg2 = rhs.lhs, rhs.rhs
        typ1, typ2 = self.typemap[arg1.name], self.typemap[arg2.name]
        if not (isinstance(typ1, DataFrameType) or isinstance(typ2, DataFrameType)):
            return [assign]

        # nodes = []
        # # TODO: support alignment, dt, etc.
        # # S3 = S1 + S2 ->
        # # S3_data = S1_data + S2_data; S3 = init_series(S3_data)
        # if isinstance(typ1, SeriesType):
        #     arg1 = self._get_series_data(arg1, nodes)
        # if isinstance(typ2, SeriesType):
        #     arg2 = self._get_series_data(arg2, nodes)

        # rhs.lhs, rhs.rhs = arg1, arg2
        # self._convert_series_calltype(rhs)

        # # output stays as Array in A += B where A is Array
        # if isinstance(self.typemap[assign.target.name], types.Array):
        #     assert isinstance(self.calltypes[rhs].return_type, types.Array)
        #     nodes.append(assign)
        #     return nodes

        # out_data = ir.Var(
        #     arg1.scope, mk_unique_var(assign.target.name+'_data'), rhs.loc)
        # self.typemap[out_data.name] = self.calltypes[rhs].return_type
        # nodes.append(ir.Assign(rhs, out_data, rhs.loc))
        # return self._replace_func(
        #     lambda data: hpat.hiframes.api.init_series(data, None, None),
        #     [out_data],
        #     pre_nodes=nodes
        # )

    def _run_unary(self, assign, rhs):
        arg = rhs.value
        typ = self.typemap[arg.name]

        if isinstance(typ, DataFrameType):
            nodes = []
            # arg = self._get_series_data(arg, nodes)
            # rhs.value = arg
            # self._convert_series_calltype(rhs)
            # out_data = ir.Var(
            #     arg.scope, mk_unique_var(assign.target.name+'_data'), rhs.loc)
            # self.typemap[out_data.name] = self.calltypes[rhs].return_type
            # nodes.append(ir.Assign(rhs, out_data, rhs.loc))
            # return self._replace_func(
            #     lambda data: hpat.hiframes.api.init_series(data),
            #     [out_data],
            #     pre_nodes=nodes
            # )

        return [assign]

    def _run_call(self, assign, lhs, rhs):
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            from numba.stencil import StencilFunc
            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == 'make_function':
                return [assign]
            if isinstance(func_def, ir.Global) and isinstance(func_def.value, StencilFunc):
                return [assign]
            warnings.warn(
                "function call couldn't be found for dataframe analysis")
            return [assign]
        else:
            func_name, func_mod = fdef

        if fdef == ('len', 'builtins') and self._is_df_var(rhs.args[0]):
            return self._run_call_len(lhs, rhs.args[0])

        if fdef == ('set_df_col', 'hpat.hiframes.api'):
            return self._run_call_set_df_column(assign, lhs, rhs)

        if fdef == ('merge', 'pandas'):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name:self.typemap[v.name]
                    for name, v in dict(rhs.kws).items()}
            impl = hpat.hiframes.pd_dataframe_ext.merge_overload(
                *arg_typs, **kw_typs)
            return self._replace_func(impl, rhs.args,
                        pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws))

        if fdef == ('merge_asof', 'pandas'):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name:self.typemap[v.name]
                    for name, v in dict(rhs.kws).items()}
            impl = hpat.hiframes.pd_dataframe_ext.merge_asof_overload(
                *arg_typs, **kw_typs)
            return self._replace_func(impl, rhs.args,
                        pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws))

        if fdef == ('join_dummy', 'hpat.hiframes.api'):
            return self._run_call_join(assign, lhs, rhs)

        if (isinstance(func_mod, ir.Var)
                and isinstance(self.typemap[func_mod.name], DataFrameType)):
            return self._run_call_dataframe(
                assign, assign.target, rhs, func_mod, func_name)

        if fdef == ('add_consts_to_type', 'hpat.hiframes.api'):
            assign.value = rhs.args[0]
            return [assign]

        if (isinstance(func_mod, ir.Var)
                and isinstance(self.typemap[func_mod.name],
                DataFrameGroupByType)):
            return self._run_call_groupby(
                assign, assign.target, rhs, func_mod, func_name)

        if (isinstance(func_mod, ir.Var)
                and isinstance(self.typemap[func_mod.name], RollingType)):
            return self._run_call_rolling(
                assign, assign.target, rhs, func_mod, func_name)

        if fdef == ('pivot_table_dummy', 'hpat.hiframes.pd_groupby_ext'):
            return self._run_call_pivot_table(assign, lhs, rhs)

        if fdef == ('crosstab', 'pandas'):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name:self.typemap[v.name]
                    for name, v in dict(rhs.kws).items()}
            impl = hpat.hiframes.pd_dataframe_ext.crosstab_overload(
                *arg_typs, **kw_typs)
            return self._replace_func(impl, rhs.args,
                        pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws))

        if fdef == ('crosstab_dummy', 'hpat.hiframes.pd_groupby_ext'):
            return self._run_call_crosstab(assign, lhs, rhs)

        return [assign]

    def _run_call_dataframe(self, assign, lhs, rhs, df_var, func_name):
        if func_name == 'merge':
            rhs.args.insert(0, df_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name:self.typemap[v.name]
                    for name, v in dict(rhs.kws).items()}
            impl = hpat.hiframes.pd_dataframe_ext.merge_overload(
                *arg_typs, **kw_typs)
            return self._replace_func(impl, rhs.args,
                        pysig=numba.utils.pysignature(pd.merge),
                        kws=dict(rhs.kws))

        if func_name == 'pivot_table':
            rhs.args.insert(0, df_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name:self.typemap[v.name]
                    for name, v in dict(rhs.kws).items()}
            impl = hpat.hiframes.pd_dataframe_ext.pivot_table_overload(
                *arg_typs, **kw_typs)
            stub = (lambda df, values=None, index=None, columns=None, aggfunc='mean',
                    fill_value=None, margins=False, dropna=True, margins_name='All',
                    _pivot_values=None: None)
            return self._replace_func(impl, rhs.args,
                        pysig=numba.utils.pysignature(stub),
                        kws=dict(rhs.kws))

        if func_name == 'rolling':
            rhs.args.insert(0, df_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name:self.typemap[v.name]
                    for name, v in dict(rhs.kws).items()}
            impl = hpat.hiframes.pd_rolling_ext.df_rolling_overload(
                *arg_typs, **kw_typs)
            stub = (lambda df, window, min_periods=None, center=False,
                        win_type=None, on=None, axis=0, closed=None: None)
            return self._replace_func(impl, rhs.args,
                        pysig=numba.utils.pysignature(stub),
                        kws=dict(rhs.kws))

        return [assign]

    def _run_call_set_df_column(self, assign, lhs, rhs):
        # replace with regular setitem if target is not dataframe
        # TODO: test non-df case
        if not self._is_df_var(rhs.args[0]):
            def _impl(target, index, val):
                target[index] = val
                return target
            return self._replace_func(_impl, rhs.args)

        df_var = rhs.args[0]
        cname = guard(find_const, self.func_ir, rhs.args[1])
        new_arr = rhs.args[2]
        df_typ = self.typemap[df_var.name]
        nodes = []

        # find df['col2'] = df['col1'][arr]
        # since columns should have the same size, output is filled with NaNs
        # TODO: make sure col1 and col2 are in the same df
        arr_def = guard(get_definition, self.func_ir, new_arr)
        if (isinstance(arr_def, ir.Expr)  and arr_def.op == 'getitem'
                and is_array(self.typemap, arr_def.value.name)
                and self.is_bool_arr(arr_def.index.name)):
            orig_arr = arr_def.value
            bool_arr = arr_def.index
            f_block = compile_to_numba_ir(
                lambda arr, bool_arr: hpat.hiframes.api.series_filter_bool(arr, bool_arr),
                {'hpat': hpat},
                self.typingctx,
                (self.typemap[orig_arr.name], self.typemap[bool_arr.name]),
                self.typemap,
                self.calltypes
            ).blocks.popitem()[1]
            replace_arg_nodes(f_block, [orig_arr, bool_arr])
            nodes += f_block.body[:-2]
            new_arr = nodes[-1].target


        # set unboxed df column with reflection
        if df_typ.has_parent:
            return self._replace_func(
                lambda df, cname, arr: hpat.hiframes.pd_dataframe_ext.set_df_column_with_reflect(
                    df, cname, hpat.hiframes.api.fix_df_array(arr)), [df_var, rhs.args[1], new_arr], pre_nodes=nodes)

        n_cols = len(df_typ.columns)
        in_arrs = [self._get_dataframe_data(df_var, c, nodes)
                    for c in df_typ.columns]
        data_args = ", ".join('data{}'.format(i) for i in range(n_cols))
        col_args = ", ".join("'{}'".format(c) for c in df_typ.columns)

        # if column is being added
        if cname not in df_typ.columns:
            data_args += ", new_arr"
            col_args += ", '{}'".format(cname)
            in_arrs.append(new_arr)
            new_arr_arg = 'new_arr'
        else:  # updating existing column
            col_ind = df_typ.columns.index(cname)
            in_arrs[col_ind] = new_arr
            new_arr_arg = 'data{}'.format(col_ind)

        # TODO: fix list, Series data
        func_text = "def _init_df({}):\n".format(data_args)
        func_text += "  {} = hpat.hiframes.api.fix_df_array({})\n".format(new_arr_arg, new_arr_arg)
        func_text += "  return hpat.hiframes.pd_dataframe_ext.init_dataframe({}, None, {})\n".format(
            data_args, col_args)
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars['_init_df']
        return self._replace_func(_init_df, in_arrs, pre_nodes=nodes)

    def _run_call_len(self, lhs, df_var):
        df_typ = self.typemap[df_var.name]

        # empty dataframe has 0 len
        if len(df_typ.columns) == 0:
            return [ir.Assign(ir.Const(0, lhs.loc), lhs, lhs.loc)]

        # run len on one of the columns
        # FIXME: it could potentially avoid remove dead for the column if
        # array analysis doesn't replace len() with it's size
        nodes = []
        arr = self._get_dataframe_data(df_var, df_typ.columns[0], nodes)
        def f(df_arr):  # pragma: no cover
            return len(df_arr)
        return self._replace_func(f, [arr], pre_nodes=nodes)

    def _run_call_join(self, assign, lhs, rhs):
        left_df, right_df, left_on_var, right_on_var, how_var = rhs.args

        left_on = self._get_const_or_list(left_on_var)
        right_on = self._get_const_or_list(right_on_var)
        how = guard(find_const, self.func_ir, how_var)
        out_typ = self.typemap[lhs.name]

        # convert right join to left join
        if how == 'right':
            how = 'left'
            left_df, right_df = right_df, left_df
            left_on, right_on = right_on, left_on

        nodes = []
        out_data_vars = {c: ir.Var(lhs.scope, mk_unique_var(c), lhs.loc)
                        for c in out_typ.columns}
        for v, t in zip(out_data_vars.values(), out_typ.data):
            self.typemap[v.name] = t

        left_arrs = {c: self._get_dataframe_data(left_df, c, nodes)
                            for c in self.typemap[left_df.name].columns}

        right_arrs = {c: self._get_dataframe_data(right_df, c, nodes)
                            for c in self.typemap[right_df.name].columns}

        nodes.append(hiframes.join.Join(lhs.name, left_df.name,
                                   right_df.name,
                                   left_on, right_on, out_data_vars, left_arrs,
                                   right_arrs, how, lhs.loc))

        _init_df = _gen_init_df(out_typ.columns)

        return self._replace_func(_init_df, list(out_data_vars.values()),
            pre_nodes=nodes)

    def _run_call_groupby(self, assign, lhs, rhs, grp_var, func_name):
        grp_typ = self.typemap[grp_var.name]
        df_var = self._get_df_obj_select(grp_var, 'groupby')
        df_type = self.typemap[df_var.name]
        out_typ = self.typemap[lhs.name]

        nodes = []
        in_vars = {c: self._get_dataframe_data(df_var, c, nodes)
                            for c in grp_typ.selection}

        in_key_arrs = [self._get_dataframe_data(df_var, c, nodes)
                            for c in grp_typ.keys]

        out_key_vars = None
        if grp_typ.as_index is False:
            out_key_vars = []
            for k in grp_typ.keys:
                out_key_var = ir.Var(lhs.scope, mk_unique_var(k), lhs.loc)
                ind = out_typ.columns.index(k)
                self.typemap[out_key_var.name] = out_typ.data[ind]
                out_key_vars.append(out_key_var)

        df_col_map = {}
        for c in grp_typ.selection:
            var = ir.Var(lhs.scope, mk_unique_var(c), lhs.loc)
            self.typemap[var.name] = (out_typ.data
                if isinstance(out_typ, SeriesType)
                else out_typ.data[out_typ.columns.index(c)])
            df_col_map[c] = var

        agg_func = get_agg_func(self.func_ir, func_name, rhs)

        agg_node = hiframes.aggregate.Aggregate(
            lhs.name, df_var.name, grp_typ.keys, out_key_vars, df_col_map,
            in_vars, in_key_arrs,
            agg_func, None, lhs.loc)

        nodes.append(agg_node)

        # XXX output becomes series if single output and explicitly selected
        if isinstance(out_typ, SeriesType):
            assert (len(grp_typ.selection) == 1
                and grp_typ.explicit_select
                and grp_typ.as_index)
            return self._replace_func(
                    lambda A: hpat.hiframes.api.init_series(A),
                    list(df_col_map.values()), pre_nodes=nodes)

        _init_df = _gen_init_df(out_typ.columns)

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = []
        for c in out_typ.columns:
            if c in grp_typ.keys:
                assert not grp_typ.as_index
                ind = grp_typ.keys.index(c)
                out_vars.append(out_key_vars[ind])
            else:
                out_vars.append(df_col_map[c])

        return self._replace_func(_init_df, out_vars,
            pre_nodes=nodes)

    def _run_call_pivot_table(self, assign, lhs, rhs):
        df_var, values, index, columns, aggfunc, _pivot_values = rhs.args
        func_name = self.typemap[aggfunc.name].literal_value
        values = self.typemap[values.name].literal_value
        index = self.typemap[index.name].literal_value
        columns = self.typemap[columns.name].literal_value
        pivot_values = self.typemap[_pivot_values.name].meta
        df_type = self.typemap[df_var.name]
        out_typ = self.typemap[lhs.name]

        nodes = []
        in_vars = {values: self._get_dataframe_data(df_var, values, nodes)}

        df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
                                for col in pivot_values})
        for v in df_col_map.values():
            self.typemap[v.name] = out_typ.data[0]

        pivot_arr = self._get_dataframe_data(df_var, columns, nodes)
        index_arr = self._get_dataframe_data(df_var, index, nodes)
        agg_func = get_agg_func(self.func_ir, func_name, rhs)

        agg_node = hiframes.aggregate.Aggregate(
            lhs.name, df_var.name, [index], None, df_col_map,
            in_vars, [index_arr],
            agg_func, None, lhs.loc, pivot_arr, pivot_values)
        nodes.append(agg_node)

        _init_df = _gen_init_df(out_typ.columns)

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = []
        for c in out_typ.columns:
            out_vars.append(df_col_map[c])

        return self._replace_func(_init_df, out_vars,
            pre_nodes=nodes)

    def _run_call_crosstab(self, assign, lhs, rhs):
        index, columns, _pivot_values = rhs.args
        pivot_values = self.typemap[_pivot_values.name].meta
        out_typ = self.typemap[lhs.name]

        in_vars = {}

        df_col_map = ({col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc)
                                for col in pivot_values})
        for i, v in enumerate(df_col_map.values()):
            self.typemap[v.name] = out_typ.data[i]

        pivot_arr = columns

        def _agg_len_impl(in_arr):  # pragma: no cover
            numba.parfor.init_prange()
            count = 0
            for i in numba.parfor.internal_prange(len(in_arr)):
                count += 1
            return count

        # TODO: make out_key_var an index column
        # TODO: check Series vs. array for index/columns
        agg_node = hpat.hiframes.aggregate.Aggregate(
            lhs.name, 'crosstab', [index.name], None, df_col_map,
            in_vars, [index],
            _agg_len_impl, None, lhs.loc, pivot_arr, pivot_values, True)
        nodes = [agg_node]

        _init_df = _gen_init_df(out_typ.columns)

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = []
        for c in out_typ.columns:
            out_vars.append(df_col_map[c])

        return self._replace_func(_init_df, out_vars,
            pre_nodes=nodes)

    def _run_call_rolling(self, assign, lhs, rhs, rolling_var, func_name):
        rolling_typ = self.typemap[rolling_var.name]
        dummy_call = guard(get_definition, self.func_ir, rolling_var)
        df_var, window, center, on = dummy_call.args
        df_type = self.typemap[df_var.name]
        out_typ = self.typemap[lhs.name]

        # handle 'on' arg
        if self.typemap[on.name] == types.none:
            on = None
        else:
            assert isinstance(self.typemap[on.name], types.StringLiteral)
            on = self.typemap[on.name].literal_value

        nodes = []
        # convert string offset window statically to nanos
        # TODO: support dynamic conversion
        # TODO: support other offsets types (time delta, etc.)
        if on is not None:
            window = guard(find_const, self.func_ir, window)
            if not isinstance(window, str):
                raise ValueError("window argument to rolling should be constant"
                                    "string in the offset case (variable window)")
            window = pd.tseries.frequencies.to_offset(window).nanos
            window_var = ir.Var(lhs.scope, mk_unique_var("window"), lhs.loc)
            self.typemap[window_var.name] = types.int64
            nodes.append(ir.Assign(ir.Const(window, lhs.loc), window_var, lhs.loc))
            window = window_var


        in_vars = {c: self._get_dataframe_data(df_var, c, nodes)
                            for c in rolling_typ.selection}

        on_arr = (self._get_dataframe_data(df_var, on, nodes)
                  if on is not None else None)

        df_col_map = {}
        for c in rolling_typ.selection:
            var = ir.Var(lhs.scope, mk_unique_var(c), lhs.loc)
            self.typemap[var.name] = (out_typ.data
                if isinstance(out_typ, SeriesType)
                else out_typ.data[out_typ.columns.index(c)])
            df_col_map[c] = var

        if on is not None:
            df_col_map[on] = on_arr  # TODO: copy array?

        other = None
        if func_name in ('cov', 'corr'):
            other = rhs.args[0]

        for cname, out_col_var in df_col_map.items():
            if cname == on:
                continue
            in_col_var = in_vars[cname]
            if func_name in ('cov', 'corr'):
                # TODO: Series as other
                if cname not in self.typemap[other.name].columns:
                    continue  # nan column handled below
                rhs.args[0] = self._get_dataframe_data(other, cname, nodes)
            nodes += self._gen_rolling_call(
                in_col_var, out_col_var, window, center, rhs.args, func_name,
                on_arr)

        # in corr/cov case, Pandas makes non-common columns NaNs
        if func_name in ('cov', 'corr'):
            nan_cols = list(set(self.typemap[other.name].columns) ^ set(df_type.columns))
            len_arr = list(in_vars.values())[0]
            for cname in nan_cols:
                def f(arr):
                    nan_arr = np.full(len(arr), np.nan)
                f_block = compile_to_numba_ir(f,
                    {'np': np},
                    self.typingctx,
                    (self.typemap[len_arr.name],),
                    self.typemap,
                    self.calltypes).blocks.popitem()[1]
                replace_arg_nodes(f_block, [len_arr])
                nodes += f_block.body[:-3]  # remove none return
                df_col_map[cname] = nodes[-1].target

        # XXX output becomes series if single output and explicitly selected
        if isinstance(out_typ, SeriesType):
            assert (len(rolling_typ.selection) == 1
                and rolling_typ.explicit_select
                and rolling_typ.as_index)
            return self._replace_func(
                    lambda A: hpat.hiframes.api.init_series(A),
                    list(df_col_map.values()), pre_nodes=nodes)

        _init_df = _gen_init_df(out_typ.columns)

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = []
        for c in out_typ.columns:
            out_vars.append(df_col_map[c])

        return self._replace_func(_init_df, out_vars,
            pre_nodes=nodes)

    def _gen_rolling_call(self, in_col_var, out_col_var, window, center, args,
                                                            func_name, on_arr):
        nodes = []
        if func_name in ('cov', 'corr'):
            other = args[0]
            if on_arr is not None:
                if func_name == 'cov':
                    def f(arr, other, on_arr, w, center):  # pragma: no cover
                        df_arr = hpat.hiframes.rolling.rolling_cov(
                                arr, other, on_arr, w, center)
                if func_name == 'corr':
                    def f(arr, other, on_arr, w, center):  # pragma: no cover
                        df_arr = hpat.hiframes.rolling.rolling_corr(
                                arr, other, on_arr, w, center)
                args = [in_col_var, other, on_arr, window, center]
            else:
                if func_name == 'cov':
                    def f(arr, other, w, center):  # pragma: no cover
                        df_arr = hpat.hiframes.rolling.rolling_cov(
                                arr, other, w, center)
                if func_name == 'corr':
                    def f(arr, other, w, center):  # pragma: no cover
                        df_arr = hpat.hiframes.rolling.rolling_corr(
                                arr, other, w, center)
                args = [in_col_var, other, window, center]
        # variable window case
        elif on_arr is not None:
            if func_name == 'apply':
                def f(arr, on_arr, w, center, func):  # pragma: no cover
                    df_arr = hpat.hiframes.rolling.rolling_variable(
                            arr, on_arr, w, center, False, func)
                args = [in_col_var, on_arr, window, center, args[0]]
            else:
                def f(arr, on_arr, w, center):  # pragma: no cover
                    df_arr = hpat.hiframes.rolling.rolling_variable(
                            arr, on_arr, w, center, False, _func_name)
                args = [in_col_var, on_arr, window, center]
        else:  # fixed window
            # apply case takes the passed function instead of just name
            if func_name == 'apply':
                def f(arr, w, center, func):  # pragma: no cover
                    df_arr = hpat.hiframes.rolling.rolling_fixed(
                            arr, w, center, False, func)
                args = [in_col_var, window, center, args[0]]
            else:
                def f(arr, w, center):  # pragma: no cover
                    df_arr = hpat.hiframes.rolling.rolling_fixed(
                            arr, w, center, False, _func_name)
                args = [in_col_var, window, center]

        arg_typs = tuple(self.typemap[v.name] for v in args)
        f_block = compile_to_numba_ir(f,
            {'hpat': hpat, '_func_name': func_name},
            self.typingctx,
            arg_typs,
            self.typemap,
            self.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, args)
        nodes += f_block.body[:-3]  # remove none return
        nodes[-1].target = out_col_var
        return nodes

    def _get_df_obj_select(self, obj_var, obj_name):
        """get df object for groupby() or rolling()
        e.g. groupby('A')['B'], groupby('A')['B', 'C'], groupby('A')
        """
        select_def = guard(get_definition, self.func_ir, obj_var)
        if isinstance(select_def, ir.Expr) and select_def.op in ('getitem', 'static_getitem'):
            obj_var = select_def.value

        obj_call = guard(get_definition, self.func_ir, obj_var)
        # find dataframe
        call_def = guard(find_callname, self.func_ir, obj_call)
        assert (call_def is not None and call_def[0] == obj_name
                and isinstance(call_def[1], ir.Var)
                and self._is_df_var(call_def[1]))
        df_var = call_def[1]

        return df_var

    def _get_const_tup(self, tup_var):
        tup_def = guard(get_definition, self.func_ir, tup_var)
        if isinstance(tup_def, ir.Expr):
            if tup_def.op == 'binop' and tup_def.fn in ('+', operator.add):
                return (self._get_const_tup(tup_def.lhs)
                        + self._get_const_tup(tup_def.rhs))
            if tup_def.op in ('build_tuple', 'build_list'):
                return tup_def.items
        raise ValueError("constant tuple expected")

    def _get_dataframe_data(self, df_var, col_name, nodes):
        # optimization: return data var directly if not ambiguous
        # (no multiple init_dataframe calls for the same df_var with control
        # flow)
        # e.g. A = init_dataframe(A, None, 'A')
        # XXX assuming init_dataframe is the only call to create a dataframe
        # and dataframe._data is never overwritten
        df_typ = self.typemap[df_var.name]
        ind = df_typ.columns.index(col_name)
        var_def = guard(get_definition, self.func_ir, df_var)
        call_def = guard(find_callname, self.func_ir, var_def)
        if call_def == ('init_dataframe', 'hpat.hiframes.pd_dataframe_ext'):
            return var_def.args[ind]

        loc = df_var.loc
        ind_var = ir.Var(df_var.scope, mk_unique_var('col_ind'), loc)
        self.typemap[ind_var.name] = types.IntegerLiteral(ind)
        nodes.append(ir.Assign(ir.Const(ind, loc), ind_var, loc))
        # XXX use get_series_data() for getting data instead of S._data
        # to enable alias analysis
        f_block = compile_to_numba_ir(
            lambda df, c_ind: hpat.hiframes.pd_dataframe_ext.get_dataframe_data(
                df, c_ind),
            {'hpat': hpat},
            self.typingctx,
            (df_typ, self.typemap[ind_var.name]),
            self.typemap,
            self.calltypes
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [df_var, ind_var])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _get_dataframe_index(self, df_var, nodes):
        df_typ = self.typemap[df_var.name]
        n_cols = len(df_typ.columns)
        var_def = guard(get_definition, self.func_ir, df_var)
        call_def = guard(find_callname, self.func_ir, var_def)
        if call_def == ('init_dataframe', 'hpat.hiframes.pd_dataframe_ext'):
            return var_def.args[n_cols]

        # XXX use get_series_data() for getting data instead of S._data
        # to enable alias analysis
        f_block = compile_to_numba_ir(
            lambda df: hpat.hiframes.pd_dataframe_ext.get_dataframe_index(df),
            {'hpat': hpat},
            self.typingctx,
            (df_typ,),
            self.typemap,
            self.calltypes
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [df_var])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _replace_func(self, func, args, const=False,
                      pre_nodes=None, extra_globals=None, pysig=None, kws=None):
        glbls = {'numba': numba, 'np': np, 'hpat': hpat}
        if extra_globals is not None:
            glbls.update(extra_globals)

        # create explicit arg variables for defaults if func has any
        # XXX: inine_closure_call() can't handle defaults properly
        if pysig is not None:
            pre_nodes = [] if pre_nodes is None else pre_nodes
            scope = next(iter(self.func_ir.blocks.values())).scope
            loc = scope.loc
            def normal_handler(index, param, default):
                return default

            def default_handler(index, param, default):
                d_var = ir.Var(scope, mk_unique_var('defaults'), loc)
                self.typemap[d_var.name] = numba.typeof(default)
                node = ir.Assign(ir.Const(default, loc), d_var, loc)
                pre_nodes.append(node)
                return d_var

            # TODO: stararg needs special handling?
            args = numba.typing.fold_arguments(
                pysig, args, kws, normal_handler, default_handler,
                normal_handler)

        arg_typs = tuple(self.typemap[v.name] for v in args)

        if const:
            new_args = []
            for i, arg in enumerate(args):
                val = guard(find_const, self.func_ir, arg)
                if val:
                    new_args.append(types.literal(val))
                else:
                    new_args.append(arg_typs[i])
            arg_typs = tuple(new_args)
        return ReplaceFunc(func, arg_typs, args, glbls, pre_nodes)

    def _is_df_var(self, var):
        return isinstance(self.typemap[var.name], DataFrameType)

    def _is_df_loc_var(self, var):
        return isinstance(self.typemap[var.name], DataFrameLocType)

    def _is_df_iloc_var(self, var):
        return isinstance(self.typemap[var.name], DataFrameILocType)

    def _is_df_iat_var(self, var):
        return isinstance(self.typemap[var.name], DataFrameIatType)

    def is_bool_arr(self, varname):
        typ = self.typemap[varname]
        return (isinstance(typ, (SeriesType, types.Array))
            and typ.dtype == types.bool_)

    def is_int_list_or_arr(self, varname):
        typ = self.typemap[varname]
        return (isinstance(typ, (SeriesType, types.Array, types.List))
            and isinstance(typ.dtype, types.Integer))

    def _is_const_none(self, var):
        var_def = guard(get_definition, self.func_ir, var)
        return isinstance(var_def, ir.Const) and var_def.value is None

    def _update_definitions(self, node_list):
        dumm_block = ir.Block(None, None)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.func_ir._definitions)
        return

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

    def _gen_arr_copy(self, in_arr, nodes):
        f_block = compile_to_numba_ir(
            lambda A: A.copy(), {}, self.typingctx,
            (self.typemap[in_arr.name],), self.typemap, self.calltypes
            ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [in_arr])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _get_const_or_list(self, by_arg, list_only=False, default=None, err_msg=None, typ=None):
        var_typ = self.typemap[by_arg.name]
        if isinstance(var_typ, types.Optional):
            var_typ = var_typ.type
        if hasattr(var_typ, 'consts'):
            return var_typ.consts

        typ = str if typ is None else typ
        by_arg_def = guard(find_build_sequence, self.func_ir, by_arg)
        if by_arg_def is None:
            # try single key column
            by_arg_def = guard(find_const, self.func_ir, by_arg)
            if by_arg_def is None:
                if default is not None:
                    return default
                raise ValueError(err_msg)
            if isinstance(var_typ, types.BaseTuple):
                assert isinstance(by_arg_def, tuple)
                return by_arg_def
            key_colnames = (by_arg_def,)
        else:
            if list_only and by_arg_def[1] != 'build_list':
                if default is not None:
                    return default
                raise ValueError(err_msg)
            key_colnames = tuple(guard(find_const, self.func_ir, v) for v in by_arg_def[0])
            if any(not isinstance(v, typ) for v in key_colnames):
                if default is not None:
                    return default
                raise ValueError(err_msg)
        return key_colnames


def _gen_init_df(columns):
    n_cols = len(columns)
    data_args = ", ".join('data{}'.format(i) for i in range(n_cols))

    func_text = "def _init_df({}):\n".format(data_args)
    func_text += "  return hpat.hiframes.pd_dataframe_ext.init_dataframe({}, None, {})\n".format(
        data_args, ", ".join("'{}'".format(c) for c in columns))
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    _init_df = loc_vars['_init_df']

    return _init_df


def _eval_const_var(func_ir, var):
    try:
        return find_const(func_ir, var)
    except GuardException:
        pass
    var_def = guard(get_definition, func_ir, var)
    if isinstance(var_def, ir.Expr) and var_def.op == 'binop':
        return var_def.fn(
            _eval_const_var(func_ir, var_def.lhs),
            _eval_const_var(func_ir, var_def.rhs))

    raise GuardException

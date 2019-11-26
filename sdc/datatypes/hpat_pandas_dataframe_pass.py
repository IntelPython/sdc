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

'''
| Procedures are required for SDC DataFrameType handling in Numba
'''

import numpy
import pandas

import numba
from numba import ir, types
from numba.compiler_machinery import FunctionPass, register_pass
from numba.ir_utils import find_topo_order, build_definitions, guard, find_callname

import sdc


@register_pass(mutates_CFG=True, analysis_only=False)
class SDC_Pandas_DataFrame_TransformationPass_Stage2(FunctionPass):
    """
    This transformation pass replaces known function definitions with implementation
    """

    _name = "sdc_dataframe_transformationpass_stage2"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        self.state = state

        blocks = self.state.func_ir.blocks

        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            new_body = []
            replaced = False
            for i, inst in enumerate(block.body):
                out_nodes = [inst]

                if isinstance(inst, ir.Assign):
                    self.state.func_ir._definitions[inst.target.name].remove(inst.value)
                    out_nodes = self._run_assign(inst)

                if isinstance(out_nodes, list):
                    new_body.extend(out_nodes)
                    self._update_definitions(out_nodes)

                if isinstance(out_nodes, sdc.utils.ReplaceFunc):
                    rp_func = out_nodes
                    if rp_func.pre_nodes is not None:
                        new_body.extend(rp_func.pre_nodes)
                        self._update_definitions(rp_func.pre_nodes)

                    inst.value = ir.Expr.call(ir.Var(block.scope, "dummy", inst.loc), rp_func.args, (), inst.loc)
                    block.body = new_body + block.body[i:]
                    sdc.utils.update_globals(rp_func.func, rp_func.glbls)
                    numba.inline_closurecall.inline_closure_call(self.state.func_ir,
                                                                 rp_func.glbls,
                                                                 block,
                                                                 len(new_body),
                                                                 rp_func.func,
                                                                 self.state.typingctx,
                                                                 rp_func.arg_types,
                                                                 self.state.typemap,
                                                                 self.state.calltypes,
                                                                 work_list)
                    replaced = True
                    break
                if isinstance(out_nodes, dict):
                    block.body = new_body + block.body[i:]
                    sdc.utils.inline_new_blocks(self.state.func_ir, block, i, out_nodes, work_list)
                    replaced = True
                    break

            if not replaced:
                blocks[label].body = new_body

        # XXX remove slice() of h5 read due to Numba's #3380 bug
        self.state.func_ir.blocks = numba.ir_utils.simplify_CFG(self.state.func_ir.blocks)
        while numba.ir_utils.remove_dead(self.state.func_ir.blocks,
                                         self.state.func_ir.arg_names,
                                         self.state.func_ir,
                                         self.state.typemap):
            pass

        self.state.func_ir._definitions = build_definitions(self.state.func_ir.blocks)
        numba.ir_utils.dprint_func_ir(self.state.func_ir, self._name)

        return True

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            if rhs.op == 'getattr':
                return self._run_getattr(assign, rhs)

            if rhs.op == 'call':
                return self._run_call(assign, lhs, rhs)

        return [assign]

    def _run_getattr(self, assign, rhs):
        rhs_type = self.state.typemap[rhs.value.name]  # get type of rhs value "S"

        if (rhs.attr == 'dtype'
            and (sdc.hiframes.pd_series_ext.is_series_type(rhs_type) or isinstance(rhs_type, types.Array))
            and isinstance(rhs_type.dtype, (types.NPDatetime, types.NPTimedelta))):

            assign.value = ir.Global("numpy.datetime64", rhs_type.dtype, rhs.loc)
            return [assign]

        if (rhs.attr == 'dtype' and isinstance(sdc.hiframes.pd_series_ext.if_series_to_array_type(rhs_type), types.Array)):
            typ_str = str(rhs_type.dtype)
            assign.value = ir.Global("numpy.dtype({})".format(typ_str), numpy.dtype(typ_str), rhs.loc)
            return [assign]

        # PR135. This needs to be commented out
        if isinstance(rhs_type, sdc.hiframes.pd_series_ext.SeriesType) and rhs.attr == 'values':
            # simply return the column
            nodes = []
            var = self._get_series_data(rhs.value, nodes)
            assign.value = var
            nodes.append(assign)
            return nodes

        return [assign]

    def _run_call(self, assign, lhs, rhs):
        func_name = None  # literal name of analyzed function
        func_mod = None  # literal name of analyzed module

        fdef = guard(find_callname, self.state.func_ir, rhs, self.state.typemap)
        if fdef is None:
            return [assign]
        else:
            func_name, func_mod = fdef

        if func_mod == 'sdc.hiframes.api':
            return self._run_call_hiframes(assign, assign.target, rhs, func_name)

        return [assign]

    def _run_call_hiframes(self, assign, lhs, rhs, func_name):
        if func_name == 'df_isin':
            nodes = []
            data, other = rhs.args

            def _isin_series(A, B):
                numba.parfor.init_prange()
                n = len(A)
                m = len(B)
                S = numpy.empty(n, numpy.bool_)

                for i in numba.parfor.internal_prange(n):
                    S[i] = (A[i] == B[i] if i < m else False)

                return S

            return self._replace_func(_isin_series, [data, other], pre_nodes=nodes)

        if func_name == 'df_isin_vals':
            nodes = []
            data = rhs.args[0]

            def _isin_series(A, vals):
                numba.parfor.init_prange()
                n = len(A)
                S = numpy.empty(n, numpy.bool_)

                for i in numba.parfor.internal_prange(n):
                    S[i] = A[i] in vals

                return S

            return self._replace_func(_isin_series, [data, rhs.args[1]], pre_nodes=nodes)

        if func_name == 'series_filter_bool':
            return self._handle_df_col_filter(assign, lhs, rhs)

        return [assign]

    def _handle_df_col_filter(self, assign, lhs, rhs):
        nodes = []
        in_arr = rhs.args[0]
        bool_arr = rhs.args[1]
        if sdc.hiframes.pd_series_ext.is_series_type(self.state.typemap[in_arr.name]):
            in_arr = self._get_series_data(in_arr, nodes)
        if sdc.hiframes.pd_series_ext.is_series_type(self.state.typemap[bool_arr.name]):
            bool_arr = self._get_series_data(bool_arr, nodes)

        return self._replace_func(sdc.hiframes.series_kernels._column_filter_impl, [in_arr, bool_arr], pre_nodes=nodes)

    def _get_series_data(self, series_var, nodes):
        var_def = guard(numba.ir_utils.get_definition, self.state.func_ir, series_var)
        call_def = guard(find_callname, self.state.func_ir, var_def)

        if call_def == ('init_series', 'sdc.hiframes.api'):
            return var_def.args[0]

        def _get_series_data_lambda(S):
            return sdc.hiframes.api.get_series_data(S)

        f_block = numba.ir_utils.compile_to_numba_ir(_get_series_data_lambda,
                                                     {'sdc': sdc},
                                                     self.state.typingctx,
                                                     (self.state.typemap[series_var.name], ),
                                                     self.state.typemap,
                                                     self.state.calltypes).blocks.popitem()[1]
        numba.ir_utils.replace_arg_nodes(f_block, [series_var])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _replace_func(self, func, args, const=False, pre_nodes=None, extra_globals=None, pysig=None, kws=None):
        glbls = {'numba': numba, 'numpy': numpy, 'sdc': sdc}
        if extra_globals is not None:
            glbls.update(extra_globals)

        if pysig is not None:
            pre_nodes = [] if pre_nodes is None else pre_nodes
            scope = next(iter(self.state.func_ir.blocks.values())).scope
            loc = scope.loc

            def normal_handler(index, param, default):
                return default

            def default_handler(index, param, default):
                d_var = ir.Var(scope, numba.ir_utils.mk_unique_var('defaults'), loc)
                self.state.typemap[d_var.name] = numba.typeof(default)
                node = ir.Assign(ir.Const(default, loc), d_var, loc)
                pre_nodes.append(node)

                return d_var

            args = numba.typing.fold_arguments(pysig, args, kws, normal_handler, default_handler, normal_handler)

        arg_typs = tuple(self.state.typemap[v.name] for v in args)

        if const:
            new_args = []

            for i, arg in enumerate(args):
                val = guard(numba.ir_utils.find_const, self.state.func_ir, arg)
                if val:
                    new_args.append(types.literal(val))
                else:
                    new_args.append(arg_typs[i])
            arg_typs = tuple(new_args)

        return sdc.utils.ReplaceFunc(func, arg_typs, args, glbls, pre_nodes)

    def _update_definitions(self, node_list):
        loc = ir.Loc("", 0)
        dumm_block = ir.Block(ir.Scope(None, loc), loc)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.state.func_ir._definitions)

        return

@register_pass(mutates_CFG=True, analysis_only=False)
class SDC_Pandas_DataFrame_TransformationPass_Stage1(FunctionPass):
    """
    This transformation pass replaces known function definitions with implementation
    """

    _name = "sdc_dataframe_transformationpass_stage1"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        self.state = state
        # replace inst variables as determined previously during the pass
        # currently use to keep lhs of Arg nodes intact
        self.replace_var_dict = {}

        # df_var -> {col1:col1_var ...}
        self.df_vars = {}
        # df_var -> label where it is defined
        self.df_labels = {}

        numba.ir_utils._max_label = max(self.state.func_ir.blocks.keys()) # shssf:  is it still needed?

        # FIXME: see why this breaks test_kmeans
        # remove_dels(self.state.func_ir.blocks)
        numba.ir_utils.dprint_func_ir(self.state.func_ir, self._name)
        blocks = self.state.func_ir.blocks
        # call build definition since rewrite pass doesn't update definitions
        # e.g. getitem to static_getitem in test_column_list_select2
        self.state.func_ir._definitions = build_definitions(blocks)
        # topo_order necessary since df vars need to be found before use
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            new_body = []
            replaced = False
            self._working_body = new_body
            for i, inst in enumerate(block.body):
                self._replace_vars(inst)
                out_nodes = [inst]

                # handle potential dataframe set column here
                # df['col'] = arr
                if (isinstance(inst, ir.StaticSetItem)
                        and isinstance(inst.index, str)):
                    # cfg needed for set df column
                    cfg = numba.ir_utils.compute_cfg_from_blocks(blocks)
                    out_nodes = self._run_df_set_column(inst, label, cfg)
                elif isinstance(inst, ir.Assign):
                    self.state.func_ir._definitions[inst.target.name].remove(inst.value)
                    out_nodes = self._run_assign(inst, label)

                if isinstance(out_nodes, list):
                    # TODO: fix scope/loc
                    new_body.extend(out_nodes)
                    self._update_definitions(out_nodes)
                if isinstance(out_nodes, sdc.utils.ReplaceFunc):
                    rp_func = out_nodes
                    if rp_func.pre_nodes is not None:
                        new_body.extend(rp_func.pre_nodes)
                        self._update_definitions(rp_func.pre_nodes)
                    # replace inst.value to a call with target args
                    # as expected by numba.inline_closurecall.inline_closure_call
                    # TODO: inst other than Assign?
                    inst.value = ir.Expr.call(
                        ir.Var(block.scope, "dummy", inst.loc),
                        rp_func.args, (), inst.loc)
                    block.body = new_body + block.body[i:]
                    sdc.utils.update_globals(rp_func.func, rp_func.glbls)
                    numba.inline_closurecall.inline_closure_call(self.state.func_ir, rp_func.glbls,
                                        block, len(new_body), rp_func.func, work_list=work_list)
                    replaced = True
                    break
                if isinstance(out_nodes, dict):
                    block.body = new_body + block.body[i:]
                    # TODO: insert new blocks in current spot of work_list
                    # instead of append?
                    # TODO: rename variables, fix scope/loc
                    sdc.utils.inline_new_blocks(self.state.func_ir, block, len(new_body), out_nodes, work_list)
                    replaced = True
                    break
            if not replaced:
                blocks[label].body = new_body

        self.state.func_ir.blocks = numba.ir_utils.simplify_CFG(self.state.func_ir.blocks)
        # self.state.func_ir._definitions = build_definitions(blocks)
        # XXX: remove dead here fixes h5 slice issue
        # iterative remove dead to make sure all extra code (e.g. df vars) is removed
        # while remove_dead(blocks, self.state.func_ir.arg_names, self.state.func_ir):
        #     pass
        self.state.func_ir._definitions = build_definitions(blocks)
        numba.ir_utils.dprint_func_ir(self.state.func_ir, self._name)

        return True

    def _replace_vars(self, inst):
        # variable replacement can affect definitions so handling assignment
        # values specifically
        if sdc.utils.is_assign(inst):
            lhs = inst.target.name
            self.state.func_ir._definitions[lhs].remove(inst.value)

        numba.ir_utils.replace_vars_stmt(inst, self.replace_var_dict)

        if sdc.utils.is_assign(inst):
            self.state.func_ir._definitions[lhs].append(inst.value)
            # if lhs changed, TODO: test
            if inst.target.name != lhs:
                self.state.func_ir._definitions[inst.target.name] = self.state.func_ir._definitions[lhs]

    def _run_assign(self, assign, label):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            if rhs.op == 'call':
                return self._run_call(assign, label)

            # HACK: delete pd.DataFrame({}) nodes to avoid typing errors
            # TODO: remove when dictionaries are implemented and typing works
            if rhs.op == 'getattr':
                val_def = guard(numba.ir_utils.get_definition, self.state.func_ir, rhs.value)
                if (isinstance(val_def, ir.Global) and val_def.value == pandas and rhs.attr in ('DataFrame', 'read_csv', 'read_parquet', 'to_numeric')):
                    # TODO: implement to_numeric in typed pass?
                    # put back the definition removed earlier but remove node
                    # enables function matching without node in IR
                    self.state.func_ir._definitions[lhs].append(rhs)
                    return []

            if rhs.op == 'getattr':
                val_def = guard(numba.ir_utils.get_definition, self.state.func_ir, rhs.value)
                if (isinstance(val_def, ir.Global) and val_def.value == numpy and rhs.attr == 'fromfile'):
                    # put back the definition removed earlier but remove node
                    self.state.func_ir._definitions[lhs].append(rhs)
                    return []

            if rhs.op == 'make_function':
                # HACK make globals availabe for typing in series.map()
                rhs.globals = self.state.func_ir.func_id.func.__globals__

        return [assign]

    def _run_call(self, assign, label):
        """handle calls and return new nodes if needed
        """
        lhs = assign.target
        rhs = assign.value

        func_name = None
        func_mod = None
        fdef = guard(find_callname, self.state.func_ir, rhs)

        if fdef is None:
            return [assign]
        else:
            func_name, func_mod = fdef

        # handling pd.DataFrame() here since input can be constant dictionary
        if fdef == ('DataFrame', 'pandas'):
            return self._handle_pd_DataFrame(assign, lhs, rhs, label)

        if fdef == ('concat', 'pandas'):
            return self._handle_concat(assign, lhs, rhs, label)

        if isinstance(func_mod, ir.Var) and self._is_df_var(func_mod):
            return self._run_call_df(assign, lhs, rhs, func_mod, func_name, label)

        if func_name == 'drop' and isinstance(func_mod, ir.Var):
            # handle potential df.drop(inplace=True) here since it needs
            # variable replacement
            return self._handle_df_drop(assign, lhs, rhs, func_mod)

        # groupby aggregate
        # e.g. df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
        if isinstance(func_mod, ir.Var) and self._is_df_obj_call(func_mod, 'groupby'):
            return self._handle_aggregate(lhs, rhs, func_mod, func_name, label)

        # rolling window
        # e.g. df.rolling(2).sum
        if isinstance(func_mod, ir.Var) and self._is_df_obj_call(func_mod, 'rolling'):
            return self._handle_rolling(lhs, rhs, func_mod, func_name, label)

        return [assign]

    def _run_call_df(self, assign, lhs, rhs, df_var, func_name, label):

        # df.isin()
        if func_name == 'isin':
            return self._handle_df_isin(lhs, rhs, df_var, label)

        # df.append()
        if func_name == 'append':
            return self._handle_df_append(lhs, rhs, df_var, label)

        # df.fillna()
        if func_name == 'fillna':
            return self._handle_df_fillna(lhs, rhs, df_var, label)

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
            other_def = guard(numba.ir_utils.get_definition, self.state.func_ir, other)
            # dict case
            if isinstance(other_def, ir.Expr) and other_def.op == 'build_map':
                for c, v in other_def.items:
                    cname = guard(numba.ir_utils.find_const, self.state.func_ir, c)
                    if not isinstance(cname, str):
                        raise ValueError("dictionary argument to isin() should have constant keys")
                    other_colmap[cname] = v
                    # HACK replace build_map to avoid inference errors
                    other_def.op = 'build_list'
                    other_def.items = [v[0] for v in other_def.items]
            else:
                # general iterable (e.g. list, set) case
                # TODO: handle passed in dict case (pass colname to func?)
                other_colmap = {c: other for c in df_col_map.keys()}

        out_df_map = {}

        def isin_func(A, B):
            # XXX df isin is different than Series.isin, df.isin considers
            #  index but Series.isin ignores it (everything is set)
            # TODO: support strings and other types
            nodes = []
            data, other = rhs.args

            def _isin_series(A, B):
                numba.parfor.init_prange()
                n = len(A)
                m = len(B)
                S = numpy.empty(n, np.bool_)
                for i in numba.parfor.internal_prange(n):
                    S[i] = (A[i] == B[i] if i < m else False)
                return S

            return self._replace_func(_isin_series, [data, other], pre_nodes=nodes)
            # return sdc.hiframes.api.df_isin(A, B)

        def isin_vals_func(A, B):
            return sdc.hiframes.api.df_isin_vals(A, B)
        # create array of False values used when other col not available
        def bool_arr_func(A):
            return sdc.hiframes.api.init_series(np.zeros(len(A), np.bool_))
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
            f_block = numba.ir_utils.compile_to_numba_ir(func, {'sdc': sdc, 'numpy': numpy}).blocks.popitem()[1]
            numba.ir_utils.replace_arg_nodes(f_block, args)
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
        df_list = guard(numba.ir_utils.get_definition, self.state.func_ir, other)
        if len(df_list.items) > 0 and self._is_df_var(df_list.items[0]):
            return self._handle_concat_df(lhs, [df_var] + df_list.items, label)
        raise ValueError("invalid df.append() input. Only dataframe and list of dataframes supported")

    def _handle_df_fillna(self, lhs, rhs, df_var, label):
        nodes = []
        inplace_default = ir.Var(lhs.scope, numba.ir_utils.mk_unique_var("fillna_default"), lhs.loc)
        nodes.append(ir.Assign(ir.Const(False, lhs.loc), inplace_default, lhs.loc))
        val_var = self._get_arg('fillna', rhs.args, dict(rhs.kws), 0, 'value')
        inplace_var = self._get_arg('fillna', rhs.args, dict(rhs.kws), 3, 'inplace', default=inplace_default)

        def _fillna_func(A, val, inplace):
            return A.fillna(val, inplace=inplace)

        out_col_map = {}
        for cname, in_var in self._get_df_cols(df_var).items():
            f_block = numba.ir_utils.compile_to_numba_ir(_fillna_func, {}).blocks.popitem()[1]
            numba.ir_utils.replace_arg_nodes(f_block, [in_var, val_var, inplace_var])
            nodes += f_block.body[:-2]
            out_col_map[cname] = nodes[-1].target

        # create output df if not inplace
        if (inplace_var.name == inplace_default.name or guard(numba.ir_utils.find_const, self.state.func_ir, inplace_var) == False):
            self._create_df(lhs.name, out_col_map, label)
        return nodes

    def _handle_df_dropna(self, lhs, rhs, df_var, label):
        nodes = []
        inplace_default = ir.Var(lhs.scope, numba.ir_utils.mk_unique_var("dropna_default"), lhs.loc)
        nodes.append(ir.Assign(ir.Const(False, lhs.loc), inplace_default, lhs.loc))
        inplace_var = self._get_arg('dropna', rhs.args, dict(rhs.kws), 4, 'inplace', default=inplace_default)

        col_names = self._get_df_col_names(df_var)
        col_vars = self._get_df_col_vars(df_var)
        arg_names = ", ".join([numba.ir_utils.mk_unique_var(cname).replace('.', '_') for cname in col_names])
        out_names = ", ".join([numba.ir_utils.mk_unique_var(cname).replace('.', '_') for cname in col_names])

        func_text = "def _dropna_imp({}, inplace):\n".format(arg_names)
        func_text += "  ({},) = sdc.hiframes.api.dropna(({},), inplace)\n".format(
            out_names, arg_names)
        loc_vars = {}
        exec(func_text, {'sdc': sdc}, loc_vars)
        _dropna_imp = loc_vars['_dropna_imp']

        f_block = numba.ir_utils.compile_to_numba_ir(_dropna_imp, {'sdc': sdc}).blocks.popitem()[1]
        numba.ir_utils.replace_arg_nodes(f_block, col_vars + [inplace_var])
        nodes += f_block.body[:-3]

        # extract column vars from output
        out_col_map = {}
        for i, cname in enumerate(col_names):
            out_col_map[cname] = nodes[-len(col_names) + i].target

        # create output df if not inplace
        if (inplace_var.name == inplace_default.name or guard(numba.ir_utils.find_const, self.state.func_ir, inplace_var) == False):
            self._create_df(lhs.name, out_col_map, label)
        else:
            # assign back to column vars for inplace case
            for i in range(len(col_vars)):
                c_var = col_vars[i]
                dropped_var = list(out_col_map.values())[i]
                nodes.append(ir.Assign(dropped_var, c_var, lhs.loc))
        return nodes

    def _handle_df_drop(self, assign, lhs, rhs, df_var):
        """handle possible df.drop(inplace=True)
        lhs = A.drop(inplace=True) -> A1, lhs = drop_inplace(...)
        replace A with A1
        """
        kws = dict(rhs.kws)
        inplace_var = self._get_arg('drop', rhs.args, kws, 5, 'inplace', '')
        inplace = guard(numba.ir_utils.find_const, self.state.func_ir, inplace_var)
        if inplace is not None and inplace:
            # TODO: make sure call post dominates df_var definition or df_var
            # is not used in other code paths
            # replace func variable with drop_inplace
            f_block = numba.ir_utils.compile_to_numba_ir(lambda: sdc.hiframes.api.drop_inplace, {'sdc': sdc}).blocks.popitem()[1]
            nodes = f_block.body[:-2]
            new_func_var = nodes[-1].target
            rhs.func = new_func_var
            rhs.args.insert(0, df_var)
            # new tuple return
            ret_tup = ir.Var(lhs.scope, numba.ir_utils.mk_unique_var('drop_ret'), lhs.loc)
            assign.target = ret_tup
            nodes.append(assign)
            new_df_var = ir.Var(df_var.scope, numba.ir_utils.mk_unique_var(df_var.name), df_var.loc)
            zero_var = ir.Var(df_var.scope, numba.ir_utils.mk_unique_var('zero'), df_var.loc)
            one_var = ir.Var(df_var.scope, numba.ir_utils.mk_unique_var('one'), df_var.loc)
            nodes.append(ir.Assign(ir.Const(0, lhs.loc), zero_var, lhs.loc))
            nodes.append(ir.Assign(ir.Const(1, lhs.loc), one_var, lhs.loc))
            getitem0 = ir.Expr.static_getitem(ret_tup, 0, zero_var, lhs.loc)
            nodes.append(ir.Assign(getitem0, new_df_var, lhs.loc))
            getitem1 = ir.Expr.static_getitem(ret_tup, 1, one_var, lhs.loc)
            nodes.append(ir.Assign(getitem1, lhs, lhs.loc))
            # replace old variable with new one
            self.replace_var_dict[df_var.name] = new_df_var
            return nodes

        return [assign]

        # df.drop(labels=None, axis=0, index=None, columns=None, level=None,
        #         inplace=False, errors='raise')
        labels_var = self._get_arg('drop', rhs.args, kws, 0, 'labels', '')
        axis_var = self._get_arg('drop', rhs.args, kws, 1, 'axis', '')
        labels = self._get_str_or_list(labels_var, default='')
        axis = guard(numba.ir_utils.find_const, self.state.func_ir, axis_var)

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
        inplace = guard(numba.ir_utils.find_const, self.state.func_ir, inplace_var)

        if inplace is not None and inplace:
            df_label = self.df_labels[df_var.name]
            cfg = numba.ir_utils.compute_cfg_from_blocks(self.state.func_ir.blocks)
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
        out_df_map = {c: _gen_arr_copy(in_df_map[c], nodes)
                      for c in in_df_map.keys() if c not in columns}
        self._create_df(lhs.name, out_df_map, label)
        return nodes

    def _handle_pd_DataFrame(self, assign, lhs, rhs, label):
        """
        Handle pd.DataFrame({'A': A}) call
        """
        kws = dict(rhs.kws)
        if 'data' in kws:
            data = kws['data']
            if len(rhs.args) != 0:  
                raise ValueError(
                    "only data argument suppoted in pd.DataFrame()")
        else:
            if len(rhs.args) != 1:  
                raise ValueError(
                    "data argument in pd.DataFrame() expected")
            data = rhs.args[0]

        arg_def = guard(numba.ir_utils.get_definition, self.state.func_ir, data)
        if (not isinstance(arg_def, ir.Expr) or arg_def.op != 'build_map'):  
            raise ValueError("Invalid DataFrame() arguments (constant dict of columns expected)")

        nodes, items = self._fix_df_arrays(arg_def.items)

        # HACK replace build_map to avoid inference errors
        arg_def.op = 'build_list'
        arg_def.items = [v[0] for v in arg_def.items]

        n_cols = len(items)
        data_args = ", ".join('data{}'.format(i) for i in range(n_cols))
        col_args = ", ".join('col{}'.format(i) for i in range(n_cols))

        func_text = "def _init_df({}, index, {}):\n".format(data_args, col_args)
        func_text += "  return sdc.hiframes.pd_dataframe_ext.init_dataframe({}, index, {})\n".format(data_args, col_args)
        loc_vars = {}
        exec(func_text, {'sdc': sdc}, loc_vars)
        _init_df = loc_vars['_init_df']

        # TODO: support index var
        index = ir.Var(lhs.scope, numba.ir_utils.mk_unique_var('df_index_none'), lhs.loc)
        nodes.append(ir.Assign(ir.Const(None, lhs.loc), index, lhs.loc))
        data_vars = [a[1] for a in items]
        col_vars = [a[0] for a in items]
        args = data_vars + [index] + col_vars

        return self._replace_func(_init_df, args, pre_nodes=nodes)

    def _get_csv_col_info(self, dtype_map, date_cols, col_names, lhs):
        if isinstance(dtype_map, types.Type):
            typ = dtype_map
            data_arrs = [ir.Var(lhs.scope, numba.ir_utils.mk_unique_var(cname), lhs.loc)
                         for cname in col_names]
            return col_names, data_arrs, [typ] * len(col_names)

        columns = []
        data_arrs = []
        out_types = []
        for i, (col_name, typ) in enumerate(dtype_map.items()):
            columns.append(col_name)
            # get array dtype
            if i in date_cols:
                typ = types.Array(types.NPDatetime('ns'), 1, 'C')
            out_types.append(typ)
            # output array variable
            data_arrs.append(
                ir.Var(lhs.scope, numba.ir_utils.mk_unique_var(col_name), lhs.loc))

        return columns, data_arrs, out_types

    def _get_const_dtype(self, dtype_var):
        dtype_def = guard(numba.ir_utils.get_definition, self.state.func_ir, dtype_var)
        if isinstance(dtype_def, ir.Const) and isinstance(dtype_def.value, str):
            typ_name = dtype_def.value
            if typ_name == 'str':
                return string_array_type
            typ_name = 'int64' if typ_name == 'int' else typ_name
            typ_name = 'float64' if typ_name == 'float' else typ_name
            typ = getattr(types, typ_name)
            typ = types.Array(typ, 1, 'C')
            return typ

        # str case
        if isinstance(dtype_def, ir.Global) and dtype_def.value == str:
            return string_array_type
        # categorical case
        if isinstance(dtype_def, ir.Expr) and dtype_def.op == 'call':
            if (not guard(find_callname, self.state.func_ir, dtype_def)
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
        glob_def = guard(numba.ir_utils.get_definition, self.state.func_ir, dtype_def.value)
        if not isinstance(glob_def, ir.Global) or glob_def.value != numpy:
            raise ValueError("pd.read_csv() invalid dtype")
        # TODO: extend to other types like string and date, check error
        typ_name = dtype_def.attr
        typ_name = 'int64' if typ_name == 'int' else typ_name
        typ_name = 'float64' if typ_name == 'float' else typ_name
        typ = getattr(types, typ_name)
        typ = types.Array(typ, 1, 'C')
        return typ

    def _handle_concat(self, assign, lhs, rhs, label):
        # converting build_list to build_tuple before type inference to avoid
        # errors
        kws = dict(rhs.kws)
        objs_arg = self._get_arg('concat', rhs.args, kws, 0, 'objs')

        df_list = guard(numba.ir_utils.get_definition, self.state.func_ir, objs_arg)
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
        def gen_nan_func(A): return np.full(len(A), np.nan)
        # gen concat function
        arg_names = ", ".join(['in{}'.format(i) for i in range(len(df_list))])
        func_text = "def _concat_imp({}):\n".format(arg_names)
        func_text += "    return sdc.hiframes.api.init_series(sdc.hiframes.api.concat(({})))\n".format(
            arg_names)
        loc_vars = {}
        exec(func_text, {'sdc': sdc}, loc_vars)
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
                    f_block = numba.ir_utils.compile_to_numba_ir(gen_nan_func,
                                                  {'sdc': sdc, 'numpy': numpy}).blocks.popitem()[1]
                    numba.ir_utils.replace_arg_nodes(f_block, [len_arr])
                    nodes += f_block.body[:-2]
                    args.append(nodes[-1].target)
                else:
                    args.append(df_col_map[cname])

            f_block = numba.ir_utils.compile_to_numba_ir(_concat_imp,
                                          {'sdc': sdc, 'numpy': numpy}).blocks.popitem()[1]
            numba.ir_utils.replace_arg_nodes(f_block, args)
            nodes += f_block.body[:-2]
            done_cols[cname] = nodes[-1].target

        self._create_df(lhs.name, done_cols, label)
        return nodes

    def _handle_concat_series(self, lhs, rhs):
        # defer to typed pass since the type might be non-numerical
        def f(arr_list):  
            return sdc.hiframes.api.init_series(sdc.hiframes.api.concat(arr_list))
        return self._replace_func(f, rhs.args)

    def _fix_df_arrays(self, items_list):
        nodes = []
        new_list = []
        for item in items_list:
            col_varname = item[0]
            col_arr = item[1]
            # fix list(multi-dim arrays) (packing images)
            # FIXME: does this break for list(other things)?
            col_arr = self._fix_df_list_of_array(col_arr)

            def f(arr):  
                df_arr = sdc.hiframes.api.fix_df_array(arr)
            f_block = numba.ir_utils.compile_to_numba_ir(
                f, {'sdc': sdc}).blocks.popitem()[1]
            numba.ir_utils.replace_arg_nodes(f_block, [col_arr])
            nodes += f_block.body[:-3]  # remove none return
            new_col_arr = nodes[-1].target
            new_list.append((col_varname, new_col_arr))
        return nodes, new_list

    def _fix_df_list_of_array(self, col_arr):
        list_call = guard(numba.ir_utils.get_definition, self.state.func_ir, col_arr)
        if guard(find_callname, self.state.func_ir, list_call) == ('list', 'builtins'):
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
                col_name = get_constant(self.state.func_ir, col_var)
                if col_name is NOT_CONSTANT:  
                    raise ValueError(
                        "data frame column names should be constant")
            # cast to series type

            def f(arr):  
                df_arr = sdc.hiframes.api.init_series(arr)
            f_block = numba.ir_utils.compile_to_numba_ir(
                f, {'sdc': sdc}).blocks.popitem()[1]
            numba.ir_utils.replace_arg_nodes(f_block, [item[1]])
            nodes += f_block.body[:-3]  # remove none return
            new_col_arr = nodes[-1].target
            df_cols[col_name] = new_col_arr
        return nodes, df_cols

    def _is_df_obj_call(self, call_var, obj_name):
        """
        determines whether variable is coming from groupby() or groupby()[], rolling(), rolling()[]
        """

        var_def = guard(numba.ir_utils.get_definition, self.state.func_ir, call_var)
        # groupby()['B'] case
        if (isinstance(var_def, ir.Expr)
                and var_def.op in ['getitem', 'static_getitem']):
            return self._is_df_obj_call(var_def.value, obj_name)
        # groupby() called on column or df
        call_def = guard(find_callname, self.state.func_ir, var_def)
        if (call_def is not None and call_def[0] == obj_name
                and isinstance(call_def[1], ir.Var)
                and self._is_df_var(call_def[1])):
            return True
        return False

    def _get_str_arg(self, f_name, args, kws, arg_no, arg_name, default=None, err_msg=None):
        arg = None
        if len(args) > arg_no:
            arg = guard(numba.ir_utils.find_const, self.state.func_ir, args[arg_no])
        elif arg_name in kws:
            arg = guard(numba.ir_utils.find_const, self.state.func_ir, kws[arg_name])

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = ("SDC error. {} requires '{}' argument as a constant string").format(f_name, arg_name)
            raise ValueError(err_msg)
        return arg

    def _get_arg(self, f_name, args, kws, arg_no, arg_name, default=None, err_msg=None):
        arg = None
        if len(args) > arg_no:
            arg = args[arg_no]
        elif arg_name in kws:
            arg = kws[arg_name]

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = "SDC error. {} requires '{}' argument".format(f_name, arg_name)
            raise ValueError(err_msg)
        return arg

    def _handle_aggregate(self, lhs, rhs, obj_var, func_name, label):
        # format df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
        # TODO: support aggregation functions sum, count, etc.
        if func_name not in supported_agg_funcs:
            raise ValueError("only {} supported in groupby".format(", ".join(supported_agg_funcs)))

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
                    out_key_var = ir.Var(lhs.scope, numba.ir_utils.mk_unique_var(k), lhs.loc)
                    out_df[k] = out_key_var
                    out_key_vars.append(out_key_var)
            df_col_map = ({col: ir.Var(lhs.scope, numba.ir_utils.mk_unique_var(col), lhs.loc)
                           for col in out_colnames})
            out_df.update(df_col_map)

            self._create_df(lhs.name, out_df, label)

        in_key_vars = [self.df_vars[df_var.name][k] for k in key_colnames]

        agg_node = aggregate.Aggregate(lhs.name, df_var.name, key_colnames, out_key_vars, df_col_map, in_vars, in_key_vars, agg_func, out_tp_vars, lhs.loc)
        nodes.append(agg_node)
        return nodes

    def _handle_agg_func(self, in_vars, out_colnames, func_name, lhs, rhs):
        agg_func = get_agg_func(self.state.func_ir, func_name, rhs)
        out_tp_vars = {}

        # sdc.jit() instead of numba.njit() to handle str arrs etc
        agg_func_dis = sdc.jit(agg_func)
        #agg_func_dis = numba.njit(agg_func)
        agg_gb_var = ir.Var(lhs.scope, numba.ir_utils.mk_unique_var("agg_gb"), lhs.loc)
        nodes = [ir.Assign(ir.Global("agg_gb", agg_func_dis, lhs.loc), agg_gb_var, lhs.loc)]
        for out_cname in out_colnames:
            in_var = in_vars[out_cname]

            def to_arr(a, _agg_f):
                b = sdc.hiframes.api.to_arr_from_series(a)
                res = sdc.hiframes.api.init_series(sdc.hiframes.api.agg_typer(b, _agg_f))
            f_block = numba.ir_utils.compile_to_numba_ir(to_arr, {'sdc': sdc, 'numpy': numpy}).blocks.popitem()[1]
            numba.ir_utils.replace_arg_nodes(f_block, [in_var, agg_gb_var])
            nodes += f_block.body[:-3]  # remove none return
            out_tp_vars[out_cname] = nodes[-1].target
        return nodes, agg_func, out_tp_vars

    def _get_agg_obj_args(self, agg_var):
        # find groupby key and as_index
        groubpy_call = guard(numba.ir_utils.get_definition, self.state.func_ir, agg_var)
        assert isinstance(groubpy_call, ir.Expr) and groubpy_call.op == 'call'
        kws = dict(groubpy_call.kws)
        as_index = True
        if 'as_index' in kws:
            as_index = guard(numba.ir_utils.find_const, self.state.func_ir, kws['as_index'])
            if as_index is None:
                raise ValueError("groupby as_index argument should be constant")
        if len(groubpy_call.args) == 1:
            by_arg = groubpy_call.args[0]
        elif 'by' in kws:
            by_arg = kws['by']
        else:  
            raise ValueError("by argument for groupby() required")

        err_msg = ("groupby() by argument should be list of column names or a column name")
        key_colnames = self._get_str_or_list(by_arg, True, err_msg=err_msg)

        return key_colnames, as_index

    def _get_str_or_list(self, by_arg, list_only=False, default=None, err_msg=None, typ=None):
        typ = str if typ is None else typ
        by_arg_def = guard(find_build_sequence, self.state.func_ir, by_arg)

        if by_arg_def is None:
            # try add_consts_to_type
            by_arg_call = guard(numba.ir_utils.get_definition, self.state.func_ir, by_arg)
            if guard(find_callname, self.state.func_ir, by_arg_call) == ('add_consts_to_type', 'sdc.hiframes.api'):
                by_arg_def = guard(find_build_sequence, self.state.func_ir, by_arg_call.args[0])

        if by_arg_def is None:
            # try dict.keys()
            by_arg_call = guard(numba.ir_utils.get_definition, self.state.func_ir, by_arg)
            call_name = guard(find_callname, self.state.func_ir, by_arg_call)
            if (call_name is not None and len(call_name) == 2
                    and call_name[0] == 'keys'
                    and isinstance(call_name[1], ir.Var)):
                var_def = guard(numba.ir_utils.get_definition, self.state.func_ir, call_name[1])
                if isinstance(var_def, ir.Expr) and var_def.op == 'build_map':
                    by_arg_def = [v[0] for v in var_def.items], 'build_map'
                    # HACK replace dict.keys getattr to avoid typing errors
                    keys_getattr = guard(
                        numba.ir_utils.get_definition, self.state.func_ir, by_arg_call.func)
                    assert isinstance(
                        keys_getattr, ir.Expr) and keys_getattr.attr == 'keys'
                    keys_getattr.attr = 'copy'

        if by_arg_def is None:
            # try single key column
            by_arg_def = guard(numba.ir_utils.find_const, self.state.func_ir, by_arg)
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
            key_colnames = [guard(numba.ir_utils.find_const, self.state.func_ir, v) for v in by_arg_def[0]]
            if any(not isinstance(v, typ) for v in key_colnames):
                if default is not None:
                    return default
                raise ValueError(err_msg)
        return key_colnames

    def _get_df_obj_select(self, obj_var, obj_name):
        """analyze selection of columns in after groupby() or rolling()
        e.g. groupby('A')['B'], groupby('A')['B', 'C'], groupby('A')
        """
        select_def = guard(numba.ir_utils.get_definition, self.state.func_ir, obj_var)
        out_colnames = None
        explicit_select = False
        if isinstance(select_def, ir.Expr) and select_def.op in ('getitem', 'static_getitem'):
            obj_var = select_def.value
            out_colnames = (select_def.index
                            if select_def.op == 'static_getitem'
                            else guard(numba.ir_utils.find_const, self.state.func_ir, select_def.index))
            if not isinstance(out_colnames, (str, tuple)):
                raise ValueError("{} output column names should be constant".format(obj_name))
            if isinstance(out_colnames, str):
                out_colnames = [out_colnames]
            explicit_select = True

        obj_call = guard(numba.ir_utils.get_definition, self.state.func_ir, obj_var)
        # find dataframe
        call_def = guard(find_callname, self.state.func_ir, obj_call)
        assert (call_def is not None and call_def[0] == obj_name
                and isinstance(call_def[1], ir.Var)
                and self._is_df_var(call_def[1]))
        df_var = call_def[1]

        return df_var, out_colnames, explicit_select, obj_var

    def _handle_rolling(self, lhs, rhs, obj_var, func_name, label):
        # format df.rolling(w)['B'].sum()
        # TODO: support aggregation functions sum, count, etc.
        if func_name not in sdc.hiframes.rolling.supported_rolling_funcs:
            raise ValueError("only ({}) supported in rolling".format(", ".join(sdc.hiframes.rolling.supported_rolling_funcs)))

        nodes = []
        # find selected output columns
        df_var, out_colnames, explicit_select, obj_var = self._get_df_obj_select(obj_var, 'rolling')
        rolling_call = guard(numba.ir_utils.get_definition, self.state.func_ir, obj_var)
        window, center, on = get_rolling_setup_args(self.state.func_ir, rolling_call, False)
        on_arr = self.df_vars[df_var.name][on] if on is not None else None
        if not isinstance(center, ir.Var):
            center_var = ir.Var(lhs.scope, numba.ir_utils.mk_unique_var("center"), lhs.loc)
            nodes.append(ir.Assign(ir.Const(center, lhs.loc), center_var, lhs.loc))
            center = center_var
        if not isinstance(window, ir.Var):
            window_var = ir.Var(lhs.scope, numba.ir_utils.mk_unique_var("window"), lhs.loc)
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
            df_col_map = ({col: ir.Var(lhs.scope, numba.ir_utils.mk_unique_var(col), lhs.loc)
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
                nan_arr = numpy.full(len(arr), np.nan)
            f_block = numba.ir_utils.compile_to_numba_ir(f, {'numpy': numpy}).blocks.popitem()[1]
            numba.ir_utils.replace_arg_nodes(f_block, [len_arr])
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
                    def f(arr, other, on_arr, w, center):  
                        df_arr = sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_cov(arr, other, on_arr, w, center))
                if func_name == 'corr':
                    def f(arr, other, on_arr, w, center):  
                        df_arr = sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_corr(arr, other, on_arr, w, center))
                args = [in_col_var, other, on_arr, window, center]
            else:
                if func_name == 'cov':
                    def f(arr, other, w, center):  
                        df_arr = sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_cov(arr, other, w, center))
                if func_name == 'corr':
                    def f(arr, other, w, center):  
                        df_arr = sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_corr(arr, other, w, center))
                args = [in_col_var, other, window, center]
        # variable window case
        elif on_arr is not None:
            if func_name == 'apply':
                def f(arr, on_arr, w, center, func):
                    df_arr = sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_variable(arr, on_arr, w, center, False, func))
                args = [in_col_var, on_arr, window, center, args[0]]
            else:
                def f(arr, on_arr, w, center):  
                    df_arr = sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_variable(arr, on_arr, w, center, False, _func_name))
                args = [in_col_var, on_arr, window, center]
        else:  # fixed window
            # apply case takes the passed function instead of just name
            if func_name == 'apply':
                def f(arr, w, center, func):  
                    df_arr = sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_fixed(arr, w, center, False, func))
                args = [in_col_var, window, center, args[0]]
            else:
                def f(arr, w, center):  
                    df_arr = sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_fixed(arr, w, center, False, _func_name))
                args = [in_col_var, window, center]

        f_block = numba.ir_utils.compile_to_numba_ir(f, {'sdc': sdc, '_func_name': func_name}).blocks.popitem()[1]
        numba.ir_utils.replace_arg_nodes(f_block, args)
        nodes += f_block.body[:-3]  # remove none return
        nodes[-1].target = out_col_var

        return nodes

    def _run_df_set_column(self, inst, label, cfg):
        """replace setitem of string index with a call to handle possible
        dataframe case where schema is changed:
        df['new_col'] = arr  ->  df2 = set_df_col(df, 'new_col', arr)
        dataframe_pass will replace set_df_col() with regular setitem if target
        is not dataframe
        """

        df_var = inst.target
        # create var for string index
        cname_var = ir.Var(inst.value.scope, numba.ir_utils.mk_unique_var("$cname_const"), inst.loc)
        nodes = [ir.Assign(ir.Const(inst.index, inst.loc), cname_var, inst.loc)]

        def func(df, cname, arr):
            return sdc.hiframes.api.set_df_col(df, cname, arr)

        f_block = numba.ir_utils.compile_to_numba_ir(func, {'sdc': sdc}).blocks.popitem()[1]
        numba.ir_utils.replace_arg_nodes(f_block, [df_var, cname_var, inst.value])
        nodes += f_block.body[:-2]

        # rename the dataframe variable to keep schema static
        new_df_var = ir.Var(df_var.scope, numba.ir_utils.mk_unique_var(df_var.name), df_var.loc)
        nodes[-1].target = new_df_var
        self.replace_var_dict[df_var.name] = new_df_var

        return nodes

    def _replace_func(self, func, args, const=False, array_typ_convert=True, pre_nodes=None, extra_globals=None):
        glbls = {'numba': numba, 'numpy': numpy, 'sdc': sdc}

        if extra_globals is not None:
            glbls.update(extra_globals)

        return sdc.utils.ReplaceFunc(func, None, args, glbls, pre_nodes)

    def _create_df(self, df_varname, df_col_map, label):
        # order is important for proper handling of itertuples, apply, etc.
        # starting pandas 0.23 and Python 3.6, regular dict order is OK
        # for <0.23 ordered_df_map = OrderedDict(sorted(df_col_map.items()))

        self.df_vars[df_varname] = df_col_map
        self.df_labels[df_varname] = label

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

    def _get_renamed_df(self, df_var):
        # XXX placeholder for df variable renaming
        assert isinstance(df_var, ir.Var)
        return df_var

    def _update_definitions(self, node_list):
        loc = ir.Loc("", 0)
        dumm_block = ir.Block(ir.Scope(None, loc), loc)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.state.func_ir._definitions)
        return

def _gen_arr_copy(in_arr, nodes):
    f_block = numba.ir_utils.compile_to_numba_ir(lambda A: A.copy(), {}).blocks.popitem()[1]
    numba.ir_utils.replace_arg_nodes(f_block, [in_arr])
    nodes += f_block.body[:-2]
    return nodes[-1].target

def sdc_nopython_pipeline_lite_register(state, name='nopython'):
    """
    This is to register some sub set of Intel SDC compiler passes in Numba NoPython pipeline
    Each pass, enabled here, is expected to be called many times on every decorated function including
    functions which are not related to Pandas.

    Test: SDC_CONFIG_PIPELINE_SDC=0 python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_sort_values1

    This function needs to be removed if SDC DataFrame support
    no more needs Numba IR transformations via DataFramePass
    """

    if sdc.config.numba_compiler_define_nopython_pipeline_orig is None:
        raise ValueError("Intel SDC. Unexpected usage of DataFrame passes registration function.")

    numba_pass_manager = sdc.config.numba_compiler_define_nopython_pipeline_orig(state, name)

    numba_pass_manager.add_pass_after(SDC_Pandas_DataFrame_TransformationPass_Stage1, numba.untyped_passes.InlineClosureLikes)

    numba_pass_manager.add_pass_after(sdc.hiframes.dataframe_pass.DataFramePass, numba.typed_passes.AnnotateTypes)
    numba_pass_manager.add_pass_after(sdc.compiler.PostprocessorPass, numba.typed_passes.AnnotateTypes)

    numba_pass_manager.add_pass_after(SDC_Pandas_DataFrame_TransformationPass_Stage2, sdc.hiframes.dataframe_pass.DataFramePass)

    numba_pass_manager.finalize()

    return numba_pass_manager

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


import operator
from collections import defaultdict, namedtuple
import re
import numpy as np
import pandas as pd
import warnings
import datetime

import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (replace_arg_nodes, compile_to_numba_ir,
                            find_topo_order, gen_np_call, get_definition, guard,
                            find_callname, mk_alloc, find_const, is_setitem,
                            is_getitem, mk_unique_var, dprint_func_ir,
                            build_definitions, find_build_sequence)
from numba.extending import overload
from numba.inline_closurecall import inline_closure_call
from numba.typing.arraydecl import ArrayAttribute
from numba.typing.templates import Signature, bound_function, signature, infer_global, AbstractTemplate, signature
from numba.compiler_machinery import FunctionPass, register_pass

import sdc
from sdc.datatypes.hpat_pandas_stringmethods_types import StringMethodsType
from sdc.utils import (debug_prints, inline_new_blocks, ReplaceFunc,
                        is_whole_slice, is_array, update_globals)
from sdc.str_ext import (string_type, unicode_to_std_str, std_str_to_unicode,
                          list_string_array_type)
from sdc.str_arr_ext import (string_array_type, StringArrayType,
                              is_str_arr_typ, pre_alloc_string_array, get_utf8_size)
from sdc import hiframes
from sdc.hiframes import series_kernels, split_impl
from sdc.hiframes.pd_series_ext import (SeriesType, is_str_series_typ,
                                         series_to_array_type, is_dt64_series_typ,
                                         if_series_to_array_type, is_series_type,
                                         SeriesRollingType, SeriesIatType,
                                         explicit_binop_funcs, series_dt_methods_type)
from sdc.hiframes.pd_index_ext import DatetimeIndexType
from sdc.hiframes.rolling import get_rolling_setup_args
from sdc.hiframes.aggregate import Aggregate
from sdc.hiframes.series_kernels import series_replace_funcs
from sdc.hiframes.split_impl import (SplitViewStringMethodsType,
                                     string_array_split_view_type, StringArraySplitViewType,
                                     getitem_c_arr, get_array_ctypes_ptr,
                                     get_split_view_index, get_split_view_data_ptr)


_dt_index_binops = ('==', '!=', '>=', '>', '<=', '<', '-',
                    operator.eq, operator.ne, operator.ge, operator.gt,
                    operator.le, operator.lt, operator.sub)

_string_array_comp_ops = ('==', '!=', '>=', '>', '<=', '<',
                          operator.eq, operator.ne, operator.ge, operator.gt,
                          operator.le, operator.lt)

_binop_to_str = {
    operator.eq: '==',
    operator.ne: '!=',
    operator.ge: '>=',
    operator.gt: '>',
    operator.le: '<=',
    operator.lt: '<',
    operator.sub: '-',
    operator.add: '+',
    operator.mul: '*',
    operator.truediv: '/',
    operator.floordiv: '//',
    operator.mod: '%',
    operator.pow: '**',
    '==': '==',
    '!=': '!=',
    '>=': '>=',
    '>': '>',
    '<=': '<=',
    '<': '<',
    '-': '-',
    '+': '+',
    '*': '*',
    '/': '/',
    '//': '//',
    '%': '%',
    '**': '**',
}


@register_pass(mutates_CFG=True, analysis_only=False)
class HiFramesTypedPass(FunctionPass):
    """Analyze and transform hiframes calls after typing"""

    _name = "sdc_extention_hi_frames_typed_pass"

    def __init__(self):
        pass

    def run_pass(self, state):
        return HiFramesTypedPassImpl(state).run_pass()


class HiFramesTypedPassImpl(object):

    def __init__(self, state):
        # keep track of tuple variables change by to_const_tuple
        self._type_changed_vars = []
        self.state = state

    def run_pass(self):
        blocks = self.state.func_ir.blocks
        # topo_order necessary so Series data replacement optimization can be
        # performed in one pass
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
                elif isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                    out_nodes = self._run_setitem(inst)
                else:
                    if isinstance(inst, (Aggregate, hiframes.sort.Sort,
                                         hiframes.join.Join, hiframes.filter.Filter,
                                         sdc.io.csv_ext.CsvReader)):
                        out_nodes = self._handle_hiframes_nodes(inst)

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
                    inst.value = ir.Expr.call(
                        ir.Var(block.scope, "dummy", inst.loc),
                        rp_func.args, (), inst.loc)
                    block.body = new_body + block.body[i:]
                    update_globals(rp_func.func, rp_func.glbls)
                    inline_closure_call(self.state.func_ir, rp_func.glbls,
                                        block, len(new_body), rp_func.func, self.state.typingctx,
                                        rp_func.arg_types,
                                        self.state.typemap, self.state.calltypes, work_list)
                    replaced = True
                    break
                if isinstance(out_nodes, dict):
                    block.body = new_body + block.body[i:]
                    inline_new_blocks(self.state.func_ir, block, i, out_nodes, work_list)
                    replaced = True
                    break

            if not replaced:
                blocks[label].body = new_body

        # XXX remove slice() of h5 read due to Numba's #3380 bug
        self.state.func_ir.blocks = ir_utils.simplify_CFG(self.state.func_ir.blocks)
        while ir_utils.remove_dead(self.state.func_ir.blocks, self.state.func_ir.arg_names,
                                   self.state.func_ir, self.state.typemap):
            pass

        self.state.func_ir._definitions = build_definitions(self.state.func_ir.blocks)
        dprint_func_ir(self.state.func_ir, "after hiframes_typed")

        return True

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value

        # fix type of lhs if type of rhs has been changed
        if isinstance(rhs, ir.Var) and rhs.name in self._type_changed_vars:
            self.state.typemap.pop(lhs)
            self.state.typemap[lhs] = self.state.typemap[rhs.name]
            self._type_changed_vars.append(lhs)

        if isinstance(rhs, ir.Expr):
            if rhs.op == 'getattr':
                return self._run_getattr(assign, rhs)

            if rhs.op == 'binop':
                return self._run_binop(assign, rhs)

            # XXX handling inplace_binop similar to binop for now
            # TODO handle inplace alignment similar to
            # add_special_arithmetic_methods() in pandas ops.py
            # TODO: inplace of str array?
            if rhs.op == 'inplace_binop':
                return self._run_binop(assign, rhs)

            if rhs.op == 'unary':
                return self._run_unary(assign, rhs)

            # replace getitems on Series.iat
            if sdc.config.config_pipeline_hpat_default:
                if rhs.op in ('getitem', 'static_getitem'):
                    return self._run_getitem(assign, rhs)

            if rhs.op == 'call':
                return self._run_call(assign, lhs, rhs)

        return [assign]

    def _run_getitem(self, assign, rhs):
        nodes = []
        # Series(bool) as index
        if (rhs.op == 'getitem'
                and self.state.typemap[rhs.index.name] == SeriesType(types.bool_)):
            rhs.index = self._get_series_data(rhs.index, nodes)

        if isinstance(self.state.typemap[rhs.value.name], SeriesIatType):
            val_def = guard(get_definition, self.state.func_ir, rhs.value)
            assert (isinstance(val_def, ir.Expr) and val_def.op == 'getattr'
                    and val_def.attr in ('iat', 'iloc', 'loc'))
            series_var = val_def.value
            rhs.value = series_var

        # replace getitems on dt_index/dt64 series with Timestamp function
        if is_dt64_series_typ(self.state.typemap[rhs.value.name]):
            if rhs.op == 'getitem':
                ind_var = rhs.index
            else:
                ind_var = rhs.index_var

            in_arr = rhs.value

            def f(_in_arr, _ind):
                dt = _in_arr[_ind]
                s = np.int64(dt)
                return sdc.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)

            data = self._get_series_data(in_arr, nodes)
            assert isinstance(self.state.typemap[ind_var.name],
                              (types.Integer, types.IntegerLiteral))
            f_block = compile_to_numba_ir(f, {'numba': numba, 'np': np,
                                              'sdc': sdc}, self.state.typingctx,
                                          (self.state.typemap[data.name], types.intp),
                                          self.state.typemap, self.state.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [data, ind_var])
            nodes += f_block.body[:-2]
            nodes[-1].target = assign.target
            return nodes

        if isinstance(self.state.typemap[rhs.value.name], SeriesType):
            rhs.value = self._get_series_data(rhs.value, nodes)
            self._convert_series_calltype(rhs)
            lhs = assign.target
            # convert output to Series from Array
            if isinstance(self.state.typemap[lhs.name], SeriesType):
                new_lhs = ir.Var(
                    lhs.scope, mk_unique_var(lhs.name + '_data'), lhs.loc)
                self.state.typemap[new_lhs.name] = series_to_array_type(
                    self.state.typemap[lhs.name])
                nodes.append(ir.Assign(rhs, new_lhs, lhs.loc))
                return self._replace_func(
                    lambda A: sdc.hiframes.api.init_series(A), [new_lhs],
                    pre_nodes=nodes)

            nodes.append(assign)
            return nodes

        nodes.append(assign)
        return nodes

    def _run_setitem(self, inst):
        target_typ = self.state.typemap[inst.target.name]
        # Series as index
        # TODO: handle all possible cases
        nodes = []
        if (isinstance(inst, ir.SetItem)
                and isinstance(self.state.typemap[inst.index.name], SeriesType)):
            inst.index = self._get_series_data(inst.index, nodes)

        if (isinstance(inst, ir.SetItem)
                and isinstance(self.state.typemap[inst.value.name], SeriesType)):
            inst.value = self._get_series_data(inst.value, nodes)

        if isinstance(target_typ, SeriesIatType):
            val_def = guard(get_definition, self.state.func_ir, inst.target)
            assert (isinstance(val_def, ir.Expr) and val_def.op == 'getattr'
                    and val_def.attr in ('iat', 'iloc', 'loc'))
            series_var = val_def.value
            inst.target = series_var
            target_typ = target_typ.stype

        if isinstance(target_typ, SeriesType):
            # TODO: handle index
            data = self._get_series_data(inst.target, nodes)
            inst.target = data
            nodes.append(inst)
            self._convert_series_calltype(inst)
            return nodes

        nodes.append(inst)
        return nodes

    def _run_getattr(self, assign, rhs):
        rhs_type = self.state.typemap[rhs.value.name]  # get type of rhs value "S"

        # replace arr.dtype for dt64 since PA replaces with
        # np.datetime64[ns] which invalid, TODO: fix PA
        if (rhs.attr == 'dtype' and (is_series_type(rhs_type)
                                     or isinstance(rhs_type, types.Array)) and isinstance(
            rhs_type.dtype,
                (types.NPDatetime, types.NPTimedelta))):
            assign.value = ir.Global("numpy.datetime64", rhs_type.dtype, rhs.loc)
            return [assign]

        # replace arr.dtype since PA replacement inserts in the
        # beginning of block, preventing fusion. TODO: fix PA
        if (rhs.attr == 'dtype' and isinstance(
                if_series_to_array_type(rhs_type), types.Array)):
            typ_str = str(rhs_type.dtype)
            assign.value = ir.Global("np.dtype({})".format(typ_str), np.dtype(typ_str), rhs.loc)
            return [assign]

        # PR135. This needs to be commented out
        if isinstance(rhs_type, SeriesType) and rhs.attr == 'values':
            # simply return the column
            nodes = []
            var = self._get_series_data(rhs.value, nodes)
            assign.value = var
            nodes.append(assign)
            return nodes

        # PR171. This needs to be commented out
        # if isinstance(rhs_type, SeriesType) and rhs.attr == 'index':
        #     nodes = []
        #     assign.value = self._get_series_index(rhs.value, nodes)
        #     nodes.append(assign)
        #     return nodes

        if isinstance(rhs_type, SeriesType) and rhs.attr == 'shape':
            nodes = []
            data = self._get_series_data(rhs.value, nodes)
            return self._replace_func(
                lambda A: (len(A),), [data], pre_nodes=nodes)

        if isinstance(rhs_type, DatetimeIndexType) and rhs.attr == 'values':
            # simply return the data array
            nodes = []
            var = self._get_dt_index_data(rhs.value, nodes)
            assign.value = var
            nodes.append(assign)
            return nodes

        if isinstance(rhs_type, DatetimeIndexType):
            if rhs.attr in sdc.hiframes.pd_timestamp_ext.date_fields:
                return self._run_DatetimeIndex_field(assign, assign.target, rhs)
            if rhs.attr == 'date':
                return self._run_DatetimeIndex_date(assign, assign.target, rhs)

        if rhs_type == series_dt_methods_type:
            dt_def = guard(get_definition, self.state.func_ir, rhs.value)
            if dt_def is None:  # TODO: check for errors
                raise ValueError("invalid series.dt")
            rhs.value = dt_def.value
            if rhs.attr in sdc.hiframes.pd_timestamp_ext.date_fields:
                return self._run_DatetimeIndex_field(assign, assign.target, rhs)
            if rhs.attr == 'date':
                return self._run_DatetimeIndex_date(assign, assign.target, rhs)

        if isinstance(rhs_type, sdc.hiframes.pd_index_ext.TimedeltaIndexType):
            if rhs.attr in sdc.hiframes.pd_timestamp_ext.timedelta_fields:
                return self._run_Timedelta_field(assign, assign.target, rhs)

        if isinstance(rhs_type, SeriesType) and rhs.attr == 'size':
            # simply return the column
            nodes = []
            var = self._get_series_data(rhs.value, nodes)
            rhs.value = var
            nodes.append(assign)
            return nodes

        return [assign]

    def _run_binop(self, assign, rhs):
        res = self._handle_string_array_expr(assign, rhs)
        if res is not None:
            return res

        if self._is_dt_index_binop(rhs):
            return self._handle_dt_index_binop(assign, rhs)

        arg1, arg2 = rhs.lhs, rhs.rhs
        typ1, typ2 = self.state.typemap[arg1.name], self.state.typemap[arg2.name]
        if not (isinstance(typ1, SeriesType) or isinstance(typ2, SeriesType)):
            return [assign]

        nodes = []
        # TODO: support alignment, dt, etc.
        # S3 = S1 + S2 ->
        # S3_data = S1_data + S2_data; S3 = init_series(S3_data)
        if isinstance(typ1, SeriesType):
            arg1 = self._get_series_data(arg1, nodes)
        if isinstance(typ2, SeriesType):
            arg2 = self._get_series_data(arg2, nodes)

        rhs.lhs, rhs.rhs = arg1, arg2
        self._convert_series_calltype(rhs)

        # output stays as Array in A += B where A is Array
        if isinstance(self.state.typemap[assign.target.name], types.Array):
            assert isinstance(self.state.calltypes[rhs].return_type, types.Array)
            nodes.append(assign)
            return nodes

        out_data = ir.Var(
            arg1.scope, mk_unique_var(assign.target.name + '_data'), rhs.loc)
        self.state.typemap[out_data.name] = self.state.calltypes[rhs].return_type
        nodes.append(ir.Assign(rhs, out_data, rhs.loc))
        return self._replace_func(
            lambda data: sdc.hiframes.api.init_series(data, None, None),
            [out_data],
            pre_nodes=nodes
        )

    def _run_unary(self, assign, rhs):
        arg = rhs.value
        typ = self.state.typemap[arg.name]

        if isinstance(typ, SeriesType):
            nodes = []
            arg = self._get_series_data(arg, nodes)
            rhs.value = arg
            self._convert_series_calltype(rhs)
            out_data = ir.Var(
                arg.scope, mk_unique_var(assign.target.name + '_data'), rhs.loc)
            self.state.typemap[out_data.name] = self.state.calltypes[rhs].return_type
            nodes.append(ir.Assign(rhs, out_data, rhs.loc))
            return self._replace_func(
                lambda data: sdc.hiframes.api.init_series(data),
                [out_data],
                pre_nodes=nodes
            )

        return [assign]

    def _run_call(self, assign, lhs, rhs):
        fdef = guard(find_callname, self.state.func_ir, rhs, self.state.typemap)
        if fdef is None:
            from numba.stencil import StencilFunc
            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.state.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == 'make_function':
                return [assign]
            if isinstance(func_def, ir.Global) and isinstance(func_def.value, StencilFunc):
                return [assign]
            warnings.warn(
                "function call couldn't be found for initial analysis")
            return [assign]
        else:
            func_name, func_mod = fdef

        string_methods_types = (SplitViewStringMethodsType, StringMethodsType)
        if isinstance(func_mod, ir.Var) and isinstance(self.state.typemap[func_mod.name], string_methods_types):
            f_def = guard(get_definition, self.state.func_ir, rhs.func)
            str_def = guard(get_definition, self.state.func_ir, f_def.value)
            if str_def is None:  # TODO: check for errors
                raise ValueError("invalid series.str")

            series_var = str_def.value

            # functions which are used from Numba directly by calling from StringMethodsType
            # other functions (for example, 'capitalize' is not presented in Numba) goes to be replaced here
            if func_name not in sdc.hiframes.pd_series_ext.str2str_methods_excluded:
                return self._run_series_str_method(assign, assign.target, series_var, func_name, rhs)

        # replace _get_type_max_value(arr.dtype) since parfors
        # arr.dtype transformation produces invalid code for dt64
        # TODO: min
        if fdef == ('_get_type_max_value', 'sdc.hiframes.hiframes_typed'):
            if self.state.typemap[rhs.args[0].name] == types.DType(types.NPDatetime('ns')):
                return self._replace_func(
                    lambda: sdc.hiframes.pd_timestamp_ext.integer_to_dt64(
                        numba.targets.builtins.get_type_max_value(
                            numba.types.int64)), [])
            return self._replace_func(
                lambda d: numba.targets.builtins.get_type_max_value(
                    d), rhs.args)

        if fdef == ('DatetimeIndex', 'pandas'):
            return self._run_pd_DatetimeIndex(assign, assign.target, rhs)

        if fdef == ('Series', 'pandas'):
            arg_typs = tuple(self.state.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.state.typemap[v.name]
                       for name, v in dict(rhs.kws).items()}

            impl = sdc.hiframes.pd_series_ext.pd_series_overload(
                *arg_typs, **kw_typs)
            return self._replace_func(impl, rhs.args,
                                      pysig=self.state.calltypes[rhs].pysig, kws=dict(rhs.kws))

        if func_mod == 'sdc.hiframes.api':
            return self._run_call_hiframes(assign, assign.target, rhs, func_name)

        if func_mod == 'sdc.hiframes.rolling':
            return self._run_call_rolling(assign, assign.target, rhs, func_name)

        if fdef == ('empty_like', 'numpy'):
            return self._handle_empty_like(assign, lhs, rhs)

        if (isinstance(func_mod, ir.Var)
                and is_series_type(self.state.typemap[func_mod.name])):
            return self._run_call_series(
                assign, assign.target, rhs, func_mod, func_name)

        if (isinstance(func_mod, ir.Var) and isinstance(
                self.state.typemap[func_mod.name], SeriesRollingType)):
            return self._run_call_series_rolling(
                assign, assign.target, rhs, func_mod, func_name)

        if (isinstance(func_mod, ir.Var)
                and isinstance(
                    self.state.typemap[func_mod.name], DatetimeIndexType)):
            return self._run_call_dt_index(
                assign, assign.target, rhs, func_mod, func_name)

        if (fdef == ('concat_dummy', 'sdc.hiframes.pd_dataframe_ext')
                and isinstance(self.state.typemap[lhs], SeriesType)):
            return self._run_call_concat(assign, lhs, rhs)

        # handle sorted() with key lambda input
        if fdef == ('sorted', 'builtins') and 'key' in dict(rhs.kws):
            return self._handle_sorted_by_key(rhs)

        if fdef == ('init_dataframe', 'sdc.hiframes.pd_dataframe_ext'):
            return [assign]

        # XXX sometimes init_dataframe() can't be resolved in dataframe_pass
        # and there are get_dataframe_data() calls that could be optimized
        # example: test_sort_parallel
        if fdef == ('get_dataframe_data', 'sdc.hiframes.pd_dataframe_ext'):
            df_var = rhs.args[0]
            df_typ = self.state.typemap[df_var.name]
            ind = guard(find_const, self.state.func_ir, rhs.args[1])
            var_def = guard(get_definition, self.state.func_ir, df_var)
            call_def = guard(find_callname, self.state.func_ir, var_def)
            if call_def == ('init_dataframe', 'sdc.hiframes.pd_dataframe_ext'):
                assign.value = var_def.args[ind]

        if fdef == ('get_dataframe_index', 'sdc.hiframes.pd_dataframe_ext'):
            df_var = rhs.args[0]
            df_typ = self.state.typemap[df_var.name]
            n_cols = len(df_typ.columns)
            var_def = guard(get_definition, self.state.func_ir, df_var)
            call_def = guard(find_callname, self.state.func_ir, var_def)
            if call_def == ('init_dataframe', 'sdc.hiframes.pd_dataframe_ext'):
                assign.value = var_def.args[n_cols]

        # convert Series to Array for unhandled calls
        # TODO check all the functions that get here and handle if necessary
        nodes = []
        new_args = []
        for arg in rhs.args:
            if isinstance(self.state.typemap[arg.name], SeriesType):
                new_args.append(self._get_series_data(arg, nodes))
            else:
                new_args.append(arg)

        self._convert_series_calltype(rhs)
        rhs.args = new_args

        # Second condition is to avoid chenging SeriesGroupBy class members
        # test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_groupby_count
        if isinstance(self.state.typemap[lhs], SeriesType) and not isinstance(func_mod, ir.Var):
            scope = assign.target.scope
            new_lhs = ir.Var(scope, mk_unique_var(lhs + '_data'), rhs.loc)
            self.state.typemap[new_lhs.name] = self.state.calltypes[rhs].return_type
            nodes.append(ir.Assign(rhs, new_lhs, rhs.loc))

            def _replace_func_param_impl(A):
                return sdc.hiframes.api.init_series(A)
            return self._replace_func(_replace_func_param_impl, [new_lhs], pre_nodes=nodes)

        nodes.append(assign)
        return nodes

    def _run_call_hiframes(self, assign, lhs, rhs, func_name):
        if func_name in ('to_arr_from_series',):
            assign.value = rhs.args[0]
            return [assign]

        # pd.DataFrame() calls init_series for even Series since it's untyped
        # remove the call since it is invalid for analysis here
        # XXX remove when df pass is typed? (test_pass_series2)
        if func_name == 'init_series' and isinstance(
                self.state.typemap[rhs.args[0].name], SeriesType):
            assign.value = rhs.args[0]
            return [assign]

        if func_name == 'get_index_data':
            # fix_df_array() calls get_index_data() for DatetimeIndex
            # but it can be removed sometimes
            var_def = guard(get_definition, self.state.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.state.func_ir, var_def)
            if call_def == ('init_datetime_index', 'sdc.hiframes.api'):
                assign.value = var_def.args[0]
                return [assign]

        if func_name == 'get_series_data':
            # fix_df_array() calls get_series_data() (e.g. for dataframes)
            # but it can be removed sometimes
            var_def = guard(get_definition, self.state.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.state.func_ir, var_def)
            if call_def == ('init_series', 'sdc.hiframes.api'):
                assign.value = var_def.args[0]
                return [assign]

        if func_name in ('str_contains_regex', 'str_contains_noregex'):
            return self._handle_str_contains(assign, lhs, rhs, func_name)

        # arr = fix_df_array(col) -> arr=col if col is array
        if func_name == 'fix_df_array':
            in_typ = self.state.typemap[rhs.args[0].name]
            impl = sdc.hiframes.api.fix_df_array_overload(in_typ)
            return self._replace_func(impl, rhs.args)

        # arr = fix_rolling_array(col) -> arr=col if col is float array
        if func_name == 'fix_rolling_array':
            in_arr = rhs.args[0]
            if isinstance(self.state.typemap[in_arr.name].dtype, types.Float):
                assign.value = rhs.args[0]
                return [assign]
            else:
                def f(column):  # pragma: no cover
                    a = column.astype(np.float64)
                f_block = compile_to_numba_ir(f,
                                              {'sdc': sdc, 'np': np}, self.state.typingctx,
                                              (if_series_to_array_type(self.state.typemap[in_arr.name]),),
                                              self.state.typemap, self.state.calltypes).blocks.popitem()[1]
                replace_arg_nodes(f_block, [in_arr])
                nodes = f_block.body[:-3]
                nodes[-1].target = assign.target
                return nodes

        if func_name == 'series_filter_bool':
            return self._handle_df_col_filter(assign, lhs, rhs)

        if func_name == 'to_const_tuple':
            tup = rhs.args[0]
            tup_items = self._get_const_tup(tup)
            new_tup = ir.Expr.build_tuple(tup_items, tup.loc)
            assign.value = new_tup
            # fix type and definition of lhs
            self.state.typemap.pop(lhs.name)
            self._type_changed_vars.append(lhs.name)
            self.state.typemap[lhs.name] = types.Tuple(tuple(
                self.state.typemap[a.name] for a in tup_items))
            return [assign]

        if func_name == 'series_tup_to_arr_tup':
            in_typ = self.state.typemap[rhs.args[0].name]
            assert isinstance(in_typ, types.BaseTuple), 'tuple expected'
            series_vars = guard(get_definition, self.state.func_ir, rhs.args[0]).items
            nodes = []
            tup_items = [self._get_series_data(v, nodes) for v in series_vars]
            new_tup = ir.Expr.build_tuple(tup_items, lhs.loc)
            assign.value = new_tup
            nodes.append(assign)
            return nodes

        if func_name == 'concat':
            # concat() case where tuple type changes by to_const_type()
            if any([a.name in self._type_changed_vars for a in rhs.args]):
                argtyps = tuple(self.state.typemap[a.name] for a in rhs.args)
                old_sig = self.state.calltypes.pop(rhs)
                new_sig = self.state.typemap[rhs.func.name].get_call_type(
                    self.state.typingctx, argtyps, rhs.kws)
                self.state.calltypes[rhs] = new_sig

            # replace tuple of Series with tuple of Arrays
            in_vars, _ = guard(find_build_sequence, self.state.func_ir, rhs.args[0])
            nodes = []
            s_arrs = [self._get_series_data(v, nodes) if isinstance(
                self.state.typemap[v.name], SeriesType) else v for v in in_vars]
            new_tup = ir.Expr.build_tuple(s_arrs, lhs.loc)
            new_arg = ir.Var(lhs.scope, mk_unique_var(
                rhs.args[0].name + '_arrs'), lhs.loc)
            self.state.typemap[new_arg.name] = if_series_to_array_type(
                self.state.typemap[rhs.args[0].name])
            nodes.append(ir.Assign(new_tup, new_arg, lhs.loc))
            rhs.args[0] = new_arg
            nodes.append(assign)
            self.state.calltypes.pop(rhs)
            new_sig = self.state.typemap[rhs.func.name].get_call_type(
                self.state.typingctx, (self.state.typemap[new_arg.name],), rhs.kws)
            self.state.calltypes[rhs] = new_sig
            return nodes

        # replace isna early to enable more optimization in PA
        # TODO: handle more types
        if func_name == 'isna':
            arr = rhs.args[0]
            ind = rhs.args[1]
            arr_typ = self.state.typemap[arr.name]
            if isinstance(arr_typ, (types.Array, SeriesType)):
                if isinstance(arr_typ.dtype, types.Float):
                    def func(arr, i):
                        return np.isnan(arr[i])
                    return self._replace_func(func, [arr, ind])
                elif isinstance(
                        arr_typ.dtype, (types.NPDatetime, types.NPTimedelta)):
                    nat = arr_typ.dtype('NaT')
                    # TODO: replace with np.isnat
                    return self._replace_func(
                        lambda arr, i: arr[i] == nat, [arr, ind])
                elif arr_typ.dtype != string_type:
                    return self._replace_func(lambda arr, i: False, [arr, ind])

        if func_name == 'df_isin':
            # XXX df isin is different than Series.isin, df.isin considers
            #  index but Series.isin ignores it (everything is set)
            # TODO: support strings and other types
            nodes = []
            data, other = rhs.args

            def _isin_series(A, B):
                numba.parfor.init_prange()
                n = len(A)
                m = len(B)
                S = np.empty(n, np.bool_)
                for i in numba.parfor.internal_prange(n):
                    S[i] = (A[i] == B[i] if i < m else False)
                return S

            return self._replace_func(
                _isin_series, [data, other], pre_nodes=nodes)

        if func_name == 'df_isin_vals':
            nodes = []
            data = rhs.args[0]

            def _isin_series(A, vals):
                numba.parfor.init_prange()
                n = len(A)
                S = np.empty(n, np.bool_)
                for i in numba.parfor.internal_prange(n):
                    S[i] = A[i] in vals
                return S

            return self._replace_func(
                _isin_series, [data, rhs.args[1]], pre_nodes=nodes)

        if func_name == 'flatten_to_series':
            arg = rhs.args[0]
            in_typ = self.state.typemap[arg.name]
            nodes = []
            if isinstance(in_typ, SeriesType):
                arg = self._get_series_data(arg, nodes)

            def _flatten_impl(A):
                numba.parfor.init_prange()
                flat_list = []
                n = len(A)
                for i in numba.parfor.internal_prange(n):
                    elems = A[i]
                    for s in elems:
                        flat_list.append(s)

                return sdc.hiframes.api.init_series(
                    sdc.hiframes.api.parallel_fix_df_array(flat_list))
            return self._replace_func(_flatten_impl, [arg], pre_nodes=nodes)

        if func_name == 'to_numeric':
            out_dtype = self.state.typemap[lhs.name].dtype
            assert out_dtype == types.int64 or out_dtype == types.float64

            # TODO: handle non-Series input

            def _to_numeric_impl(A):
                # TODO: fix distributed
                numba.parfor.init_prange()
                n = len(A)
                B = np.empty(n, out_dtype)
                for i in numba.parfor.internal_prange(n):
                    sdc.str_arr_ext.str_arr_item_to_numeric(B, i, A, i)

                return sdc.hiframes.api.init_series(B)

            nodes = []
            data = self._get_series_data(rhs.args[0], nodes)
            return self._replace_func(_to_numeric_impl, [data],
                                      pre_nodes=nodes,
                                      extra_globals={'out_dtype': out_dtype})

        if func_name == 'parse_datetimes_from_strings':
            nodes = []
            data = self._get_series_data(rhs.args[0], nodes)

            def parse_impl(data):
                numba.parfor.init_prange()
                n = len(data)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                for i in numba.parfor.internal_prange(n):
                    S[i] = sdc.hiframes.pd_timestamp_ext.parse_datetime_str(data[i])
                return S

            return self._replace_func(parse_impl, [data], pre_nodes=nodes)

        if func_name == 'get_itertuples':
            nodes = []
            new_args = []
            for arg in rhs.args:
                if isinstance(self.state.typemap[arg.name], SeriesType):
                    new_args.append(self._get_series_data(arg, nodes))
                else:
                    new_args.append(arg)

            self._convert_series_calltype(rhs)
            rhs.args = new_args

            nodes.append(assign)
            return nodes

        return self._handle_df_col_calls(assign, lhs, rhs, func_name)

    def _run_call_series(self, assign, lhs, rhs, series_var, func_name):
        # single arg functions
        if func_name in ('sum', 'count', 'mean', 'var', 'min', 'max'):
            if rhs.args or rhs.kws:
                raise ValueError("SDC pipeline does not support arguments for Series.{}()".format(func_name))

            # TODO: handle skipna, min_count arguments
            series_typ = self.state.typemap[series_var.name]
            series_dtype = series_typ.dtype
            func = series_replace_funcs[func_name]
            if isinstance(func, dict):
                func = func[series_dtype]
            nodes = []
            data = self._get_series_data(series_var, nodes)
            return self._replace_func(func, [data], pre_nodes=nodes)

        if func_name in ('std', 'nunique', 'describe',
                         'isnull', 'median', 'unique'):
            if rhs.args or rhs.kws:
                raise ValueError("unsupported Series.{}() arguments".format(
                    func_name))
            func = series_replace_funcs[func_name]
            # TODO: handle skipna, min_count arguments
            nodes = []
            if func_name == 'describe':
                data = series_var
            else:
                data = self._get_series_data(series_var, nodes)
            return self._replace_func(func, [data], pre_nodes=nodes)

        if func_name == 'quantile':
            nodes = []
            data = self._get_series_data(series_var, nodes)

            def run_call_series_quantile(A, q):
                return sdc.hiframes.api.quantile(A, q)

            def run_call_series_quantile_default(A):
                return sdc.hiframes.api.quantile(A, 0.5)

            if len(rhs.args) == 0:
                args = [data]
                replacement_func = run_call_series_quantile_default
            else:
                assert len(rhs.args) == 1, "invalid args for " + func_name
                args = [data, rhs.args[0]]
                replacement_func = run_call_series_quantile

            return self._replace_func(replacement_func, args, pre_nodes=nodes)

        if func_name == 'fillna':
            return self._run_call_series_fillna(assign, lhs, rhs, series_var)

        if func_name == 'dropna':
            return self._run_call_series_dropna(assign, lhs, rhs, series_var)

        # if func_name == 'rename':
        #     nodes = []
        #     data = self._get_series_data(series_var, nodes)
        #     index = self._get_series_index(series_var, nodes)
        #     name = rhs.args[0]
        #     return self._replace_func(
        #         lambda data, index, name: sdc.hiframes.api.init_series(
        #             data, index, name),
        #         [data, index, name], pre_nodes=nodes)

        if func_name == 'pct_change':
            nodes = []
            data = self._get_series_data(series_var, nodes)
            # TODO: support default period argument
            if len(rhs.args) == 0:
                args = [data]
                func = series_replace_funcs[func_name + "_default"]
            else:
                assert len(rhs.args) == 1, "invalid args for " + func_name
                shift_const = rhs.args[0]
                args = [data, shift_const]
                func = series_replace_funcs[func_name]
            return self._replace_func(func, args, pre_nodes=nodes)

        if func_name in ('nlargest', 'nsmallest'):
            # TODO: kws
            nodes = []
            data = self._get_series_data(series_var, nodes)
            name = self._get_series_name(series_var, nodes)
            if len(rhs.args) == 0 and not rhs.kws:
                return self._replace_func(
                    series_replace_funcs[func_name + '_default'], [data, name],
                    extra_globals={'gt_f': series_kernels.gt_f,
                                   'lt_f': series_kernels.lt_f},
                    pre_nodes=nodes)
            n_arg = rhs.args[0]
            func = series_replace_funcs[func_name]
            return self._replace_func(func, [data, n_arg, name],
                                      extra_globals={'gt_f': series_kernels.gt_f,
                                                     'lt_f': series_kernels.lt_f},
                                      pre_nodes=nodes)

        if func_name == 'head':
            nodes = []
            n_arg = self._get_arg('Series.head', rhs.args, dict(rhs.kws), 0,
                                  'n', default=False)  # TODO: proper default handling
            if n_arg is False:
                n_arg = ir.Var(lhs.scope, mk_unique_var('head_n'), lhs.loc)
                # default is 5
                self.state.typemap[n_arg.name] = types.IntegerLiteral(5)
                nodes.append(ir.Assign(
                    ir.Const(5, lhs.loc), n_arg, lhs.loc))

            data = self._get_series_data(series_var, nodes)
            func = series_replace_funcs[func_name]

            if self.state.typemap[series_var.name].index != types.none:
                index = self._get_series_index(series_var, nodes)
                func = series_replace_funcs['head_index']
            else:
                index = self._get_index_values(data, nodes)

            name = self._get_series_name(series_var, nodes)

            return self._replace_func(
                func, (data, index, n_arg, name), pre_nodes=nodes)

        if func_name in ('cov', 'corr'):
            S2 = rhs.args[0]
            func = series_replace_funcs[func_name]
            return self._replace_func(func, [series_var, S2])

        if func_name in ('argsort', 'sort_values'):
            return self._handle_series_sort(
                lhs, rhs, series_var, func_name == 'argsort')

        if func_name == 'rolling':
            # XXX: remove rolling setup call, assuming still available in definitions
            self.state.func_ir._definitions[lhs.name].append(rhs)
            return []

        if func_name == 'combine':
            return self._handle_series_combine(assign, lhs, rhs, series_var)

        if func_name in ('map', 'apply'):
            return self._handle_series_map(assign, lhs, rhs, series_var)

        if func_name == 'append':
            nodes = []
            data = self._get_series_data(series_var, nodes)
            other = rhs.args[0]
            if isinstance(self.state.typemap[other.name], SeriesType):
                func = series_replace_funcs['append_single']
                other = self._get_series_data(other, nodes)
            else:
                func = series_replace_funcs['append_tuple']
            return self._replace_func(func, [data, other], pre_nodes=nodes)

        # if func_name == 'notna':
        #     # TODO: make sure this is fused and optimized properly
        #     return self._replace_func(
        #         lambda S: S.isna() == False, [series_var])

        if func_name == 'value_counts':
            nodes = []
            data = self._get_series_data(series_var, nodes)
            # reusing aggregate/count
            # TODO: write optimized implementation
            # data of input becomes both key and data for aggregate input
            # data of output is the counts
            out_key_var = ir.Var(lhs.scope, mk_unique_var(lhs.name + '_index'), lhs.loc)
            self.state.typemap[out_key_var.name] = self.state.typemap[data.name]
            out_data_var = ir.Var(lhs.scope, mk_unique_var(lhs.name + '_data'), lhs.loc)
            self.state.typemap[out_data_var.name] = self.state.typemap[lhs.name].data
            agg_func = series_replace_funcs['count']
            agg_node = hiframes.aggregate.Aggregate(
                lhs.name, 'series', ['series'], [out_key_var], {
                    'data': out_data_var}, {
                    'data': data}, [data], agg_func, None, lhs.loc)
            nodes.append(agg_node)
            # TODO: handle args like sort=False

            def func(A, B):
                return sdc.hiframes.api.init_series(A, B).sort_values(ascending=False)
            return self._replace_func(func, [out_data_var, out_key_var], pre_nodes=nodes)

        # astype with string output
        # if func_name == 'astype' and is_str_series_typ(self.state.typemap[lhs.name]):
        #     # just return input if string
        #     if is_str_series_typ(self.state.typemap[series_var.name]):
        #         return self._replace_func(lambda a: a, [series_var])
        #     func = series_replace_funcs['astype_str']
        #     nodes = []
        #     data = self._get_series_data(series_var, nodes)
        #     return self._replace_func(func, [data], pre_nodes=nodes)

        if func_name in explicit_binop_funcs.keys():
            binop_map = {k: _binop_to_str[v] for k, v in explicit_binop_funcs.items()}
            func_text = "def _binop_impl(A, B):\n"
            func_text += "  return A {} B\n".format(binop_map[func_name])

            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _binop_impl = loc_vars['_binop_impl']
            return self._replace_func(_binop_impl, [series_var] + rhs.args)

        # functions we revert to Numpy for now, otherwise warning
        _conv_to_np_funcs = ('cumsum', 'cumprod')
        # TODO: handle series-specific cases for this funcs
        if (not func_name.startswith("values.") and func_name
                not in _conv_to_np_funcs):
            warnings.warn("unknown Series call {}, reverting to Numpy".format(
                func_name))

        if func_name in _conv_to_np_funcs:
            nodes = []
            data = self._get_series_data(series_var, nodes)

            n_args = len(rhs.args)
            arg_names = ", ".join("arg{}".format(i) for i in range(n_args))
            sep_comma = ", " if n_args > 0 else ""
            func_text = "def _func_impl(A{}{}):\n".format(sep_comma, arg_names)
            func_text += ("  return sdc.hiframes.api.init_series(A.{}({}))\n"
                          ).format(func_name, arg_names)

            loc_vars = {}
            exec(func_text, {'sdc': sdc}, loc_vars)
            _func_impl = loc_vars['_func_impl']
            return self._replace_func(_func_impl, [data] + rhs.args,
                                      pre_nodes=nodes)

        return [assign]

    def _handle_series_sort(self, lhs, rhs, series_var, is_argsort):
        """creates an index list and passes it to a Sort node as data
        """
        # get data array
        nodes = []
        data = self._get_series_data(series_var, nodes)

        # get index array
        if self.state.typemap[series_var.name].index != types.none:
            index_var = self._get_series_index(series_var, nodes)
        else:
            index_var = self._get_index_values(data, nodes)

        # output data arrays for results, before conversion to Series
        out_data = ir.Var(lhs.scope, mk_unique_var(lhs.name + '_data'), lhs.loc)
        self.state.typemap[out_data.name] = self.state.typemap[lhs.name].data
        out_index = ir.Var(lhs.scope, mk_unique_var(lhs.name + '_index'), lhs.loc)
        self.state.typemap[out_index.name] = self.state.typemap[index_var.name]

        # indexes are input/output Sort data
        in_df = {'inds': index_var}
        out_df = {'inds': out_index}
        # data arrays are Sort key
        in_keys = [data]
        out_keys = [out_data]
        args = [out_data, out_index]

        if is_argsort:
            # output of argsort doesn't have new index so assign None
            none_index = ir.Var(lhs.scope, mk_unique_var(lhs.name + '_index'), lhs.loc)
            self.state.typemap[none_index.name] = types.none
            nodes.append(ir.Assign(
                ir.Const(None, lhs.loc), none_index, lhs.loc))
            args = [out_index, none_index]
            ascending = True
        else:
            # TODO refactor to use overload_method
            ascending = self._get_arg(
                'sort_values', rhs.args, dict(rhs.kws), 1, 'ascending',
                default=True)
            if isinstance(ascending, ir.Var):  # TODO: check error
                ascending = find_const(self.state.func_ir, ascending)

        # Sort node
        nodes.append(hiframes.sort.Sort(data.name, lhs.name, in_keys,
                                        out_keys, in_df, out_df, False, lhs.loc, ascending))

        # create output Series
        return self._replace_func(
            lambda A, B: sdc.hiframes.api.init_series(A, B),
            args,
            pre_nodes=nodes)

    def _run_call_series_fillna(self, assign, lhs, rhs, series_var):
        dtype = self.state.typemap[series_var.name].dtype
        val = rhs.args[0]
        nodes = []
        data = self._get_series_data(series_var, nodes)
        name = self._get_series_name(series_var, nodes)
        kws = dict(rhs.kws)
        inplace = False
        if 'inplace' in kws:
            inplace = guard(find_const, self.state.func_ir, kws['inplace'])
            if inplace is None:  # pragma: no cover
                raise ValueError("inplace arg to fillna should be constant")

        if inplace:
            if dtype == string_type:
                # optimization: just set null bit if fill is empty
                if guard(find_const, self.state.func_ir, val) == "":
                    return self._replace_func(
                        lambda A: sdc.str_arr_ext.set_null_bits(A),
                        [data],
                        pre_nodes=nodes)
                # Since string arrays can't be changed, we have to create a new
                # array and assign it back to the same Series variable
                # result back to the same variable
                # TODO: handle string array reflection

                def str_fillna_impl(A, fill, name):
                    # not using A.fillna since definition list is not working
                    # for A to find callname
                    return sdc.hiframes.api.fillna_str_alloc(A, fill, name)
                    # A.fillna(fill)

                assign.target = series_var  # replace output
                return self._replace_func(str_fillna_impl, [data, val, name], pre_nodes=nodes)
            else:
                return self._replace_func(
                    lambda a, b, c: sdc.hiframes.api.fillna(a, b, c),
                    [data, data, val],
                    pre_nodes=nodes)
        else:
            if dtype == string_type:
                func = series_replace_funcs['fillna_str_alloc']
            else:
                func = series_replace_funcs['fillna_alloc']
            return self._replace_func(func, [data, val, name], pre_nodes=nodes)

    def _run_call_series_dropna(self, assign, lhs, rhs, series_var):
        dtype = self.state.typemap[series_var.name].dtype
        kws = dict(rhs.kws)
        inplace = False
        if 'inplace' in kws:
            inplace = guard(find_const, self.state.func_ir, kws['inplace'])
            if inplace is None:  # pragma: no cover
                raise ValueError("inplace arg to dropna should be constant")

        nodes = []
        data = self._get_series_data(series_var, nodes)
        name = self._get_series_name(series_var, nodes)

        if inplace:
            # Since arrays can't resize inplace, we have to create a new
            # array and assign it back to the same Series variable
            # result back to the same variable
            def dropna_impl(A, name):
                # not using A.dropna since definition list is not working
                # for A to find callname
                return sdc.hiframes.api.dropna(A, name)

            assign.target = series_var  # replace output
            return self._replace_func(dropna_impl, [data, name], pre_nodes=nodes)
        else:
            if dtype == string_type:
                func = series_replace_funcs['dropna_str_alloc']
            elif isinstance(dtype, types.Float):
                func = series_replace_funcs['dropna_float']
            else:
                # integer case, TODO: bool, date etc.
                def func(A, name):
                    return sdc.hiframes.api.init_series(A, None, name)
            return self._replace_func(func, [data, name], pre_nodes=nodes)

    def _handle_series_map(self, assign, lhs, rhs, series_var):
        """translate df.A.map(lambda a:...) to prange()
        """
        # error checking: make sure there is function input only
        if len(rhs.args) != 1:
            raise ValueError("map expects 1 argument")
        func = guard(get_definition, self.state.func_ir, rhs.args[0]).value.py_func

        dtype = self.state.typemap[series_var.name].dtype
        nodes = []
        data = self._get_series_data(series_var, nodes)
        out_typ = self.state.typemap[lhs.name].dtype

        # TODO: handle non numpy alloc types like string array
        # prange func to inline
        func_text = "def f(A):\n"
        func_text += "  numba.parfor.init_prange()\n"
        func_text += "  n = len(A)\n"
        func_text += "  S = numba.unsafe.ndarray.empty_inferred((n,))\n"
        func_text += "  for i in numba.parfor.internal_prange(n):\n"
        if dtype == types.NPDatetime('ns'):
            func_text += "    t = sdc.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(np.int64(A[i]))\n"
        elif isinstance(dtype, types.BaseTuple):
            func_text += "    t = sdc.hiframes.api.convert_rec_to_tup(A[i])\n"
        else:
            func_text += "    t = A[i]\n"
        func_text += "    v = map_func(t)\n"
        if isinstance(out_typ, types.BaseTuple):
            func_text += "    S[i] = sdc.hiframes.api.convert_tup_to_rec(v)\n"
        else:
            func_text += "    S[i] = v\n"
        # func_text += "    print(S[i])\n"
        func_text += "  return sdc.hiframes.api.init_series(S)\n"
        #func_text += "  return ret\n"

        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'np': np, 'numba': numba}, loc_vars)
        f = loc_vars['f']

        _globals = self.state.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(f, {'numba': numba, 'np': np, 'sdc': sdc})

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
                        update_globals(func, _globals)
                        inline_closure_call(f_ir, _globals, block, i, func)
                        break

        # remove sentinel global to avoid type inference issues
        ir_utils.remove_dead(f_ir.blocks, f_ir.arg_names, f_ir)
        f_ir._definitions = build_definitions(f_ir.blocks)
        arg_typs = (self.state.typemap[data.name],)
        f_typemap, _f_ret_t, f_calltypes = numba.typed_passes.type_inference_stage(
            self.state.typingctx, f_ir, arg_typs, None)
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        self.state.typemap.update(f_typemap)
        self.state.calltypes.update(f_calltypes)
        replace_arg_nodes(f_ir.blocks[topo_order[0]], [data])
        f_ir.blocks[topo_order[0]].body = nodes + f_ir.blocks[topo_order[0]].body
        return f_ir.blocks

    def _run_call_rolling(self, assign, lhs, rhs, func_name):
        # replace input arguments with data arrays from Series
        nodes = []
        new_args = []
        for arg in rhs.args:
            if isinstance(self.state.typemap[arg.name], SeriesType):
                new_args.append(self._get_series_data(arg, nodes))
            else:
                new_args.append(arg)

        self._convert_series_calltype(rhs)
        rhs.args = new_args

        if func_name == 'rolling_corr':
            def rolling_corr_impl(arr, other, win, center):
                cov = sdc.hiframes.rolling.rolling_cov(
                    arr, other, win, center)
                a_std = sdc.hiframes.rolling.rolling_fixed(
                    arr, win, center, False, 'std')
                b_std = sdc.hiframes.rolling.rolling_fixed(
                    other, win, center, False, 'std')
                return cov / (a_std * b_std)
            return self._replace_func(
                rolling_corr_impl, rhs.args, pre_nodes=nodes)
        if func_name == 'rolling_cov':
            def rolling_cov_impl(arr, other, w, center):  # pragma: no cover
                ddof = 1
                X = arr.astype(np.float64)
                Y = other.astype(np.float64)
                XpY = X + Y
                XtY = X * Y
                count = sdc.hiframes.rolling.rolling_fixed(
                    XpY, w, center, False, 'count')
                mean_XtY = sdc.hiframes.rolling.rolling_fixed(
                    XtY, w, center, False, 'mean')
                mean_X = sdc.hiframes.rolling.rolling_fixed(
                    X, w, center, False, 'mean')
                mean_Y = sdc.hiframes.rolling.rolling_fixed(
                    Y, w, center, False, 'mean')
                bias_adj = count / (count - ddof)
                return (mean_XtY - mean_X * mean_Y) * bias_adj
            return self._replace_func(
                rolling_cov_impl, rhs.args, pre_nodes=nodes)
        # replace apply function with dispatcher obj, now the type is known
        if (func_name == 'rolling_fixed' and isinstance(
                self.state.typemap[rhs.args[4].name], types.MakeFunctionLiteral)):
            # for apply case, create a dispatcher for the kernel and pass it
            # TODO: automatically handle lambdas in Numba
            dtype = self.state.typemap[rhs.args[0].name].dtype
            out_dtype = self.state.typemap[lhs.name].dtype
            func_node = guard(get_definition, self.state.func_ir, rhs.args[4])
            imp_dis = self._handle_rolling_apply_func(
                func_node, dtype, out_dtype)

            def f(arr, w, center):  # pragma: no cover
                df_arr = sdc.hiframes.rolling.rolling_fixed(
                    arr, w, center, False, _func)
            f_block = compile_to_numba_ir(f, {'sdc': sdc, '_func': imp_dis},
                                          self.state.typingctx,
                                          tuple(self.state.typemap[v.name] for v in rhs.args[:-2]),
                                          self.state.typemap, self.state.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, rhs.args[:-2])
            nodes += f_block.body[:-3]  # remove none return
            nodes[-1].target = lhs
            return nodes
        elif (func_name == 'rolling_variable' and isinstance(
                self.state.typemap[rhs.args[5].name], types.MakeFunctionLiteral)):
            # for apply case, create a dispatcher for the kernel and pass it
            # TODO: automatically handle lambdas in Numba
            dtype = self.state.typemap[rhs.args[0].name].dtype
            out_dtype = self.state.typemap[lhs.name].dtype
            func_node = guard(get_definition, self.state.func_ir, rhs.args[5])
            imp_dis = self._handle_rolling_apply_func(
                func_node, dtype, out_dtype)

            def f(arr, on_arr, w, center):  # pragma: no cover
                df_arr = sdc.hiframes.rolling.rolling_variable(
                    arr, on_arr, w, center, False, _func)
            f_block = compile_to_numba_ir(f, {'sdc': sdc, '_func': imp_dis},
                                          self.state.typingctx,
                                          tuple(self.state.typemap[v.name] for v in rhs.args[:-2]),
                                          self.state.typemap, self.state.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, rhs.args[:-2])
            nodes += f_block.body[:-3]  # remove none return
            nodes[-1].target = lhs
            return nodes

        nodes.append(assign)
        return nodes

    def _handle_series_combine(self, assign, lhs, rhs, series_var):
        """translate s1.combine(s2,lambda x1,x2 :...) to prange()
        """
        # error checking: make sure there is function input only
        if len(rhs.args) < 2:
            raise ValueError("not enough arguments in call to combine")
        if len(rhs.args) > 3:
            raise ValueError("too many arguments in call to combine")
        func = guard(get_definition, self.state.func_ir, rhs.args[1]).value.py_func
        out_typ = self.state.typemap[lhs.name].dtype
        other = rhs.args[0]
        nodes = []
        data = self._get_series_data(series_var, nodes)
        other_data = self._get_series_data(other, nodes)

        # If we are called with 3 arguments, we must use 3rd arg as a fill value,
        # instead of Nan.
        use_nan = len(rhs.args) == 2

        # prange func to inline
        if use_nan:
            func_text = "def f(A, B):\n"
        else:
            func_text = "def f(A, B, C):\n"
        func_text += "  n1 = len(A)\n"
        func_text += "  n2 = len(B)\n"
        func_text += "  n = max(n1, n2)\n"
        if not isinstance(self.state.typemap[series_var.name].dtype, types.Float) and use_nan:
            func_text += "  assert n1 == n, 'can not use NAN for non-float series, with different length'\n"
        if not isinstance(self.state.typemap[other.name].dtype, types.Float) and use_nan:
            func_text += "  assert n2 == n, 'can not use NAN for non-float series, with different length'\n"
        func_text += "  numba.parfor.init_prange()\n"
        func_text += "  S = numba.unsafe.ndarray.empty_inferred((n,))\n"
        func_text += "  for i in numba.parfor.internal_prange(n):\n"
        if use_nan and isinstance(self.state.typemap[series_var.name].dtype, types.Float):
            func_text += "    t1 = np.nan\n"
            func_text += "    if i < n1:\n"
            func_text += "      t1 = A[i]\n"
        # length is equal, due to assertion above
        elif use_nan:
            func_text += "    t1 = A[i]\n"
        else:
            func_text += "    t1 = C\n"
            func_text += "    if i < n1:\n"
            func_text += "      t1 = A[i]\n"
        # same, but for 2nd argument
        if use_nan and isinstance(self.state.typemap[other.name].dtype, types.Float):
            func_text += "    t2 = np.nan\n"
            func_text += "    if i < n2:\n"
            func_text += "      t2 = B[i]\n"
        elif use_nan:
            func_text += "    t2 = B[i]\n"
        else:
            func_text += "    t2 = C\n"
            func_text += "    if i < n2:\n"
            func_text += "      t2 = B[i]\n"
        func_text += "    S[i] = map_func(t1, t2)\n"
        func_text += "  return sdc.hiframes.api.init_series(S)\n"

        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'np': np, 'numba': numba}, loc_vars)
        f = loc_vars['f']

        _globals = self.state.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(f, {'numba': numba, 'np': np, 'sdc': sdc})

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
                        update_globals(func, _globals)
                        inline_closure_call(f_ir, _globals, block, i, func)
                        break

        # remove sentinel global to avoid type inference issues
        ir_utils.remove_dead(f_ir.blocks, f_ir.arg_names, f_ir)
        f_ir._definitions = build_definitions(f_ir.blocks)
        arg_typs = (self.state.typemap[data.name], self.state.typemap[other_data.name],)
        if not use_nan:
            arg_typs += (self.state.typemap[rhs.args[2].name],)
        f_typemap, _f_ret_t, f_calltypes = numba.typed_passes.type_inference_stage(
            self.state.typingctx, f_ir, arg_typs, None)
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        self.state.typemap.update(f_typemap)
        self.state.calltypes.update(f_calltypes)
        func_args = [data, other_data]
        if not use_nan:
            func_args.append(rhs.args[2])
        first_block = f_ir.blocks[topo_order[0]]
        replace_arg_nodes(first_block, func_args)
        first_block.body = nodes + first_block.body
        return f_ir.blocks

    def _run_call_series_rolling(self, assign, lhs, rhs, rolling_var, func_name):
        """
        Handle Series rolling calls like:
          A = df.column.rolling(3).sum()
        """
        rolling_call = guard(get_definition, self.state.func_ir, rolling_var)
        assert isinstance(rolling_call, ir.Expr) and rolling_call.op == 'call'
        call_def = guard(get_definition, self.state.func_ir, rolling_call.func)
        assert isinstance(call_def, ir.Expr) and call_def.op == 'getattr'
        series_var = call_def.value
        nodes = []
        data = self._get_series_data(series_var, nodes)

        window, center, on = get_rolling_setup_args(self.state.func_ir, rolling_call, False)
        if not isinstance(center, ir.Var):
            center_var = ir.Var(lhs.scope, mk_unique_var("center"), lhs.loc)
            self.state.typemap[center_var.name] = types.bool_
            nodes.append(ir.Assign(ir.Const(center, lhs.loc), center_var, lhs.loc))
            center = center_var

        if func_name in ('cov', 'corr'):
            # TODO: variable window
            if len(rhs.args) == 1:
                other = self._get_series_data(rhs.args[0], nodes)
            else:
                other = data
            if func_name == 'cov':
                def f(a, b, w, c):
                    return sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_cov(a, b, w, c))
            if func_name == 'corr':
                def f(a, b, w, c):
                    return sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_corr(a, b, w, c))
            return self._replace_func(f, [data, other, window, center],
                                      pre_nodes=nodes)
        elif func_name == 'apply':
            func_node = guard(get_definition, self.state.func_ir, rhs.args[0])
            dtype = self.state.typemap[data.name].dtype
            out_dtype = self.state.typemap[lhs.name].dtype
            func_global = self._handle_rolling_apply_func(func_node, dtype, out_dtype)
        else:
            func_global = func_name

        def f(arr, w, center):  # pragma: no cover
            return sdc.hiframes.api.init_series(sdc.hiframes.rolling.rolling_fixed(arr, w, center, False, _func))
        args = [data, window, center]
        return self._replace_func(f, args, pre_nodes=nodes, extra_globals={'_func': func_global})

    def _handle_rolling_apply_func(self, func_node, dtype, out_dtype):
        if func_node is None:
            raise ValueError("cannot find kernel function for rolling.apply() call")
        func_node = func_node.value.py_func
        # TODO: more error checking on the kernel to make sure it doesn't
        # use global/closure variables
        # create a function from the code object
        glbs = self.state.func_ir.func_id.func.__globals__
        lcs = {}
        exec("def f(A): return A", glbs, lcs)
        kernel_func = lcs['f']
        kernel_func.__code__ = func_node.__code__
        kernel_func.__name__ = func_node.__code__.co_name
        # use hpat's sequential pipeline to enable pandas operations
        # XXX seq pipeline used since dist pass causes a hang
        m = numba.ir_utils._max_label
        impl_disp = numba.njit(
            kernel_func, pipeline_class=sdc.compiler.SDCPipelineSeq)
        # precompile to avoid REP counting conflict in testing
        sig = out_dtype(types.Array(dtype, 1, 'C'))
        impl_disp.compile(sig)
        numba.ir_utils._max_label += m
        return impl_disp

    def _run_DatetimeIndex_field(self, assign, lhs, rhs):
        """transform DatetimeIndex.<field> and Series.dt.<field>
        """
        nodes = []
        in_typ = self.state.typemap[rhs.value.name]
        if isinstance(in_typ, DatetimeIndexType):
            arr = self._get_dt_index_data(rhs.value, nodes)
            is_dt_index = True
        else:
            arr = self._get_series_data(rhs.value, nodes)
            is_dt_index = False
        field = rhs.attr

        func_text = 'def f(dti):\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(dti)\n'
        #func_text += '    S = numba.unsafe.ndarray.empty_inferred((n,))\n'
        # TODO: why doesn't empty_inferred work for t4 mortgage test?
        func_text += '    S = np.empty(n, np.int64)\n'
        func_text += '    for i in numba.parfor.internal_prange(n):\n'
        func_text += '        dt64 = sdc.hiframes.pd_timestamp_ext.dt64_to_integer(dti[i])\n'
        func_text += '        ts = sdc.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n'
        func_text += '        S[i] = ts.' + field + '\n'
        if is_dt_index:  # TODO: support Int64Index
            func_text += '    return sdc.hiframes.api.init_series(S)\n'
        else:
            func_text += '    return sdc.hiframes.api.init_series(S)\n'
        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'np': np, 'numba': numba}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, [arr], pre_nodes=nodes)

    def _run_DatetimeIndex_date(self, assign, lhs, rhs):
        """transform DatetimeIndex.date and Series.dt.date
        """
        nodes = []
        in_typ = self.state.typemap[rhs.value.name]
        if isinstance(in_typ, DatetimeIndexType):
            arr = self._get_dt_index_data(rhs.value, nodes)
            is_dt_index = True
        else:
            arr = self._get_series_data(rhs.value, nodes)
            is_dt_index = False

        func_text = 'def f(dti):\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(dti)\n'
        func_text += '    S = numba.unsafe.ndarray.empty_inferred((n,))\n'
        func_text += '    for i in numba.parfor.internal_prange(n):\n'
        func_text += '        dt64 = sdc.hiframes.pd_timestamp_ext.dt64_to_integer(dti[i])\n'
        func_text += '        ts = sdc.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n'
        func_text += '        S[i] = sdc.hiframes.pd_timestamp_ext.datetime_date_ctor(ts.year, ts.month, ts.day)\n'
        #func_text += '        S[i] = datetime.date(ts.year, ts.month, ts.day)\n'
        #func_text += '        S[i] = ts.day + (ts.month << 16) + (ts.year << 32)\n'
        if is_dt_index:  # DatetimeIndex returns Array but Series.dt returns Series
            func_text += '    return sdc.hiframes.datetime_date_ext.np_arr_to_array_datetime_date(S)\n'
        else:
            func_text += '    return sdc.hiframes.api.init_series(S)\n'
        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'numba': numba}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, [arr], pre_nodes=nodes)

    def _run_call_dt_index(self, assign, lhs, rhs, dt_index_var, func_name):
        if func_name in ('min', 'max'):
            if rhs.args or rhs.kws:
                raise ValueError(
                    "unsupported DatetimeIndex.{}() arguments".format(
                        func_name))
            func = series_replace_funcs[func_name][types.NPDatetime('ns')]
            nodes = []
            data = self._get_dt_index_data(dt_index_var, nodes)
            return self._replace_func(func, [data], pre_nodes=nodes)

    def _run_Timedelta_field(self, assign, lhs, rhs):
        """transform Timedelta.<field>
        """
        nodes = []
        arr = self._get_timedelta_index_data(rhs.value, nodes)
        field = rhs.attr

        func_text = 'def f(dti):\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(dti)\n'
        func_text += '    S = numba.unsafe.ndarray.empty_inferred((n,))\n'
        func_text += '    for i in numba.parfor.internal_prange(n):\n'
        func_text += '        dt64 = sdc.hiframes.pd_timestamp_ext.timedelta64_to_integer(dti[i])\n'
        if field == 'nanoseconds':
            func_text += '        S[i] = dt64 % 1000\n'
        elif field == 'microseconds':
            func_text += '        S[i] = dt64 // 1000 % 100000\n'
        elif field == 'seconds':
            func_text += '        S[i] = dt64 // (1000 * 1000000) % (60 * 60 * 24)\n'
        elif field == 'days':
            func_text += '        S[i] = dt64 // (1000 * 1000000 * 60 * 60 * 24)\n'
        else:
            assert(0)
        func_text += '    return S\n'
        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'numba': numba}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, [arr], pre_nodes=nodes)

    def _run_pd_DatetimeIndex(self, assign, lhs, rhs):
        """transform pd.DatetimeIndex() call with string array argument
        """
        arg_typs = tuple(self.state.typemap[v.name] for v in rhs.args)
        kw_typs = {name: self.state.typemap[v.name]
                   for name, v in dict(rhs.kws).items()}
        impl = sdc.hiframes.pd_index_ext.pd_datetimeindex_overload(
            *arg_typs, **kw_typs)
        return self._replace_func(impl, rhs.args,
                                  pysig=self.state.calltypes[rhs].pysig, kws=dict(rhs.kws))

    def _run_series_str_method(self, assign, lhs, series_var, func_name, rhs):

        supported_methods = (sdc.hiframes.pd_series_ext.str2str_methods
                             + ['len', 'replace', 'split', 'get', 'contains'])
        if func_name not in supported_methods:
            raise NotImplementedError("Series.str.{} is not supported yet".format(func_name))

        nodes = []
        arr = self._get_series_data(series_var, nodes)

        # string 2 string methods
        if func_name in sdc.hiframes.pd_series_ext.str2str_methods:
            func_text = 'def f(str_arr):\n'
            func_text += '    numba.parfor.init_prange()\n'
            func_text += '    n = len(str_arr)\n'
            # functions that don't change the number of characters
            if func_name in ('capitalize', 'lower', 'swapcase', 'title', 'upper'):
                func_text += '    num_chars = num_total_chars(str_arr)\n'
            else:
                func_text += '    num_chars = 0\n'
                func_text += '    for i in numba.parfor.internal_prange(n):\n'
                func_text += '        num_chars += get_utf8_size(str_arr[i].{}())\n'.format(func_name)
            func_text += '    S = sdc.str_arr_ext.pre_alloc_string_array(n, num_chars)\n'
            func_text += '    for i in numba.parfor.internal_prange(n):\n'
            func_text += '        S[i] = str_arr[i].{}()\n'.format(func_name)
            func_text += '    return sdc.hiframes.api.init_series(S)\n'
            loc_vars = {}
            # print(func_text)
            exec(func_text, {'sdc': sdc, 'numba': numba}, loc_vars)
            f = loc_vars['f']
            return self._replace_func(f, [arr], pre_nodes=nodes,
                                      extra_globals={
                'num_total_chars': sdc.str_arr_ext.num_total_chars,
                'get_utf8_size': sdc.str_arr_ext.get_utf8_size,
            })

        if func_name == 'contains':
            return self._run_series_str_contains(rhs, arr, nodes)

        if func_name == 'replace':
            return self._run_series_str_replace(assign, lhs, arr, rhs, nodes)

        if func_name == 'split':
            return self._run_series_str_split(assign, lhs, arr, rhs, nodes)

        if func_name == 'get':
            return self._run_series_str_get(assign, lhs, arr, rhs, nodes)

        if func_name == 'len':
            out_typ = 'np.int64'

        func_text = 'def f(str_arr):\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(str_arr)\n'
        #func_text += '    S = numba.unsafe.ndarray.empty_inferred((n,))\n'
        # TODO: use empty_inferred after it is fixed
        func_text += '    S = np.empty(n, {})\n'.format(out_typ)
        func_text += '    for i in numba.parfor.internal_prange(n):\n'
        func_text += '        val = str_arr[i]\n'
        func_text += '        S[i] = len(val)\n'
        func_text += '    return sdc.hiframes.api.init_series(S)\n'
        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'np': np, 'numba': numba}, loc_vars)
        f = loc_vars['f']

        return self._replace_func(f, [arr], pre_nodes=nodes)

    def _run_series_str_replace(self, assign, lhs, arr, rhs, nodes):
        regex = True
        # TODO: refactor arg parsing
        kws = dict(rhs.kws)
        if 'regex' in kws:
            regex = guard(find_const, self.state.func_ir, kws['regex'])
            if regex is None:
                raise ValueError(
                    "str.replace regex argument should be constant")

        impl = (series_kernels._str_replace_regex_impl if regex
                else series_kernels._str_replace_noregex_impl)

        return self._replace_func(
            impl,
            [arr, rhs.args[0], rhs.args[1]], pre_nodes=nodes,
            extra_globals={'unicode_to_std_str': unicode_to_std_str,
                           'std_str_to_unicode': std_str_to_unicode,
                           'pre_alloc_string_array': pre_alloc_string_array,
                           'get_utf8_size': get_utf8_size,
                           're': re}
        )

    def _run_series_str_split(self, assign, lhs, arr, rhs, nodes):
        sep = self._get_arg('str.split', rhs.args, dict(rhs.kws), 0, 'pat',
                            default=False)  # TODO: proper default handling
        if sep is False:
            sep = ir.Var(lhs.scope, mk_unique_var('split_sep'), lhs.loc)
            sep_typ = types.none
            self.state.typemap[sep.name] = types.none
            nodes.append(ir.Assign(
                ir.Const(None, lhs.loc), sep, lhs.loc))
        else:
            sep_typ = self.state.typemap[sep.name]

        def _str_split_impl(str_arr, sep):
            numba.parfor.init_prange()
            n = len(str_arr)
            out_arr = sdc.str_ext.alloc_list_list_str(n)
            for i in numba.parfor.internal_prange(n):
                in_str = str_arr[i]
                out_arr[i] = in_str.split(sep)

            return sdc.hiframes.api.init_series(out_arr)

        if isinstance(sep_typ, types.StringLiteral) and len(sep_typ.literal_value) == 1:
            def _str_split_impl(str_arr, sep):
                out_arr = sdc.hiframes.split_impl.compute_split_view(
                    str_arr, sep)
                return sdc.hiframes.api.init_series(out_arr)

        return self._replace_func(_str_split_impl, [arr, sep], pre_nodes=nodes)

    def _run_series_str_get(self, assign, lhs, arr, rhs, nodes):
        arr_typ = self.state.typemap[arr.name]
        # XXX only supports get for list(list(str)) input and split view
        assert (arr_typ == types.List(types.List(string_type))
                or arr_typ == string_array_split_view_type)
        ind_var = rhs.args[0]

        def _str_get_impl(str_arr, ind):
            numba.parfor.init_prange()
            n = len(str_arr)
            n_total_chars = 0
            str_list = sdc.str_ext.alloc_str_list(n)
            for i in numba.parfor.internal_prange(n):
                # TODO: support NAN
                in_list_str = str_arr[i]
                out_str = in_list_str[ind]
                str_list[i] = out_str
                n_total_chars += get_utf8_size(out_str)
            numba.parfor.init_prange()
            out_arr = pre_alloc_string_array(n, n_total_chars)
            for i in numba.parfor.internal_prange(n):
                _str = str_list[i]
                out_arr[i] = _str
            return sdc.hiframes.api.init_series(out_arr)

        if arr_typ == string_array_split_view_type:
            # TODO: refactor and enable distributed
            def _str_get_impl(arr, ind):
                numba.parfor.init_prange()
                n = len(arr)
                n_total_chars = 0
                for i in numba.parfor.internal_prange(n):
                    data_start, length = get_split_view_index(arr, i, ind)
                    n_total_chars += length
                numba.parfor.init_prange()
                out_arr = pre_alloc_string_array(n, n_total_chars)
                for i in numba.parfor.internal_prange(n):
                    data_start, length = get_split_view_index(arr, i, ind)
                    ptr = get_split_view_data_ptr(arr, data_start)
                    sdc.str_arr_ext.setitem_str_arr_ptr(out_arr, i, ptr, length)
                return sdc.hiframes.api.init_series(out_arr)

        return self._replace_func(_str_get_impl, [arr, ind_var],
                                  pre_nodes=nodes,
                                  extra_globals={'pre_alloc_string_array': pre_alloc_string_array,
                                                 'get_array_ctypes_ptr': get_array_ctypes_ptr,
                                                 'getitem_c_arr': getitem_c_arr,
                                                 'get_split_view_index': get_split_view_index,
                                                 'get_split_view_data_ptr': get_split_view_data_ptr,
                                                 'get_utf8_size': get_utf8_size})

    def _is_dt_index_binop(self, rhs):
        if rhs.op != 'binop':
            return False

        if rhs.fn not in _dt_index_binops:
            return False

        arg1, arg2 = self.state.typemap[rhs.lhs.name], self.state.typemap[rhs.rhs.name]
        # one of them is dt_index but not both
        if ((is_dt64_series_typ(arg1) or is_dt64_series_typ(arg2))
                and not (is_dt64_series_typ(arg1) and is_dt64_series_typ(arg2))):
            return True

        if ((isinstance(arg1, DatetimeIndexType) or isinstance(arg2, DatetimeIndexType))
                and not (isinstance(arg1, DatetimeIndexType) and isinstance(arg2, DatetimeIndexType))):
            return True

        return False

    def _handle_dt_index_binop(self, assign, rhs):
        arg1, arg2 = rhs.lhs, rhs.rhs

        def _is_allowed_type(t):
            return is_dt64_series_typ(t) or t == string_type

        # TODO: this has to be more generic to support all combinations.
        if (is_dt64_series_typ(self.state.typemap[arg1.name])
                and self.state.typemap[arg2.name] == sdc.hiframes.pd_timestamp_ext.pandas_timestamp_type
                and rhs.fn in ('-', operator.sub)):
            return self._replace_func(
                series_kernels._column_sub_impl_datetime_series_timestamp,
                [arg1, arg2])

        if (isinstance(self.state.typemap[arg1.name], DatetimeIndexType)
                and self.state.typemap[arg2.name] == sdc.hiframes.pd_timestamp_ext.pandas_timestamp_type
                and rhs.fn in ('-', operator.sub)):
            nodes = []
            arg1 = self._get_dt_index_data(arg1, nodes)
            return self._replace_func(
                series_kernels._column_sub_impl_datetimeindex_timestamp, [
                    arg1, arg2], pre_nodes=nodes)

        if (not _is_allowed_type(types.unliteral(self.state.typemap[arg1.name]))
                or not _is_allowed_type(types.unliteral(self.state.typemap[arg2.name]))):
            raise ValueError("DatetimeIndex operation not supported")

        # string comparison with DatetimeIndex
        op_str = _binop_to_str[rhs.fn]
        typ1 = self.state.typemap[arg1.name]
        typ2 = self.state.typemap[arg2.name]
        nodes = []
        is_out_series = False

        func_text = 'def f(arg1, arg2):\n'
        if is_dt64_series_typ(typ1) or isinstance(typ1, DatetimeIndexType):
            if is_dt64_series_typ(typ1):
                is_out_series = True
                arg1 = self._get_series_data(arg1, nodes)
            else:
                arg1 = self._get_dt_index_data(arg1, nodes)
            func_text += '  dt_index, _str = arg1, arg2\n'
            comp = 'dt_index[i] {} other'.format(op_str)
        else:
            if is_dt64_series_typ(typ2):
                is_out_series = True
                arg2 = self._get_series_data(arg2, nodes)
            else:
                arg2 = self._get_dt_index_data(arg2, nodes)
            func_text += '  dt_index, _str = arg2, arg1\n'
            comp = 'other {} dt_index[i]'.format(op_str)
        func_text += '  l = len(dt_index)\n'
        func_text += '  other = sdc.hiframes.pd_timestamp_ext.parse_datetime_str(_str)\n'
        func_text += '  S = numba.unsafe.ndarray.empty_inferred((l,))\n'
        func_text += '  for i in numba.parfor.internal_prange(l):\n'
        func_text += '    S[i] = {}\n'.format(comp)
        if is_out_series:  # TODO: test
            func_text += '  return sdc.hiframes.api.init_series(S)\n'
        else:
            func_text += '  return S\n'
        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'numba': numba}, loc_vars)
        f = loc_vars['f']
        # print(func_text)
        return self._replace_func(f, [arg1, arg2])

    def _handle_string_array_expr(self, assign, rhs):
        # convert str_arr==str into parfor
        if (rhs.fn in _string_array_comp_ops
                and is_str_arr_typ(self.state.typemap[rhs.lhs.name])
                or is_str_arr_typ(self.state.typemap[rhs.rhs.name])):
            nodes = []
            arg1 = rhs.lhs
            arg2 = rhs.rhs
            is_series = False
            if is_str_series_typ(self.state.typemap[arg1.name]):
                arg1 = self._get_series_data(arg1, nodes)
                is_series = True
            if is_str_series_typ(self.state.typemap[arg2.name]):
                arg2 = self._get_series_data(arg2, nodes)
                is_series = True

            arg1_access = 'A'
            arg2_access = 'B'
            len_call = 'len(A)'
            if is_str_arr_typ(self.state.typemap[arg1.name]):
                arg1_access = 'A[i]'
                # replace type now for correct typing of len, etc.
                self.state.typemap.pop(arg1.name)
                self.state.typemap[arg1.name] = string_array_type

            if is_str_arr_typ(self.state.typemap[arg2.name]):
                arg1_access = 'B[i]'
                len_call = 'len(B)'
                self.state.typemap.pop(arg2.name)
                self.state.typemap[arg2.name] = string_array_type

            op_str = _binop_to_str[rhs.fn]

            func_text = 'def f(A, B):\n'
            func_text += '  l = {}\n'.format(len_call)
            func_text += '  S = np.empty(l, dtype=np.bool_)\n'
            func_text += '  for i in numba.parfor.internal_prange(l):\n'
            func_text += '    S[i] = {} {} {}\n'.format(arg1_access, op_str,
                                                        arg2_access)
            if is_series:
                func_text += '  return sdc.hiframes.api.init_series(S)\n'
            else:
                func_text += '  return S\n'

            loc_vars = {}
            exec(func_text, {'sdc': sdc, 'np': np, 'numba': numba}, loc_vars)
            f = loc_vars['f']
            return self._replace_func(f, [arg1, arg2], pre_nodes=nodes)

        return None

    def _run_series_str_contains(self, rhs, series_var, nodes):
        """
        Handle string contains like:
          B = df.column.str.contains('oo*', regex=True)
        """
        kws = dict(rhs.kws)
        pat = rhs.args[0]
        regex = True  # default regex arg is True
        if 'regex' in kws:
            regex = guard(find_const, self.state.func_ir, kws['regex'])
            if regex is None:
                raise ValueError("str.contains expects constant regex argument")
        if regex:
            fname = "str_contains_regex"
        else:
            fname = "str_contains_noregex"

        return self._replace_func(
            series_replace_funcs[fname], [series_var, pat], pre_nodes=nodes)

    def _handle_empty_like(self, assign, lhs, rhs):
        # B = empty_like(A) -> B = empty(len(A), dtype)
        in_arr = rhs.args[0]

        if self.state.typemap[in_arr.name].ndim == 1:
            # generate simpler len() for 1D case
            def f(_in_arr):  # pragma: no cover
                _alloc_size = len(_in_arr)
                _out_arr = np.empty(_alloc_size, _in_arr.dtype)
        else:
            def f(_in_arr):  # pragma: no cover
                _alloc_size = _in_arr.shape
                _out_arr = np.empty(_alloc_size, _in_arr.dtype)

        f_block = compile_to_numba_ir(f, {'np': np}, self.state.typingctx, (if_series_to_array_type(
            self.state.typemap[in_arr.name]),), self.state.typemap, self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [in_arr])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = assign.target
        return nodes

    def _handle_str_contains(self, assign, lhs, rhs, fname):

        if fname == 'str_contains_regex':
            comp_func = 'sdc.str_ext.contains_regex'
        elif fname == 'str_contains_noregex':
            comp_func = 'sdc.str_ext.contains_noregex'
        else:
            assert False

        func_text = 'def f(str_arr, pat):\n'
        func_text += '  l = len(str_arr)\n'
        func_text += '  S = np.empty(l, dtype=np.bool_)\n'
        func_text += '  for i in numba.parfor.internal_prange(l):\n'
        func_text += '    S[i] = {}(str_arr[i], pat)\n'.format(comp_func)
        func_text += '  return sdc.hiframes.api.init_series(S)\n'
        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'np': np, 'numba': numba}, loc_vars)
        f = loc_vars['f']
        return self._replace_func(f, rhs.args)

    def _handle_df_col_filter(self, assign, lhs, rhs):
        nodes = []
        in_arr = rhs.args[0]
        bool_arr = rhs.args[1]
        if is_series_type(self.state.typemap[in_arr.name]):
            in_arr = self._get_series_data(in_arr, nodes)
        if is_series_type(self.state.typemap[bool_arr.name]):
            bool_arr = self._get_series_data(bool_arr, nodes)

        return self._replace_func(series_kernels._column_filter_impl,
                                  [in_arr, bool_arr],
                                  pre_nodes=nodes)

    def _handle_df_col_calls(self, assign, lhs, rhs, func_name):

        if func_name == 'count':
            return self._replace_func(
                series_kernels._column_count_impl, rhs.args)

        if func_name == 'fillna':
            return self._replace_func(
                series_kernels._column_fillna_impl, rhs.args)

        if func_name == 'fillna_str_alloc':
            return self._replace_func(
                series_kernels._series_fillna_str_alloc_impl, rhs.args)

        if func_name == 'dropna':
            # df.dropna case
            if isinstance(self.state.typemap[rhs.args[0].name], types.BaseTuple):
                return self._handle_df_dropna(assign, lhs, rhs)
            dtype = self.state.typemap[rhs.args[0].name].dtype
            if dtype == string_type:
                func = series_replace_funcs['dropna_str_alloc']
            elif isinstance(dtype, types.Float):
                func = series_replace_funcs['dropna_float']
            else:
                # integer case, TODO: bool, date etc.
                def func(A):
                    return sdc.hiframes.api.init_series(A)
            return self._replace_func(func, rhs.args)

        if func_name == 'column_sum':
            return self._replace_func(series_kernels._column_sum_impl_basic, rhs.args)

        if func_name == 'mean':
            return self._replace_func(series_kernels._column_mean_impl, rhs.args)

        if func_name == 'var':
            return self._replace_func(series_kernels._column_var_impl, rhs.args)

        return [assign]

    def _handle_df_dropna(self, assign, lhs, rhs):
        in_typ = self.state.typemap[rhs.args[0].name]

        in_vars, _ = guard(find_build_sequence, self.state.func_ir, rhs.args[0])
        in_names = [mk_unique_var(in_vars[i].name).replace('.', '_') for i in range(len(in_vars))]
        out_names = [mk_unique_var(in_vars[i].name).replace('.', '_') for i in range(len(in_vars))]
        str_colnames = [in_names[i] for i, t in enumerate(in_typ.types) if is_str_arr_typ(t)]
        list_str_colnames = [in_names[i] for i, t in enumerate(in_typ.types) if t == list_string_array_type]
        split_view_colnames = [in_names[i] for i, t in enumerate(in_typ.types) if t == string_array_split_view_type]
        isna_calls = ['sdc.hiframes.api.isna({}, i)'.format(v) for v in in_names]

        func_text = "def _dropna_impl(arr_tup, inplace):\n"
        func_text += "  ({},) = arr_tup\n".format(", ".join(in_names))
        func_text += "  old_len = len({})\n".format(in_names[0])
        func_text += "  new_len = 0\n"
        for c in str_colnames:
            func_text += "  num_chars_{} = 0\n".format(c)
        func_text += "  for i in numba.parfor.internal_prange(old_len):\n"
        func_text += "    if not ({}):\n".format(' or '.join(isna_calls))
        func_text += "      new_len += 1\n"
        for c in str_colnames:
            func_text += "      num_chars_{} += len({}[i])\n".format(c, c)
        for v, out in zip(in_names, out_names):
            if v in str_colnames:
                func_text += "  {} = sdc.str_arr_ext.pre_alloc_string_array(new_len, num_chars_{})\n".format(out, v)
            elif v in list_str_colnames:
                func_text += "  {} = sdc.str_ext.alloc_list_list_str(new_len)\n".format(out)
            elif v in split_view_colnames:
                # TODO support dropna() for split view
                func_text += "  {} = {}\n".format(out, v)
            else:
                func_text += "  {} = np.empty(new_len, {}.dtype)\n".format(out, v)
        func_text += "  curr_ind = 0\n"
        func_text += "  for i in numba.parfor.internal_prange(old_len):\n"
        func_text += "    if not ({}):\n".format(' or '.join(isna_calls))
        for v, out in zip(in_names, out_names):
            if v in split_view_colnames:
                continue
            func_text += "      {}[curr_ind] = {}[i]\n".format(out, v)
        func_text += "      curr_ind += 1\n"
        func_text += "  return ({},)\n".format(", ".join(out_names))

        loc_vars = {}
        exec(func_text, {'sdc': sdc, 'np': np, 'numba': numba}, loc_vars)
        _dropna_impl = loc_vars['_dropna_impl']
        return self._replace_func(_dropna_impl, rhs.args)

    def _run_call_concat(self, assign, lhs, rhs):
        nodes = []
        series_list = guard(get_definition, self.state.func_ir, rhs.args[0]).items
        arrs = [self._get_series_data(v, nodes) for v in series_list]
        arr_tup = ir.Var(rhs.args[0].scope, mk_unique_var('arr_tup'), rhs.args[0].loc)
        self.state.typemap[arr_tup.name] = types.Tuple([self.state.typemap[a.name] for a in arrs])
        tup_expr = ir.Expr.build_tuple(arrs, arr_tup.loc)
        nodes.append(ir.Assign(tup_expr, arr_tup, arr_tup.loc))
        return self._replace_func(
            lambda arr_list: sdc.hiframes.api.init_series(sdc.hiframes.api.concat(arr_list)),
            [arr_tup], pre_nodes=nodes)

    def _handle_sorted_by_key(self, rhs):
        """generate a sort function with the given key lambda
        """
        # TODO: handle reverse
        from numba.targets import quicksort
        # get key lambda
        key_lambda_var = dict(rhs.kws)['key']
        key_lambda = guard(get_definition, self.state.func_ir, key_lambda_var)
        if key_lambda is None or not (isinstance(key_lambda, ir.Expr) and key_lambda.op == 'make_function'):
            raise ValueError("sorted(): lambda for key not found")

        # wrap lambda in function
        def key_lambda_wrapper(A):
            return A
        key_lambda_wrapper.__code__ = key_lambda.code
        key_func = numba.njit(key_lambda_wrapper)

        # make quicksort with new lt
        def lt(a, b):
            return key_func(a) < key_func(b)
        sort_func = quicksort.make_jit_quicksort(lt=lt).run_quicksort

        return self._replace_func(
            lambda a: _sort_func(a), rhs.args,
            extra_globals={'_sort_func': numba.njit(sort_func)})

    def _get_const_tup(self, tup_var):
        tup_def = guard(get_definition, self.state.func_ir, tup_var)
        if isinstance(tup_def, ir.Expr):
            if tup_def.op == 'binop' and tup_def.fn in ('+', operator.add):
                return (self._get_const_tup(tup_def.lhs) + self._get_const_tup(tup_def.rhs))
            if tup_def.op in ('build_tuple', 'build_list'):
                return tup_def.items
        raise ValueError("constant tuple expected")

    def _get_dt_index_data(self, dt_var, nodes):
        var_def = guard(get_definition, self.state.func_ir, dt_var)
        call_def = guard(find_callname, self.state.func_ir, var_def)
        if call_def == ('init_datetime_index', 'sdc.hiframes.api'):
            return var_def.args[0]

        f_block = compile_to_numba_ir(
            lambda S: sdc.hiframes.api.get_index_data(S),
            {'sdc': sdc},
            self.state.typingctx,
            (self.state.typemap[dt_var.name],),
            self.state.typemap,
            self.state.calltypes
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [dt_var])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _get_series_data(self, series_var, nodes):
        # optimization: return data var directly if series has a single
        # definition by init_series()
        # e.g. S = init_series(A, None)
        # XXX assuming init_series() is the only call to create a series
        # and series._data is never overwritten
        var_def = guard(get_definition, self.state.func_ir, series_var)
        call_def = guard(find_callname, self.state.func_ir, var_def)
        if call_def == ('init_series', 'sdc.hiframes.api'):
            return var_def.args[0]

        # XXX use get_series_data() for getting data instead of S._data
        # to enable alias analysis
        f_block = compile_to_numba_ir(
            lambda S: sdc.hiframes.api.get_series_data(S),
            {'sdc': sdc},
            self.state.typingctx,
            (self.state.typemap[series_var.name],),
            self.state.typemap,
            self.state.calltypes
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [series_var])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _get_series_index(self, series_var, nodes):
        # XXX assuming init_series is the only call to create a series
        # and series._index is never overwritten
        var_def = guard(get_definition, self.state.func_ir, series_var)
        call_def = guard(find_callname, self.state.func_ir, var_def)
        if (call_def == ('init_series', 'sdc.hiframes.api')
                and (len(var_def.args) >= 2
                     and not self._is_const_none(var_def.args[1]))):
            return var_def.args[1]

        # XXX use get_series_index() for getting data instead of S._index
        # to enable alias analysis
        f_block = compile_to_numba_ir(
            lambda S: sdc.hiframes.api.get_series_index(S),
            {'sdc': sdc},
            self.state.typingctx,
            (self.state.typemap[series_var.name],),
            self.state.typemap,
            self.state.calltypes
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [series_var])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _get_index_values(self, series_data, nodes):
        """
        Generate index values by numpy.arange

        :param series_data: numba ir.Var for series data
        :param nodes: list of all irs
        :return: numba ir.Var for generated index
        """

        def _gen_arange(S):  # pragma: no cover
            n = len(S)
            return np.arange(n)

        f_block = compile_to_numba_ir(
            _gen_arange, {'np': np}, self.state.typingctx,
            (self.state.typemap[series_data.name],),
            self.state.typemap, self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [series_data])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _get_series_name(self, series_var, nodes):
        var_def = guard(get_definition, self.state.func_ir, series_var)
        call_def = guard(find_callname, self.state.func_ir, var_def)
        if (call_def == ('init_series', 'sdc.hiframes.api')
                and len(var_def.args) == 3):
            return var_def.args[2]

        f_block = compile_to_numba_ir(
            lambda S: sdc.hiframes.api.get_series_name(S),
            {'sdc': sdc},
            self.state.typingctx,
            (self.state.typemap[series_var.name],),
            self.state.typemap,
            self.state.calltypes
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [series_var])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _get_timedelta_index_data(self, dt_var, nodes):
        var_def = guard(get_definition, self.state.func_ir, dt_var)
        call_def = guard(find_callname, self.state.func_ir, var_def)
        if call_def == ('init_timedelta_index', 'sdc.hiframes.api'):
            return var_def.args[0]

        f_block = compile_to_numba_ir(
            lambda S: sdc.hiframes.api.get_index_data(S),
            {'sdc': sdc},
            self.state.typingctx,
            (self.state.typemap[dt_var.name],),
            self.state.typemap,
            self.state.calltypes
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [dt_var])
        nodes += f_block.body[:-2]
        return nodes[-1].target

    def _replace_func(self, func, args, const=False,
                      pre_nodes=None, extra_globals=None, pysig=None, kws=None):
        glbls = {'numba': numba, 'np': np, 'sdc': sdc}
        if extra_globals is not None:
            glbls.update(extra_globals)

        # create explicit arg variables for defaults if func has any
        # XXX: inine_closure_call() can't handle defaults properly
        if pysig is not None:
            pre_nodes = [] if pre_nodes is None else pre_nodes
            scope = next(iter(self.state.func_ir.blocks.values())).scope
            loc = scope.loc

            def normal_handler(index, param, default):
                return default

            def default_handler(index, param, default):
                d_var = ir.Var(scope, mk_unique_var('defaults'), loc)
                self.state.typemap[d_var.name] = numba.typeof(default)
                node = ir.Assign(ir.Const(default, loc), d_var, loc)
                pre_nodes.append(node)
                return d_var

            # TODO: stararg needs special handling?
            args = numba.typing.fold_arguments(
                pysig, args, kws, normal_handler, default_handler,
                normal_handler)

        arg_typs = tuple(self.state.typemap[v.name] for v in args)

        if const:
            new_args = []
            for i, arg in enumerate(args):
                val = guard(find_const, self.state.func_ir, arg)
                if val:
                    new_args.append(types.literal(val))
                else:
                    new_args.append(arg_typs[i])
            arg_typs = tuple(new_args)
        return ReplaceFunc(func, arg_typs, args, glbls, pre_nodes)

    def _convert_series_calltype(self, call):
        sig = self.state.calltypes[call]
        if sig is None:
            return
        assert isinstance(sig, Signature)

        # XXX using replace() since it copies, otherwise cached overload
        # functions fail
        new_sig = sig.replace(return_type=if_series_to_array_type(sig.return_type))
        new_sig.args = tuple(map(if_series_to_array_type, sig.args))

        # XXX: side effect: force update of call signatures
        if isinstance(call, ir.Expr) and call.op == 'call':
            # StencilFunc requires kws for typing so sig.args can't be used
            # reusing sig.args since some types become Const in sig
            argtyps = new_sig.args[:len(call.args)]
            kwtyps = {name: self.state.typemap[v.name] for name, v in call.kws}
            sig = new_sig
            new_sig = self.state.typemap[call.func.name].get_call_type(
                self.state.typingctx, argtyps, kwtyps)
            # calltypes of things like BoundFunction (array.call) need to
            # be updated for lowering to work
            # XXX: new_sig could be None for things like np.int32()
            if call in self.state.calltypes and new_sig is not None:
                old_sig = self.state.calltypes[call]
                # fix types with undefined dtypes in empty_inferred, etc.
                return_type = _fix_typ_undefs(new_sig.return_type, old_sig.return_type)
                args = tuple(_fix_typ_undefs(a, b) for a, b in zip(new_sig.args, old_sig.args))
                new_sig = Signature(return_type, args, new_sig.recvr, new_sig.pysig)

        if new_sig is not None:
            # XXX sometimes new_sig is None for some reason
            # FIXME e.g. test_series_nlargest_parallel1 np.int32()
            self.state.calltypes.pop(call)
            self.state.calltypes[call] = new_sig
        return

    def is_bool_arr(self, varname):
        typ = self.state.typemap[varname]
        return (isinstance(if_series_to_array_type(typ), types.Array)
                and typ.dtype == types.bool_)

    def _is_const_none(self, var):
        var_def = guard(get_definition, self.state.func_ir, var)
        return isinstance(var_def, ir.Const) and var_def.value is None

    def _handle_hiframes_nodes(self, inst):
        if isinstance(inst, Aggregate):
            # now that type inference is done, remove type vars to
            # enable dead code elimination
            inst.out_typer_vars = None
            use_vars = inst.key_arrs + list(inst.df_in_vars.values())
            if inst.pivot_arr is not None:
                use_vars.append(inst.pivot_arr)
            def_vars = list(inst.df_out_vars.values())
            if inst.out_key_vars is not None:
                def_vars += inst.out_key_vars
            apply_copies_func = hiframes.aggregate.apply_copies_aggregate
        elif isinstance(inst, hiframes.sort.Sort):
            use_vars = inst.key_arrs + list(inst.df_in_vars.values())
            def_vars = []
            if not inst.inplace:
                def_vars = inst.out_key_arrs + list(inst.df_out_vars.values())
            apply_copies_func = hiframes.sort.apply_copies_sort
        elif isinstance(inst, hiframes.join.Join):
            use_vars = list(inst.right_vars.values()) + list(inst.left_vars.values())
            def_vars = list(inst.df_out_vars.values())
            apply_copies_func = hiframes.join.apply_copies_join
        elif isinstance(inst, sdc.io.csv_ext.CsvReader):
            use_vars = []
            def_vars = inst.out_vars
            apply_copies_func = sdc.io.csv_ext.apply_copies_csv
        else:
            assert isinstance(inst, hiframes.filter.Filter)
            use_vars = list(inst.df_in_vars.values())
            if isinstance(self.state.typemap[inst.bool_arr.name], SeriesType):
                use_vars.append(inst.bool_arr)
            def_vars = list(inst.df_out_vars.values())
            apply_copies_func = hiframes.filter.apply_copies_filter

        out_nodes = self._convert_series_hiframes_nodes(
            inst, use_vars, def_vars, apply_copies_func)

        return out_nodes

    def _update_definitions(self, node_list):
        loc = ir.Loc("", 0)
        dumm_block = ir.Block(ir.Scope(None, loc), loc)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.state.func_ir._definitions)
        return

    def _convert_series_hiframes_nodes(self, inst, use_vars, def_vars,
                                       apply_copies_func):
        #
        out_nodes = []
        varmap = {v.name: self._get_series_data(v, out_nodes) for v in use_vars
                  if isinstance(self.state.typemap[v.name], SeriesType)}
        apply_copies_func(inst, varmap, None, None, None, None)
        out_nodes.append(inst)

        for v in def_vars:
            self.state.func_ir._definitions[v.name].remove(inst)
        varmap = {}
        for v in def_vars:
            if not isinstance(self.state.typemap[v.name], SeriesType):
                continue
            data_var = ir.Var(
                v.scope, mk_unique_var(v.name + 'data'), v.loc)
            self.state.typemap[data_var.name] = series_to_array_type(self.state.typemap[v.name])
            f_block = compile_to_numba_ir(
                lambda A: sdc.hiframes.api.init_series(A),
                {'sdc': sdc},
                self.state.typingctx,
                (self.state.typemap[data_var.name],),
                self.state.typemap,
                self.state.calltypes
            ).blocks.popitem()[1]
            replace_arg_nodes(f_block, [data_var])
            out_nodes += f_block.body[:-2]
            out_nodes[-1].target = v
            varmap[v.name] = data_var

        apply_copies_func(inst, varmap, None, None, None, None)
        return out_nodes

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


def _fix_typ_undefs(new_typ, old_typ):
    if isinstance(old_typ, (types.Array, SeriesType)):
        assert isinstance(new_typ, (types.Array, SeriesType, StringArrayType,
                                    types.List, StringArraySplitViewType))
        if new_typ.dtype == types.undefined:
            return new_typ.copy(old_typ.dtype)
    if isinstance(old_typ, (types.Tuple, types.UniTuple)):
        return types.Tuple([_fix_typ_undefs(t, u)
                            for t, u in zip(new_typ.types, old_typ.types)])
    # TODO: fix List, Set
    return new_typ

from __future__ import print_function, division, absolute_import
import types as pytypes  # avoid confusion with numba.types

import numba
from numba import ir, config
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc)
import hpat
from hpat import hiframes_api
import numpy
import pandas

import copy
def global_deepcopy(self, memo):
    return ir.Global(self.name, self.value, copy.deepcopy(self.loc))
ir.Global.__deepcopy__ = global_deepcopy

class HiFrames(object):
    """analyze and transform hiframes calls"""
    def __init__(self, func_ir):
        self.func_ir = func_ir

        # varname -> 'str'
        self.const_table = {}

        # var -> list
        self.map_calls = {}
        self.pd_globals = []
        self.pd_df_calls = []
        self.make_functions = {}

        # rolling_varname -> column_var
        self.rolling_vars = {}
        # rolling call name -> [column_varname, win_size]
        self.rolling_calls = {}
        # rolling call agg name -> [column_varname, win_size, func]
        self.rolling_calls_agg = {}

        # df_var -> {col1:col1_var ...}
        self.df_vars = {}
        # df_column -> df_var
        self.df_cols = {}
        self.df_col_calls = {}

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
            if (rhs.op=='getattr' and rhs.value.name in self.pd_globals
                    and rhs.attr=='DataFrame'):
                self.pd_df_calls.append(lhs)

            # df = pd.DataFrame(map_var)
            if rhs.op=='call' and rhs.func.name in self.pd_df_calls:
                # only map input allowed now
                assert len(rhs.args) is 1 and rhs.args[0].name in self.map_calls

                self.df_vars[lhs] = self._process_df_build_map(
                                            self.map_calls[rhs.args[0].name])
                self._update_df_cols()
                # remove DataFrame call
                return []

            # d = df['column']
            if (rhs.op == 'static_getitem' and rhs.value.name in self.df_vars
                                            and isinstance(rhs.index, str)):
                df = rhs.value.name
                assign.value = self.df_vars[df][rhs.index]
                self.df_cols[lhs] = df  # save lhs as column

            # df1 = df[df.A > .5]
            if (rhs.op == 'getitem' and rhs.value.name in self.df_vars):
                # output df1 has same columns as df, create new vars
                scope = assign.target.scope
                loc = assign.target.loc
                self.df_vars[lhs] = {}
                for col, _ in self.df_vars[rhs.value.name].items():
                    self.df_vars[lhs][col] = ir.Var(scope, mk_unique_var(col),
                                                                            loc)
                self._update_df_cols()
                return [hiframes_api.Filter(lhs, rhs.value.name, rhs.index,
                                                        self.df_vars, rhs.loc)]

            # d = df.column
            if rhs.op=='getattr' and rhs.value.name in self.df_vars:
                df = rhs.value.name
                df_cols = self.df_vars[df]
                assert rhs.attr in df_cols
                assign.value = df_cols[rhs.attr]
                self.df_cols[lhs] = df  # save lhs as column

            # c = df.column.shift
            if (rhs.op=='getattr' and rhs.value.name in self.df_cols and
                                        rhs.attr in ['shift', 'pct_change']):
                self.df_col_calls[lhs] = (rhs.value, rhs.attr)

            # A = df.column.shift(3)
            if rhs.op=='call' and rhs.func.name in self.df_col_calls:
                return self._gen_column_call(assign.target, rhs.args,
                                            *self.df_col_calls[rhs.func.name])

            # d.rolling
            if rhs.op=='getattr' and rhs.value.name in self.df_cols:
                if rhs.attr=='rolling':
                    self.rolling_vars[lhs] = rhs.value
                    return []  # remove node

            # d.rolling(3)
            if rhs.op=='call' and rhs.func.name in self.rolling_vars:
                window = -1
                center = False
                kws = dict(rhs.kws)
                if rhs.args:
                    window = self.const_table[rhs.args[0].name]
                elif 'window' in kws:
                    window = self.const_table[kws['window'].name]
                assert window >= 0
                if 'center' in kws:
                    center = self.const_table[kws['center'].name]
                self.rolling_calls[lhs] = [self.rolling_vars[rhs.func.name],
                        window, center]
                return []  # remove

            # d.rolling(3).sum
            if rhs.op=='getattr' and rhs.value.name in self.rolling_calls:
                self.rolling_calls_agg[lhs] = self.rolling_calls[rhs.value.name]
                self.rolling_calls_agg[lhs].append(rhs.attr)
                return []  # remove

            # d.rolling(3).sum()
            if rhs.op=='call' and rhs.func.name in self.rolling_calls_agg:
                return self._gen_rolling_call(rhs.args,
                    *self.rolling_calls_agg[rhs.func.name]+[assign.target])

            if rhs.op == 'build_map':
                self.map_calls[lhs] = rhs.items

            if rhs.op == 'make_function':
                self.make_functions[lhs] = rhs

        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.df_vars:
            self.df_vars[lhs] = self.df_vars[rhs.name]
        if isinstance(rhs, ir.Var) and rhs.name in self.df_cols:
            self.df_cols[lhs] = self.df_cols[rhs.name]

        if isinstance(rhs, ir.Const):
            self.const_table[lhs] = rhs.value
        return [assign]

    def _process_df_build_map(self, items_list):
        df_cols = {}
        for item in items_list:
            col_var = item[0].name
            assert col_var in self.const_table
            col_name = self.const_table[col_var]
            df_cols[col_name] = item[1]
        return df_cols

    def _update_df_cols(self):
        self.df_cols = {}  # reset
        for df_name, cols_map in self.df_vars.items():
            for col_name, col_var in cols_map.items():
                self.df_cols[col_var.name] = df_name
        return

    def _gen_column_call(self, out_var, args, col_var, func):
        loc = col_var.loc
        if func == 'pct_change':
            shift_const = 1
            if args:
                shift_const = self.const_table[args[0].name]
            func_text = 'def g(a):\n  return (a[0]-a[{}])/a[{}]\n'.format(
                                                    -shift_const, -shift_const)
        else:
            assert func == 'shift'
            shift_const = self.const_table[args[0].name]
            func_text = 'def g(a):\n  return a[{}]\n'.format(-shift_const)

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        code_obj = loc_vars['g'].__code__
        code_expr = ir.Expr.make_function(None, code_obj, None, None, loc)
        index_offsets = [0]
        return gen_stencil_call(col_var, out_var, code_expr, index_offsets)

    def _gen_rolling_call(self, args, col_var, win_size, center, func, out_var):
        loc = col_var.loc
        if func == 'apply':
            code_expr = self.make_functions[args[0].name]
        elif func in ['sum', 'mean', 'min', 'max', 'std', 'var']:
            kernel_args = ','.join(['a[{}]'.format(-i) for i in range(win_size)])
            kernel_expr = 'np.{}(np.array([{}]))'.format(func, kernel_args)
            if func == 'sum':  # simplify sum
                kernel_expr = '+'.join(['a[{}]'.format(-i) for i in range(win_size)])
            func_text = 'def g(a):\n  return {}\n'.format(kernel_expr)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            code_obj = loc_vars['g'].__code__
            code_expr = ir.Expr.make_function(None, code_obj, None, None, loc)
        index_offsets = [0]
        if func == 'apply':
            index_offsets = [-win_size+1]
        if center:
            index_offsets[0] += win_size//2

        return gen_stencil_call(col_var, out_var, code_expr, index_offsets)

def gen_empty_like(in_arr, out_arr):
    scope = in_arr.scope
    loc = in_arr.loc
    # g_np_var = Global(numpy)
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    g_np = ir.Global('np', numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: empty_attr = getattr(g_np_var, empty_like)
    empty_attr_call = ir.Expr.getattr(g_np_var, "zeros_like", loc)
    attr_var = ir.Var(scope, mk_unique_var("$empty_attr_attr"), loc)
    attr_assign = ir.Assign(empty_attr_call, attr_var, loc)
    # alloc call: out_arr = empty_attr(in_arr)
    alloc_call = ir.Expr.call(attr_var, [in_arr], (), loc)
    alloc_assign = ir.Assign(alloc_call, out_arr, loc)
    return [g_np_assign, attr_assign, alloc_assign]

def gen_stencil_call(in_arr, out_arr, code_expr, index_offsets):
    scope = in_arr.scope
    loc = in_arr.loc
    alloc_nodes = gen_empty_like(in_arr, out_arr)
    # generate stencil call
    # g_numba_var = Global(numba)
    g_numba_var = ir.Var(scope, mk_unique_var("$g_numba_var"), loc)
    g_dist = ir.Global('numba', numba, loc)
    g_numba_assign = ir.Assign(g_dist, g_numba_var, loc)
    # attr call: stencil_attr = getattr(g_numba_var, stencil)
    stencil_attr_call = ir.Expr.getattr(g_numba_var, "stencil", loc)
    stencil_attr_var = ir.Var(scope, mk_unique_var("$stencil_attr"), loc)
    stencil_attr_assign = ir.Assign(stencil_attr_call, stencil_attr_var, loc)
    # stencil_out = numba.stencil()
    stencil_out = ir.Var(scope, mk_unique_var("$stencil_out"), loc)
    stencil_call = ir.Expr.call(stencil_attr_var, [in_arr, out_arr], (), loc)
    stencil_call.stencil_def = code_expr
    stencil_call.index_offsets = index_offsets
    stencil_assign = ir.Assign(stencil_call, stencil_out, loc)
    return alloc_nodes + [g_numba_assign, stencil_attr_assign, stencil_assign]

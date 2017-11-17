from __future__ import print_function, division, absolute_import

import numpy as np
import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (replace_arg_nodes, compile_to_numba_ir,
                            find_topo_order, gen_np_call, get_definition, guard,
                            find_callname, mk_alloc)

import hpat
from hpat.utils import get_definitions
from hpat.hiframes import include_new_blocks, gen_empty_like
from hpat.str_arr_ext import string_array_type, StringArrayType

class HiFramesTyped(object):
    """Analyze and transform hiframes calls after typing"""
    def __init__(self, func_ir, typingctx, typemap, calltypes):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.typemap = typemap
        self.calltypes = calltypes
        self.df_cols = func_ir.df_cols

    def run(self):
        blocks = self.func_ir.blocks
        call_table, _ = ir_utils.get_call_table(blocks)
        topo_order = find_topo_order(blocks)
        for label in topo_order:
            new_body = []
            for inst in blocks[label].body:
                if isinstance(inst, ir.Assign):
                    out_nodes = self._run_assign(inst, call_table)
                    if isinstance(out_nodes, list):
                        new_body.extend(out_nodes)
                    if isinstance(out_nodes, dict):
                        label = include_new_blocks(blocks, out_nodes, label,
                                                                    new_body)
                        new_body = []
                    if isinstance(out_nodes, tuple):
                        gen_blocks, post_nodes = out_nodes
                        label = include_new_blocks(blocks, gen_blocks, label,
                                                                    new_body)
                        new_body = post_nodes
                else:
                    new_body.append(inst)
            blocks[label].body = new_body

        self.func_ir._definitions = get_definitions(self.func_ir.blocks)
        return

    def _run_assign(self, assign, call_table):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            res = self._handle_string_array_expr(lhs, rhs, assign)
            if res is not None:
                return res

            res = self._handle_fix_df_array(lhs, rhs, assign, call_table)
            if res is not None:
                return res

            res = self._handle_empty_like(lhs, rhs, assign, call_table)
            if res is not None:
                return res

            res = self._handle_df_col_filter(lhs, rhs, assign)
            if res is not None:
                return res

            if rhs.op == 'call':
                res = self._handle_df_col_calls(lhs, rhs, assign)
                if res is not None:
                    return res

        return [assign]

    def _handle_string_array_expr(self, lhs, rhs, assign):
        # convert str_arr==str into parfor
        if (rhs.op == 'binop'
                and rhs.fn in ['==', '!=']
                and (self.typemap[rhs.lhs.name] == string_array_type
                or self.typemap[rhs.rhs.name] == string_array_type)):
            arg1 = rhs.lhs
            arg2 = rhs.rhs
            arg1_access = 'A'
            arg2_access = 'B'
            len_call = 'A.size'
            if self.typemap[arg1.name] == string_array_type:
                arg1_access = 'A[i]'
            if self.typemap[arg2.name] == string_array_type:
                arg1_access = 'B[i]'
                len_call = 'B.size'
            func_text = 'def f(A, B):\n'
            func_text += '  l = {}\n'.format(len_call)
            func_text += '  S = np.empty(l, dtype=np.bool_)\n'
            func_text += '  for i in numba.parfor.internal_prange(l):\n'
            func_text += '    S[i] = {} {} {}\n'.format(arg1_access, rhs.fn,
                                                                    arg2_access)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            f = loc_vars['f']
            f_blocks = compile_to_numba_ir(f,
                    {'numba': numba, 'np': np}, self.typingctx,
                    (self.typemap[arg1.name], self.typemap[arg2.name]),
                    self.typemap, self.calltypes).blocks
            replace_arg_nodes(f_blocks[min(f_blocks.keys())], [arg1, arg2])
            # replace == expression with result of parfor (S)
            # S is target of last statement in 1st block of f
            assign.value = f_blocks[min(f_blocks.keys())].body[-2].target
            return (f_blocks, [assign])

        return None

    def _handle_fix_df_array(self, lhs, rhs, assign, call_table):
        # arr = fix_df_array(col) -> arr=col if col is array
        if (rhs.op == 'call'
                and rhs.func.name in call_table
                and call_table[rhs.func.name] ==
                            ['fix_df_array', 'hiframes_api', hpat]
                and isinstance(self.typemap[rhs.args[0].name],
                                    (types.Array, StringArrayType))):
            assign.value = rhs.args[0]
            return [assign]

        return None

    def _handle_empty_like(self, lhs, rhs, assign, call_table):
        # B = empty_like(A) -> B = empty(len(A), dtype)
        if (rhs.op == 'call'
                and rhs.func.name in call_table
                and call_table[rhs.func.name] == ['empty_like', np]):
            in_arr= rhs.args[0]
            def f(A):
                c = len(A)
            f_block = compile_to_numba_ir(f, {}, self.typingctx, (self.typemap[in_arr.name],),
                                self.typemap, self.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [in_arr])
            nodes = f_block.body[:-3]  # remove none return
            size_var = nodes[-1].target
            alloc_nodes = mk_alloc(self.typemap, self.calltypes, assign.target,
                            size_var,
                            self.typemap[in_arr.name].dtype, in_arr.scope, in_arr.loc)
            return nodes + alloc_nodes
        return None

    def _handle_df_col_filter(self, lhs_name, rhs, assign):
        # find df['col2'] = df['col1'][arr]
        # since columns should have the same size, output is filled with NaNs
        # TODO: check for float, make sure col1 and col2 are in the same df
        if (rhs.op=='getitem'
                and rhs.value.name in self.df_cols
                and lhs_name in self.df_cols
                and self.is_bool_arr(rhs.index.name)):
            lhs = assign.target
            in_arr = rhs.value
            index_var = rhs.index
            f_blocks = compile_to_numba_ir(_column_filter_impl_float,
                    {'numba': numba, 'np': np}, self.typingctx,
                    (self.typemap[lhs.name], self.typemap[in_arr.name],
                    self.typemap[index_var.name]),
                    self.typemap, self.calltypes).blocks
            first_block = min(f_blocks.keys())
            replace_arg_nodes(f_blocks[first_block], [lhs, in_arr, index_var])
            alloc_nodes = gen_np_call('empty_like', np.empty_like, lhs, [in_arr],
                        self.typingctx, self.typemap, self.calltypes)
            f_blocks[first_block].body = alloc_nodes + f_blocks[first_block].body
            return f_blocks

    def _handle_df_col_calls(self, lhs_name, rhs, assign):
        if guard(find_callname, self.func_ir, rhs) == ('fillna', 'hpat.hiframes_api'):
            out_arr = rhs.args[0]
            in_arr = rhs.args[1]
            val = rhs.args[2]
            f_blocks = compile_to_numba_ir(_column_fillna_impl,
                    {'numba': numba, 'np': np}, self.typingctx,
                    (self.typemap[out_arr.name], self.typemap[in_arr.name],
                    self.typemap[val.name]),
                    self.typemap, self.calltypes).blocks
            first_block = min(f_blocks.keys())
            replace_arg_nodes(f_blocks[first_block], [out_arr, in_arr, val])
            return f_blocks

        if guard(find_callname, self.func_ir, rhs) == ('column_sum', 'hpat.hiframes_api'):
            in_arr = rhs.args[0]
            f_blocks = compile_to_numba_ir(_column_sum_impl,
                    {'numba': numba, 'np': np, 'hpat': hpat}, self.typingctx,
                    (self.typemap[in_arr.name],),
                    self.typemap, self.calltypes).blocks
            topo_order = find_topo_order(f_blocks)
            first_block = topo_order[0]
            last_block = topo_order[-1]
            replace_arg_nodes(f_blocks[first_block], [in_arr])
            # assign results to lhs output
            f_blocks[last_block].body[-4].target = assign.target
            return f_blocks

        if guard(find_callname, self.func_ir, rhs) == ('mean', 'hpat.hiframes_api'):
            in_arr = rhs.args[0]
            f_blocks = compile_to_numba_ir(_column_mean_impl,
                    {'numba': numba, 'np': np, 'hpat': hpat}, self.typingctx,
                    (self.typemap[in_arr.name],),
                    self.typemap, self.calltypes).blocks
            topo_order = find_topo_order(f_blocks)
            first_block = topo_order[0]
            last_block = topo_order[-1]
            replace_arg_nodes(f_blocks[first_block], [in_arr])
            # assign results to lhs output
            f_blocks[last_block].body[-4].target = assign.target
            return f_blocks

        if guard(find_callname, self.func_ir, rhs) == ('var', 'hpat.hiframes_api'):
            in_arr = rhs.args[0]
            f_blocks = compile_to_numba_ir(_column_var_impl,
                    {'numba': numba, 'np': np, 'hpat': hpat}, self.typingctx,
                    (self.typemap[in_arr.name],),
                    self.typemap, self.calltypes).blocks
            topo_order = find_topo_order(f_blocks)
            first_block = topo_order[0]
            last_block = topo_order[-1]
            replace_arg_nodes(f_blocks[first_block], [in_arr])
            # assign results to lhs output
            f_blocks[last_block].body[-4].target = assign.target
            return f_blocks

        return

    def is_bool_arr(self, varname):
        typ = self.typemap[varname]
        return isinstance(typ, types.npytypes.Array) and typ.dtype==types.bool_

# float columns can have regular np.nan
def _column_filter_impl_float(A, B, ind):
    for i in numba.parfor.internal_prange(len(A)):
        s = 0
        if ind[i]:
            s = B[i]
        else:
            s = np.nan
        A[i] = s

def _column_fillna_impl(A, B, fill):
    for i in numba.parfor.internal_prange(len(A)):
        s = B[i]
        if np.isnan(s):
            s = fill
        A[i] = s

@numba.njit
def _sum_handle_nan(s, count):
    if not count:
        s = np.nan
    return s

def _column_sum_impl(A):
    count = 0
    s = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += val
            count += 1

    res = hpat.hiframes_typed._sum_handle_nan(s, count)

@numba.njit
def _mean_handle_nan(s, count):
    if not count:
        s = np.nan
    else:
        s = s/count
    return s

def _column_mean_impl(A):
    count = 0
    s = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += val
            count += 1

    res = hpat.hiframes_typed._mean_handle_nan(s, count)

@numba.njit
def _var_handle_nan(s, count):
    if count <= 1:
        s = np.nan
    else:
        s = s/(count-1)
    return s

def _column_var_impl(A):
    count_m = 0
    m = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            m += val
            count_m += 1

    m = hpat.hiframes_typed._mean_handle_nan(m, count_m)
    s = 0
    count = 0
    for i in numba.parfor.internal_prange(len(A)):
        val = A[i]
        if not np.isnan(val):
            s += (val-m)**2
            count += 1

    res = hpat.hiframes_typed._var_handle_nan(s, count)

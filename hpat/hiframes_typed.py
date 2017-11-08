from __future__ import print_function, division, absolute_import

import numpy as np
import numba
from numba import ir, ir_utils, types
from numba.ir_utils import replace_arg_nodes, compile_to_numba_ir, find_topo_order, gen_np_call

import hpat
from hpat.utils import get_definitions
from hpat.hiframes import include_new_blocks, gen_empty_like, get_inner_ir, replace_var_names
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
            for stmt in blocks[label].body:
                # convert str_arr==str into parfor
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'binop'
                        and stmt.value.fn in ['==', '!=']
                        and (self.typemap[stmt.value.lhs.name] == string_array_type
                        or self.typemap[stmt.value.rhs.name] == string_array_type)):
                    lhs = stmt.value.lhs
                    rhs = stmt.value.rhs
                    lhs_access = 'A'
                    rhs_access = 'B'
                    len_call = 'A.size'
                    if self.typemap[lhs.name] == string_array_type:
                        lhs_access = 'A[i]'
                    if self.typemap[rhs.name] == string_array_type:
                        lhs_access = 'B[i]'
                        len_call = 'B.size'
                    func_text = 'def f(A, B):\n'
                    func_text += '  l = {}\n'.format(len_call)
                    func_text += '  S = np.empty(l, dtype=np.bool_)\n'
                    func_text += '  for i in numba.parfor.prange(l):\n'
                    func_text += '    S[i] = {} {} {}\n'.format(lhs_access, stmt.value.fn, rhs_access)
                    loc_vars = {}
                    exec(func_text, {}, loc_vars)
                    f = loc_vars['f']
                    f_blocks = compile_to_numba_ir(f,
                            {'numba': numba, 'np': np}, self.typingctx,
                            (self.typemap[lhs.name], self.typemap[rhs.name]),
                            self.typemap, self.calltypes).blocks
                    replace_arg_nodes(f_blocks[min(f_blocks.keys())], [lhs, rhs])
                    label = include_new_blocks(blocks, f_blocks, label, new_body)
                    new_body = []
                    # replace == expression with result of parfor (S)
                    # S is target of last statement in 1st block of f
                    stmt.value = f_blocks[min(f_blocks.keys())].body[-2].target
                # arr = fix_df_array(col) -> arr=col if col is array
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'
                        and stmt.value.func.name in call_table
                        and call_table[stmt.value.func.name] ==
                                    ['fix_df_array', 'hiframes_api', hpat]
                        and isinstance(self.typemap[stmt.value.args[0].name],
                                            (types.Array, StringArrayType))):
                    stmt.value = stmt.value.args[0]
                # find df['col2'] = df['col1'][arr]
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op=='getitem'
                        and stmt.value.value.name in self.df_cols
                        and stmt.target.name in self.df_cols
                        and self.is_bool_arr(stmt.value.index.name)):
                    lhs = stmt.target
                    in_arr = stmt.value.value
                    index_var = stmt.value.index
                    def f(A, B, ind):
                        for i in numba.parfor.prange(len(A)):
                            s = 0
                            if ind[i]:
                                s = B[i]
                            else:
                                s= np.nan
                            A[i] = s
                    f_blocks = compile_to_numba_ir(f,
                            {'numba': numba, 'np': np}, self.typingctx,
                            (self.typemap[lhs.name], self.typemap[in_arr.name], self.typemap[index_var.name]),
                            self.typemap, self.calltypes).blocks
                    first_block = min(f_blocks.keys())
                    replace_arg_nodes(f_blocks[first_block], [lhs, in_arr, index_var])
                    alloc_nodes = gen_np_call('empty_like', np.empty_like, lhs, [in_arr],
                                self.typingctx, self.typemap, self.calltypes)
                    f_blocks[first_block].body = alloc_nodes + f_blocks[first_block].body
                    label = include_new_blocks(blocks, f_blocks, label, new_body)
                    new_body = []
                else:
                    new_body.append(stmt)
            blocks[label].body = new_body

        self.func_ir._definitions = get_definitions(self.func_ir.blocks)
        return

    def is_bool_arr(self, varname):
        typ = self.typemap[varname]
        return isinstance(typ, types.npytypes.Array) and typ.dtype==types.bool_

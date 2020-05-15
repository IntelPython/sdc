# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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

from numba.core.rewrites import register_rewrite, Rewrite
from numba.core.ir_utils import find_callname, guard, mk_unique_var
from numba import errors
from numba.core import ir
from numba.core import consts

from sdc.rewrites.ir_utils import remove_unused_recursively, make_assign, find_operations


def find_build_sequence(func_ir, var):
    """Reimplemented from numba.core.ir_utils.find_build_sequence
    Added 'build_map' to build_ops list.
    """
    from numba.core.ir_utils import (require, get_definition)

    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)
    require(isinstance(var_def, ir.Expr))
    build_ops = ['build_tuple', 'build_list', 'build_set', 'build_map']
    require(var_def.op in build_ops)
    return var_def.items, var_def.op


class ConstantInference(consts.ConstantInference):

    def _infer_expr(self, expr):
        if expr.op == 'build_map':
            def inf_const(value):
                return self.infer_constant(value.name, loc=expr.loc)
            return {inf_const(k): inf_const(v) for k, v in expr.items}
        return super()._infer_expr(expr)


@register_rewrite('before-inference')
class RewriteReadCsv(Rewrite):
    """
    Searches for calls of pandas.read_csv() and replace it with calls of read_csv.
    """

    _pandas_read_csv_calls = [
        ('read_csv', 'pandas'),             # for calls like pandas.read_csv()
        ('read_csv', 'pandas.io.parsers'),  # for calls like read_csv = pandas.read_csv, read_csv()
    ]

    _read_csv_const_args = ('names', 'dtype', 'usecols')

    def match(self, func_ir, block, typemap, calltypes):
        # TODO: 1. save instructions of build_map, build_list for read_csv params
        # 2. check that vars are used only in read_csv
        # 3. replace vars with build_tuple inplace

        self.func_ir = func_ir
        self.block = block
        self.consts = consts = {}

        # Find all assignments with a right-hand read_csv() call
        for inst in find_operations(block=block, op_name='call'):
            expr = inst.value
            call = guard(find_callname, func_ir, expr)
            if call not in self._pandas_read_csv_calls:
                continue
            # collect constant parameters with type list and dict
            # in order to replace with tuple
            for key, var in expr.kws:
                if key not in self._read_csv_const_args:
                    continue
                try:
                    const = func_ir.infer_constant(var)
                except errors.ConstantInferenceError:
                    try:
                        const = ConstantInference(func_ir).infer_constant(var.name)
                    except errors.ConstantInferenceError:
                        continue
                if isinstance(const, (list, dict)):
                    consts.setdefault(inst, {})[key] = const

        return len(consts) > 0

    def apply(self):
        new_block = self.block.copy()
        new_block.clear()
        vars_to_remove = []

        for inst in self.block.body:
            if inst in self.consts:
                consts = self.consts[inst]

                for key, value in consts.items():
                    if key not in dict(inst.value.kws):
                        continue

                    # collecting data from current variable
                    current_var = [var for name, var in inst.value.kws if name == key][0]
                    loc = current_var.loc

                    seq, _ = guard(find_build_sequence, self.func_ir, current_var)
                    if not seq:
                        continue
                    if isinstance(value, list):
                        items = seq
                    elif isinstance(value, dict):
                        items = sum(map(list, seq), [])
                    else:
                        continue

                    # create tuple variable
                    stmt = make_assign(ir.Expr.build_tuple(items=items, loc=loc), new_block.scope,
                                       self.func_ir, loc, name=f"{key}_tuple")
                    new_block.append(stmt)

                    # replace variable in call
                    inst.value.kws = [(kw[0], stmt.target) if kw[0] == key else kw for kw in inst.value.kws]

                    # save old variable for removing
                    vars_to_remove.append(current_var)

            new_block.append(inst)

        # remove old variables
        for var in vars_to_remove:
            # unsused variables are removed after new block is created b/c
            # remove_unused_recursively should see all del statements of variables
            remove_unused_recursively(var, new_block, self.func_ir)

        return new_block

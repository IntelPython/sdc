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
from numba import errors
from numba.core import ir
from numba.core.ir_utils import guard, get_definition


class TuplifyArgs(Rewrite):
    """
    Base rewrite calls to *callee*. Replaces *arg* from list and set to tuple.

    Redefine callee and arg in subclass.
    """

    # need to be defined in subclasses
    callee = None
    arg = None
    expr_checker = None

    def match_expr(self, expr, func_ir, block, typemap, calltypes):
        """For extended checks in supbclasses."""
        if self.expr_checker:
            return self.expr_checker(expr, func_ir, block, typemap, calltypes)
        return True

    def match(self, func_ir, block, typemap, calltypes):
        self.args = args = []
        self.block = block
        for inst in block.find_insts(ir.Assign):
            if isinstance(inst.value, ir.Expr) and inst.value.op == 'call':
                expr = inst.value
                try:
                    callee = func_ir.infer_constant(expr.func)
                except errors.ConstantInferenceError:
                    continue
                if callee is self.callee:
                    if not self.match_expr(expr, func_ir, block, typemap, calltypes):
                        continue

                    arg_var = None
                    if len(expr.args):
                        arg_var = expr.args[0]
                    elif len(expr.kws) and expr.kws[0][0] == self.arg:
                        arg_var = expr.kws[0][1]
                    if arg_var:
                        arg_var_def = guard(get_definition, func_ir, arg_var)
                        if arg_var_def and arg_var_def.op in ('build_list', 'build_set'):
                            args.append(arg_var_def)
        return len(args) > 0

    def apply(self):
        """
        Replace list expression with tuple.
        """
        block = self.block
        for inst in block.body:
            if isinstance(inst, ir.Assign) and inst.value in self.args:
                inst.value.op = 'build_tuple'
        return block


def register_tuplify(_callee, _arg, _expr_checker=None):
    @register_rewrite('before-inference')
    class Tuplifier(TuplifyArgs):
        callee = _callee
        arg = _arg
        expr_checker = _expr_checker

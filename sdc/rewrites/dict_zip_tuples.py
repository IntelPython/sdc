# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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
from numba.core.ir_utils import guard, get_definition
from numba import errors
from numba.core import ir

from sdc.rewrites.ir_utils import find_operations, import_function
from sdc.functions.tuple_utils import sdc_tuple_zip


@register_rewrite('before-inference')
class RewriteDictZip(Rewrite):
    """
    Searches for calls like dict(zip(arg1, arg2)) and replaces zip with sdc_zip.
    """

    def match(self, func_ir, block, typemap, calltypes):

        self._block = block
        self._func_ir = func_ir
        self._calls_to_rewrite = set()

        # Find all assignments with a RHS expr being a call to dict, and where arg
        # is a call to zip and store these ir.Expr for further modification
        for inst in find_operations(block=block, op_name='call'):
            expr = inst.value
            try:
                callee = func_ir.infer_constant(expr.func)
            except errors.ConstantInferenceError:
                continue

            if (callee is dict and len(expr.args) == 1):
                dict_arg_expr = guard(get_definition, func_ir, expr.args[0])
                if (getattr(dict_arg_expr, 'op', None) == 'call'):
                    called_func = guard(get_definition, func_ir, dict_arg_expr.func)
                    if (called_func.value is zip and len(dict_arg_expr.args) == 2):
                        self._calls_to_rewrite.add(dict_arg_expr)

        return len(self._calls_to_rewrite) > 0

    def apply(self):
        """
        Replace call to zip in matched expressions with call to sdc_zip.
        """
        new_block = self._block.copy()
        new_block.clear()
        zip_spec_stmt = import_function(sdc_tuple_zip, new_block, self._func_ir)
        for inst in self._block.body:
            if isinstance(inst, ir.Assign) and inst.value in self._calls_to_rewrite:
                expr = inst.value
                expr.func = zip_spec_stmt.target  # injects the new function
            new_block.append(inst)
        return new_block

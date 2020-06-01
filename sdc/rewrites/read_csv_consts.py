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
from numba.core.ir_utils import guard, get_definition
from numba import errors
from numba.core import ir

from sdc.rewrites.ir_utils import find_operations

import pandas as pd


@register_rewrite('before-inference')
class RewriteReadCsv(Rewrite):
    """
    Searches for calls to Pandas read_csv() and replace its arguments with tuples.
    """

    _read_csv_const_args = ('names', 'dtype', 'usecols')

    def match(self, func_ir, block, typemap, calltypes):
        # TODO: check that vars are used only in read_csv

        self.block = block
        self.args = args = []

        # Find all assignments with a right-hand read_csv() call
        for inst in find_operations(block=block, op_name='call'):
            expr = inst.value
            try:
                callee = func_ir.infer_constant(expr.func)
            except errors.ConstantInferenceError:
                continue
            if callee is not pd.read_csv:
                continue
            # collect arguments with list, set and dict
            # in order to replace with tuple
            for key, var in expr.kws:
                if key in self._read_csv_const_args:
                    arg_def = guard(get_definition, func_ir, var)
                    ops = ['build_list', 'build_set', 'build_map']
                    if arg_def.op in ops:
                        args.append(arg_def)

        return len(args) > 0

    def apply(self):
        """
        Replace list, set and dict expressions with tuple.
        """
        block = self.block
        for inst in block.body:
            if isinstance(inst, ir.Assign) and inst.value in self.args:
                if inst.value.op == 'build_map':
                    inst.value.items = sum(map(list, inst.value.items), [])
                inst.value.op = 'build_tuple'
        return block

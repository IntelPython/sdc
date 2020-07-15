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

from numba.core.ir import Assign, Const, Expr, Var
from numba.core.ir_utils import mk_unique_var
from numba.core.rewrites import register_rewrite, Rewrite
from numba.core.types import StringLiteral
from numba.core.typing import signature

from sdc.hiframes.pd_dataframe_type import DataFrameType


@register_rewrite('after-inference')
class RewriteDataFrameGetItemAttr(Rewrite):
    """
    Search for calls of df.attr and replace it with calls of df['attr']:
    $0.2 = getattr(value=df, attr=A) -> $const0.0 = const(str, A)
                                        $0.2 = static_getitem(value=df, index=A, index_var=$const0.0)
    """

    def match(self, func_ir, block, typemap, calltypes):
        self.func_ir = func_ir
        self.block = block
        self.typemap = typemap
        self.calltypes = calltypes
        self.getattrs = getattrs = set()
        for expr in block.find_exprs(op='getattr'):
            obj = typemap[expr.value.name]
            if not isinstance(obj, DataFrameType):
                continue
            if expr.attr in obj.columns:
                getattrs.add(expr)

        return len(getattrs) > 0

    def apply(self):
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if isinstance(inst, Assign) and inst.value in self.getattrs:
                const_assign = self._assign_const(inst)
                new_block.append(const_assign)

                inst = self._assign_getitem(inst, index=const_assign.target)

            new_block.append(inst)

        return new_block

    def _mk_unique_var(self, prefix):
        """Make unique var name checking self.func_ir._definitions"""
        name = mk_unique_var(prefix)
        while name in self.func_ir._definitions:
            name = mk_unique_var(prefix)

        return name

    def _assign_const(self, inst, prefix='$const0'):
        """Create constant from attribute of the instruction."""
        const_node = Const(inst.value.attr, inst.loc)
        unique_var_name = self._mk_unique_var(prefix)
        const_var = Var(inst.target.scope, unique_var_name, inst.loc)

        self.func_ir._definitions[const_var.name] = [const_node]
        self.typemap[const_var.name] = StringLiteral(inst.value.attr)

        return Assign(const_node, const_var, inst.loc)

    def _assign_getitem(self, inst, index):
        """Create getitem instruction from the getattr instruction."""
        new_expr = Expr.getitem(inst.value.value, index, inst.loc)
        new_inst = Assign(value=new_expr, target=inst.target, loc=inst.loc)

        self.func_ir._definitions[inst.target] = [new_expr]
        self.calltypes[new_expr] = signature(
            self.typemap[inst.target.name],
            self.typemap[new_expr.value.name],
            self.typemap[new_expr.index.name]
        )

        return new_inst

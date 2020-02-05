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

from numba.ir import Assign, Const, Expr, Var
from numba.ir_utils import mk_unique_var
from numba.rewrites import register_rewrite, Rewrite
from numba.types import StringLiteral
from numba.typing import signature

from sdc.config import config_pipeline_hpat_default
from sdc.hiframes.pd_dataframe_type import DataFrameType


if not config_pipeline_hpat_default:
    @register_rewrite('after-inference')
    class RewriteDataFrameGetItemAttr(Rewrite):
        """
        Search for calls of df.attr and replace it with calls of df['attr']:
        $0.2 = getattr(value=df, attr=A) -> $const0.0 = const(str, A)
                                            $0.2 = static_getitem(value=df, index=A, index_var=$const0.0)
        """

        def match(self, func_ir, block, typemap, calltypes):
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
                if isinstance(inst, Assign):
                    expr = inst.value
                    if expr in self.getattrs:
                        const_assign = self._assign_const(inst)
                        new_block.append(const_assign)

                        new_expr = Expr.getitem(expr.value, const_assign.target, inst.loc)
                        self.calltypes[new_expr] = signature(
                            self.typemap[inst.target.name],
                            self.typemap[expr.value.name],
                            self.typemap[new_expr.index.name]
                        )
                        inst = Assign(value=new_expr, target=inst.target, loc=inst.loc)
                new_block.append(inst)

            return new_block

        def _assign_const(self, inst, prefix='$const0'):
            """Create constant from attribute of specified instruction."""
            const_node = Const(inst.value.attr, inst.loc)
            const_var = Var(inst.target.scope, mk_unique_var(prefix), inst.loc)
            self.typemap[const_var.name] = StringLiteral(inst.value.attr)

            return Assign(const_node, const_var, inst.loc)

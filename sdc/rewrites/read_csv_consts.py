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

from numba.rewrites import (register_rewrite, Rewrite)
from numba.ir_utils import (
    find_build_sequence,
    find_callname,
    get_definition,
    guard,
    mk_unique_var,
    require,
)
from numba.ir import (Expr)
from numba import ir, errors
from numba.extending import overload
from numba import types
from numba.types import (
    literal
)

from pandas import DataFrame

from sdc.rewrites.ir_utils import (find_operations, is_dict,
                                   get_tuple_items, get_dict_items, remove_unused_recursively,
                                   get_call_parameters,
                                   declare_constant,
                                   import_function, make_call,
                                   insert_before)
from sdc.hiframes.pd_dataframe_ext import (init_dataframe, DataFrameType)
from sdc.io.csv_ext import (pandas_read_csv)

from sdc.hiframes.api import fix_df_array

from sdc.config import config_pipeline_hpat_default


def find_build_sequence(func_ir, var):
    """Check if a variable is constructed via build_tuple or
    build_list or build_set, and return the sequence and the
    operator, or raise GuardException otherwise.
    Note: only build_tuple is immutable, so use with care.
    """
    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)
    require(isinstance(var_def, ir.Expr))
    build_ops = ['build_tuple', 'build_list', 'build_set', 'build_map']
    require(var_def.op in build_ops)
    return var_def.items, var_def.op


def _new_definition(func_ir, var, value, loc):
    func_ir._definitions[var.name] = [value]
    return ir.Assign(value=value, target=var, loc=loc)


@register_rewrite('before-inference')
class RewriteReadCsv(Rewrite):
    """
    Searches for calls of pandas.read_csv() and replace it with calls of read_csv.
    """

    _pandas_read_csv = ('read_csv', 'pandas.io.parsers')
    _read_csv_arg_list = ('dtype', 'names')

    def match(self, func_ir, block, typemap, calltypes):
        self.func_ir = func_ir
        self.block = block
        self.consts = consts = {}

        # Find all assignments with a right-hand read_csv() call
        for inst in block.find_insts(ir.Assign):
            if isinstance(inst.value, ir.Expr) and inst.value.op == 'call':
                expr = inst.value
                call = guard(find_callname, func_ir, inst.value)
                if call is not None and call == ('read_csv', 'pandas.io.parsers'):
                    if hasattr(inst, 'consts'):
                        # Already rewritten
                        continue

                    for key, var in expr.kws:
                        try:
                            const = func_ir.infer_constant(var)
                        except errors.ConstantInferenceError:
                            continue
                        consts.setdefault(inst, {})[key] = const

        return len(consts) > 0

    def apply(self):
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if inst in self.consts:
                # mark that inst is processed
                inst.consts = self.consts[inst]

                if 'names' in dict(inst.value.kws):
                    # create names tuple
                    names_var = [val for name, val in inst.value.kws if name == 'names'][0]
                    loc = names_var.loc

                    seq, _ = guard(find_build_sequence, self.func_ir, names_var)
                    names_tuple_var = ir.Var(new_block.scope, mk_unique_var("names_tuple"), loc)

                    new_block.append(_new_definition(self.func_ir, names_tuple_var,
                            ir.Expr.build_tuple(items=seq, loc=loc), loc))

                    # replace names in call
                    inst.value.kws = [(kw[0], names_tuple_var) if kw[0] == "names" else kw for kw in inst.value.kws]

                if 'dtype' in dict(inst.value.kws):
                    # create dtype typle
                    dtype_var = [val for name, val in inst.value.kws if name == 'dtype'][0]
                    loc = dtype_var.loc

                    seq, _ = guard(find_build_sequence, self.func_ir, dtype_var)
                    dtype_tuple_var = ir.Var(new_block.scope, mk_unique_var("dtype_tuple"), loc)

                    new_block.append(_new_definition(self.func_ir, dtype_tuple_var,
                            ir.Expr.build_tuple(items=sum(map(list, seq), []), loc=loc), loc))

                    # replace dtype in call
                    inst.value.kws = [(kw[0], dtype_tuple_var) if kw[0] == "dtype" else kw for kw in inst.value.kws]

                if 'usecols' in dict(inst.value.kws):
                    # create usecols typle
                    usecols_var = [val for name, val in inst.value.kws if name == 'usecols'][0]
                    loc = usecols_var.loc

                    seq, _ = guard(find_build_sequence, self.func_ir, usecols_var)
                    usecols_tuple_var = ir.Var(new_block.scope, mk_unique_var("usecols_tuple"), loc)

                    new_block.append(_new_definition(self.func_ir, usecols_tuple_var,
                            ir.Expr.build_tuple(items=seq, loc=loc), loc))

                    # replace usecols in call
                    inst.value.kws = [(kw[0], usecols_tuple_var) if kw[0] == "usecols" else kw for kw in inst.value.kws]

            new_block.append(inst)
        return new_block

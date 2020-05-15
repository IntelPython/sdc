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

import pandas as pd
from sdc.datatypes.common.rewriteutils import register_tuplify

from numba.core import ir
from numba.core.ir_utils import guard, get_definition


def check_dtype_is_categorical(self, expr, func_ir, block, typemap, calltypes):
    dtype_var = None
    for name, var in expr.kws:
        if name == 'dtype':
            dtype_var = var
    if not dtype_var:
        return False

    dtype_var_def = guard(get_definition, func_ir, dtype_var)
    is_alias = isinstance(dtype_var_def, ir.Const) and dtype_var_def.value == 'category'
    is_categoricaldtype = (hasattr(dtype_var_def, 'func') and
                           func_ir.infer_constant(dtype_var_def.func) == pd.CategoricalDtype)
    if not (is_alias or is_categoricaldtype):
        return False

    return True


def expr_checker(self, expr, func_ir, block, typemap, calltypes):
    return check_dtype_is_categorical(self, expr, func_ir, block, typemap, calltypes)


register_tuplify(pd.Series, 'data', expr_checker)

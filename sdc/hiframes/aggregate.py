# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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


from __future__ import print_function, division, absolute_import

import numba
from numba import compiler, ir, ir_utils, typed_passes, typeinfer, types
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.extending import overload, lower_builtin


@infer_global(bool)
class BoolNoneTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        val_t = args[0]
        if val_t == types.none:
            return signature(types.boolean, *args)


@lower_builtin(bool, types.none)
def lower_column_mean_impl(context, builder, sig, args):
    res = context.compile_internal(builder, lambda a: False, sig, args)
    return res  # impl_ret_untracked(context, builder, sig.return_type, res)

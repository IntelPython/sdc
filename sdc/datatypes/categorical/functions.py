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

from sdc.utilities.utils import sdc_overload_attribute, sdc_overload
from numba.extending import intrinsic
from numba import types

from .types import CategoricalDtypeType, Categorical


@sdc_overload_attribute(CategoricalDtypeType, 'ordered')
def pd_CategoricalDtype_categories_overload(self):
    ordered = self.ordered

    def impl(self):
        return ordered
    return impl


@intrinsic
def _categorical_len(tyctx, arr_type):
    ret_type = types.intp

    def codegen(context, builder, sig, args):
        arr_val, = args
        arr_info = context.make_helper(builder, arr_type, arr_val)
        res = builder.load(arr_info._get_ptr_by_name('nitems'))
        return res

    return ret_type(arr_type), codegen


@sdc_overload(len)
def pd_Categorical_len_overload(self):
    if not isinstance(self, Categorical):
        return None

    # Categorical use ArrayModel and don't expose be_type members
    # hence we use intrinsic to access those fields. TO-DO: refactor
    def impl(self):
        return _categorical_len(self)

    return impl

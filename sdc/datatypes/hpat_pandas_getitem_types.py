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


import pandas

from numba import types, cgutils
from numba.extending import (models, overload, register_model, make_attribute_wrapper, intrinsic)
from numba.datamodel import (register_default, StructModel)
from numba.typing.templates import signature


class SeriesGetitemAccessorType(types.Type):
    def __init__(self, series, accessor):
        self.series = series
        self.accessor = accessor
        super(SeriesGetitemAccessorType, self).__init__('SeriesGetitemAccessorType({}, {})\
            '.format(series, accessor))


@register_model(SeriesGetitemAccessorType)
class SeriesGetitemAccessorTypeModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('series', fe_type.series),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(SeriesGetitemAccessorType, 'series', '_series')


@intrinsic
def series_getitem_accessor_init(typingctx, series, accessor):
    def series_getitem_accessor_init_codegen(context, builder, signature, args):
        series_val, accessor_val = args
        getitem_accessor = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        getitem_accessor.series = series_val

        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], series_val)

        return getitem_accessor._getvalue()

    ret_typ = SeriesGetitemAccessorType(series, accessor)
    sig = signature(ret_typ, series, accessor)

    return sig, series_getitem_accessor_init_codegen

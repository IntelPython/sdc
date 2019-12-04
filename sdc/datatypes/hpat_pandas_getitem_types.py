# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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


class SeriesGetitemSelectorType(types.IterableType):
    def __init__(self, data, selector):
        self.data = data
        self.selector = selector
        super(SeriesGetitemSelectorType, self).__init__('SeriesGetitemSelectorType')

    @property
    def iterator_type(self):
        return None


@register_model(SeriesGetitemSelectorType)
class SeriesGetitemSelectorTypeModel(StructModel):
    def __init__(self, dmm, fe_type):
        selector_typ = types.uint64
        members = [
            ('data', fe_type.data),
            ('selector', selector_typ),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(SeriesGetitemSelectorType, 'data', '_data')
make_attribute_wrapper(SeriesGetitemSelectorType, 'selector', '_selector')


@intrinsic
def series_getitem_selector_init(typingctx, data, selector=0):
    def series_getitem_selector_init_codegen(context, builder, signature, args):
        data_val, selector_val = args
        getitem_selector = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        getitem_selector.data = data_val
        getitem_selector.selector = selector_val

        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], selector_val)

        return getitem_selector._getvalue()

    ret_typ = SeriesGetitemSelectorType(data, selector)
    sig = signature(ret_typ, data, selector)

    return sig, series_getitem_selector_init_codegen

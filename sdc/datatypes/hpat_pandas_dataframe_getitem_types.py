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

from numba import types
from numba.core import cgutils
from numba.extending import models, overload, register_model, make_attribute_wrapper, intrinsic
from numba.core.datamodel import register_default, StructModel
from numba.core.typing.templates import signature
from sdc.hiframes.pd_dataframe_type import DataFrameType


class DataFrameGetitemAccessorType(types.Type):
    def __init__(self, dataframe, accessor):
        self.dataframe = dataframe
        self.accessor = accessor
        super(DataFrameGetitemAccessorType, self).__init__('DataFrameGetitemAccessorType({}, {})\
            '.format(dataframe, accessor))


@register_model(DataFrameGetitemAccessorType)
class DataFrameGetitemAccessorTypeModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('dataframe', fe_type.dataframe),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DataFrameGetitemAccessorType, 'dataframe', '_dataframe')


@intrinsic
def dataframe_getitem_accessor_init(typingctx, dataframe, accessor):

    if not (isinstance(dataframe, DataFrameType)
            and isinstance(accessor, types.StringLiteral)):
        return None, None

    def dataframe_getitem_accessor_init_codegen(context, builder, signature, args):
        dataframe_val, accessor_val = args
        getitem_accessor = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        getitem_accessor.dataframe = dataframe_val

        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], dataframe_val)

        return getitem_accessor._getvalue()

    ret_typ = DataFrameGetitemAccessorType(dataframe, accessor)
    sig = signature(ret_typ, dataframe, accessor)

    return sig, dataframe_getitem_accessor_init_codegen

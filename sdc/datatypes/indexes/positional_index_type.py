# -*- coding: utf-8 -*-
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

from numba import types
from numba.extending import (
    models,
    register_model,
    make_attribute_wrapper
)

from sdc.datatypes.indexes import RangeIndexType


class PositionalIndexType(types.IterableType):
    dtype = types.int64

    def __init__(self, is_named=False):
        self.data = RangeIndexType(is_named)
        self.is_named = is_named
        super(PositionalIndexType, self).__init__(
            name='PositionalIndexType({})'.format(is_named))

    @property
    def iterator_type(self):
        res = self.data.iterator_type
        return res


@register_model(PositionalIndexType)
class PositionalIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        members = [
            ('data', fe_type.data),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


# FIXME_Numba#3372: add into numba.types to allow returning from objmode
types.PositionalIndexType = PositionalIndexType


make_attribute_wrapper(PositionalIndexType, 'data', '_data')

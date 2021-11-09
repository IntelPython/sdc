# *****************************************************************************
# Copyright (c) 2021, Intel Corporation All rights reserved.
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
from numba.extending import (models, register_model, make_attribute_wrapper, )


# class StdStringViewType(types.IterableType):  # TO-DO: make iterable?
class StdStringViewType(types.Type):
    def __init__(self):
        super(StdStringViewType, self).__init__(
            name='StdStringViewType()'.format())


@register_model(StdStringViewType)
class StdStringViewModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        members = [
            ('data_ptr', types.voidptr),                        # pointer to std::string_view
            ('meminfo', types.MemInfoPointer(types.voidptr)),   # NRT meminfo managing this memory
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(StdStringViewType, 'data_ptr', '_data_ptr')

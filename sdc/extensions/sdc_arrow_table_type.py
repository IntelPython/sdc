# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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


# class ArrowTableType(IterableType):    # TO-DO: make iterable?
class ArrowTableType(types.Type):

    def __init__(self, dtypes, names):
        self.dtypes = dtypes
        self.names = names
        super(ArrowTableType, self).__init__(
            name='ArrowTableType(dtypes=({}), names=({}))'.format(
                ', '.join([str(d) for d in dtypes]),
                ', '.join([str(n) for n in names])))

#     @property
#     def iterator_type(self):
#         return ArrowTableIteratorType(self).iterator_type


@register_model(ArrowTableType)
class ArrowTableModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        members = [
            ('table_ptr', types.CPointer(types.uint8)),
            ('meminfo', types.MemInfoPointer(types.voidptr)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(ArrowTableType, 'table_ptr', '_table_ptr')


class PyarrowTableType(types.Type):

    def __init__(self):
        super(PyarrowTableType, self).__init__(
            name="PyarrowTableType()")


register_model(PyarrowTableType)(models.OpaqueModel)

# FIXME_Numba#3372: add into numba.types to allow returning from objmode
types.PyarrowTableType = PyarrowTableType

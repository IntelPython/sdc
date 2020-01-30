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
from numba.extending import (models, register_model, make_attribute_wrapper)
from sdc.str_ext import string_type


char_typ = types.uint8
offset_typ = types.uint32

data_ctypes_type = types.ArrayCTypes(types.Array(char_typ, 1, 'C'))
offset_ctypes_type = types.ArrayCTypes(types.Array(offset_typ, 1, 'C'))


class StringArray(object):
    def __init__(self, str_list):
        # dummy constructor
        self.num_strings = len(str_list)
        self.offsets = str_list
        self.data = str_list

    def __repr__(self):
        return 'StringArray({})'.format(self.data)


class StringArrayType(types.IterableType):
    def __init__(self):
        super(StringArrayType, self).__init__(
            name='StringArrayType()')

    @property
    def dtype(self):
        return string_type

    @property
    def iterator_type(self):
        return StringArrayIterator()

    def copy(self):
        return StringArrayType()


string_array_type = StringArrayType()


class StringArrayPayloadType(types.Type):
    def __init__(self):
        super(StringArrayPayloadType, self).__init__(
            name='StringArrayPayloadType()')


str_arr_payload_type = StringArrayPayloadType()


# XXX: C equivalent in _str_ext.cpp
@register_model(StringArrayPayloadType)
class StringArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('offsets', types.CPointer(offset_typ)),
            ('data', types.CPointer(char_typ)),
            ('null_bitmap', types.CPointer(char_typ)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)

str_arr_model_members = [
    ('num_items', types.uint64),
    ('num_total_chars', types.uint64),
    ('offsets', types.CPointer(offset_typ)),
    ('data', types.CPointer(char_typ)),
    ('null_bitmap', types.CPointer(char_typ)),
    ('meminfo', types.MemInfoPointer(str_arr_payload_type)),
]


make_attribute_wrapper(StringArrayType, 'null_bitmap', 'null_bitmap')


@register_model(StringArrayType)
class StringArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        models.StructModel.__init__(self, dmm, fe_type, str_arr_model_members)


class StringArrayIterator(types.SimpleIteratorType):
    """
    Type class for iterators of string arrays.
    """

    def __init__(self):
        name = "iter(String)"
        yield_type = string_type
        super(StringArrayIterator, self).__init__(name, yield_type)


@register_model(StringArrayIterator)
class StrArrayIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('array', string_array_type)]
        super(StrArrayIteratorModel, self).__init__(dmm, fe_type, members)


def is_str_arr_typ(typ):
    from sdc.hiframes.pd_series_ext import is_str_series_typ
    return typ == string_array_type or is_str_series_typ(typ)

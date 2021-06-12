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
from numba.core.types import IterableType, SimpleIterableType, SimpleIteratorType
from numba.extending import (models, register_model, make_attribute_wrapper, )

from collections.abc import MutableMapping


class ConcDictIteratorType(SimpleIteratorType):
    def __init__(self, iterable):
        self.parent = iterable.parent
        self.iterable = iterable
        yield_type = iterable.yield_type
        name = "iter[{}->{}],{}".format(
            iterable.parent, yield_type, iterable.name
        )
        super(ConcDictIteratorType, self).__init__(name, yield_type)


class ConcDictKeysIterableType(SimpleIterableType):
    """Concurrent Dictionary iterable type for .keys()
    """

    def __init__(self, parent):
        assert isinstance(parent, ConcurrentDictType)
        self.parent = parent
        self.yield_type = self.parent.key_type
        name = "keys[{}]".format(self.parent.name)
        self.name = name
        iterator_type = ConcDictIteratorType(self)
        super(ConcDictKeysIterableType, self).__init__(name, iterator_type)


class ConcDictItemsIterableType(SimpleIterableType):
    """Concurrent Dictionary iterable type for .items()
    """

    def __init__(self, parent):
        assert isinstance(parent, ConcurrentDictType)
        self.parent = parent
        self.yield_type = self.parent.keyvalue_type
        name = "items[{}]".format(self.parent.name)
        self.name = name
        iterator_type = ConcDictIteratorType(self)
        super(ConcDictItemsIterableType, self).__init__(name, iterator_type)


class ConcDictValuesIterableType(SimpleIterableType):
    """Concurrent Dictionary iterable type for .values()
    """

    def __init__(self, parent):
        assert isinstance(parent, ConcurrentDictType)
        self.parent = parent
        self.yield_type = self.parent.value_type
        name = "values[{}]".format(self.parent.name)
        self.name = name
        iterator_type = ConcDictIteratorType(self)
        super(ConcDictValuesIterableType, self).__init__(name, iterator_type)


@register_model(ConcDictItemsIterableType)
@register_model(ConcDictKeysIterableType)
@register_model(ConcDictValuesIterableType)
@register_model(ConcDictIteratorType)
class ConcDictIterModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('parent', fe_type.parent),  # reference to the dict
            ('state', types.voidptr),    # iterator state in native code
            ('meminfo', types.MemInfoPointer(types.voidptr)),   # meminfo for allocated iter state
        ]
        super(ConcDictIterModel, self).__init__(dmm, fe_type, members)


class ConcurrentDictType(IterableType):
    def __init__(self, keyty, valty):
        self.key_type = keyty
        self.value_type = valty
        self.keyvalue_type = types.Tuple([keyty, valty])
        super(ConcurrentDictType, self).__init__(
            name='ConcurrentDictType({}, {})'.format(keyty, valty))

    @property
    def iterator_type(self):
        return ConcDictKeysIterableType(self).iterator_type


@register_model(ConcurrentDictType)
class ConcurrentDictModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        members = [
            ('data_ptr', types.CPointer(types.uint8)),
            ('meminfo', types.MemInfoPointer(types.voidptr)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(ConcurrentDictType, 'data_ptr', '_data_ptr')


class ConcurrentDict(MutableMapping):
    def __new__(cls, dcttype=None, meminfo=None):
        return object.__new__(cls)

    @classmethod
    def empty(cls, key_type, value_type):
        return cls(dcttype=ConcurrentDictType(key_type, value_type))

    @classmethod
    def from_arrays(cls, keys, values):
        return cls(dcttype=ConcurrentDictType(keys.dtype, values.dtype))

    @classmethod
    def fromkeys(cls, keys, value):
        return cls(dcttype=ConcurrentDictType(keys.dtype, value))

    def __init__(self, **kwargs):
        if kwargs:
            self._dict_type, self._opaque = self._parse_arg(**kwargs)
        else:
            self._dict_type = None

    @property
    def _numba_type_(self):
        if self._dict_type is None:
            raise TypeError("invalid operation on untyped dictionary")
        return self._dict_type

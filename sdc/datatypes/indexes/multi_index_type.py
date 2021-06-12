# -*- coding: utf-8 -*-
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
from numba.extending import (
    models,
    register_model,
    make_attribute_wrapper,
    typeof_impl,
)
from numba.core.typing.typeof import _typeof_type as numba_typeof_type


class MultiIndexIteratorType(types.SimpleIteratorType):
    def __init__(self, iterable):
        self.parent = iterable
        yield_type = iterable.dtype
        name = "iter[{}->{}],{}".format(
            iterable, yield_type, iterable.name
        )
        super(MultiIndexIteratorType, self).__init__(name, yield_type)


@register_model(MultiIndexIteratorType)
class MultiIndexIterModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('parent', fe_type.parent),                 # reference to the index object
            ('state', types.CPointer(types.int64)),     # iterator state (i.e. counter)
        ]
        super(MultiIndexIterModel, self).__init__(dmm, fe_type, members)


class MultiIndexType(types.IterableType):

    def __init__(self, levels, codes, is_named=False):
        self.levels = levels
        self.codes = codes
        self.is_named = is_named
        super(MultiIndexType, self).__init__(
            name='MultiIndexType({}, {}, {})'.format(levels, codes, is_named))

    @property
    def iterator_type(self):
        return MultiIndexIteratorType(self).iterator_type

    @property
    def dtype(self):
        nlevels = len(self.levels)
        levels_types = [self.levels.dtype] * nlevels if isinstance(self.levels, types.UniTuple) else self.levels
        return types.Tuple.from_types([level.dtype for level in levels_types])

    @property
    def nlevels(self):
        return len(self.levels)

    @property
    def levels_types(self):
        if isinstance(self.levels, types.UniTuple):
            return [self.levels.dtype] * self.levels.count

        return self.levels

    @property
    def codes_types(self):
        if isinstance(self.codes, types.UniTuple):
            return [self.codes.dtype] * self.codes.count

        return self.codes


@register_model(MultiIndexType)
class MultiIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        levels_type = fe_type.levels
        codes_type = fe_type.codes
        name_type = types.unicode_type if fe_type.is_named else types.none  # TO-DO: change to types.Optional
        members = [
            ('levels', levels_type),
            ('codes', codes_type),
            ('name', name_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(MultiIndexType, 'levels', '_levels')
make_attribute_wrapper(MultiIndexType, 'codes', '_codes')
make_attribute_wrapper(MultiIndexType, 'name', '_name')


#### FIXME: move below to one common place:
 
# FIXME_Numba#6781: due to overlapping of overload_methods for Numba TypeRef
# we have to use our new SdcTypeRef to type objects created from types.Type
# (i.e. ConcurrentDict meta-type). This should be removed once it's fixed.
class SdcTypeRef(types.Dummy):
    """Reference to a type.
    Used when a type is passed as a value.
    """
    def __init__(self, instance_type):
        self.instance_type = instance_type
        super(SdcTypeRef, self).__init__('sdc_typeref[{}]'.format(self.instance_type))
 
 
@register_model(SdcTypeRef)
class SdcTypeRefModel(models.OpaqueModel):
    def __init__(self, dmm, fe_type):
 
        models.OpaqueModel.__init__(self, dmm, fe_type)
 

import pandas as pd
@typeof_impl.register(type)
def mynew_typeof_type(val, c):
    """ This function is a workaround for """

    # print("DEBUG: val=", val)
    if not issubclass(val, pd.MultiIndex):
    # if not issubclass(val, MultiIndex):
        return numba_typeof_type(val, c)
    else:
        return SdcTypeRef(MultiIndexType)

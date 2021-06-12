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

import pandas as pd

from numba.core import types
from numba.extending import (models, register_model, typeof_impl, )
from numba.core.typing.typeof import _typeof_type as numba_typeof_type

from sdc.extensions.sdc_hashmap_type import ConcurrentDict, ConcurrentDictType
from sdc.datatypes.indexes import MultiIndexType


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


@typeof_impl.register(type)
def mynew_typeof_type(val, c):
    """ This function is a workaround for """

    if issubclass(val, ConcurrentDict):
        return SdcTypeRef(ConcurrentDictType)
    elif issubclass(val, pd.MultiIndex):
        return SdcTypeRef(MultiIndexType)
    else:
        return numba_typeof_type(val, c)

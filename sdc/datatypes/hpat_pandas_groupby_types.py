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


import numba
from numba import types, cgutils
from numba.extending import (models, register_model, make_attribute_wrapper)
from numba.typed import Dict, List
from sdc.str_ext import string_type


class DataFrameGroupByType(types.Type):
    """
    Type definition for DataFrameGroupBy functions handling.
    """

    def __init__(self, parent, col_id):
        self.parent = parent
        self.col_id = col_id
        super(DataFrameGroupByType, self).__init__(
            name="DataFrameGroupByType({}, {})".format(parent, col_id))

    @property
    def key(self):
        return self.parent, self.col_id


@register_model(DataFrameGroupByType)
class DataFrameGroupByModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        by_series_dtype = fe_type.parent.data[fe_type.col_id.literal_value].dtype
        ty_data = types.containers.DictType(
            by_series_dtype,
            types.containers.ListType(types.int64)
        )
        members = [
            ('parent', fe_type.parent),
            ('col_id', types.int64),
            ('data', ty_data),
            ('sort', types.bool_)
        ]
        super(DataFrameGroupByModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameGroupByType, 'parent', '_parent')
make_attribute_wrapper(DataFrameGroupByType, 'col_id', '_col_id')
make_attribute_wrapper(DataFrameGroupByType, 'data', '_data')
make_attribute_wrapper(DataFrameGroupByType, 'sort', '_sort')

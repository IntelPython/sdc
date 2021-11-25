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

import re
from typing import NamedTuple

import numba
from numba import types
from numba.core import cgutils
from numba.extending import (models, register_model, make_attribute_wrapper)

from sdc.str_ext import string_type
from sdc.hiframes.pd_series_type import SeriesType


class DataFrameType(types.Type):  # TODO: IterableType over column names
    """Temporary type class for DataFrame objects.
    """

    def __init__(self, data=None, index=None, columns=None, has_parent=False, column_loc=None):
        self.data = data
        if index is None:
            index = types.none
        self.index = index
        self.columns = columns
        # keeping whether it is unboxed from Python to enable reflection of new
        # columns
        self.has_parent = has_parent
        self.column_loc = column_loc
        super(DataFrameType, self).__init__(
            name="DataFrameType({}, {}, {}, {})".format(data, index, columns, has_parent))

    def copy(self, index=None, has_parent=None):
        # XXX is copy necessary?
        if index is None:
            index = types.none if self.index == types.none else self.index.copy()
        data = tuple(a.copy() for a in self.data)
        if has_parent is None:
            has_parent = self.has_parent
        return DataFrameType(data, index, self.columns, has_parent)

    @property
    def key(self):
        # needed?
        return self.data, self.index, self.columns, self.has_parent

    def get_series_type(self, col_idx):
        col_type = self.data[col_idx]
        return SeriesType(col_type.dtype, col_type, self.index, is_named=True)

    def unify(self, typingctx, other):
        if (isinstance(other, DataFrameType)
                and len(other.data) == len(self.data)
                and other.columns == self.columns
                and other.has_parent == self.has_parent):
            new_index = types.none
            if self.index != types.none and other.index != types.none:
                new_index = self.index.unify(typingctx, other.index)
            elif other.index != types.none:
                new_index = other.index
            elif self.index != types.none:
                new_index = self.index

            data = tuple(a.unify(typingctx, b) for a, b in zip(self.data, other.data))
            return DataFrameType(data, new_index, self.columns, self.has_parent)

    def is_precise(self):
        return all(a.is_precise() for a in self.data) and self.index.is_precise()

    def __repr__(self):

        # To have correct repr of DataFrame we need some changes to what types.Type gives:
        # (1) e.g. array(int64, 1d, C) should be Array(int64, 1, 'C')
        # (2) ColumnLoc is not part of DataFrame name, so we need to add it
        default_repr = super(DataFrameType, self).__repr__()
        res = re.sub(r'array\((\w+), 1d, C\)', r'Array(\1, 1, \'C\')', default_repr)
        res = re.sub(r'\)$', f', column_loc={self.column_loc})', res)
        return res


@register_model(DataFrameType)
class DataFrameModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        types_unique = set()
        df_types = []
        for col_type in fe_type.data:
            if col_type in types_unique:
                continue
            types_unique.add(col_type)
            df_types.append(col_type)

        members = [
            ('data', types.Tuple([types.List(typ) for typ in df_types])),
            ('index', fe_type.index),
            ('parent', types.pyobject),
        ]
        super(DataFrameModel, self).__init__(dmm, fe_type, members)


class ColumnLoc(NamedTuple):
    type_id: int
    col_id: int


# FIXME_Numba#3372: add into numba.types to allow returning from objmode
types.DataFrameType = DataFrameType
types.ColumnLoc = ColumnLoc

make_attribute_wrapper(DataFrameType, 'data', '_data')
make_attribute_wrapper(DataFrameType, 'index', '_index')
make_attribute_wrapper(DataFrameType, 'parent', '_parent')

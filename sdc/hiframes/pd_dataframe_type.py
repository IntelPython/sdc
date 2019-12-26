# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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

from sdc.str_ext import string_type


class DataFrameType(types.Type):  # TODO: IterableType over column names
    """Temporary type class for DataFrame objects.
    """

    def __init__(self, data=None, index=None, columns=None, has_parent=False):
        self.data = data
        if index is None:
            index = types.none
        self.index = index
        self.columns = columns
        # keeping whether it is unboxed from Python to enable reflection of new
        # columns
        self.has_parent = has_parent
        super(DataFrameType, self).__init__(
            name="dataframe({}, {}, {}, {})".format(data, index, columns, has_parent))

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


@register_model(DataFrameType)
class DataFrameModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        n_cols = len(fe_type.columns)
        members = [
            ('data', types.Tuple(fe_type.data)),
            ('index', fe_type.index),
            ('columns', types.UniTuple(string_type, n_cols)),
            # for lazy unboxing of df coming from Python (usually argument)
            # list of flags noting which columns and index are unboxed
            # index flag is last
            ('unboxed', types.UniTuple(types.int8, n_cols + 1)),
            ('parent', types.pyobject),
        ]
        super(DataFrameModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameType, 'data', '_data')
make_attribute_wrapper(DataFrameType, 'index', '_index')
make_attribute_wrapper(DataFrameType, 'columns', '_columns')
make_attribute_wrapper(DataFrameType, 'unboxed', '_unboxed')
make_attribute_wrapper(DataFrameType, 'parent', '_parent')

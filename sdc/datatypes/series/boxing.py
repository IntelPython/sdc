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

from numba.core.imputils import lower_constant
from numba.core import cgutils, types

from .types import SeriesType
from sdc.datatypes.indexes.positional_index_type import PositionalIndexType
from sdc.hiframes.boxing import _unbox_index_data
from sdc.extensions.indexes.range_index_ext import unbox_range_index
from sdc.datatypes.indexes.range_index_type import RangeIndexDataType


@lower_constant(SeriesType)
def constant_Series(context, builder, ty, pyval):
    """
    Create a constant Series.

    See @unbox(SeriesType)
    """
    series = cgutils.create_struct_proxy(ty)(context, builder)
    series.data = _constant_Series_data(context, builder, ty, pyval)

    # TODO: index and name (this only handles PositionalIndexType(False)
    # and repeats unboxing, need to refactor to support all indexes)
    native_range_index = cgutils.create_struct_proxy(RangeIndexDataType)(context, builder)
    native_range_index.start = context.get_constant(types.int64, pyval.index.start)
    native_range_index.stop = context.get_constant(types.int64, pyval.index.stop)
    native_range_index.step = context.get_constant(types.int64, pyval.index.step)

    range_index = cgutils.create_struct_proxy(ty.index.data)(context, builder)
    range_index.data = native_range_index._getvalue()

    positional_index = cgutils.create_struct_proxy(PositionalIndexType(False))(context, builder)
    positional_index.data = range_index._getvalue()

    series.index = positional_index._getvalue()
    return series._getvalue()


def _constant_Series_data(context, builder, ty, pyval):
    """
    Create a constant for Series data.

    Mostly reuses constant creation for pandas arrays.
    """

    from ..categorical.types import CategoricalDtypeType

    # TO-DO: this requires lower_constant to be implemented for other types
    # like indices and so on, until that raise NotImplementedError
    if (isinstance(ty.dtype, CategoricalDtypeType)
            and ty.index is PositionalIndexType(False)
            and ty.is_named is False):
        from ..categorical.boxing import constant_Categorical
        return constant_Categorical(context, builder, ty.data, pyval.array)

    raise NotImplementedError()

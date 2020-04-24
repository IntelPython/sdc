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

from numba.targets.imputils import lower_constant
from numba import cgutils

from .types import SeriesType


@lower_constant(SeriesType)
def constant_Series(context, builder, ty, pyval):
    """
    Create a constant Series.

    See @unbox(SeriesType)
    """
    series = cgutils.create_struct_proxy(ty)(context, builder)
    series.data = _constant_Series_data(context, builder, ty, pyval)
    # TODO: index and name
    return series._getvalue()


def _constant_Series_data(context, builder, ty, pyval):
    """
    Create a constant for Series data.

    Mostly reuses constant creation for pandas arrays.
    """

    from ..categorical.types import CategoricalDtypeType

    if isinstance(ty.dtype, CategoricalDtypeType):
        from ..categorical.boxing import constant_Categorical
        return constant_Categorical(context, builder, ty.data, pyval.array)

    raise NotImplementedError()

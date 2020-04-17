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

import pandas as pd

import numba
from numba import types, cgutils
from numba.extending import (
    models,
    register_model,
    lower_cast,
    lower_builtin,
    infer_getattr,
    type_callable,
    infer,
    overload)

import sdc
from sdc.hiframes.pd_categorical_ext import PDCategoricalDtype

from sdc.str_arr_ext import string_array_type
from sdc.str_ext import string_type, list_string_array_type

from sdc.hiframes.pd_series_type import SeriesType


def is_str_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == string_type


def is_dt64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPDatetime('ns')


def series_to_array_type(typ, replace_boxed=False):
    return typ.data


def arr_to_series_type(arr):
    series_type = None
    if isinstance(arr, types.Array):
        series_type = SeriesType(arr.dtype, arr)
    elif arr == string_array_type:
        # StringArray is readonly
        series_type = SeriesType(string_type)
    elif arr == list_string_array_type:
        series_type = SeriesType(types.List(string_type))
    return series_type


def if_series_to_array_type(typ, replace_boxed=False):
    if isinstance(typ, SeriesType):
        return series_to_array_type(typ, replace_boxed)

    if isinstance(typ, (types.Tuple, types.UniTuple)):
        return types.Tuple(
            [if_series_to_array_type(t, replace_boxed) for t in typ.types])
    if isinstance(typ, types.List):
        return types.List(if_series_to_array_type(typ.dtype, replace_boxed))
    if isinstance(typ, types.Set):
        return types.Set(if_series_to_array_type(typ.dtype, replace_boxed))
    # TODO: other types that can have Series inside?
    return typ


def if_arr_to_series_type(typ):
    if isinstance(typ, (types.Tuple, types.UniTuple)):
        return types.Tuple([if_arr_to_series_type(t) for t in typ.types])
    if isinstance(typ, types.List):
        return types.List(if_arr_to_series_type(typ.dtype))
    if isinstance(typ, types.Set):
        return types.Set(if_arr_to_series_type(typ.dtype))
    # TODO: other types that can have Arrays inside?
    return typ


@overload(pd.Series)
def pd_series_overload(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series

    Limitations
    -----------
    - Parameters ``dtype`` and ``copy`` are currently unsupported.
    - Types iterable and dict as ``data`` parameter are currently unsupported.

    Examples
    --------
    Create Series with data [1, 2, 3] and index ['A', 'B', 'C'].

    >>> pd.Series([1, 2, 3], ['A', 'B', 'C'])

    .. seealso::

        :ref:`DataFrame <pandas.DataFrame>`
            DataFrame constructor.
    """

    is_index_none = isinstance(index, types.NoneType) or index is None

    def hpat_pandas_series_ctor_impl(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):

        '''' use binop here as otherwise Numba's dead branch pruning doesn't work
        TODO: replace with 'if not is_index_none' when resolved '''
        if is_index_none == False:  # noqa
            fix_index = sdc.hiframes.api.fix_df_array(index)
        else:
            fix_index = index

        return sdc.hiframes.api.init_series(sdc.hiframes.api.fix_df_array(data), fix_index, name)

    return hpat_pandas_series_ctor_impl

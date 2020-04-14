# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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


import numpy as np

import numba
from numba import generated_jit, ir, ir_utils, typeinfer, types
from numba.extending import overload
import sdc
from sdc import distributed, distributed_analysis

from sdc.str_arr_ext import (string_array_type, str_arr_set_na)
from sdc.hiframes.pd_categorical_ext import CategoricalArray


def write_send_buff(shuffle_meta, node_id, i, val, data):
    return i


def setitem_arr_nan(arr, ind):
    arr[ind] = np.nan


@overload(setitem_arr_nan)
def setitem_arr_nan_overload(arr, ind):
    if isinstance(arr.dtype, types.Float):
        return setitem_arr_nan

    if isinstance(arr.dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = arr.dtype('NaT')

        def _setnan_impl(arr, ind):
            arr[ind] = nat
        return _setnan_impl

    if arr == string_array_type:
        return lambda arr, ind: str_arr_set_na(arr, ind)
    # TODO: support strings, bools, etc.
    # XXX: set NA values in bool arrays to False
    # FIXME: replace with proper NaN
    if arr.dtype == types.bool_:
        def b_set(arr, ind):
            arr[ind] = False
        return b_set

    if isinstance(arr, CategoricalArray):
        def setitem_arr_nan_cat(arr, ind):
            int_arr = sdc.hiframes.pd_categorical_ext.cat_array_to_int(arr)
            int_arr[ind] = -1
        return setitem_arr_nan_cat

    # XXX set integer NA to 0 to avoid unexpected errors
    # TODO: convert integer to float if nan
    if isinstance(arr.dtype, types.Integer):
        def setitem_arr_nan_int(arr, ind):
            arr[ind] = 0
        return setitem_arr_nan_int
    return lambda arr, ind: None

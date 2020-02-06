# -*- coding: utf-8 -*-
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

"""

| This file contains SDC numpy modified functions, rewritten according to concurrency

"""

import numba
import numpy

from numba import types, jit, prange, numpy_support
from numba.errors import TypingError

import sdc
from sdc.utilities.sdc_typing_utils import TypeChecker
from sdc.str_arr_ext import (StringArrayType, pre_alloc_string_array, get_utf8_size)
from sdc.utilities.utils import sdc_overload, sdc_register_jitable


def astype(self, dtype):
    pass


@sdc_overload(astype)
def sdc_astype_overload(self, dtype):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel replacement of numpy.astype.

    .. only:: developer
       Test: python -m sdc.runtests sdc.tests.test_sdc_numpy -k astype

    """

    if not isinstance(self, types.Array):
        return None

    if not isinstance(dtype, (types.StringLiteral, types.UnicodeType, types.Function, types.functions.NumberClass)):
        ty_checker.raise_exc(dtype, 'string or type', 'dtype')

    if (
        (isinstance(dtype, types.Function) and dtype.typing_key == str) or
        (isinstance(dtype, types.StringLiteral) and dtype.literal_value == 'str')
    ):
        def sdc_astype_number_to_string_impl(self, dtype):
            num_bytes = 0
            arr_len = len(self)

            # Get total bytes for new array
            for i in prange(arr_len):
                item = self[i]
                num_bytes += get_utf8_size(str(item))

            data = pre_alloc_string_array(arr_len, num_bytes)

            for i in range(arr_len):
                item = self[i]
                data[i] = str(item)  # TODO: check NA

            return data

        return sdc_astype_number_to_string_impl

    if (isinstance(self, types.Array) and isinstance(dtype, (types.StringLiteral, types.functions.NumberClass))):
        def sdc_astype_number_impl(self, dtype):
            arr = numpy.empty(len(self), dtype=numpy.dtype(dtype))
            for i in numba.prange(len(self)):
                arr[i] = self[i]

            return arr

        return sdc_astype_number_impl

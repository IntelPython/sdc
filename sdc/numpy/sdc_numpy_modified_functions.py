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

from numba.errors import TypingError
from numba import types
from numba import jit
from numba import prange
from numba import numpy_support

import sdc
from sdc.datatypes.common_functions import TypeChecker
from sdc.str_arr_ext import (StringArrayType, pre_alloc_string_array)
from sdc.utils import sdc_overload


def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
    pass


@sdc_overload(astype, jit_options={'parallel': True})
def sdc_astype_overload(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
    """
    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************
    Parallel analogue of numpy 'astype' implementation.

    .. only:: developer

       Test: python -m sdc.runtests -k sdc.tests.test_sdc_numpy.TestArrays.test_astype*

    """

    if not isinstance(self, types.Array):
        return None

    if not isinstance(dtype, (types.StringLiteral, types.UnicodeType, types.Function, types.functions.NumberClass)):
        ty_checker.raise_exc(dtype, 'string or type', 'dtype')

    if not isinstance(order, (str, types.Omitted, types.StringLiteral, types.UnicodeType)):
        ty_checker.raise_exc(order, 'string', 'order')

    if not isinstance(casting, (str, types.Omitted, types.StringLiteral, types.UnicodeType)):
        ty_checker.raise_exc(casting, 'string', 'casting')

    if not isinstance(subok, (bool, types.Omitted, types.Boolean)):
        ty_checker.raise_exc(subok, 'boolean', 'subok')

    if not isinstance(copy, (bool, types.Omitted, types.Boolean)):
        ty_checker.raise_exc(copy, 'boolean', 'copy')

    if ((isinstance(dtype, types.Function) and dtype.typing_key == str)
        or (isinstance(dtype, types.StringLiteral) and dtype.literal_value == 'str')):
        def sdc_astype_number_to_string_impl(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
            num_chars = 0
            arr_len = len(self)

            # Get total chars for new array
            for i in prange(arr_len):
                item = self[i]
                num_chars += len(str(item))  # TODO: check NA

            data = pre_alloc_string_array(arr_len, num_chars)

            for i in range(arr_len):
                item = self[i]
                data[i] = str(item)  # TODO: check NA

            return data

        return sdc_astype_number_to_string_impl

    if (isinstance(self, types.Array) and isinstance(dtype, types.functions.NumberClass)):
        other_numba_dtype = dtype.instance_type
        numba_self = self.dtype
        def sdc_astype_number_impl(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
            if numba_self == other_numba_dtype:
                return self
            arr = numpy.empty(len(self), dtype=dtype)
            for i in numba.prange(len(self)):
                arr[i] = self[i]

            return arr

        return sdc_astype_number_impl

    if (isinstance(self, types.Array) and isinstance(dtype, (types.StringLiteral, types.UnicodeType))):
        other_numpy_dtype = numpy.dtype(dtype.literal_value)
        other_numba_dtype = numpy_support.from_dtype(other_numpy_dtype)
        numba_self = self.dtype
        def sdc_astype_number_literal_impl(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
            if other_numba_dtype == numba_self:
                return self

            arr = numpy.empty(len(self), dtype=numpy.dtype(dtype))
            for i in numba.prange(len(self)):
                arr[i] = self[i]

            return arr

        return sdc_astype_number_literal_impl

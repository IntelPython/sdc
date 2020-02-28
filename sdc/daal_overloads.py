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
import ctypes as ct

from numba import types
from numba.extending import overload

# from numba import typing, generated_jit
# from numba.extending import models, register_model
# from numba.extending import lower_builtin, overload_method, intrinsic

# from llvmlite import ir as lir
import llvmlite.binding as ll

from . import daal


ll.add_symbol('test', daal.test)
ll.add_symbol('sum', daal.sum)


_test = types.ExternalFunction("test", types.int_(types.int_))
_sum = types.ExternalFunction("sum", types.float64(types.voidptr, types.int_))


def test(x):
    pass


@overload(test)
def test_overload(x):
    return lambda x: _test(x)


functype_test = ct.CFUNCTYPE(ct.c_int, ct.c_int)
ctypes_test = functype_test(daal.test)

# functype_sum = ct.CFUNCTYPE(ct.c_double, ct.POINTER(ct.c_double), ct.c_int)
functype_sum = ct.CFUNCTYPE(ct.c_double, ct.c_void_p, ct.c_int)
ctypes_sum = functype_sum(daal.sum)


median = ct.CFUNCTYPE(ct.c_double, ct.c_int, ct.c_void_p)(daal.median)
quantile = ct.CFUNCTYPE(ct.c_double, ct.c_int, ct.c_void_p, ct.c_double)(daal.quantile)

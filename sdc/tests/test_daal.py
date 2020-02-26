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

import numpy as np
import ctypes

from sdc.tests.test_base import TestCase

from sdc.daal_overloads import test, ctypes_test, ctypes_sum, quantile


class TestDaal(TestCase):

    def test_test(self):
        def pyfunc():
            return test(10)

        def ctypes_pyfunc():
            return ctypes_test(10)

        cfunc = self.jit(pyfunc)
        ctypes_cfunc = self.jit(ctypes_pyfunc)
        # self.assertEqual(cfunc(), pyfunc())
        self.assertEqual(cfunc(), ctypes_pyfunc())
        self.assertEqual(ctypes_cfunc(), ctypes_pyfunc())

    def test_sum(self):
        def pyfunc(arr):
            # return ctypes_sum(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(arr))
            return ctypes_sum(arr.ctypes, len(arr))
        cfunc = self.jit(pyfunc)

        arr = np.arange(10, dtype=np.float64)
        expected = np.sum(arr)

        # print(ctypes_sum.argtypes)

        self.assertEqual(pyfunc(arr), expected)
        self.assertEqual(cfunc(arr), expected)

    def test_quantile(self):
        def pyfunc(arr, q):
            return quantile(len(arr), arr.ctypes, q)
        cfunc = self.jit(pyfunc)

        arr = np.arange(10, dtype=np.float64)

        # print(ctypes_sum.argtypes)

        for q in [0., 0.25, 0.5, 0.75, 1.]:
            with self.subTest(q=q):
                expected = np.quantile(arr, q)
                self.assertEqual(pyfunc(arr, q), expected)
                self.assertEqual(cfunc(arr, q), expected)

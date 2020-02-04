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


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sdc
import unittest

from sdc.str_arr_ext import StringArray
from sdc.str_ext import std_str_to_unicode, unicode_to_std_str
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import skip_numba_jit
from sdc.functions.numpy_like import astype


class TestArrays(TestCase):

    def test_astype_to_num(self):
        def ref_impl(a, t):
            return a.astype(t)

        def sdc_impl(a, t):
            return astype(a, t)

        sdc_func = self.jit(sdc_impl)

        cases = [[5, 2, 0, 333, -4], [3.3, 5.4, np.nan]]
        cases_type = [np.float64, np.int64]
        for case in cases:
            a = np.array(case)
            for type_ in cases_type:
                with self.subTest(data=case, type=type_):
                    np.testing.assert_array_equal(sdc_func(a, type_), ref_impl(a, type_))

    @unittest.skip("Need implement support for non-literal dtype parameter")
    def test_astype_to_num_unbox(self):
        def ref_impl(a, t):
            return a.astype(t)

        def sdc_impl(a, t):
            return astype(a, t)

        sdc_func = self.jit(sdc_impl)

        cases = [[5, 2, 0, 333, -4], [3.3, 5.4, 3.4]]
        cases_type = ['float64', 'int64', 'str']
        for case in cases:
            a = np.array(case)
            for type_ in cases_type:
                with self.subTest(data=case, type=type_):
                    np.testing.assert_array_equal(sdc_func(a, type_), ref_impl(a, type_))

    def test_astype_to_float(self):
        def ref_impl(a):
            return a.astype('float64')

        def sdc_impl(a):
            return astype(a, 'float64')

        sdc_func = self.jit(sdc_impl)

        cases = [[2, 3, 0], [4., 5.6, np.nan]]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_astype_to_int(self):
        def ref_impl(a):
            return a.astype(np.int64)

        def sdc_impl(a):
            return astype(a, np.int64)

        sdc_func = self.jit(sdc_impl)

        cases = [[2, 3, 0], [4., 5.6, np.nan]]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_astype_int_to_str(self):
        def ref_impl(a):
            return a.astype(str)

        def sdc_impl(a):
            return astype(a, str)

        sdc_func = self.jit(sdc_impl)

        a = np.array([2, 3, 0])
        np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    @unittest.skip('Numba converts float to string with incorrect precision')
    def test_astype_float_to_str(self):
        def ref_impl(a):
            return a.astype(str)

        def sdc_impl(a):
            return astype(a, str)

        sdc_func = self.jit(sdc_impl)

        a = np.array([4., 5.6, np.nan])
        np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_astype_num_to_str(self):
        def ref_impl(a):
            return a.astype('str')

        def sdc_impl(a):
            return astype(a, 'str')

        sdc_func = self.jit(sdc_impl)

        a = np.array([5, 2, 0, 333, -4])
        np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    @unittest.skip('Needs Numba astype impl support converting unicode_type to other type')
    def test_astype_str_to_num(self):
        def ref_impl(a, t):
            return a.astype(t)

        def sdc_impl(a, t):
            return astype(a, t)

        sdc_func = self.jit(sdc_impl)

        cases = [['a', 'cc', 'd'], ['3.3', '5', '.4'], ['¡Y', 'tú quién ', 'te crees']]
        cases_type = [np.float64, np.int64]
        for case in cases:
            a = np.array(case)
            for type_ in cases_type:
                with self.subTest(data=case, type=type_):
                    np.testing.assert_array_equal(sdc_func(a, type_), ref_impl(a, type_))

if __name__ == "__main__":
    unittest.main()

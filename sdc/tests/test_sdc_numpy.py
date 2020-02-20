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
from sdc.functions import numpy_like


class TestArrays(TestCase):

    def test_astype_to_num(self):
        def ref_impl(a, t):
            return a.astype(t)

        def sdc_impl(a, t):
            return numpy_like.astype(a, t)

        sdc_func = self.jit(sdc_impl)

        cases = [[5, 2, 0, 333, -4], [3.3, 5.4, np.nan]]
        cases_type = [np.float64, np.int64, 'float64', 'int64']
        for case in cases:
            a = np.array(case)
            for type_ in cases_type:
                with self.subTest(data=case, type=type_):
                    np.testing.assert_array_equal(sdc_func(a, type_), ref_impl(a, type_))

    def test_astype_to_float(self):
        def ref_impl(a):
            return a.astype('float64')

        def sdc_impl(a):
            return numpy_like.astype(a, 'float64')

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
            return numpy_like.astype(a, np.int64)

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
            return numpy_like.astype(a, str)

        sdc_func = self.jit(sdc_impl)

        a = np.array([2, 3, 0])
        np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    @unittest.skip('Numba converts float to string with incorrect precision')
    def test_astype_float_to_str(self):
        def ref_impl(a):
            return a.astype(str)

        def sdc_impl(a):
            return numpy_like.astype(a, str)

        sdc_func = self.jit(sdc_impl)

        a = np.array([4., 5.6, np.nan])
        np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_astype_num_to_str(self):
        def ref_impl(a):
            return a.astype('str')

        def sdc_impl(a):
            return numpy_like.astype(a, 'str')

        sdc_func = self.jit(sdc_impl)

        a = np.array([5, 2, 0, 333, -4])
        np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    @unittest.skip('Needs Numba astype impl support converting unicode_type to other type')
    def test_astype_str_to_num(self):
        def ref_impl(a, t):
            return a.astype(t)

        def sdc_impl(a, t):
            return numpy_like.astype(a, t)

        sdc_func = self.jit(sdc_impl)

        cases = [['a', 'cc', 'd'], ['3.3', '5', '.4'], ['¡Y', 'tú quién ', 'te crees']]
        cases_type = [np.float64, np.int64]
        for case in cases:
            a = np.array(case)
            for type_ in cases_type:
                with self.subTest(data=case, type=type_):
                    np.testing.assert_array_equal(sdc_func(a, type_), ref_impl(a, type_))

    def test_isnan(self):
        def ref_impl(a):
            return np.isnan(a)

        def sdc_impl(a):
            return numpy_like.isnan(a)

        sdc_func = self.jit(sdc_impl)

        cases = [[5, 2, 0, 333, -4], [3.3, 5.4, np.nan, 7.9, np.nan]]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    @unittest.skip('Needs provide String Array boxing')
    def test_isnan_str(self):
        def ref_impl(a):
            return np.isnan(a)

        def sdc_impl(a):
            return numpy_like.isnan(a)

        sdc_func = self.jit(sdc_impl)

        cases = [['a', 'cc', np.nan], ['se', None, 'vvv']]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_notnan(self):
        def ref_impl(a):
            return np.invert(np.isnan(a))

        def sdc_impl(a):
            return numpy_like.notnan(a)

        sdc_func = self.jit(sdc_impl)

        cases = [[5, 2, 0, 333, -4], [3.3, 5.4, np.nan, 7.9, np.nan]]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_copy(self):
        def ref_impl(a):
            return np.copy(a)

        def sdc_impl(a):
            return numpy_like.copy(a)

        sdc_func = self.jit(sdc_impl)

        cases = [[5, 2, 0, 333, -4], [3.3, 5.4, np.nan, 7.9, np.nan], [True, False, True], ['a', 'vv', 'o12oo']]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_copy_int(self):
        def ref_impl():
            a = np.array([5, 2, 0, 333, -4])
            return np.copy(a)

        def sdc_impl():
            a = np.array([5, 2, 0, 333, -4])
            return numpy_like.copy(a)

        sdc_func = self.jit(sdc_impl)
        np.testing.assert_array_equal(sdc_func(), ref_impl())

    def test_copy_bool(self):
        def ref_impl():
            a = np.array([True, False, True])
            return np.copy(a)

        def sdc_impl():
            a = np.array([True, False, True])
            return numpy_like.copy(a)

        sdc_func = self.jit(sdc_impl)
        np.testing.assert_array_equal(sdc_func(), ref_impl())

    @unittest.skip("Numba doesn't have string array")
    def test_copy_str(self):
        def ref_impl():
            a = np.array(['a', 'vv', 'o12oo'])
            return np.copy(a)

        def sdc_impl():
            a = np.array(['a', 'vv', 'o12oo'])
            return numpy_like.copy(a)

        sdc_func = self.jit(sdc_impl)
        np.testing.assert_array_equal(sdc_func(), ref_impl())

    def test_argmin(self):
        def ref_impl(a):
            return np.argmin(a)

        def sdc_impl(a):
            return numpy_like.argmin(a)

        sdc_func = self.jit(sdc_impl)

        cases = [[5, 2, 0, 333, -4], [3.3, 5.4, np.nan, 7.9, np.nan]]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_argmax(self):
        def ref_impl(a):
            return np.argmax(a)

        def sdc_impl(a):
            return numpy_like.argmax(a)

        sdc_func = self.jit(sdc_impl)

        cases = [[np.nan, np.nan, np.inf, np.nan], [5, 2, 0, 333, -4], [3.3, 5.4, np.nan, 7.9, np.nan]]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_nanargmin(self):
        def ref_impl(a):
            return np.nanargmin(a)

        def sdc_impl(a):
            return numpy_like.nanargmin(a)

        sdc_func = self.jit(sdc_impl)

        cases = [[5, 2, 0, 333, -4], [3.3, 5.4, np.nan, 7.9, np.nan]]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))

    def test_nanargmax(self):
        def ref_impl(a):
            return np.nanargmax(a)

        def sdc_impl(a):
            return numpy_like.nanargmax(a)

        sdc_func = self.jit(sdc_impl)

        cases = [[np.nan, np.nan, np.inf, np.nan], [5, 2, -9, 333, -4], [3.3, 5.4, np.nan, 7.9]]
        for case in cases:
            a = np.array(case)
            with self.subTest(data=case):
                np.testing.assert_array_equal(sdc_func(a), ref_impl(a))


class TestArrayReductions(TestCase):

    def check_reduction_basic(self, pyfunc, alt_pyfunc, all_nans=True, comparator=None):
        if not comparator:
            comparator = np.testing.assert_array_equal

        alt_cfunc = self.jit(alt_pyfunc)

        def cases():
            yield np.array([5, 2, 0, 333, -4])
            yield np.array([3.3, 5.4, np.nan, 7.9, np.nan])
            yield np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5])
            yield np.float64([-0.0, -1.5])
            yield np.float64([-1.5, 2.5, 'inf'])
            yield np.float64([-1.5, 2.5, '-inf'])
            yield np.float64([-1.5, 2.5, 'inf', '-inf'])
            yield np.float64(['nan', -1.5, 2.5, 'nan', 3.0])
            yield np.float64(['nan', -1.5, 2.5, 'nan', 'inf', '-inf', 3.0])
            if all_nans:
                # Only NaNs
                yield np.float64(['nan', 'nan'])

        for case in cases():
            with self.subTest(data=case):
                comparator(alt_cfunc(case), pyfunc(case))

    def test_nanmean(self):
        def ref_impl(a):
            return np.nanmean(a)

        def sdc_impl(a):
            return numpy_like.nanmean(a)

        self.check_reduction_basic(ref_impl, sdc_impl)

    def test_nanmin(self):
        def ref_impl(a):
            return np.nanmin(a)

        def sdc_impl(a):
            return numpy_like.nanmin(a)

        self.check_reduction_basic(ref_impl, sdc_impl)

    def test_nanmax(self):
        def ref_impl(a):
            return np.nanmax(a)

        def sdc_impl(a):
            return numpy_like.nanmax(a)

        self.check_reduction_basic(ref_impl, sdc_impl)

    def test_nanprod(self):
        def ref_impl(a):
            return np.nanprod(a)

        def sdc_impl(a):
            return numpy_like.nanprod(a)

        self.check_reduction_basic(ref_impl, sdc_impl)

    def test_nansum(self):
        def ref_impl(a):
            return np.nansum(a)

        def sdc_impl(a):
            return numpy_like.nansum(a)

        self.check_reduction_basic(ref_impl, sdc_impl)

    def test_nanvar(self):
        def ref_impl(a):
            return np.nanvar(a)

        def sdc_impl(a):
            return numpy_like.nanvar(a)

        self.check_reduction_basic(ref_impl, sdc_impl,
                                   comparator=np.testing.assert_array_almost_equal)

    def test_sum(self):
        def ref_impl(a):
            return np.sum(a)

        def sdc_impl(a):
            return numpy_like.sum(a)

        self.check_reduction_basic(ref_impl, sdc_impl)

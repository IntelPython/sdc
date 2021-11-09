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
from itertools import product

from sdc.str_arr_ext import StringArray
from sdc.tests.test_base import TestCase
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
        from sdc.str_arr_ext import StringArray

        def ref_impl(a):
            return np.copy(a)

        @self.jit
        def sdc_func(a):
            _a = StringArray(a) if as_str_arr == True else a  # noqa
            return numpy_like.copy(_a)

        cases = {
            'int': [5, 2, 0, 333, -4],
            'float': [3.3, 5.4, np.nan, 7.9, np.nan],
            'bool': [True, False, True],
            'str': ['a', 'vv', 'o12oo']
        }

        for dtype, data in cases.items():
            a = data if dtype == 'str' else np.asarray(data)
            as_str_arr = True if dtype == 'str' else False
            with self.subTest(case=data):
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

    def test_sort(self):
        np.random.seed(0)

        def ref_impl(a, kind):
            return np.sort(a, kind=kind)

        def sdc_impl(a, kind):
            numpy_like.sort(a, kind=kind)
            return a

        sdc_func = self.jit(sdc_impl)

        float_array = np.random.ranf(10**2)
        int_arryay = np.random.randint(0, 127, 10**2)

        for kind in [None, 'quicksort', 'mergesort']:
            float_cases = ['float32', 'float64']
            for case in float_cases:
                array0 = float_array.astype(case)
                array1 = np.copy(array0)
                with self.subTest(data=case, kind=kind):
                    np.testing.assert_array_equal(ref_impl(array0, kind), sdc_func(array1, kind))

            int_cases = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            for case in int_cases:
                array0 = int_arryay.astype(case)
                array1 = np.copy(array0)
                with self.subTest(data=case, kind=kind):
                    np.testing.assert_array_equal(ref_impl(array0, kind), sdc_func(array1, kind))

    def test_argsort(self):
        np.random.seed(0)

        def ref_impl(a, kind):
            return np.argsort(a, kind=kind)

        def sdc_impl(a, kind):
            return numpy_like.argsort(a, kind=kind)

        def run_test(ref_impl, sdc_impl, data, kind):
            if kind == 'mergesort':
                np.testing.assert_array_equal(ref_impl(data, kind), sdc_func(data, kind))
            else:
                sorted_ref = data[ref_impl(data, kind)]
                sorted_sdc = data[sdc_impl(data, kind)]
                np.testing.assert_array_equal(sorted_ref, sorted_sdc)

        sdc_func = self.jit(sdc_impl)

        float_arrays = [np.random.ranf(10**5),
                        np.random.ranf(10**4)]

        # make second float array to contain nan in every second element
        for i in range(len(float_arrays[1])//2):
            float_arrays[1][i*2] = np.nan

        int_arrays = [np.random.randint(0, 2, 10**6 + 1),
                      np.ones(10**5 + 1, dtype=np.int64),
                      np.random.randint(0, 255, 10**4)]

        for kind in [None, 'quicksort', 'mergesort']:
            float_cases = ['float32', 'float64']
            for case in float_cases:
                for float_array in float_arrays:
                    data = float_array.astype(case)
                    with self.subTest(data=case, kind=kind, size=len(float_array)):
                        run_test(ref_impl, sdc_func, data, kind)

            int_cases = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            for case in int_cases:
                for int_array in int_arrays:
                    data = int_array.astype(case)
                    array0 = np.copy(data)
                    array1 = np.copy(data)
                    with self.subTest(data=case, kind=kind, size=len(int_array)):
                        run_test(ref_impl, sdc_func, data, kind)

    def test_argsort_param_ascending(self):

        def ref_impl(a, kind, ascending):
            return pd.Series(a).sort_values(kind=kind, ascending=ascending).index

        def sdc_impl(a, kind, ascending):
            return numpy_like.argsort(a, kind=kind, ascending=ascending)

        def run_test(ref_impl, sdc_impl, data, kind, ascending):
            if kind == 'mergesort':
                np.testing.assert_array_equal(
                    ref_impl(data, kind, ascending),
                    sdc_func(data, kind, ascending))
            else:
                sorted_ref = data[ref_impl(data, kind, ascending)]
                sorted_sdc = data[sdc_impl(data, kind, ascending)]
                np.testing.assert_array_equal(sorted_ref, sorted_sdc)

        sdc_func = self.jit(sdc_impl)

        n = 100
        np.random.seed(0)
        data_values = {
            'float': [np.inf, np.NINF, np.nan, 0, -1, 2.1, 2/3, -3/4, 0.777],
            'int': [1, -1, 3, 5, -60, 21, 22, 23],
        }
        all_dtypes = {
            'float': ['float32', 'float64'],
            'int': ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
        }

        for kind, ascending in product([None, 'quicksort', 'mergesort'], [True, False]):
            for dtype_group, arr_values in data_values.items():
                for dtype in all_dtypes[dtype_group]:
                    data = np.random.choice(arr_values, n).astype(dtype)
                    with self.subTest(data=data, kind=kind, ascending=ascending):
                        run_test(ref_impl, sdc_func, data, kind, ascending)


    def _test_fillna_numeric(self, pyfunc, cfunc, inplace):
        data_to_test = [
            [True, False, False, True, True],
            [5, 2, 0, 333, -4],
            [3.3, 5.4, 7.9],
            [3.3, 5.4, np.nan, 7.9, np.nan],
        ]
        values_to_test = [
            None,
            np.nan,
            2.1,
            2
        ]

        for data, value in product(data_to_test, values_to_test):
            a1 = np.asarray(data)
            a2 = pd.Series(np.copy(a1)) if inplace else pd.Series(a1)

            with self.subTest(data=data, value=value):
                result = cfunc(a1, value)
                result_ref = pyfunc(a2, value)
                if inplace:
                    result, result_ref = a1, a2
                np.testing.assert_array_equal(result, result_ref)

    def test_fillna_numeric_inplace_false(self):
        def ref_impl(S, value):
            if value is None:
                return S.values.copy()
            else:
                return S.fillna(value=value, inplace=False).values

        def sdc_impl(a, value):
            return numpy_like.fillna(a, inplace=False, value=value)
        sdc_func = self.jit(sdc_impl)

        self._test_fillna_numeric(ref_impl, sdc_func, inplace=False)

    def test_fillna_numeric_inplace_true(self):
        def ref_impl(S, value):
            if value is None:
                return None
            else:
                S.fillna(value=value, inplace=True)
                return None

        def sdc_impl(a, value):
            return numpy_like.fillna(a, inplace=True, value=value)
        sdc_func = self.jit(sdc_impl)

        self._test_fillna_numeric(ref_impl, sdc_func, inplace=True)

    def test_fillna_str_inplace_false(self):
        def ref_impl(S, value):
            if value is None:
                return S.values.copy()
            else:
                return S.fillna(value=value, inplace=False).values

        def sdc_impl(S, value):
            str_arr = S.values
            return numpy_like.fillna(str_arr, inplace=False, value=value)
        sdc_func = self.jit(sdc_impl)

        data_to_test = [
            ['a', 'b', 'c', 'd'],
            ['a', 'b', None, 'c', None, 'd'],
        ]
        values_to_test = [
            None,
            '',
            'asd'
        ]
        for data, value in product(data_to_test, values_to_test):
            S = pd.Series(data)
            with self.subTest(data=data, value=value):
                result = sdc_func(S, value)
                result_ref = ref_impl(S, value)

                # FIXME: str_arr unifies None with np.nan and StringArray boxing always return np.nan
                # to avoid mismatch in results for fill value == None use custom comparing func
                def is_same_unify_nones(a, b):
                    return a == b or ((a is None or np.isnan(a)) and (b is None or np.isnan(b)))
                cmp_result = np.asarray(
                    list(map(is_same_unify_nones, result, result_ref))
                )
                self.assertEqual(np.all(cmp_result), True)


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

    def test_cumsum(self):
        def ref_impl(a):
            return np.cumsum(a)

        def sdc_impl(a):
            return numpy_like.cumsum(a)

        self.check_reduction_basic(ref_impl, sdc_impl)

    def test_nancumsum(self):
        def ref_impl(a):
            return np.nancumsum(a)

        def sdc_impl(a):
            return numpy_like.nancumsum(a)

        self.check_reduction_basic(ref_impl, sdc_impl)

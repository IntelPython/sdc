# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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


import unittest
import platform
import sdc
import numba
import numpy as np
import pandas as pd
from sdc import *
from numba.typed import Dict
from collections import defaultdict
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import skip_numba_jit


class TestHpatJitIssues(TestCase):

    @unittest.skip("Dict is not supported as class member")
    def test_class_with_dict(self):
        @jitclass([('d', Dict)])
        class ClassWithDict:
            def __init__(self):
                self.d = Dict.empty(key_type=int32, value_type=int32)

        @numba.njit
        def test_impl():
            c = ClassWithDict()

            c.d[0] = 1

            return c.d[0]

        test_impl()

    @unittest.skip("Type infer from __init__ is not supported")
    def test_class_from_init(self):
        @jitclass()
        class ClassWithInt:
            def __init__(self):
                self.i = 0

        @numba.njit
        def test_impl():
            c = ClassWithInt()

            print(c.i)

        test_impl()

    @unittest.skip("list.sort with lambda is not supported")
    def test_list_sort_lambda(self):
        @numba.njit
        def sort_with_list_and_lambda():
            data = [5, 4, 3, 2, 1, 0]

            data.sort(key=lambda x: x)

            return data

        sort_with_list_and_lambda()

    @unittest.skip("list.sort with key is not supported")
    def test_list_sort_with_func(self):
        @numba.njit
        def key_func(x):
            return x

        @numba.njit
        def sort_with_list():
            data = [5, 4, 3, 2, 1, 0]

            data.sort(key=key_func)

            return data

        sort_with_list()

    @unittest.skip("sorted with lambda is not supported")
    def test_sorted_lambda(self):
        @numba.njit
        def sorted_with_list():
            data = [5, 4, 3, 2, 1, 0]

            sorted(data, key=lambda x: x)

            return data

        sorted_with_list()

    @unittest.skip("sorted with key is not supported")
    def test_sorted_with_func(self):
        @numba.njit
        def key_func(x):
            return x

        @numba.njit
        def sorted_with_list():
            data = [5, 4, 3, 2, 1, 0]

            sorted(data, key=key_func)

            return data

        sorted_with_list()

    @unittest.skip("iterate over tuple is not supported")
    def test_iterate_over_tuple(self):
        @numba.njit
        def func_iterate_over_tuple():
            t = ('1', 1, 1.)

            for i in t:
                print(i)

        func_iterate_over_tuple()

    @unittest.skip("try/except is not supported")
    def test_with_try_except(self):
        @numba.njit
        def func_with_try_except():
            try:
                return 0
            except BaseException:
                return 1

        func_with_try_except()

    @unittest.skip("raise is not supported")
    def test_with_raise(self):
        @numba.njit
        def func_with_raise(b):
            if b:
                return b
            else:
                raise "error"

        func_with_raise(True)

    @unittest.skip("defaultdict is not supported")
    def test_default_dict(self):
        @numba.njit
        def func_with_dict():
            d = defaultdict(int)

            return d['a']

        func_with_dict()

    @unittest.skip('TODO: needs different integer typing in Numba\n'
                   'AssertionError - Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_series_binop_int_casting(self):
        def test_impl(A):
            res = A + 42
            return res.dtype
        hpat_func = sdc.jit(test_impl)

        A = np.ones(1, dtype='int32')
        self.assertEqual(hpat_func(A), test_impl(A))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_box1_issue(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        hpat_func = sdc.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column1_issue(self):
        # set existing column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
            df['A'] = np.arange(n)
            return df

        hpat_func = sdc.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column_reflect4(self):
        # set existing column
        def test_impl(df, n):
            df['A'] = np.arange(n)

        hpat_func = sdc.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
        df2 = df1.copy()
        hpat_func(df1, n)
        test_impl(df2, n)
        pd.testing.assert_frame_equal(df1, df2)

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column_new_type1(self):
        # set existing column with a new type
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n) + 3.0})
            df['A'] = np.arange(n)
            return df

        hpat_func = sdc.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column2(self):
        # create new column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n) + 1.0})
            df['C'] = np.arange(n)
            return df

        hpat_func = sdc.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column_reflect3(self):
        # create new column
        def test_impl(df, n):
            df['C'] = np.arange(n)

        hpat_func = sdc.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
        df2 = df1.copy()
        hpat_func(df1, n)
        test_impl(df2, n)
        pd.testing.assert_frame_equal(df1, df2)

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_series_op2_issue(self):
        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')

        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = sdc.jit(test_impl)

            n = 11
            df = pd.DataFrame({'A': np.arange(1, n)})
            pd.testing.assert_series_equal(hpat_func(df.A, 1), test_impl(df.A, 1), check_names=False)

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_series_op5_integer_scalar_issue(self):
        arithmetic_methods = ('add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow')

        for method in arithmetic_methods:
            test_impl = _make_func_use_method_arg1(method)
            hpat_func = sdc.jit(test_impl)

            n = 11
            operand_series = pd.Series(np.arange(1, n))
            operand_scalar = 10
            pd.testing.assert_series_equal(
                hpat_func(operand_series, operand_scalar),
                test_impl(operand_series, operand_scalar),
                check_names=False)

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_series_fusion1_issue(self):
        def test_impl(A, B):
            return A + B + 1
        hpat_func = sdc.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 1)

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_series_fusion2_issue(self):
        # make sure getting data var avoids incorrect single def assumption
        def test_impl(A, B):
            S = B + 2
            if A[0] == 0:
                S = A + 1
            return S + B
        hpat_func = sdc.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 3)

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows',
                     'AssertionError: Attributes are different'
                     'Attribute "dtype" are different'
                     '[left]:  int64'
                     '[right]: int32')
    def test_csv1_issue(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int},
                               )
        hpat_func = sdc.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows',
                     'AssertionError: Attributes are different'
                     'Attribute "dtype" are different'
                     '[left]:  int64'
                     '[right]: int32')
    def test_csv_keys1_issue(self):
        def test_impl():
            dtype = {'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int}
            return pd.read_csv("csv_data1.csv",
                               names=dtype.keys(),
                               dtype=dtype,
                               )
        hpat_func = sdc.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows',
                     'AssertionError: Attributes are different'
                     'Attribute "dtype" are different'
                     '[left]:  int64'
                     '[right]: int32')
    def test_csv_const_dtype1_issue(self):
        def test_impl():
            dtype = {'A': 'int', 'B': 'float64', 'C': 'float', 'D': 'int64'}
            return pd.read_csv("csv_data1.csv",
                               names=dtype.keys(),
                               dtype=dtype,
                               )
        hpat_func = sdc.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows',
                     'AssertionError: Attributes are different'
                     'Attribute "dtype" are different'
                     '[left]:  int64'
                     '[right]: int32')
    def test_csv_skip1_issue(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int},
                               skiprows=2,
                               )
        hpat_func = sdc.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows',
                     'AssertionError: Attributes are different'
                     'Attribute "dtype" are different'
                     '[left]:  int64'
                     '[right]: int32')
    def test_csv_date1_issue(self):
        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int},
                               parse_dates=[2])
        hpat_func = sdc.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows',
                     'AssertionError: Attributes are different'
                     'Attribute "dtype" are different'
                     '[left]:  int64'
                     '[right]: int32')
    def test_csv_str1_issue(self):
        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int})
        hpat_func = sdc.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())


if __name__ == "__main__":
    unittest.main()

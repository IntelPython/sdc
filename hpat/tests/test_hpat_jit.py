import unittest
import platform
import hpat
import numba
import numpy as np
import pandas as pd
from hpat import *
from numba.typed import Dict
from collections import defaultdict


class TestHpatJitIssues(unittest.TestCase):

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
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

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
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

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
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

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
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

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
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

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
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())


if __name__ == "__main__":
    unittest.main()

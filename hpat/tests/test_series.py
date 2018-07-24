import unittest
import pandas as pd
import numpy as np
import random
import string
import pyarrow.parquet as pq
import numba
import hpat
from hpat import hiframes_sort
from hpat.str_arr_ext import StringArray
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs, dist_IR_contains)

class TestSeries(unittest.TestCase):
    def test_create1(self):
        def test_impl():
            df = pd.DataFrame({'A': [1,2,3]})
            return (df.A == 1).sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_create2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)})
            return (df.A == 2).sum()

        n = 11
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_create_str(self):
        def test_impl():
            df = pd.DataFrame({'A': ['a', 'b', 'c']})
            return (df.A == 'a').sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_pass_df1(self):
        def test_impl(df):
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_pass_df_str(self):
        def test_impl(df):
            return (df.A == 'a').sum()

        df = pd.DataFrame({'A': ['a', 'b', 'c']})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_pass_series1(self):
        # TODO: check to make sure it is series type
        def test_impl(A):
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_pass_series2(self):
        # test creating dataframe from passed series
        def test_impl(A):
            df = pd.DataFrame({'A': A})
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_pass_series_str(self):
        def test_impl(A):
            return (A == 'a').sum()

        df = pd.DataFrame({'A': ['a', 'b', 'c']})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_series_attr1(self):
        def test_impl(A):
            return A.size

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_series_attr2(self):
        def test_impl(A):
            return A.copy().values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr3(self):
        def test_impl(A):
            return A.min()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_series_attr4(self):
        def test_impl(A):
            return A.cumsum().values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    @unittest.skip("needs argsort fix in canonicalize_array_math")
    def test_series_attr5(self):
        def test_impl(A):
            return A.argsort().values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr6(self):
        def test_impl(A):
            return A.take([2,3]).values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_np_call_on_series1(self):
        def test_impl(A):
            return np.min(A)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_values1(self):
        def test_impl(A):
            return (A == 2).values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_static_setitem_series1(self):
        def test_impl(A):
            A[0] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_setitem_series1(self):
        def test_impl(A, i):
            A[i] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_setitem_series2(self):
        def test_impl(A, i):
            A[i] = 100
            # TODO: remove return after aliasing fix
            return A

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        hpat_func = hpat.jit(test_impl)
        hpat_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_static_getitem_series1(self):
        def test_impl(A):
            return A[0]

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_getitem_series1(self):
        def test_impl(A, i):
            return A[i]

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_list_convert(self):
        def test_impl():
            df = pd.DataFrame({'one': np.array([-1, np.nan, 2.5]),
                        'two': ['foo', 'bar', 'baz'],
                        'three': [True, False, True]})
            return df.one.values, df.two.values, df.three.values

        hpat_func = hpat.jit(test_impl)
        one, two, three = hpat_func()
        self.assertTrue(isinstance(one, np.ndarray))
        self.assertTrue(isinstance(two,  np.ndarray))
        self.assertTrue(isinstance(three, np.ndarray))

if __name__ == "__main__":
    unittest.main()

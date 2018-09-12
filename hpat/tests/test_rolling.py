import unittest
import itertools
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import numba
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains)
from hpat.hiframes_rolling import supported_rolling_funcs

class TestRolling(unittest.TestCase):
    def test_fixed1(self):
        # test sequentially with manually created dfs
        # all functions except apply
        for func_name in supported_rolling_funcs[:-1]:
            func_text = "def test_impl(df, w, c):\n  return df.rolling(w, center=c).{}()\n".format(func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            wins = (2, 3, 5)
            centers = (False, True)
            for args in itertools.product(wins, centers):
                df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
                pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))
                df = pd.DataFrame({'B': [0, 1, 2, -2, 4]})
                pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))

    def test_fixed2(self):
        # test sequentially with generated dfs
        # all functions except apply
        for func_name in supported_rolling_funcs[:-1]:
            func_text = "def test_impl(df, w, c):\n  return df.rolling(w, center=c).{}()\n".format(func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            sizes = (1, 2, 10, 11, 121, 1000)
            wins = (2, 3, 5)
            centers = (False, True)
            for n, w, c in itertools.product(sizes, wins, centers):
                df = pd.DataFrame({'B': np.arange(n)})
                pd.testing.assert_frame_equal(hpat_func(df, w, c), test_impl(df, w, c))

    def test_fixed_apply1(self):
        # test sequentially with manually created dfs
            def test_impl(df, w, c):
                return df.rolling(w, center=c).apply(lambda a: a.sum())
            hpat_func = hpat.jit(test_impl)
            wins = (2, 3, 5)
            centers = (False, True)
            for args in itertools.product(wins, centers):
                df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
                pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))
                df = pd.DataFrame({'B': [0, 1, 2, -2, 4]})
                pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))

    def test_fixed_apply2(self):
        # test sequentially with generated dfs
        def test_impl(df, w, c):
            return df.rolling(w, center=c).apply(lambda a: a.sum())
        hpat_func = hpat.jit(test_impl)
        sizes = (1, 2, 10, 11, 121, 1000)
        wins = (2, 3, 5)
        centers = (False, True)
        for n, w, c in itertools.product(sizes, wins, centers):
            df = pd.DataFrame({'B': np.arange(n)})
            pd.testing.assert_frame_equal(hpat_func(df, w, c), test_impl(df, w, c))

    def test_fixed_parallel1(self):
        def test_impl(n, w, center):
            df = pd.DataFrame({'B': np.arange(n)})
            R = df.rolling(w, center=center).sum()
            return R.B.sum()

        hpat_func = hpat.jit(test_impl)
        sizes = (1, 2, 10, 11, 121, 1000)
        wins = (2, 4, 5, 10, 11)
        centers = (False, True)
        for args in itertools.product(sizes, wins, centers):
            self.assertEqual(hpat_func(*args), test_impl(*args),
                             "rolling fixed window with {}".format(args))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_fixed_parallel_apply1(self):
        def test_impl(n, w, center):
            df = pd.DataFrame({'B': np.arange(n)})
            R = df.rolling(w, center=center).apply(lambda a: a.sum())
            return R.B.sum()

        hpat_func = hpat.jit(test_impl)
        sizes = (1, 2, 10, 11, 121, 1000)
        wins = (2, 4, 5, 10, 11)
        centers = (False, True)
        for args in itertools.product(sizes, wins, centers):
            self.assertEqual(hpat_func(*args), test_impl(*args),
                             "rolling fixed window with {}".format(args))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

if __name__ == "__main__":
    unittest.main()

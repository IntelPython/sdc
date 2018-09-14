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

    def test_variable1(self):
        # test sequentially with manually created dfs
        df1 = pd.DataFrame({'B': [0, 1, 2, np.nan, 4],
                'time': [pd.Timestamp('20130101 09:00:00'),
                        pd.Timestamp('20130101 09:00:02'),
                        pd.Timestamp('20130101 09:00:03'),
                        pd.Timestamp('20130101 09:00:05'),
                        pd.Timestamp('20130101 09:00:06')]})
        df2 = pd.DataFrame({'B': [0, 1, 2, -2, 4],
                    'time': [pd.Timestamp('20130101 09:00:01'),
                        pd.Timestamp('20130101 09:00:02'),
                        pd.Timestamp('20130101 09:00:03'),
                        pd.Timestamp('20130101 09:00:04'),
                        pd.Timestamp('20130101 09:00:09')]})
        wins = ('1s', '2s', '3s', '4s')
        # all functions except apply
        for w, func_name in itertools.product(wins, supported_rolling_funcs[:-1]):
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').{}()\n".format(w, func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            pd.testing.assert_frame_equal(hpat_func(df1), test_impl(df1))
            pd.testing.assert_frame_equal(hpat_func(df2), test_impl(df2))

    def test_variable2(self):
        # test sequentially with generated dfs
        wins = ('1s', '2s', '3s', '4s')
        sizes = (1, 2, 10, 11, 121, 1000)
        # all functions except apply
        for w, n, func_name in itertools.product(wins, sizes, supported_rolling_funcs[:-1]):
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').{}()\n".format(w, func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            time = pd.date_range(start='1/1/2018', periods=n, freq='s')
            df = pd.DataFrame({'B': np.arange(n), 'time': time})
            pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_variable_apply1(self):
        # test sequentially with manually created dfs
        df1 = pd.DataFrame({'B': [0, 1, 2, np.nan, 4],
                'time': [pd.Timestamp('20130101 09:00:00'),
                        pd.Timestamp('20130101 09:00:02'),
                        pd.Timestamp('20130101 09:00:03'),
                        pd.Timestamp('20130101 09:00:05'),
                        pd.Timestamp('20130101 09:00:06')]})
        df2 = pd.DataFrame({'B': [0, 1, 2, -2, 4],
                    'time': [pd.Timestamp('20130101 09:00:01'),
                        pd.Timestamp('20130101 09:00:02'),
                        pd.Timestamp('20130101 09:00:03'),
                        pd.Timestamp('20130101 09:00:04'),
                        pd.Timestamp('20130101 09:00:09')]})
        wins = ('1s', '2s', '3s', '4s')
        # all functions except apply
        for w in wins:
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').apply(lambda a: a.sum())\n".format(w)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            pd.testing.assert_frame_equal(hpat_func(df1), test_impl(df1))
            pd.testing.assert_frame_equal(hpat_func(df2), test_impl(df2))

    def test_variable_apply2(self):
        # test sequentially with generated dfs
        wins = ('1s', '2s', '3s', '4s')
        sizes = (1, 2, 10, 11, 121, 1000)
        # all functions except apply
        for w, n in itertools.product(wins, sizes):
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').apply(lambda a: a.sum())\n".format(w)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            time = pd.date_range(start='1/1/2018', periods=n, freq='s')
            df = pd.DataFrame({'B': np.arange(n), 'time': time})
            pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_variable_parallel1(self):
        wins = ('1s', '2s', '3s', '4s')
        # XXX: Pandas returns time = [np.nan] for size==1 for some reason
        sizes = (2, 10, 11, 121, 1000)
        # all functions except apply
        for w, n, func_name in itertools.product(wins, sizes, supported_rolling_funcs[:-1]):
            func_text = "def test_impl(n):\n"
            func_text += "  df = pd.DataFrame({'B': np.arange(n), 'time': "
            func_text += "    pd.DatetimeIndex(np.arange(n) * 1000000000)})\n"
            func_text += "  res = df.rolling('{}', on='time').{}()\n".format(w, func_name)
            func_text += "  return res.B.sum()\n"
            loc_vars = {}
            exec(func_text, {'pd': pd, 'np': np}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

if __name__ == "__main__":
    unittest.main()

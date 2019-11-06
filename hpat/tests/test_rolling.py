import unittest
import itertools
import os
import pandas as pd
import platform
import numpy as np
import numba
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                   count_parfor_OneDs, count_array_OneDs, dist_IR_contains)
from hpat.hiframes.rolling import supported_rolling_funcs

LONG_TEST = (int(os.environ['SDC_LONG_ROLLING_TEST']) != 0
             if 'SDC_LONG_ROLLING_TEST' in os.environ else False)

test_funcs = ('mean', 'max',)
if LONG_TEST:
    # all functions except apply, cov, corr
    test_funcs = supported_rolling_funcs[:-3]


class TestRolling(unittest.TestCase):
    def test_fixed1(self):
        # test sequentially with manually created dfs
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)

        for func_name in test_funcs:
            func_text = "def test_impl(df, w, c):\n  return df.rolling(w, center=c).{}()\n".format(func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)

            for args in itertools.product(wins, centers):
                df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
                pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))
                df = pd.DataFrame({'B': [0, 1, 2, -2, 4]})
                pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))

    def test_fixed2(self):
        # test sequentially with generated dfs
        sizes = (121,)
        wins = (3,)
        if LONG_TEST:
            sizes = (1, 2, 10, 11, 121, 1000)
            wins = (2, 3, 5)
        centers = (False, True)
        for func_name in test_funcs:
            func_text = "def test_impl(df, w, c):\n  return df.rolling(w, center=c).{}()\n".format(func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            for n, w, c in itertools.product(sizes, wins, centers):
                df = pd.DataFrame({'B': np.arange(n)})
                pd.testing.assert_frame_equal(hpat_func(df, w, c), test_impl(df, w, c))

    def test_fixed_apply1(self):
        # test sequentially with manually created dfs
        def test_impl(df, w, c):
            return df.rolling(w, center=c).apply(lambda a: a.sum())
        hpat_func = hpat.jit(test_impl)
        wins = (3,)
        if LONG_TEST:
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
        sizes = (121,)
        wins = (3,)
        if LONG_TEST:
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
        sizes = (121,)
        wins = (5,)
        if LONG_TEST:
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
        sizes = (121,)
        wins = (5,)
        if LONG_TEST:
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
        wins = ('2s',)
        if LONG_TEST:
            wins = ('1s', '2s', '3s', '4s')
        # all functions except apply
        for w, func_name in itertools.product(wins, test_funcs):
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').{}()\n".format(w, func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            # XXX: skipping min/max for this test since the behavior of Pandas
            # is inconsistent: it assigns NaN to last output instead of 4!
            if func_name not in ('min', 'max'):
                pd.testing.assert_frame_equal(hpat_func(df1), test_impl(df1))
            pd.testing.assert_frame_equal(hpat_func(df2), test_impl(df2))

    def test_variable2(self):
        # test sequentially with generated dfs
        wins = ('2s',)
        sizes = (121,)
        if LONG_TEST:
            wins = ('1s', '2s', '3s', '4s')
            sizes = (1, 2, 10, 11, 121, 1000)
        # all functions except apply
        for w, func_name in itertools.product(wins, test_funcs):
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').{}()\n".format(w, func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            for n in sizes:
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
        wins = ('2s',)
        if LONG_TEST:
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
        wins = ('2s',)
        sizes = (121,)
        if LONG_TEST:
            wins = ('1s', '2s', '3s', '4s')
            # TODO: this crashes on Travis (3 process config) with size 1
            sizes = (2, 10, 11, 121, 1000)
        # all functions except apply
        for w in wins:
            func_text = "def test_impl(df):\n  return df.rolling('{}', on='time').apply(lambda a: a.sum())\n".format(w)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            for n in sizes:
                time = pd.date_range(start='1/1/2018', periods=n, freq='s')
                df = pd.DataFrame({'B': np.arange(n), 'time': time})
                pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @unittest.skipIf(platform.system() == 'Windows', "ValueError: time must be monotonic")
    def test_variable_parallel1(self):
        wins = ('2s',)
        sizes = (121,)
        if LONG_TEST:
            wins = ('1s', '2s', '3s', '4s')
            # XXX: Pandas returns time = [np.nan] for size==1 for some reason
            sizes = (2, 10, 11, 121, 1000)
        # all functions except apply
        for w, func_name in itertools.product(wins, test_funcs):
            func_text = "def test_impl(n):\n"
            func_text += "  df = pd.DataFrame({'B': np.arange(n), 'time': "
            func_text += "    pd.DatetimeIndex(np.arange(n) * 1000000000)})\n"
            func_text += "  res = df.rolling('{}', on='time').{}()\n".format(w, func_name)
            func_text += "  return res.B.sum()\n"
            loc_vars = {}
            exec(func_text, {'pd': pd, 'np': np}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            for n in sizes:
                np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skipIf(platform.system() == 'Windows', "ValueError: time must be monotonic")
    def test_variable_apply_parallel1(self):
        wins = ('2s',)
        sizes = (121,)
        if LONG_TEST:
            wins = ('1s', '2s', '3s', '4s')
            # XXX: Pandas returns time = [np.nan] for size==1 for some reason
            sizes = (2, 10, 11, 121, 1000)
        # all functions except apply
        for w in wins:
            func_text = "def test_impl(n):\n"
            func_text += "  df = pd.DataFrame({'B': np.arange(n), 'time': "
            func_text += "    pd.DatetimeIndex(np.arange(n) * 1000000000)})\n"
            func_text += "  res = df.rolling('{}', on='time').apply(lambda a: a.sum())\n".format(w)
            func_text += "  return res.B.sum()\n"
            loc_vars = {}
            exec(func_text, {'pd': pd, 'np': np}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            for n in sizes:
                np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_series_fixed1(self):
        # test series rolling functions
        # all functions except apply
        S1 = pd.Series([0, 1, 2, np.nan, 4])
        S2 = pd.Series([0, 1, 2, -2, 4])
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)
        for func_name in test_funcs:
            func_text = "def test_impl(S, w, c):\n  return S.rolling(w, center=c).{}()\n".format(func_name)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = hpat.jit(test_impl)
            for args in itertools.product(wins, centers):
                pd.testing.assert_series_equal(hpat_func(S1, *args), test_impl(S1, *args))
                pd.testing.assert_series_equal(hpat_func(S2, *args), test_impl(S2, *args))
        # test apply

        def apply_test_impl(S, w, c):
            return S.rolling(w, center=c).apply(lambda a: a.sum())
        hpat_func = hpat.jit(apply_test_impl)
        for args in itertools.product(wins, centers):
            pd.testing.assert_series_equal(hpat_func(S1, *args), apply_test_impl(S1, *args))
            pd.testing.assert_series_equal(hpat_func(S2, *args), apply_test_impl(S2, *args))

    def test_series_cov1(self):
        # test series rolling functions
        # all functions except apply
        S1 = pd.Series([0, 1, 2, np.nan, 4])
        S2 = pd.Series([0, 1, 2, -2, 4])
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)

        def test_impl(S, S2, w, c):
            return S.rolling(w, center=c).cov(S2)
        hpat_func = hpat.jit(test_impl)
        for args in itertools.product([S1, S2], [S1, S2], wins, centers):
            pd.testing.assert_series_equal(hpat_func(*args), test_impl(*args))
            pd.testing.assert_series_equal(hpat_func(*args), test_impl(*args))

        def test_impl2(S, S2, w, c):
            return S.rolling(w, center=c).corr(S2)
        hpat_func = hpat.jit(test_impl2)
        for args in itertools.product([S1, S2], [S1, S2], wins, centers):
            pd.testing.assert_series_equal(hpat_func(*args), test_impl2(*args))
            pd.testing.assert_series_equal(hpat_func(*args), test_impl2(*args))

    def test_df_cov1(self):
        # test series rolling functions
        # all functions except apply
        df1 = pd.DataFrame({'A': [0, 1, 2, np.nan, 4], 'B': np.ones(5)})
        df2 = pd.DataFrame({'A': [0, 1, 2, -2, 4], 'C': np.ones(5)})
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)

        def test_impl(df, df2, w, c):
            return df.rolling(w, center=c).cov(df2)
        hpat_func = hpat.jit(test_impl)
        for args in itertools.product([df1, df2], [df1, df2], wins, centers):
            pd.testing.assert_frame_equal(hpat_func(*args), test_impl(*args))
            pd.testing.assert_frame_equal(hpat_func(*args), test_impl(*args))

        def test_impl2(df, df2, w, c):
            return df.rolling(w, center=c).corr(df2)
        hpat_func = hpat.jit(test_impl2)
        for args in itertools.product([df1, df2], [df1, df2], wins, centers):
            pd.testing.assert_frame_equal(hpat_func(*args), test_impl2(*args))
            pd.testing.assert_frame_equal(hpat_func(*args), test_impl2(*args))


if __name__ == "__main__":
    unittest.main()

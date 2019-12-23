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
import itertools
import os
import pandas as pd
import platform
import numpy as np
import numba
import sdc
from itertools import product
from numba.errors import TypingError
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                  count_parfor_OneDs, count_array_OneDs, dist_IR_contains,
                                  skip_numba_jit, skip_sdc_jit,
                                  test_global_input_data_float64)
from sdc.hiframes.rolling import supported_rolling_funcs

LONG_TEST = (int(os.environ['SDC_LONG_ROLLING_TEST']) != 0
             if 'SDC_LONG_ROLLING_TEST' in os.environ else False)

test_funcs = ('mean', 'max',)
if LONG_TEST:
    # all functions except apply, cov, corr
    test_funcs = supported_rolling_funcs[:-3]


def series_rolling_std_usecase(series, window, min_periods, ddof):
    return series.rolling(window, min_periods).std(ddof)


def series_rolling_var_usecase(series, window, min_periods, ddof):
    return series.rolling(window, min_periods).var(ddof)


class TestRolling(TestCase):
    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)

            for args in itertools.product(wins, centers):
                df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
                pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))
                df = pd.DataFrame({'B': [0, 1, 2, -2, 4]})
                pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))

    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)
            for n, w, c in itertools.product(sizes, wins, centers):
                df = pd.DataFrame({'B': np.arange(n)})
                pd.testing.assert_frame_equal(hpat_func(df, w, c), test_impl(df, w, c))

    @skip_numba_jit
    def test_fixed_apply1(self):
        # test sequentially with manually created dfs
        def test_impl(df, w, c):
            return df.rolling(w, center=c).apply(lambda a: a.sum())
        hpat_func = self.jit(test_impl)
        wins = (3,)
        if LONG_TEST:
            wins = (2, 3, 5)
        centers = (False, True)
        for args in itertools.product(wins, centers):
            df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
            pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))
            df = pd.DataFrame({'B': [0, 1, 2, -2, 4]})
            pd.testing.assert_frame_equal(hpat_func(df, *args), test_impl(df, *args))

    @skip_numba_jit
    def test_fixed_apply2(self):
        # test sequentially with generated dfs
        def test_impl(df, w, c):
            return df.rolling(w, center=c).apply(lambda a: a.sum())
        hpat_func = self.jit(test_impl)
        sizes = (121,)
        wins = (3,)
        if LONG_TEST:
            sizes = (1, 2, 10, 11, 121, 1000)
            wins = (2, 3, 5)
        centers = (False, True)
        for n, w, c in itertools.product(sizes, wins, centers):
            df = pd.DataFrame({'B': np.arange(n)})
            pd.testing.assert_frame_equal(hpat_func(df, w, c), test_impl(df, w, c))

    @skip_numba_jit
    def test_fixed_parallel1(self):
        def test_impl(n, w, center):
            df = pd.DataFrame({'B': np.arange(n)})
            R = df.rolling(w, center=center).sum()
            return R.B.sum()

        hpat_func = self.jit(test_impl)
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

    @skip_numba_jit
    def test_fixed_parallel_apply1(self):
        def test_impl(n, w, center):
            df = pd.DataFrame({'B': np.arange(n)})
            R = df.rolling(w, center=center).apply(lambda a: a.sum())
            return R.B.sum()

        hpat_func = self.jit(test_impl)
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

    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)
            # XXX: skipping min/max for this test since the behavior of Pandas
            # is inconsistent: it assigns NaN to last output instead of 4!
            if func_name not in ('min', 'max'):
                pd.testing.assert_frame_equal(hpat_func(df1), test_impl(df1))
            pd.testing.assert_frame_equal(hpat_func(df2), test_impl(df2))

    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)
            for n in sizes:
                time = pd.date_range(start='1/1/2018', periods=n, freq='s')
                df = pd.DataFrame({'B': np.arange(n), 'time': time})
                pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)
            pd.testing.assert_frame_equal(hpat_func(df1), test_impl(df1))
            pd.testing.assert_frame_equal(hpat_func(df2), test_impl(df2))

    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)
            for n in sizes:
                time = pd.date_range(start='1/1/2018', periods=n, freq='s')
                df = pd.DataFrame({'B': np.arange(n), 'time': time})
                pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)
            for n in sizes:
                np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)
            for n in sizes:
                np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
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
            hpat_func = self.jit(test_impl)
            for args in itertools.product(wins, centers):
                pd.testing.assert_series_equal(hpat_func(S1, *args), test_impl(S1, *args))
                pd.testing.assert_series_equal(hpat_func(S2, *args), test_impl(S2, *args))
        # test apply

        def apply_test_impl(S, w, c):
            return S.rolling(w, center=c).apply(lambda a: a.sum())
        hpat_func = self.jit(apply_test_impl)
        for args in itertools.product(wins, centers):
            pd.testing.assert_series_equal(hpat_func(S1, *args), apply_test_impl(S1, *args))
            pd.testing.assert_series_equal(hpat_func(S2, *args), apply_test_impl(S2, *args))

    @skip_numba_jit
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
        hpat_func = self.jit(test_impl)
        for args in itertools.product([S1, S2], [S1, S2], wins, centers):
            pd.testing.assert_series_equal(hpat_func(*args), test_impl(*args))
            pd.testing.assert_series_equal(hpat_func(*args), test_impl(*args))

        def test_impl2(S, S2, w, c):
            return S.rolling(w, center=c).corr(S2)
        hpat_func = self.jit(test_impl2)
        for args in itertools.product([S1, S2], [S1, S2], wins, centers):
            pd.testing.assert_series_equal(hpat_func(*args), test_impl2(*args))
            pd.testing.assert_series_equal(hpat_func(*args), test_impl2(*args))

    @skip_numba_jit
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
        hpat_func = self.jit(test_impl)
        for args in itertools.product([df1, df2], [df1, df2], wins, centers):
            pd.testing.assert_frame_equal(hpat_func(*args), test_impl(*args))
            pd.testing.assert_frame_equal(hpat_func(*args), test_impl(*args))

        def test_impl2(df, df2, w, c):
            return df.rolling(w, center=c).corr(df2)
        hpat_func = self.jit(test_impl2)
        for args in itertools.product([df1, df2], [df1, df2], wins, centers):
            pd.testing.assert_frame_equal(hpat_func(*args), test_impl2(*args))
            pd.testing.assert_frame_equal(hpat_func(*args), test_impl2(*args))

    @skip_sdc_jit('Series.rolling.min() unsupported exceptions')
    def test_series_rolling_unsupported_values(self):
        def test_impl(series, window, min_periods, center,
                      win_type, on, axis, closed):
            return series.rolling(window, min_periods, center,
                                  win_type, on, axis, closed).min()

        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_float64[0])

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, -1, None, False, None, None, 0, None)
        self.assertIn('window must be non-negative', str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, 1, -1, False, None, None, 0, None)
        self.assertIn('min_periods must be >= 0', str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, 1, 2, False, None, None, 0, None)
        self.assertIn('min_periods must be <= window', str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, 1, 2, False, None, None, 0, None)
        self.assertIn('min_periods must be <= window', str(raises.exception))

        msg_tmpl = 'Method rolling(). The object {}\n expected: {}'

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, 1, None, True, None, None, 0, None)
        msg = msg_tmpl.format('center', 'False')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, 1, None, False, 'None', None, 0, None)
        msg = msg_tmpl.format('win_type', 'None')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, 1, None, False, None, 'None', 0, None)
        msg = msg_tmpl.format('on', 'None')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, 1, None, False, None, None, 1, None)
        msg = msg_tmpl.format('axis', '0')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, 1, None, False, None, None, 0, 'None')
        msg = msg_tmpl.format('closed', 'None')
        self.assertIn(msg, str(raises.exception))

    @skip_sdc_jit('Series.rolling.min() unsupported exceptions')
    def test_series_rolling_unsupported_types(self):
        def test_impl(series, window, min_periods, center,
                      win_type, on, axis, closed):
            return series.rolling(window, min_periods, center,
                                  win_type, on, axis, closed).min()

        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_float64[0])
        msg_tmpl = 'Method rolling(). The object {}\n given: {}\n expected: {}'

        with self.assertRaises(TypingError) as raises:
            hpat_func(series, '1', None, False, None, None, 0, None)
        msg = msg_tmpl.format('window', 'unicode_type', 'int')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(series, 1, '1', False, None, None, 0, None)
        msg = msg_tmpl.format('min_periods', 'unicode_type', 'None, int')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(series, 1, None, 0, None, None, 0, None)
        msg = msg_tmpl.format('center', 'int64', 'bool')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(series, 1, None, False, -1, None, 0, None)
        msg = msg_tmpl.format('win_type', 'int64', 'str')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(series, 1, None, False, None, -1, 0, None)
        msg = msg_tmpl.format('on', 'int64', 'str')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(series, 1, None, False, None, None, None, None)
        msg = msg_tmpl.format('axis', 'none', 'int, str')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(series, 1, None, False, None, None, 0, -1)
        msg = msg_tmpl.format('closed', 'int64', 'str')
        self.assertIn(msg, str(raises.exception))

    @skip_sdc_jit('Series.rolling.corr() unsupported Series index')
    def test_series_rolling_corr(self):
        def test_impl(series, window, min_periods, other):
            return series.rolling(window, min_periods).corr(other)

        hpat_func = self.jit(test_impl)

        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        for main_data, other_data in product(all_data, all_data):
            series = pd.Series(main_data)
            other = pd.Series(other_data)
            for window in range(0, len(series) + 3, 2):
                for min_periods in range(0, window, 2):
                    with self.subTest(series=series, other=other,
                                      window=window, min_periods=min_periods):
                        ref_result = test_impl(series, window, min_periods, other)
                        jit_result = hpat_func(series, window, min_periods, other)
                        pd.testing.assert_series_equal(ref_result, jit_result)

    @skip_sdc_jit('Series.rolling.corr() unsupported Series index')
    def test_series_rolling_corr_with_no_other(self):
        def test_impl(series, window, min_periods):
            return series.rolling(window, min_periods).corr()

        hpat_func = self.jit(test_impl)

        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        for data in all_data:
            series = pd.Series(data)
            for window in range(0, len(series) + 3, 2):
                for min_periods in range(0, window, 2):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods):
                        jit_result = hpat_func(series, window, min_periods)
                        ref_result = test_impl(series, window, min_periods)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.corr() unsupported exceptions')
    def test_series_rolling_corr_unsupported_types(self):
        def test_impl(pairwise):
            series = pd.Series([1., -1., 0., 0.1, -0.1])
            return series.rolling(3, 3).corr(pairwise=pairwise)

        hpat_func = self.jit(test_impl)

        with self.assertRaises(TypingError) as raises:
            hpat_func(1)
        msg = 'Method rolling.corr(). The object pairwise\n given: int64\n expected: bool'
        self.assertIn(msg, str(raises.exception))

    @skip_sdc_jit('Series.rolling.count() unsupported Series index')
    def test_series_rolling_count(self):
        def test_impl(series, window, min_periods):
            return series.rolling(window, min_periods).count()

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            for window in range(0, len(series) + 3, 2):
                for min_periods in range(0, window + 1, 2):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods):
                        jit_result = hpat_func(series, window, min_periods)
                        ref_result = test_impl(series, window, min_periods)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.kurt() unsupported Series index')
    def test_series_rolling_kurt(self):
        def test_impl(series, window, min_periods):
            return series.rolling(window, min_periods).kurt()

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            for window in range(4, len(series) + 1):
                for min_periods in range(window + 1):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods):
                        ref_result = test_impl(series, window, min_periods)
                        jit_result = hpat_func(series, window, min_periods)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.max() unsupported Series index')
    def test_series_rolling_max(self):
        def test_impl(series, window, min_periods):
            return series.rolling(window, min_periods).max()

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            # TODO: fix the issue when window = 0
            for window in range(1, len(series) + 2):
                for min_periods in range(window + 1):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods):
                        jit_result = hpat_func(series, window, min_periods)
                        ref_result = test_impl(series, window, min_periods)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.mean() unsupported Series index')
    def test_series_rolling_mean(self):
        def test_impl(series, window, min_periods):
            return series.rolling(window, min_periods).mean()

        hpat_func = self.jit(test_impl)

        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            for window in range(0, len(series) + 3, 2):
                for min_periods in range(0, window + 1, 2):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods):
                        jit_result = hpat_func(series, window, min_periods)
                        ref_result = test_impl(series, window, min_periods)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.median() unsupported Series index')
    def test_series_rolling_median(self):
        def test_impl(series, window, min_periods):
            return series.rolling(window, min_periods).median()

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            for window in range(0, len(series) + 3, 2):
                for min_periods in range(0, window + 1, 2):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods):
                        jit_result = hpat_func(series, window, min_periods)
                        ref_result = test_impl(series, window, min_periods)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.min() unsupported Series index')
    def test_series_rolling_min(self):
        def test_impl(series, window, min_periods):
            return series.rolling(window, min_periods).min()

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            # TODO: fix the issue when window = 0
            for window in range(1, len(series) + 2):
                for min_periods in range(window + 1):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods):
                        jit_result = hpat_func(series, window, min_periods)
                        ref_result = test_impl(series, window, min_periods)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.std() unsupported Series index')
    def test_series_rolling_std(self):
        test_impl = series_rolling_std_usecase
        hpat_func = self.jit(test_impl)

        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            for window in range(0, len(series) + 3, 2):
                for min_periods, ddof in product(range(0, window, 2), [0, 1]):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods, ddof=ddof):
                        jit_result = hpat_func(series, window, min_periods, ddof)
                        ref_result = test_impl(series, window, min_periods, ddof)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.std() unsupported exceptions')
    def test_series_rolling_std_exception_unsupported_ddof(self):
        test_impl = series_rolling_std_usecase
        hpat_func = self.jit(test_impl)

        series = pd.Series([1., -1., 0., 0.1, -0.1])
        with self.assertRaises(TypingError) as raises:
            hpat_func(series, 3, 2, '1')
        msg = 'Method rolling.std(). The object ddof\n given: unicode_type\n expected: int'
        self.assertIn(msg, str(raises.exception))

    @skip_sdc_jit('Series.rolling.sum() unsupported Series index')
    def test_series_rolling_sum(self):
        def test_impl(series, window, min_periods):
            return series.rolling(window, min_periods).sum()

        hpat_func = self.jit(test_impl)

        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            for window in range(0, len(series) + 3, 2):
                for min_periods in range(0, window + 1, 2):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods):
                        jit_result = hpat_func(series, window, min_periods)
                        ref_result = test_impl(series, window, min_periods)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.var() unsupported Series index')
    def test_series_rolling_var(self):
        test_impl = series_rolling_var_usecase
        hpat_func = self.jit(test_impl)

        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            for window in range(0, len(series) + 3, 2):
                for min_periods, ddof in product(range(0, window, 2), [0, 1]):
                    with self.subTest(series=series, window=window,
                                      min_periods=min_periods, ddof=ddof):
                        jit_result = hpat_func(series, window, min_periods, ddof)
                        ref_result = test_impl(series, window, min_periods, ddof)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.var() unsupported exceptions')
    def test_series_rolling_var_exception_unsupported_ddof(self):
        test_impl = series_rolling_var_usecase
        hpat_func = self.jit(test_impl)

        series = pd.Series([1., -1., 0., 0.1, -0.1])
        with self.assertRaises(TypingError) as raises:
            hpat_func(series, 3, 2, '1')
        msg = 'Method rolling.var(). The object ddof\n given: unicode_type\n expected: int'
        self.assertIn(msg, str(raises.exception))


if __name__ == "__main__":
    unittest.main()

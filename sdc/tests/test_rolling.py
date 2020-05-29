# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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

import itertools
import os
import platform
import string
import unittest
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from numba.core.errors import TypingError
from sdc.hiframes.rolling import supported_rolling_funcs
from sdc.tests.test_base import TestCase
from sdc.tests.test_series import gen_frand_array
from sdc.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                  skip_numba_jit, skip_sdc_jit,
                                  test_global_input_data_float64,
                                  dfRefactoringNotImplemented)


LONG_TEST = (int(os.environ['SDC_LONG_ROLLING_TEST']) != 0
             if 'SDC_LONG_ROLLING_TEST' in os.environ else False)

test_funcs = ('mean', 'max',)
if LONG_TEST:
    # all functions except apply, cov, corr
    test_funcs = supported_rolling_funcs[:-3]


def rolling_std_usecase(obj, window, min_periods, ddof):
    return obj.rolling(window, min_periods).std(ddof)


def rolling_var_usecase(obj, window, min_periods, ddof):
    return obj.rolling(window, min_periods).var(ddof)


class TestRolling(TestCase):

    @skip_numba_jit
    def test_series_rolling1(self):
        def test_impl(S):
            return S.rolling(3).sum()
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

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

    def _get_assert_equal(self, obj):
        if isinstance(obj, pd.Series):
            return pd.testing.assert_series_equal
        elif isinstance(obj, pd.DataFrame):
            return pd.testing.assert_frame_equal
        elif isinstance(obj, np.ndarray):
            return np.testing.assert_array_equal

        return self.assertEqual

    def _test_rolling_unsupported_values(self, obj):
        def test_impl(obj, window, min_periods, center,
                      win_type, on, axis, closed):
            return obj.rolling(window, min_periods, center,
                               win_type, on, axis, closed).min()

        hpat_func = self.jit(test_impl)

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, -1, None, False, None, None, 0, None)
        self.assertIn('window must be non-negative', str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 1, -1, False, None, None, 0, None)
        self.assertIn('min_periods must be >= 0', str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 1, 2, False, None, None, 0, None)
        self.assertIn('min_periods must be <= window', str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 1, 2, False, None, None, 0, None)
        self.assertIn('min_periods must be <= window', str(raises.exception))

        msg_tmpl = 'Method rolling(). The object {}\n expected: {}'

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 1, None, True, None, None, 0, None)
        msg = msg_tmpl.format('center', 'False')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 1, None, False, 'None', None, 0, None)
        msg = msg_tmpl.format('win_type', 'None')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 1, None, False, None, 'None', 0, None)
        msg = msg_tmpl.format('on', 'None')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 1, None, False, None, None, 1, None)
        msg = msg_tmpl.format('axis', '0')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 1, None, False, None, None, 0, 'None')
        msg = msg_tmpl.format('closed', 'None')
        self.assertIn(msg, str(raises.exception))

    def _test_rolling_unsupported_types(self, obj):
        def test_impl(obj, window, min_periods, center,
                      win_type, on, axis, closed):
            return obj.rolling(window, min_periods, center,
                               win_type, on, axis, closed).min()

        hpat_func = self.jit(test_impl)
        msg_tmpl = 'Method rolling(). The object {}\n given: {}\n expected: {}'

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, '1', None, False, None, None, 0, None)
        msg = msg_tmpl.format('window', 'unicode_type', 'int')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1, '1', False, None, None, 0, None)
        msg = msg_tmpl.format('min_periods', 'unicode_type', 'None, int')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1, None, 0, None, None, 0, None)
        msg = msg_tmpl.format('center', 'int64', 'bool')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1, None, False, -1, None, 0, None)
        msg = msg_tmpl.format('win_type', 'int64', 'str')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1, None, False, None, -1, 0, None)
        msg = msg_tmpl.format('on', 'int64', 'str')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1, None, False, None, None, None, None)
        msg = msg_tmpl.format('axis', 'none', 'int, str')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1, None, False, None, None, 0, -1)
        msg = msg_tmpl.format('closed', 'int64', 'str')
        self.assertIn(msg, str(raises.exception))

    def _test_rolling_apply_mean(self, obj):
        def test_impl(obj, window, min_periods):
            def func(x):
                if len(x) == 0:
                    return np.nan
                return x.mean()

            return obj.rolling(window, min_periods).apply(func)

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods in range(0, window + 1, 2):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_apply_unsupported_types(self, obj):
        def test_impl(obj, raw):
            def func(x):
                if len(x) == 0:
                    return np.nan
                return np.median(x)

            return obj.rolling(3).apply(func, raw=raw)

        hpat_func = self.jit(test_impl)

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1)
        msg = 'Method rolling.apply(). The object raw\n given: int64\n expected: bool'
        self.assertIn(msg, str(raises.exception))

    def _test_rolling_apply_args(self, obj):
        def test_impl(obj, window, min_periods, q):
            def func(x, q):
                if len(x) == 0:
                    return np.nan
                return np.quantile(x, q)

            return obj.rolling(window, min_periods).apply(func, raw=None, args=(q,))

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods in range(0, window + 1, 2):
                for q in [0.25, 0.5, 0.75]:
                    with self.subTest(obj=obj, window=window,
                                      min_periods=min_periods, q=q):
                        jit_result = hpat_func(obj, window, min_periods, q)
                        ref_result = test_impl(obj, window, min_periods, q)
                        assert_equal(jit_result, ref_result)

    def _test_rolling_corr(self, obj, other):
        def test_impl(obj, window, min_periods, other):
            return obj.rolling(window, min_periods).corr(other)

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods in range(0, window, 2):
                with self.subTest(obj=obj, other=other,
                                  window=window, min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods, other)
                    ref_result = test_impl(obj, window, min_periods, other)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_corr_with_no_other(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).corr(pairwise=False)

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods in range(0, window, 2):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_corr_unsupported_types(self, obj):
        def test_impl(obj, pairwise):
            return obj.rolling(3, 3).corr(pairwise=pairwise)

        hpat_func = self.jit(test_impl)

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1)
        msg = 'Method rolling.corr(). The object pairwise\n given: int64\n expected: bool'
        self.assertIn(msg, str(raises.exception))

    def _test_rolling_count(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).count()

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods in range(0, window + 1, 2):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_cov(self, obj, other):
        def test_impl(obj, window, min_periods, other, ddof):
            return obj.rolling(window, min_periods).cov(other, ddof=ddof)

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods, ddof in product(range(0, window, 2), [0, 1]):
                with self.subTest(obj=obj, other=other, window=window,
                                  min_periods=min_periods, ddof=ddof):
                    jit_result = hpat_func(obj, window, min_periods, other, ddof)
                    ref_result = test_impl(obj, window, min_periods, other, ddof)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_cov_with_no_other(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).cov(pairwise=False)

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods in range(0, window, 2):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_cov_unsupported_types(self, obj):
        def test_impl(obj, pairwise, ddof):
            return obj.rolling(3, 3).cov(pairwise=pairwise, ddof=ddof)

        hpat_func = self.jit(test_impl)

        msg_tmpl = 'Method rolling.cov(). The object {}\n given: {}\n expected: {}'

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 1, 1)
        msg = msg_tmpl.format('pairwise', 'int64', 'bool')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, None, '1')
        msg = msg_tmpl.format('ddof', 'unicode_type', 'int')
        self.assertIn(msg, str(raises.exception))

    def _test_rolling_kurt(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).kurt()

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(4, len(obj) + 1):
            for min_periods in range(window + 1):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    ref_result = test_impl(obj, window, min_periods)
                    jit_result = hpat_func(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_max(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).max()

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        # python implementation crashes if window = 0, jit works correctly
        for window in range(1, len(obj) + 2):
            for min_periods in range(window + 1):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_mean(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).mean()

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(len(obj) + 2):
            for min_periods in range(window):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_median(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).median()

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods in range(0, window + 1, 2):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_min(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).min()

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        # python implementation crashes if window = 0, jit works correctly
        for window in range(1, len(obj) + 2):
            for min_periods in range(window + 1):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_quantile(self, obj):
        def test_impl(obj, window, min_periods, quantile):
            return obj.rolling(window, min_periods).quantile(quantile)

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)
        quantiles = [0, 0.25, 0.5, 0.75, 1]

        for window in range(0, len(obj) + 3, 2):
            for min_periods, q in product(range(0, window, 2), quantiles):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods, quantiles=q):
                    jit_result = hpat_func(obj, window, min_periods, q)
                    ref_result = test_impl(obj, window, min_periods, q)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_quantile_exception_unsupported_types(self, obj):
        def test_impl(obj, quantile, interpolation):
            return obj.rolling(3, 2).quantile(quantile, interpolation)

        hpat_func = self.jit(test_impl)

        msg_tmpl = 'Method rolling.quantile(). The object {}\n given: {}\n expected: {}'

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, '0.5', 'linear')
        msg = msg_tmpl.format('quantile', 'unicode_type', 'float')
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, 0.5, None)
        msg = msg_tmpl.format('interpolation', 'none', 'str')
        self.assertIn(msg, str(raises.exception))

    def _test_rolling_quantile_exception_unsupported_values(self, obj):
        def test_impl(obj, quantile, interpolation):
            return obj.rolling(3, 2).quantile(quantile, interpolation)

        hpat_func = self.jit(test_impl)

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 2, 'linear')
        self.assertIn('quantile value not in [0, 1]', str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(obj, 0.5, 'lower')
        self.assertIn('interpolation value not "linear"', str(raises.exception))

    def _test_rolling_skew(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).skew()

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(3, len(obj) + 1):
            for min_periods in range(window + 1):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    ref_result = test_impl(obj, window, min_periods)
                    jit_result = hpat_func(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_std(self, obj):
        test_impl = rolling_std_usecase
        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods, ddof in product(range(0, window, 2), [0, 1]):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods, ddof=ddof):
                    jit_result = hpat_func(obj, window, min_periods, ddof)
                    ref_result = test_impl(obj, window, min_periods, ddof)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_std_exception_unsupported_ddof(self, obj):
        test_impl = rolling_std_usecase
        hpat_func = self.jit(test_impl)

        window, min_periods, invalid_ddof = 3, 2, '1'
        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, window, min_periods, invalid_ddof)
        msg = 'Method rolling.std(). The object ddof\n given: unicode_type\n expected: int'
        self.assertIn(msg, str(raises.exception))

    def _test_rolling_sum(self, obj):
        def test_impl(obj, window, min_periods):
            return obj.rolling(window, min_periods).sum()

        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(len(obj) + 2):
            for min_periods in range(window):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods):
                    jit_result = hpat_func(obj, window, min_periods)
                    ref_result = test_impl(obj, window, min_periods)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_var(self, obj):
        test_impl = rolling_var_usecase
        hpat_func = self.jit(test_impl)
        assert_equal = self._get_assert_equal(obj)

        for window in range(0, len(obj) + 3, 2):
            for min_periods, ddof in product(range(0, window, 2), [0, 1]):
                with self.subTest(obj=obj, window=window,
                                  min_periods=min_periods, ddof=ddof):
                    jit_result = hpat_func(obj, window, min_periods, ddof)
                    ref_result = test_impl(obj, window, min_periods, ddof)
                    assert_equal(jit_result, ref_result)

    def _test_rolling_var_exception_unsupported_ddof(self, obj):
        test_impl = rolling_var_usecase
        hpat_func = self.jit(test_impl)

        window, min_periods, invalid_ddof = 3, 2, '1'
        with self.assertRaises(TypingError) as raises:
            hpat_func(obj, window, min_periods, invalid_ddof)
        msg = 'Method rolling.var(). The object ddof\n given: unicode_type\n expected: int'
        self.assertIn(msg, str(raises.exception))

    @skip_sdc_jit('DataFrame.rolling.min() unsupported exceptions')
    def test_df_rolling_unsupported_values(self):
        all_data = test_global_input_data_float64
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_unsupported_values(df)

    @skip_sdc_jit('DataFrame.rolling.min() unsupported exceptions')
    def test_df_rolling_unsupported_types(self):
        all_data = test_global_input_data_float64
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_unsupported_types(df)

    @skip_sdc_jit('DataFrame.rolling.apply() unsupported')
    def test_df_rolling_apply_mean(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_apply_mean(df)

    def test_df_rolling_apply_mean_no_unboxing(self):
        def test_impl(window, min_periods):
            def func(x):
                if len(x) == 0:
                    return np.nan
                return x.mean()

            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).apply(func)

        hpat_func = self.jit(test_impl)
        for window in range(0, 8, 2):
            for min_periods in range(0, window + 1, 2):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.apply() unsupported exceptions')
    def test_df_rolling_apply_unsupported_types(self):
        all_data = [[1., -1., 0., 0.1, -0.1], [-1., 1., 0., -0.1, 0.1]]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_apply_unsupported_types(df)

    @unittest.skip('DataFrame.rolling.apply() unsupported args')
    def test_df_rolling_apply_args(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_apply_args(df)

    @skip_sdc_jit('DataFrame.rolling.corr() unsupported')
    def test_df_rolling_corr(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)
        for d in all_data:
            other = pd.Series(d)
            self._test_rolling_corr(df, other)

        other_all_data = deepcopy(all_data) + [list(range(10))[::-1]]
        other_all_data[1] = [-1., 1., 0., -0.1, 0.1, 0.]
        other_length = min(len(d) for d in other_all_data)
        other_data = {n: d[:other_length] for n, d in zip(string.ascii_uppercase, other_all_data)}
        other = pd.DataFrame(other_data)

        self._test_rolling_corr(df, other)

    def test_df_rolling_corr_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            other = pd.DataFrame({
                'A': [0, 1, 2, 3, 4, 5],
                'B': [-1., 1., 0., -0.1, 0.1, 0.],
                'C': [1., np.inf, np.inf, -1., 0., np.inf],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan],
                'E': [9, 8, 7, 6, 5, 4],
            })
            return df.rolling(window, min_periods).corr(other)

        hpat_func = self.jit(test_impl)
        for window in range(0, 8, 2):
            for min_periods in range(0, window, 2):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.corr() unsupported')
    def test_df_rolling_corr_no_other(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_corr_with_no_other(df)

    @skip_sdc_jit('DataFrame.rolling.corr() unsupported exceptions')
    def test_df_rolling_corr_unsupported_types(self):
        all_data = [[1., -1., 0., 0.1, -0.1], [-1., 1., 0., -0.1, 0.1]]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_corr_unsupported_types(df)

    @skip_sdc_jit('DataFrame.rolling.corr() unsupported exceptions')
    def test_df_rolling_corr_unsupported_values(self):
        def test_impl(df, other, pairwise):
            return df.rolling(3, 3).corr(other=other, pairwise=pairwise)

        hpat_func = self.jit(test_impl)
        msg_tmpl = 'Method rolling.corr(). The object pairwise\n expected: {}'

        df = pd.DataFrame({'A': [1., -1., 0., 0.1, -0.1],
                           'B': [-1., 1., 0., -0.1, 0.1]})
        for pairwise in [None, True]:
            with self.assertRaises(ValueError) as raises:
                hpat_func(df, None, pairwise)
            self.assertIn(msg_tmpl.format('False'), str(raises.exception))

        other = pd.DataFrame({'A': [-1., 1., 0., -0.1, 0.1],
                              'C': [1., -1., 0., 0.1, -0.1]})
        with self.assertRaises(ValueError) as raises:
            hpat_func(df, other, True)
        self.assertIn(msg_tmpl.format('False, None'), str(raises.exception))

    @skip_sdc_jit('DataFrame.rolling.count() unsupported')
    def test_df_rolling_count(self):
        all_data = test_global_input_data_float64
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_count(df)

    def test_df_rolling_count_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).count()

        hpat_func = self.jit(test_impl)
        for window in range(0, 8, 2):
            for min_periods in range(0, window + 1, 2):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.cov() unsupported')
    def test_df_rolling_cov(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)
        for d in all_data:
            other = pd.Series(d)
            self._test_rolling_cov(df, other)

        other_all_data = deepcopy(all_data) + [list(range(10))[::-1]]
        other_all_data[1] = [-1., 1., 0., -0.1, 0.1]
        other_length = min(len(d) for d in other_all_data)
        other_data = {n: d[:other_length] for n, d in zip(string.ascii_uppercase, other_all_data)}
        other = pd.DataFrame(other_data)

        self._test_rolling_cov(df, other)

    def test_df_rolling_cov_no_unboxing(self):
        def test_impl(window, min_periods, ddof):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            other = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [-1., 1., 0., -0.1, 0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
                'E': [9, 8, 7, 6, 5],
            })
            return df.rolling(window, min_periods).cov(other, ddof=ddof)

        hpat_func = self.jit(test_impl)
        for window in range(0, 8, 2):
            for min_periods, ddof in product(range(0, window, 2), [0, 1]):
                with self.subTest(window=window, min_periods=min_periods,
                                  ddof=ddof):
                    jit_result = hpat_func(window, min_periods, ddof)
                    ref_result = test_impl(window, min_periods, ddof)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.cov() unsupported')
    def test_df_rolling_cov_no_other(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_cov_with_no_other(df)

    @skip_sdc_jit('DataFrame.rolling.cov() unsupported exceptions')
    def test_df_rolling_cov_unsupported_types(self):
        all_data = [[1., -1., 0., 0.1, -0.1], [-1., 1., 0., -0.1, 0.1]]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_cov_unsupported_types(df)

    @skip_sdc_jit('DataFrame.rolling.cov() unsupported exceptions')
    def test_df_rolling_cov_unsupported_values(self):
        def test_impl(df, other, pairwise):
            return df.rolling(3, 3).cov(other=other, pairwise=pairwise)

        hpat_func = self.jit(test_impl)
        msg_tmpl = 'Method rolling.cov(). The object pairwise\n expected: {}'

        df = pd.DataFrame({'A': [1., -1., 0., 0.1, -0.1],
                           'B': [-1., 1., 0., -0.1, 0.1]})
        for pairwise in [None, True]:
            with self.assertRaises(ValueError) as raises:
                hpat_func(df, None, pairwise)
            self.assertIn(msg_tmpl.format('False'), str(raises.exception))

        other = pd.DataFrame({'A': [-1., 1., 0., -0.1, 0.1],
                              'C': [1., -1., 0., 0.1, -0.1]})
        with self.assertRaises(ValueError) as raises:
            hpat_func(df, other, True)
        self.assertIn(msg_tmpl.format('False, None'), str(raises.exception))

    @skip_sdc_jit('Series.rolling.cov() unsupported Series index')
    @unittest.expectedFailure
    def test_df_rolling_cov_issue_floating_point_rounding(self):
        """
            Cover issue of different float rounding in Python and SDC/Numba:

            s = np.Series([1., -1., 0., 0.1, -0.1])
            s.rolling(2, 0).mean()

            Python:                  SDC/Numba:
            0    1.000000e+00        0    1.00
            1    0.000000e+00        1    0.00
            2   -5.000000e-01        2   -0.50
            3    5.000000e-02        3    0.05
            4   -1.387779e-17        4    0.00
            dtype: float64           dtype: float64

            BTW: cov uses mean inside itself
        """
        def test_impl(df, window, min_periods, other, ddof):
            return df.rolling(window, min_periods).cov(other, ddof=ddof)

        hpat_func = self.jit(test_impl)

        df = pd.DataFrame({'A': [1., -1., 0., 0.1, -0.1]})
        other = pd.DataFrame({'A': [-1., 1., 0., -0.1, 0.1, 0.]})

        jit_result = hpat_func(df, 2, 0, other, 1)
        ref_result = test_impl(df, 2, 0, other, 1)
        pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.kurt() unsupported')
    def test_df_rolling_kurt(self):
        all_data = test_global_input_data_float64
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_kurt(df)

    def test_df_rolling_kurt_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).kurt()

        hpat_func = self.jit(test_impl)
        for window in range(4, 6):
            for min_periods in range(window + 1):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.max() unsupported')
    def test_df_rolling_max(self):
        all_data = test_global_input_data_float64
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_max(df)

    def test_df_rolling_max_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).max()

        hpat_func = self.jit(test_impl)
        for window in range(1, 7):
            for min_periods in range(window + 1):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.mean() unsupported')
    def test_df_rolling_mean(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_mean(df)

    def test_df_rolling_mean_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).mean()

        hpat_func = self.jit(test_impl)
        for window in range(7):
            for min_periods in range(window):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.median() unsupported')
    def test_df_rolling_median(self):
        all_data = test_global_input_data_float64
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_median(df)

    def test_df_rolling_median_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).median()

        hpat_func = self.jit(test_impl)
        for window in range(0, 8, 2):
            for min_periods in range(0, window + 1, 2):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.min() unsupported')
    def test_df_rolling_min(self):
        all_data = test_global_input_data_float64
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_min(df)

    def test_df_rolling_min_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).min()

        hpat_func = self.jit(test_impl)
        for window in range(1, 7):
            for min_periods in range(window + 1):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @unittest.expectedFailure
    @unittest.skipIf(platform.system() == 'Darwin', 'Segmentation fault on Mac')
    @skip_sdc_jit('DataFrame.rolling.min() unsupported')
    def test_df_rolling_min_exception_many_columns(self):
        def test_impl(df):
            return df.rolling(3).min()

        hpat_func = self.jit(test_impl)

        # more than 19 columns raise SystemError: CPUDispatcher() returned a result with an error set
        all_data = test_global_input_data_float64 * 5
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_sdc_jit('DataFrame.rolling.quantile() unsupported')
    def test_df_rolling_quantile(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_quantile(df)

    def test_df_rolling_quantile_no_unboxing(self):
        def test_impl(window, min_periods, quantile):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).quantile(quantile)

        hpat_func = self.jit(test_impl)
        quantiles = [0, 0.25, 0.5, 0.75, 1]
        for window in range(0, 8, 2):
            for min_periods, q in product(range(0, window, 2), quantiles):
                with self.subTest(window=window, min_periods=min_periods,
                                  quantiles=q):
                    jit_result = hpat_func(window, min_periods, q)
                    ref_result = test_impl(window, min_periods, q)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.quantile() unsupported exceptions')
    def test_df_rolling_quantile_exception_unsupported_types(self):
        all_data = [[1., -1., 0., 0.1, -0.1], [-1., 1., 0., -0.1, 0.1]]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_quantile_exception_unsupported_types(df)

    @skip_sdc_jit('DataFrame.rolling.quantile() unsupported exceptions')
    def test_df_rolling_quantile_exception_unsupported_values(self):
        all_data = [[1., -1., 0., 0.1, -0.1], [-1., 1., 0., -0.1, 0.1]]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_quantile_exception_unsupported_values(df)

    @skip_sdc_jit('DataFrame.rolling.skew() unsupported')
    def test_df_rolling_skew(self):
        all_data = test_global_input_data_float64
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_skew(df)

    def test_df_rolling_skew_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).skew()

        hpat_func = self.jit(test_impl)
        for window in range(3, 6):
            for min_periods in range(window + 1):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.std() unsupported')
    def test_df_rolling_std(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_std(df)

    def test_df_rolling_std_no_unboxing(self):
        def test_impl(window, min_periods, ddof):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).std(ddof)

        hpat_func = self.jit(test_impl)
        for window in range(0, 8, 2):
            for min_periods, ddof in product(range(0, window, 2), [0, 1]):
                with self.subTest(window=window, min_periods=min_periods,
                                  ddof=ddof):
                    jit_result = hpat_func(window, min_periods, ddof)
                    ref_result = test_impl(window, min_periods, ddof)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.std() unsupported exceptions')
    def test_df_rolling_std_exception_unsupported_ddof(self):
        all_data = [[1., -1., 0., 0.1, -0.1], [-1., 1., 0., -0.1, 0.1]]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_std_exception_unsupported_ddof(df)

    @skip_sdc_jit('DataFrame.rolling.sum() unsupported')
    def test_df_rolling_sum(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_sum(df)

    def test_df_rolling_sum_no_unboxing(self):
        def test_impl(window, min_periods):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).sum()

        hpat_func = self.jit(test_impl)
        for window in range(7):
            for min_periods in range(window):
                with self.subTest(window=window, min_periods=min_periods):
                    jit_result = hpat_func(window, min_periods)
                    ref_result = test_impl(window, min_periods)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.var() unsupported')
    def test_df_rolling_var(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_var(df)

    def test_df_rolling_var_no_unboxing(self):
        def test_impl(window, min_periods, ddof):
            df = pd.DataFrame({
                'A': [0, 1, 2, 3, 4],
                'B': [1., -1., 0., 0.1, -0.1],
                'C': [1., np.inf, np.inf, -1., 0.],
                'D': [np.nan, np.inf, np.inf, np.nan, np.nan],
            })
            return df.rolling(window, min_periods).var(ddof)

        hpat_func = self.jit(test_impl)
        for window in range(0, 8, 2):
            for min_periods, ddof in product(range(0, window, 2), [0, 1]):
                with self.subTest(window=window, min_periods=min_periods,
                                  ddof=ddof):
                    jit_result = hpat_func(window, min_periods, ddof)
                    ref_result = test_impl(window, min_periods, ddof)
                    pd.testing.assert_frame_equal(jit_result, ref_result)

    @skip_sdc_jit('DataFrame.rolling.var() unsupported exceptions')
    def test_df_rolling_var_exception_unsupported_ddof(self):
        all_data = [[1., -1., 0., 0.1, -0.1], [-1., 1., 0., -0.1, 0.1]]
        length = min(len(d) for d in all_data)
        data = {n: d[:length] for n, d in zip(string.ascii_uppercase, all_data)}
        df = pd.DataFrame(data)

        self._test_rolling_var_exception_unsupported_ddof(df)

    @skip_sdc_jit('Series.rolling.min() unsupported exceptions')
    def test_series_rolling_unsupported_values(self):
        series = pd.Series(test_global_input_data_float64[0])
        self._test_rolling_unsupported_values(series)

    @skip_sdc_jit('Series.rolling.min() unsupported exceptions')
    def test_series_rolling_unsupported_types(self):
        series = pd.Series(test_global_input_data_float64[0])
        self._test_rolling_unsupported_types(series)

    @skip_sdc_jit('Series.rolling.apply() unsupported Series index')
    def test_series_rolling_apply_mean(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_apply_mean(series)

    @skip_sdc_jit('Series.rolling.apply() unsupported exceptions')
    def test_series_rolling_apply_unsupported_types(self):
        series = pd.Series([1., -1., 0., 0.1, -0.1])
        self._test_rolling_apply_unsupported_types(series)

    @unittest.skip('Series.rolling.apply() unsupported args')
    def test_series_rolling_apply_args(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_apply_args(series)

    @skip_sdc_jit('Series.rolling.corr() unsupported Series index')
    def test_series_rolling_corr(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [-1., 1., 0., -0.1, 0.1, 0.],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        for main_data, other_data in product(all_data, all_data):
            series = pd.Series(main_data)
            other = pd.Series(other_data)
            self._test_rolling_corr(series, other)

    @skip_sdc_jit('Series.rolling.corr() unsupported Series index')
    def test_series_rolling_corr_diff_length(self):
        def test_impl(series, window, other):
            return series.rolling(window).corr(other)

        hpat_func = self.jit(test_impl)

        series = pd.Series([1., -1., 0., 0.1, -0.1])
        other = pd.Series(gen_frand_array(40))
        window = 5
        jit_result = hpat_func(series, window, other)
        ref_result = test_impl(series, window, other)
        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.corr() unsupported Series index')
    def test_series_rolling_corr_with_no_other(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        for data in all_data:
            series = pd.Series(data)
            self._test_rolling_corr_with_no_other(series)

    @skip_sdc_jit('Series.rolling.corr() unsupported exceptions')
    def test_series_rolling_corr_unsupported_types(self):
        series = pd.Series([1., -1., 0., 0.1, -0.1])
        self._test_rolling_corr_unsupported_types(series)

    @skip_sdc_jit('Series.rolling.corr() unsupported Series index')
    @unittest.expectedFailure  # https://jira.devtools.intel.com/browse/SAT-2377
    def test_series_rolling_corr_index(self):
        def test_impl(S1, S2):
            return S1.rolling(window=3).corr(S2)

        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        index_values = np.arange(n)

        np.random.shuffle(index_values)
        S1 = pd.Series(np.arange(n), index=index_values, name='A')
        np.random.shuffle(index_values)
        S2 = pd.Series(2 * np.arange(n) - 5, index=index_values, name='B')

        result = hpat_func(S1, S2)
        result_ref = test_impl(S1, S2)
        pd.testing.assert_series_equal(result, result_ref)

    @skip_sdc_jit('Series.rolling.count() unsupported Series index')
    def test_series_rolling_count(self):
        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_count(series)

    @skip_sdc_jit('Series.rolling.cov() unsupported Series index')
    def test_series_rolling_cov(self):
        all_data = [
            list(range(5)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        for main_data, other_data in product(all_data, all_data):
            series = pd.Series(main_data)
            other = pd.Series(other_data)
            self._test_rolling_cov(series, other)

    @skip_sdc_jit('Series.rolling.cov() unsupported Series index')
    def test_series_rolling_cov_diff_length(self):
        def test_impl(series, window, other):
            return series.rolling(window).cov(other)

        hpat_func = self.jit(test_impl)

        series = pd.Series([1., -1., 0., 0.1, -0.1])
        other = pd.Series(gen_frand_array(40))
        window = 5
        jit_result = hpat_func(series, window, other)
        ref_result = test_impl(series, window, other)
        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.cov() unsupported Series index')
    def test_series_rolling_cov_no_other(self):
        all_data = [
            list(range(5)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        for data in all_data:
            series = pd.Series(data)
            self._test_rolling_cov_with_no_other(series)

    @skip_sdc_jit('Series.rolling.cov() unsupported Series index')
    @unittest.expectedFailure
    def test_series_rolling_cov_issue_floating_point_rounding(self):
        """Cover issue of different float rounding in Python and SDC/Numba"""
        def test_impl(series, window, min_periods, other, ddof):
            return series.rolling(window, min_periods).cov(other, ddof=ddof)

        hpat_func = self.jit(test_impl)

        series = pd.Series(list(range(10)))
        other = pd.Series([1., -1., 0., 0.1, -0.1])
        jit_result = hpat_func(series, 6, 0, other, 1)
        ref_result = test_impl(series, 6, 0, other, 1)
        pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_sdc_jit('Series.rolling.cov() unsupported exceptions')
    def test_series_rolling_cov_unsupported_types(self):
        series = pd.Series([1., -1., 0., 0.1, -0.1])
        self._test_rolling_cov_unsupported_types(series)

    @skip_sdc_jit('Series.rolling.kurt() unsupported Series index')
    def test_series_rolling_kurt(self):
        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_kurt(series)

    @skip_sdc_jit('Series.rolling.max() unsupported Series index')
    def test_series_rolling_max(self):
        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_max(series)

    @skip_sdc_jit('Series.rolling.mean() unsupported Series index')
    def test_series_rolling_mean(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_mean(series)

    @skip_sdc_jit('Series.rolling.median() unsupported Series index')
    def test_series_rolling_median(self):
        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_median(series)

    @skip_sdc_jit('Series.rolling.min() unsupported Series index')
    def test_series_rolling_min(self):
        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_min(series)

    @skip_sdc_jit('Series.rolling.quantile() unsupported Series index')
    def test_series_rolling_quantile(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_quantile(series)

    @skip_sdc_jit('Series.rolling.quantile() unsupported exceptions')
    def test_series_rolling_quantile_exception_unsupported_types(self):
        series = pd.Series([1., -1., 0., 0.1, -0.1])
        self._test_rolling_quantile_exception_unsupported_types(series)

    @skip_sdc_jit('Series.rolling.quantile() unsupported exceptions')
    def test_series_rolling_quantile_exception_unsupported_values(self):
        series = pd.Series([1., -1., 0., 0.1, -0.1])
        self._test_rolling_quantile_exception_unsupported_values(series)

    @skip_sdc_jit('Series.rolling.skew() unsupported Series index')
    def test_series_rolling_skew(self):
        all_data = test_global_input_data_float64
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_skew(series)

    @skip_sdc_jit('Series.rolling.std() unsupported Series index')
    def test_series_rolling_std(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_std(series)

    @skip_sdc_jit('Series.rolling.std() unsupported exceptions')
    def test_series_rolling_std_exception_unsupported_ddof(self):
        series = pd.Series([1., -1., 0., 0.1, -0.1])
        self._test_rolling_std_exception_unsupported_ddof(series)

    @skip_sdc_jit('Series.rolling.sum() unsupported Series index')
    def test_series_rolling_sum(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_sum(series)

    @skip_sdc_jit('Series.rolling.var() unsupported Series index')
    def test_series_rolling_var(self):
        all_data = [
            list(range(10)), [1., -1., 0., 0.1, -0.1],
            [1., np.inf, np.inf, -1., 0., np.inf, np.NINF, np.NINF],
            [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
        ]
        indices = [list(range(len(data)))[::-1] for data in all_data]
        for data, index in zip(all_data, indices):
            series = pd.Series(data, index, name='A')
            self._test_rolling_var(series)

    @skip_sdc_jit('Series.rolling.var() unsupported exceptions')
    def test_series_rolling_var_exception_unsupported_ddof(self):
        series = pd.Series([1., -1., 0., 0.1, -0.1])
        self._test_rolling_var_exception_unsupported_ddof(series)


if __name__ == "__main__":
    unittest.main()

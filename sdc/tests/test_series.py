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
import platform
import pyarrow.parquet as pq
import sdc
import string
import unittest
from itertools import combinations, combinations_with_replacement, islice, permutations, product
import numba
from numba import types
from numba.core.config import IS_32BITS
from numba.core.errors import TypingError
from numba import literally

from sdc.tests.test_series_apply import TestSeries_apply
from sdc.tests.test_series_map import TestSeries_map
from sdc.tests.test_series_ops import TestSeries_ops
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_OneDs,
                                  count_array_REPs,
                                  count_parfor_REPs,
                                  get_start_end,
                                  sdc_limitation,
                                  skip_inline,
                                  skip_numba_jit,
                                  skip_parallel,
                                  create_series_from_values,
                                  take_k_elements)
from sdc.tests.gen_test_data import ParquetGenerator

from sdc.tests.test_utils import (test_global_input_data_unicode_kind1,
                                  assert_raises_ty_checker,
                                  gen_srand_array,
                                  gen_frand_array,
                                  gen_strlist,
                                  _make_func_from_text)
from sdc.datatypes.common_functions import SDCLimitation


_cov_corr_series = [(pd.Series(x), pd.Series(y)) for x, y in [
    (
        [np.nan, -2., 3., 9.1],
        [np.nan, -2., 3., 5.0],
    ),
    # TODO(quasilyte): more intricate data for complex-typed series.
    # Some arguments make assert_almost_equal fail.
    # Functions that yield mismaching results:
    # _column_corr_impl and _column_cov_impl.
    (
        [complex(-2., 1.0), complex(3.0, 1.0)],
        [complex(-3., 1.0), complex(2.0, 1.0)],
    ),
    (
        [complex(-2.0, 1.0), complex(3.0, 1.0)],
        [1.0, -2.0],
    ),
    (
        [1.0, -4.5],
        [complex(-4.5, 1.0), complex(3.0, 1.0)],
    ),
]]

min_float64 = np.finfo('float64').min
max_float64 = np.finfo('float64').max

test_global_input_data_float64 = [
    [11., 35.2, -24., 0., np.NZERO, np.NINF, np.PZERO, min_float64],
    [1., np.nan, -1., 0., min_float64, max_float64, max_float64, min_float64],
    [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
]

min_int64 = np.iinfo('int64').min
max_int64 = np.iinfo('int64').max
max_uint64 = np.iinfo('uint64').max

test_global_input_data_signed_integer64 = [
    [1, -1, 0],
    [min_int64, max_int64, max_int64, min_int64],
]

test_global_input_data_integer64 = test_global_input_data_signed_integer64 + [[max_uint64, max_uint64]]

test_global_input_data_numeric = test_global_input_data_integer64 + test_global_input_data_float64

test_global_input_data_unicode_kind4 = [
    'ascii',
    '12345',
    '1234567890',
    '¡Y tú quién te crees?',
    '🐍⚡',
    '大处着眼，小处着手。',
]


def series_values_from_argsort_result(series, argsorted):
    """
        Rearranges series values according to pandas argsort result.
        Used in tests to verify correct work of Series.argsort implementation for unstable sortings.
    """
    argsort_indices = argsorted.values
    result = np.empty_like(series.values)
    # pandas argsort returns -1 in positions of NaN elements
    nan_values_mask = argsort_indices == -1
    if np.any(nan_values_mask):
        result[nan_values_mask] = np.nan

    # pandas argsort returns indexes in series values after all nans were dropped from it
    # hence drop the NaN values, rearrange the rest with argsort result and assign them back to their positions
    series_notna_values = series.dropna().values
    result[~nan_values_mask] = series_notna_values.take(argsort_indices[~nan_values_mask])

    return result


#   Restores a series and checks the correct arrangement of indices,
#   taking into account the same elements for unstable sortings
#   Example: pd.Series([15, 3, 7, 3, 1],[2, 4, 6, 8, 10])
#   Result can be pd.Series([1, 3, 3, 7, 15],[10, 4, 8, 6, 2]) or pd.Series([1, 3, 3, 7, 15],[10, 8, 4, 6, 2])
#   if indices correct - return 0; wrong - return 1
def restore_series_sort_values(series, my_result_index, ascending):
    value_dict = {}
    nan_list = []
    data = np.copy(series.values)
    index = np.copy(series.index)
    for value in range(len(data)):
        # if np.isnan(data[value]):
        if series.isna()[index[value]]:
            nan_list.append(index[value])
        if data[value] in value_dict:
            value_dict[data[value]].append(index[value])
        else:
            value_dict[data[value]] = [index[value]]
    na = series.isna().sum()
    sort = np.argsort(data)
    result = np.copy(my_result_index)
    if not ascending:
        sort[:len(result)-na] = sort[:len(result)-na][::-1]
    for i in range(len(result)-na):
        check = 0
        for j in value_dict[data[sort[i]]]:
            if j == result[i]:
                check = 1
        if check == 0:
            return 1
    for i in range(len(result)-na, len(result)):
        check = 0
        for j in nan_list:
            if result[i] == j:
                check = 1
        if check == 0:
            return 1
    return 0


def ljust_usecase(series, width):
    return series.str.ljust(width)


def ljust_with_fillchar_usecase(series, width, fillchar):
    return series.str.ljust(width, fillchar)


def rjust_usecase(series, width):
    return series.str.rjust(width)


def rjust_with_fillchar_usecase(series, width, fillchar):
    return series.str.rjust(width, fillchar)


def istitle_usecase(series):
    return series.str.istitle()


def isspace_usecase(series):
    return series.str.isspace()


def isalpha_usecase(series):
    return series.str.isalpha()


def islower_usecase(series):
    return series.str.islower()


def isalnum_usecase(series):
    return series.str.isalnum()


def isnumeric_usecase(series):
    return series.str.isnumeric()


def isdigit_usecase(series):
    return series.str.isdigit()


def isdecimal_usecase(series):
    return series.str.isdecimal()


def isupper_usecase(series):
    return series.str.isupper()


def lower_usecase(series):
    return series.str.lower()


def upper_usecase(series):
    return series.str.upper()


def strip_usecase(series, to_strip=None):
    return series.str.strip(to_strip)


def lstrip_usecase(series, to_strip=None):
    return series.str.lstrip(to_strip)


def rstrip_usecase(series, to_strip=None):
    return series.str.rstrip(to_strip)


def contains_usecase(series, pat, case=True, flags=0, na=None, regex=True):
    return series.str.contains(pat, case, flags, na, regex)


class TestSeries(
    TestSeries_apply,
    TestSeries_map,
    TestSeries_ops,
    TestCase
):

    @unittest.skip('Feature request: implement Series::ctor with list(list(type))')
    def test_create_list_list_unicode(self):
        def test_impl():
            S = pd.Series([
                          ['abc', 'defg', 'ijk'],
                          ['lmn', 'opq', 'rstuvwxyz']
                          ])
            return S
        hpat_func = self.jit(test_impl)

        result_ref = test_impl()
        result = hpat_func()
        pd.testing.assert_series_equal(result, result_ref)

    @unittest.skip('Feature request: implement Series::ctor with list(list(type))')
    def test_create_list_list_integer(self):
        def test_impl():
            S = pd.Series([
                          [123, 456, -789],
                          [-112233, 445566, 778899]
                          ])
            return S
        hpat_func = self.jit(test_impl)

        result_ref = test_impl()
        result = hpat_func()
        pd.testing.assert_series_equal(result, result_ref)

    @unittest.skip('Feature request: implement Series::ctor with list(list(type))')
    def test_create_list_list_float(self):
        def test_impl():
            S = pd.Series([
                          [1.23, -4.56, 7.89],
                          [11.2233, 44.5566, -778.899]
                          ])
            return S
        hpat_func = self.jit(test_impl)

        result_ref = test_impl()
        result = hpat_func()
        pd.testing.assert_series_equal(result, result_ref)

    def test_create_series1(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    @skip_numba_jit("Numba creates array with dtype=intp by default"
                    "On Win this fails with int32 vs int64 dtype mismatch")
    def test_create_series2(self):
        def test_impl(n):
            return pd.Series(np.arange(n))

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_create_series_param_name_literal(self):
        def test_impl():
            A = pd.Series([1, 2, 3], index=['A', 'C', 'B'], name='A')
            return A
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_create_series_param_name(self):
        def test_impl(name):
            A = pd.Series([1, 2, 3], index=['A', 'C', 'B'], name=name)
            return A
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func('A'), test_impl('A'))

    @skip_numba_jit
    def test_pass_series1(self):
        # TODO: check to make sure it is series type
        def test_impl(A):
            return (A == 2).sum()
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        self.assertEqual(hpat_func(S), test_impl(S))

    @skip_numba_jit
    def test_pass_series_str(self):
        def test_impl(A):
            return (A == 'a').sum()
        hpat_func = self.jit(test_impl)

        S = pd.Series(['a', 'b', 'c'], name='A')
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_pass_series_all_indexes(self):
        def test_impl(A):
            return A
        hpat_func = self.jit(test_impl)

        n = 11
        indexes_to_test = [
            None,
            list(np.arange(n)),
            np.arange(n),
            pd.RangeIndex(n),
            pd.Int64Index(np.arange(n)),
            gen_strlist(n)
        ]
        for index in indexes_to_test:
            with self.subTest(df_index=index):
                S = pd.Series(np.arange(n), index, name='A')
                pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_getattr_size(self):
        def test_impl(S):
            return S.size
        hpat_func = self.jit(test_impl)

        n = 11
        for S, expected in [
            (pd.Series(), 0),
            (pd.Series([]), 0),
            (pd.Series(np.arange(n)), n),
            (pd.Series([np.nan, 1, 2]), 3),
            (pd.Series(['1', '2', '3']), 3),
        ]:
            with self.subTest(S=S, expected=expected):
                self.assertEqual(hpat_func(S), expected)
                self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_argsort1(self):
        def test_impl(A):
            return A.argsort()
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        A = pd.Series(np.random.ranf(n))
        pd.testing.assert_series_equal(hpat_func(A), test_impl(A))

    def test_series_argsort2(self):
        def test_impl(S):
            return S.argsort()
        hpat_func = self.jit(test_impl)

        S = pd.Series([1, -1, 0, 2, np.nan], [1, 2, 3, 4, 5])
        pd.testing.assert_series_equal(test_impl(S), hpat_func(S))

    def test_series_argsort_full(self):
        def test_impl(series, kind):
            return series.argsort(axis=0, kind=kind, order=None)

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_numeric

        for data in all_data:
            S = pd.Series(data * 3)
            for kind in ['quicksort', 'mergesort']:
                result = test_impl(S, kind=kind)
                result_ref = hpat_func(S, kind=kind)
                if kind == 'mergesort':
                    pd.testing.assert_series_equal(result, result_ref)
                else:
                    # for non-stable sorting check that values of restored series are equal
                    np.testing.assert_array_equal(
                        series_values_from_argsort_result(S, result),
                        series_values_from_argsort_result(S, result_ref)
                    )

    def test_series_argsort_full_idx(self):
        def test_impl(series, kind):
            return series.argsort(axis=0, kind=kind, order=None)

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_numeric

        for data in all_data:
            data = data * 3
            for index in [gen_srand_array(len(data)), gen_frand_array(len(data)), range(len(data))]:
                S = pd.Series(data, index)
                for kind in ['quicksort', 'mergesort']:
                    result = test_impl(S, kind=kind)
                    result_ref = hpat_func(S, kind=kind)
                    if kind == 'mergesort':
                        pd.testing.assert_series_equal(result, result_ref)
                    else:
                        # for non-stable sorting check that values of restored series are equal
                        np.testing.assert_array_equal(
                            series_values_from_argsort_result(S, result),
                            series_values_from_argsort_result(S, result_ref)
                        )

    def test_series_attr6(self):
        def test_impl(A):
            return A.take([2, 3]).values
        hpat_func = self.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr7(self):
        def test_impl(A):
            return A.astype(np.float64)
        hpat_func = self.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_getattr_ndim(self):
        """Verifies getting Series attribute ndim is supported"""
        def test_impl(S):
            return S.ndim
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_getattr_T(self):
        """Verifies getting Series attribute T is supported"""
        def test_impl(S):
            return S.T
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_copy_str1(self):
        def test_impl(A):
            return A.copy()
        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_copy_int1(self):
        def test_impl(A):
            return A.copy()
        hpat_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_copy_deep(self):
        def test_impl(A, deep):
            return A.copy(deep=deep)
        hpat_func = self.jit(test_impl)

        for S in [
            pd.Series([1, 2]),
            pd.Series([1, 2], index=["a", "b"]),
            pd.Series([1, 2], name='A'),
            pd.Series([1, 2], index=["a", "b"], name='A'),
        ]:
            with self.subTest(S=S):
                for deep in (True, False):
                    with self.subTest(deep=deep):
                        actual = hpat_func(S, deep)
                        expected = test_impl(S, deep)

                        pd.testing.assert_series_equal(actual, expected)

                        self.assertEqual(actual.values is S.values, expected.values is S.values)
                        self.assertEqual(actual.values is S.values, not deep)

                        # Shallow copy of index is not supported yet
                        if deep:
                            self.assertEqual(actual.index is S.index, expected.index is S.index)
                            self.assertEqual(actual.index is S.index, not deep)

    def test_series_corr(self):
        def test_series_corr_impl(s1, s2, min_periods=None):
            return s1.corr(s2, min_periods=min_periods)

        hpat_func = self.jit(test_series_corr_impl)
        test_input_data1 = [[.2, .0, .6, .2],
                            [.2, .0, .6, .2, .5, .6, .7, .8],
                            [],
                            [2, 0, 6, 2],
                            [.2, .1, np.nan, .5, .3],
                            [-1, np.nan, 1, np.inf]]
        test_input_data2 = [[.3, .6, .0, .1],
                            [.3, .6, .0, .1, .8],
                            [],
                            [3, 6, 0, 1],
                            [.3, .2, .9, .6, np.nan],
                            [np.nan, np.nan, np.inf, np.nan]]
        for input_data1 in test_input_data1:
            for input_data2 in test_input_data2:
                s1 = pd.Series(input_data1)
                s2 = pd.Series(input_data2)
                for period in [None, 2, 1, 8, -4]:
                    result_ref = test_series_corr_impl(s1, s2, min_periods=period)
                    result = hpat_func(s1, s2, min_periods=period)
                    np.testing.assert_allclose(result, result_ref)

    def test_series_corr_unsupported_dtype(self):
        def test_series_corr_impl(s1, s2, min_periods=None):
            return s1.corr(s2, min_periods=min_periods)

        hpat_func = self.jit(test_series_corr_impl)
        s1 = pd.Series([.2, .0, .6, .2])
        s2 = pd.Series(['abcdefgh', 'a', 'abcdefg', 'ab', 'abcdef', 'abc'])
        s3 = pd.Series(['aaaaa', 'bbbb', 'ccc', 'dd', 'e'])
        s4 = pd.Series([.3, .6, .0, .1])

        with self.assertRaises(TypingError) as raises:
            hpat_func(s1, s2, min_periods=5)
        msg = 'Method corr(). The object other.data'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(s3, s4, min_periods=5)
        msg = 'Method corr(). The object self.data'
        self.assertIn(msg, str(raises.exception))

    def test_series_corr_unsupported_period(self):
        def test_series_corr_impl(s1, s2, min_periods=None):
            return s1.corr(s2, min_periods=min_periods)

        hpat_func = self.jit(test_series_corr_impl)
        s1 = pd.Series([.2, .0, .6, .2])
        s2 = pd.Series([.3, .6, .0, .1])

        with self.assertRaises(TypingError) as raises:
            hpat_func(s1, s2, min_periods='aaaa')
        msg = 'Method corr(). The object min_periods'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(s1, s2, min_periods=0.5)
        msg = 'Method corr(). The object min_periods'
        self.assertIn(msg, str(raises.exception))

    @skip_parallel
    @skip_inline
    def test_series_astype_int_to_str1(self):
        """Verifies Series.astype implementation with function 'str' as argument
           converts integer series to series of strings
        """
        def test_impl(S):
            return S.astype(str)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_parallel
    @skip_inline
    def test_series_astype_int_to_str2(self):
        """Verifies Series.astype implementation with a string literal dtype argument
           converts integer series to series of strings
        """
        def test_impl(S):
            return S.astype('str')
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_parallel
    @skip_inline
    def test_series_astype_str_to_str1(self):
        """Verifies Series.astype implementation with function 'str' as argument
           handles string series not changing it
        """
        def test_impl(S):
            return S.astype(str)
        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_parallel
    @skip_inline
    def test_series_astype_str_to_str2(self):
        """Verifies Series.astype implementation with a string literal dtype argument
           handles string series not changing it
        """
        def test_impl(S):
            return S.astype('str')
        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_parallel
    @skip_inline
    def test_series_astype_str_to_str_index_str(self):
        """Verifies Series.astype implementation with function 'str' as argument
           handles string series not changing it
        """

        def test_impl(S):
            return S.astype(str)

        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'], index=['d', 'e', 'f'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_parallel
    @skip_inline
    def test_series_astype_str_to_str_index_int(self):
        """Verifies Series.astype implementation with function 'str' as argument
           handles string series not changing it
        """

        def test_impl(S):
            return S.astype(str)

        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'], index=[1, 2, 3])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @unittest.skip('TODO: requires str(datetime64) support in Numba')
    def test_series_astype_dt_to_str1(self):
        """Verifies Series.astype implementation with function 'str' as argument
           converts datetime series to series of strings
        """
        def test_impl(A):
            return A.astype(str)
        hpat_func = self.jit(test_impl)

        S = pd.Series([pd.Timestamp('20130101 09:00:00'),
                       pd.Timestamp('20130101 09:00:02'),
                       pd.Timestamp('20130101 09:00:03')
                       ])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @unittest.skip('AssertionError: Series are different'
                   '[left]:  [0.000000, 1.000000, 2.000000, 3.000000, ...'
                   '[right]:  [0.0, 1.0, 2.0, 3.0, ...'
                   'TODO: needs alignment to NumPy on Numba side')
    def test_series_astype_float_to_str1(self):
        """Verifies Series.astype implementation with function 'str' as argument
           converts float series to series of strings
        """
        def test_impl(A):
            return A.astype(str)
        hpat_func = self.jit(test_impl)

        n = 11.0
        S = pd.Series(np.arange(n))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_astype_int32_to_int64(self):
        """Verifies Series.astype implementation with NumPy dtype argument
           converts series with dtype=int32 to series with dtype=int64
        """
        def test_impl(A):
            return A.astype(np.int64)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), dtype=np.int32)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_astype_int_to_float64(self):
        """Verifies Series.astype implementation with NumPy dtype argument
           converts named integer series to series of float
        """
        def test_impl(A):
            return A.astype(np.float64)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_astype_float_to_int32(self):
        """Verifies Series.astype implementation with NumPy dtype argument
           converts float series to series of integers
        """
        def test_impl(A):
            return A.astype(np.int32)
        hpat_func = self.jit(test_impl)

        n = 11.0
        S = pd.Series(np.arange(n))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_astype_literal_dtype1(self):
        """Verifies Series.astype implementation with a string literal dtype argument
           converts float series to series of integers
        """
        def test_impl(A):
            return A.astype('int32')
        hpat_func = self.jit(test_impl)

        n = 11.0
        S = pd.Series(np.arange(n))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @unittest.skip('TODO: needs Numba astype impl support converting unicode_type to int')
    def test_series_astype_str_to_int32(self):
        """Verifies Series.astype implementation with NumPy dtype argument
           converts series of strings to series of integers
        """
        import numba

        def test_impl(A):
            return A.astype(np.int32)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series([str(x) for x in np.arange(n) - n // 2])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @unittest.skip('TODO: needs Numba astype impl support converting unicode_type to float')
    def test_series_astype_str_to_float64(self):
        """Verifies Series.astype implementation with NumPy dtype argument
           converts series of strings to series of float
        """
        def test_impl(A):
            return A.astype(np.float64)
        hpat_func = self.jit(test_impl)

        S = pd.Series(['3.24', '1E+05', '-1', '-1.3E-01', 'nan', 'inf'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_inline
    def test_series_astype_str_index_str(self):
        """Verifies Series.astype implementation with function 'str' as argument
           handles string series not changing it
        """

        def test_impl(S):
            return S.astype(str)
        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'], index=['a', 'b', 'c'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_inline
    def test_series_astype_str_index_int(self):
        """Verifies Series.astype implementation with function 'str' as argument
           handles string series not changing it
        """

        def test_impl(S):
            return S.astype(str)

        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'], index=[2, 3, 5])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_astype_errors_ignore_return_self_str(self):
        """Verifies Series.astype implementation return self object on error
           if errors='ignore' is passed in arguments
        """

        def test_impl(S):
            return S.astype(np.float64, errors='ignore')

        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'], index=[2, 3, 5])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_numba_jit('TODO: implement np.call on Series in new-pipeline')
    def test_np_call_on_series1(self):
        def test_impl(A):
            return np.min(A)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_getattr_values(self):
        def test_impl(A):
            return A.values
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_values1(self):
        def test_impl(A):
            return (A == 2).values
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_getattr_shape1(self):
        def test_impl(A):
            return A.shape
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_static_setitem(self):
        def test_impl(A):
            A[0] = 2
            return (A == 2).sum()
        hpat_func = self.jit(test_impl)

        n = 11
        S1 = pd.Series(np.arange(n), name='A')
        S2 = S1.copy()
        self.assertEqual(hpat_func(S1), test_impl(S2))

    def test_series_setitem1(self):
        def test_impl(A, i):
            A[i] = 2
            return (A == 2).sum()
        hpat_func = self.jit(test_impl)

        n, i = 11, 0
        S1 = pd.Series(np.arange(n), name='A')
        S2 = S1.copy()
        self.assertEqual(hpat_func(S1, i), test_impl(S2, i))

    def test_series_setitem2(self):
        def test_impl(A, i):
            A[i] = 100
        hpat_func = self.jit(test_impl)

        n = 11
        S1 = pd.Series(np.arange(n), name='A')
        S2 = S1.copy()
        hpat_func(S1, 0)
        test_impl(S2, 0)
        pd.testing.assert_series_equal(S1, S2)

    @skip_numba_jit("Assertion Error. Effects of set are not observed due to dead code elimination"
                    "TODO: investigate how to support this in Numba")
    def test_series_setitem3(self):
        def test_impl(A, i):
            S = pd.Series(A)
            S[i] = 100
        hpat_func = self.jit(test_impl)

        n = 11
        A = np.arange(n)
        A1 = A.copy()
        A2 = A
        hpat_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1, A2)

    def test_series_setitem_with_filter1(self):
        def test_impl(A):
            A[A > 3] = 100
        hpat_func = self.jit(test_impl)

        n = 11
        S1 = pd.Series(np.arange(n))
        S2 = S1.copy()
        hpat_func(S1)
        test_impl(S2)
        pd.testing.assert_series_equal(S1, S2)

    @skip_numba_jit("Series.setitem misses specialization for OptionalType")
    def test_series_setitem_with_filter2(self):
        def test_impl(A, B):
            A[A > 3] = B[A > 3]
        hpat_func = self.jit(test_impl)

        n = 11
        A1 = pd.Series(np.arange(n), name='A')
        B = pd.Series(np.arange(n)**2, name='B')
        A2 = A1.copy()
        hpat_func(A1, B)
        test_impl(A2, B)
        pd.testing.assert_series_equal(A1, A2)

    def test_series_static_getitem(self):
        def test_impl(A):
            return A[1]
        hpat_func = self.jit(test_impl)

        A = pd.Series([1, 3, 5], ['1', '4', '2'], name='A')
        self.assertEqual(hpat_func(A), test_impl(A))

    def test_series_getitem_idx_int1(self):
        def test_impl(A, i):
            return A[i]
        hpat_func = self.jit(test_impl)

        n, i = 11, 0
        S = pd.Series(np.arange(n), name='A')
        # SDC and pandas results differ due to type limitation requirements:
        # SDC returns Series of one element, whereas pandas returns scalar, hence we align result_ref
        result = hpat_func(S, i)
        result_ref = pd.Series(test_impl(S, i), dtype=S.dtype, name='A')
        pd.testing.assert_series_equal(result, result_ref)
        self.assertEqual(len(result), 1)

    def test_series_getitem_idx_int2(self):
        def test_impl(A, i):
            return A[i]
        hpat_func = self.jit(test_impl)

        n, i = 11, 0
        S = pd.Series(np.arange(n), name='A')
        # SDC and pandas results differ due to type limitation requirements:
        # SDC returns Series of one element, whereas pandas returns scalar, hence we align result
        result = hpat_func(S, i).values[0]
        result_ref = test_impl(S, i)
        self.assertEqual(result, result_ref)

    def test_series_getitem_idx_int3(self):
        def test_impl(A, i):
            return A[i]
        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'], name='A')
        # SDC and pandas results differ due to type limitation requirements:
        # SDC returns Series of one element, whereas pandas returns scalar, hence we align result_ref
        result = hpat_func(S, 0)
        result_ref = pd.Series(test_impl(S, 0), name=S.name)
        pd.testing.assert_series_equal(result, result_ref)

    def test_series_iat1(self):
        def test_impl(A):
            return A.iat[3]
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n)**2, name='A')
        self.assertEqual(hpat_func(S), test_impl(S))

    @skip_numba_jit('TODO: implement setitem for SeriesGetitemAccessorType')
    def test_series_iat2(self):
        def test_impl(A):
            A.iat[3] = 1
            return A
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n)**2, name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_getitem_idx_series(self):
        def test_impl(A, B):
            return A[B]
        hpat_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3, 4, 5], [6, 0, 8, 0, 0], name='A')
        S2 = pd.Series([8, 6, 0], [12, 11, 14])
        pd.testing.assert_series_equal(hpat_func(S, S2), test_impl(S, S2))

    def test_series_getitem_idx_series_noidx(self):
        def test_impl(A, B):
            return A[B]
        hpat_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3, 4, 5], name='A')
        S2 = pd.Series([3, 2, 0])
        pd.testing.assert_series_equal(hpat_func(S, S2), test_impl(S, S2))

    def test_series_getitem_idx_series_index_str(self):
        def test_impl(A, B):
            return A[B]
        hpat_func = self.jit(test_impl)

        S = pd.Series(['1', '2', '3', '4', '5'], ['6', '7', '8', '9', '0'], name='A')
        S2 = pd.Series(['8', '6', '0'], ['12', '11', '14'])
        pd.testing.assert_series_equal(hpat_func(S, S2), test_impl(S, S2))

    @skip_numba_jit("TODO: support named Series indexes")
    def test_series_getitem_idx_series_named(self):
        def test_impl(A, B):
            return A[B]
        hpat_func = self.jit(test_impl)

        S = pd.Series(['1', '2', '3', '4', '5'], ['6', '7', '8', '9', '0'], name='A')
        S2 = pd.Series(['8', '6', '0'], ['12', '11', '14'], name='B')
        pd.testing.assert_series_equal(hpat_func(S, S2), test_impl(S, S2))

    def test_series_iloc1(self):
        def test_impl(A):
            return A.iloc[3]
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n)**2)
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_loc_return_ser(self):
        def test_impl(A):
            return A.loc[3]
        hpat_func = self.jit(test_impl)

        S = pd.Series([2, 4, 6, 5, 7], [1, 3, 5, 3, 3])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_getitem_idx_int4(self):
        def test_impl(S, key):
            return S[key]

        jit_impl = self.jit(test_impl)

        keys = [2, 2, 3]
        indices = [[2, 3, 5], [2, 3, 5], [2, 3, 5]]
        for key, index in zip(keys, indices):
            S = pd.Series([11, 22, 33], index, name='A')
            np.testing.assert_array_equal(jit_impl(S, key).values, np.array(test_impl(S, key)))

    def test_series_getitem_duplicate_index(self):
        def test_impl(A):
            return A[3]
        hpat_func = self.jit(test_impl)

        S = pd.Series([2, 4, 6, 33, 7], [1, 3, 5, 3, 3], name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_getitem_idx_int_slice(self):
        def test_impl(S, start, end):
            return S[start:end]

        jit_impl = self.jit(test_impl)

        starts = [0, 0]
        ends = [2, 2]
        indices = [[2, 3, 5], ['2', '3', '5'], ['2', '3', '5']]
        for start, end, index in zip(starts, ends, indices):
            S = pd.Series([11, 22, 33], index, name='A')
            ref_result = test_impl(S, start, end)
            jit_result = jit_impl(S, start, end)
            pd.testing.assert_series_equal(jit_result, ref_result)

    @skip_numba_jit('TODO: implement String slice support')
    def test_series_getitem_idx_str_slice(self):
        def test_impl(A):
            return A['1':'7']
        hpat_func = self.jit(test_impl)

        S = pd.Series(['1', '4', '6', '33', '7'], ['1', '3', '5', '4', '7'], name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_at(self):
        def test_impl(S, key):
            return S.at[key]

        jit_impl = self.jit(test_impl)

        keys = ['2', '2']
        all_data = [[11, 22, 33], [11, 22, 33]]
        indices = [['2', '22', '0'], ['2', '3', '5']]
        for key, data, index in zip(keys, all_data, indices):
            S = pd.Series(data, index, name='A')
            self.assertEqual(jit_impl(S, key), test_impl(S, key))

    def test_series_at_array(self):
        def test_impl(A):
            return A.at[3]
        hpat_func = self.jit(test_impl)

        S = pd.Series([2, 4, 6, 12, 0], [1, 3, 5, 3, 3], name='A')
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_loc(self):
        def test_impl(S, key):
            return S.loc[key]

        jit_impl = self.jit(test_impl)

        keys = [2, 2, 2]
        all_data = [[11, 22, 33], [11, 22, 33], [11, 22, 33]]
        indices = [[2, 3, 5], [2, 2, 2], [2, 4, 15]]
        for key, data, index in zip(keys, all_data, indices):
            S = pd.Series(data, index, name='A')
            np.testing.assert_array_equal(jit_impl(S, key).values, np.array(test_impl(S, key)))

    def test_series_loc_str(self):
        def test_impl(A):
            return A.loc['1']
        hpat_func = self.jit(test_impl)

        S = pd.Series([2, 4, 6], ['1', '3', '5'], name='A')
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_loc_array(self):
        def test_impl(A, n):
            return A.loc[n]
        hpat_func = self.jit(test_impl)

        S = pd.Series([1, 2, 4, 8, 6, 0], [1, 2, 4, 0, 6, 0], name='A')
        n = [0, 4, 2]
        cases = [n, np.array(n)]
        for n in cases:
            pd.testing.assert_series_equal(hpat_func(S, n), test_impl(S, n))

    def test_series_at_str(self):
        def test_impl(A):
            return A.at['1']
        hpat_func = self.jit(test_impl)

        S = pd.Series([2, 4, 6], ['1', '3', '5'], name='A')
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_loc_slice_nonidx(self):
        def test_impl(A):
            return A.loc[1:3]
        hpat_func = self.jit(test_impl)

        S = pd.Series([2, 4, 6, 6, 3], name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @unittest.skip('Slice string index not impl')
    def test_series_loc_slice_empty(self):
        def test_impl(A):
            return A.loc['301':'-4']
        hpat_func = self.jit(test_impl)

        S = pd.Series([2, 4, 6, 6, 3], ['-22', '-5', '-2', '300', '40000'], name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_iloc2(self):
        def test_impl(A):
            return A.iloc[3:8]
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n)**2, name='A')
        pd.testing.assert_series_equal(
            hpat_func(S), test_impl(S))

    def test_series_slice_loc_start(self):
        def test_impl(A, n):
            return A.loc[n:]
        hpat_func = self.jit(test_impl)

        all_data = [[1, 3, 5, 13, 22], [1, 3, 3, 13, 22], [22, 13, 5, 3, 1], [100, 3, 1, -3, -3]]
        key = [1, 3, 18]
        for index in all_data:
            for n in key:
                with self.subTest(index=index, start=n):
                    S = pd.Series([2, 4, 6, 6, 3], index, name='A')
                    pd.testing.assert_series_equal(hpat_func(S, n), test_impl(S, n))

    """
        For a pandas series: S = pd.Series([2, 4, 6, 6, 3], index=[100, 3, 0, -3, -3])
        pandas implementation of loc with slice returns different results if slice has start defined,
        and if it's omitted:
            >>>S.loc[:-3]
             100    2
             3      4
             0      6
            -3      6
            -3      3
            dtype: int64
            >>>S.loc[0:-3]
             0    6
            -3    6
            -3    3
            dtype: int64
        Current Numba SliceType implementation doesn't allow to distinguish these cases.
    """
    @unittest.expectedFailure  # add reference to Numba issue!
    def test_series_slice_loc_stop(self):
        def test_impl(A, n):
            return A.loc[:n]
        hpat_func = self.jit(test_impl)

        all_data = [[1, 3, 5, 13, 22], [1, 3, 3, 13, 22], [22, 13, 5, 3, 1], [100, 3, 0, -3, -3]]
        key = [1, 3, 18]
        for index in all_data:
            for n in key:
                with self.subTest(index=index, stop=n):
                    S = pd.Series([2, 4, 6, 6, 3], index, name='A')
                    pd.testing.assert_series_equal(hpat_func(S, n), test_impl(S, n))

    def test_series_slice_loc_start_stop(self):
        def test_impl(A, n, k):
            return A.loc[n:k]
        hpat_func = self.jit(test_impl)

        all_data = [[1, 3, 5, 13, 22], [1, 3, 3, 13, 22], [22, 13, 5, 3, 1], [100, 3, 0, -3, -3]]
        key = [-100, 1, 3, 18, 22, 100]
        for index in all_data:
            for data_left, data_right in combinations_with_replacement(key, 2):
                with self.subTest(index=index, left=data_left, right=data_right):
                    S = pd.Series([2, 4, 6, 6, 3], index, name='A')
                    pd.testing.assert_series_equal(hpat_func(S, data_left, data_right),
                                                   test_impl(S, data_left, data_right))

    def test_series_len(self):
        def test_impl(A):
            return len(A)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_box(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_box2(self):
        def test_impl():
            A = pd.Series(['1', '2', '3'])
            return A
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_list_str_unbox1(self):
        def test_impl(A):
            return A.iloc[0]
        hpat_func = self.jit(test_impl)

        S = pd.Series([['aa', 'b'], ['ccc'], []])
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

        # call twice to test potential refcount errors
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_iloc_array(self):
        def test_impl(A, n):
            return A.iloc[n]
        hpat_func = self.jit(test_impl)

        S = pd.Series([1, 2, 4, 8, 6, 0], [1, 2, 4, 8, 6, 0])
        n = np.array([0, 4, 2])
        pd.testing.assert_series_equal(hpat_func(S, n), test_impl(S, n))

    def test_series_iloc_callable(self):
        def test_impl(S):
            return S.iloc[(lambda a: abs(4 - a))]
        hpat_func = self.jit(test_impl)
        S = pd.Series([0, 6, 4, 7, 8], [0, 6, 66, 6, 8])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_np_typ_call_replace(self):
        # calltype replacement is tricky for np.typ() calls since variable
        # type can't provide calltype
        def test_impl(i):
            return np.int32(i)
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(1), test_impl(1))

    @skip_numba_jit('TODO: implement np.call on Series in new-pipeline')
    def test_series_ufunc1(self):
        def test_impl(A):
            return np.isinf(A).values
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    @skip_numba_jit('TODO: implement np.call on Series in new-pipeline')
    def test_series_empty_like(self):
        def test_impl(A):
            return np.empty_like(A)
        hpat_func = self.jit(test_impl)
        n = 11
        S = pd.Series(np.arange(n), name='A')
        result = hpat_func(S)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.dtype is S.dtype)

    def test_series_fillna_axis1(self):
        """Verifies Series.fillna() implementation handles 'index' as axis argument"""
        def test_impl(S):
            return S.fillna(5.0, axis='index')
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_fillna_axis2(self):
        """Verifies Series.fillna() implementation handles 0 as axis argument"""
        def test_impl(S):
            return S.fillna(5.0, axis=0)
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_fillna_axis3(self):
        """Verifies Series.fillna() implementation handles correct non-literal axis argument"""
        def test_impl(S, axis):
            return S.fillna(5.0, axis=axis)
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        for axis in [0, 'index']:
            pd.testing.assert_series_equal(hpat_func(S, axis), test_impl(S, axis))

    def test_series_fillna_float(self):
        """Verifies Series.fillna() applied to a named float Series with default index"""
        def test_impl(S):
            return S.fillna(5.0)
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf], name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_fillna_float_index1(self):
        """Verifies Series.fillna() implementation for float series with default index"""
        def test_impl(S):
            return S.fillna(5.0)
        hpat_func = self.jit(test_impl)

        for data in test_global_input_data_float64:
            S = pd.Series(data)
            pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_fillna_float_index2(self):
        """Verifies Series.fillna() implementation for float series with string index"""
        def test_impl(S):
            return S.fillna(5.0)
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf], ['a', 'b', 'c', 'd', 'e'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_fillna_float_index3(self):
        def test_impl(S):
            return S.fillna(5.0)
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf], index=[1, 2, 5, 7, 10])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_inline
    def test_series_fillna_str_index1(self):
        """Verifies Series.fillna() applied to a named string Series with default index"""
        def test_impl(S):
            return S.fillna("dd")
        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'b', None, 'cccd', ''], name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_inline
    def test_series_fillna_str_index2(self):
        """Verifies Series.fillna() implementation for series of strings with string index"""
        def test_impl(S):
            return S.fillna("dd")
        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'b', None, 'cccd', ''], ['a', 'b', 'c', 'd', 'e'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_inline
    def test_series_fillna_str_index3(self):
        def test_impl(S):
            return S.fillna("dd")

        hpat_func = self.jit(test_impl)

        S = pd.Series(['aa', 'b', None, 'cccd', ''], index=[1, 2, 5, 7, 10])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_fillna_float_inplace1(self):
        """Verifies Series.fillna() implementation for float series with default index and inplace argument True"""
        def test_impl(S):
            S.fillna(5.0, inplace=True)
            return S
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    @unittest.skip('TODO: add reflection support and check method return value')
    def test_series_fillna_float_inplace2(self):
        """Verifies Series.fillna(inplace=True) results are reflected back in the original float series"""
        def test_impl(S):
            return S.fillna(inplace=True)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        S2 = S1.copy()
        self.assertIsNone(hpat_func(S1))
        self.assertIsNone(test_impl(S2))
        pd.testing.assert_series_equal(S1, S2)

    def test_series_fillna_float_inplace3(self):
        """Verifies Series.fillna() implementation correcly handles omitted inplace argument as default False"""
        def test_impl(S):
            return S.fillna(5.0)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S1))
        pd.testing.assert_series_equal(S1, S2)

    def test_series_fillna_inplace_non_literal(self):
        """Verifies Series.fillna() implementation handles only Boolean literals as inplace argument"""
        def test_impl(S, param):
            S.fillna(5.0, inplace=param)
            return S
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        expected = TypingError
        self.assertRaises(expected, hpat_func, S, True)

    @skip_numba_jit('TODO: investigate why Numba types inplace as bool (non-literal value)')
    def test_series_fillna_str_inplace1(self):
        """Verifies Series.fillna() implementation for series of strings
           with default index and inplace argument True
        """
        def test_impl(S):
            S.fillna("dd", inplace=True)
            return S
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'cccd', ''])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    @unittest.skip('TODO (both): support StringArrayType reflection'
                   'TODO (new-style): investigate why Numba infers inplace type as bool (non-literal value)')
    def test_series_fillna_str_inplace2(self):
        """Verifies Series.fillna(inplace=True) results are reflected back in the original string series"""
        def test_impl(S):
            return S.fillna("dd", inplace=True)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'cccd', ''])
        S2 = S1.copy()
        self.assertIsNone(hpat_func(S1))
        self.assertIsNone(test_impl(S2))
        pd.testing.assert_series_equal(S1, S2)

    @skip_numba_jit('TODO: investigate why Numba types inplace as bool (non-literal value)')
    def test_series_fillna_str_inplace_empty1(self):
        def test_impl(A):
            A.fillna("", inplace=True)
            return A
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'cccd', ''])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    @unittest.skip('AssertionError: Series are different\n'
                   'Series length are different\n'
                   '[left]:  [NaT, 1970-12-01T00:00:00.000000000, 2012-07-25T00:00:00.000000000]\n'
                   '[right]: [2020-05-03T00:00:00.000000000, 1970-12-01T00:00:00.000000000, 2012-07-25T00:00:00.000000000]')
    def test_series_fillna_dt_no_index1(self):
        """Verifies Series.fillna() implementation for datetime series and np.datetime64 value"""
        def test_impl(S, value):
            return S.fillna(value)
        hpat_func = self.jit(test_impl)

        value = np.datetime64('2020-05-03', 'ns')
        S = pd.Series([pd.NaT, pd.Timestamp('1970-12-01'), pd.Timestamp('2012-07-25'), None])
        pd.testing.assert_series_equal(hpat_func(S, value), test_impl(S, value))

    @unittest.skip('TODO: change unboxing of pd.Timestamp Series or support conversion between PandasTimestampType and datetime64')
    def test_series_fillna_dt_no_index2(self):
        """Verifies Series.fillna() implementation for datetime series and pd.Timestamp value"""
        def test_impl(S):
            value = pd.Timestamp('2020-05-03')
            return S.fillna(value)
        hpat_func = self.jit(test_impl)

        S = pd.Series([pd.NaT, pd.Timestamp('1970-12-01'), pd.Timestamp('2012-07-25')])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_fillna_bool_no_index1(self):
        """Verifies Series.fillna() implementation for bool series with default index"""
        def test_impl(S):
            return S.fillna(True)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([True, False, False, True])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_fillna_int_no_index1(self):
        """Verifies Series.fillna() implementation for integer series with default index"""
        def test_impl(S):
            return S.fillna(7)
        hpat_func = self.jit(test_impl)

        n = 11
        S1 = pd.Series(np.arange(n, dtype=np.int64))
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_axis1(self):
        """Verifies Series.dropna() implementation handles 'index' as axis argument"""
        def test_impl(S):
            return S.dropna(axis='index')
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_axis2(self):
        """Verifies Series.dropna() implementation handles 0 as axis argument"""
        def test_impl(S):
            return S.dropna(axis=0)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_axis3(self):
        """Verifies Series.dropna() implementation handles correct non-literal axis argument"""
        def test_impl(S, axis):
            return S.dropna(axis=axis)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        S2 = S1.copy()
        for axis in [0, 'index']:
            pd.testing.assert_series_equal(hpat_func(S1, axis), test_impl(S2, axis))

    def test_series_dropna_float_index1(self):
        """Verifies Series.dropna() implementation for float series with default index"""
        def test_impl(S):
            return S.dropna()
        hpat_func = self.jit(test_impl)

        for data in test_global_input_data_float64:
            S1 = pd.Series(data)
            S2 = S1.copy()
            pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_float_index2(self):
        """Verifies Series.dropna() implementation for float series with string index"""
        def test_impl(S):
            return S.dropna()
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf], ['a', 'b', 'c', 'd', 'e'])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_str_index1(self):
        """Verifies Series.dropna() implementation for series of strings with default index"""
        def test_impl(S):
            return S.dropna()
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'cccd', ''])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_str_index2(self):
        """Verifies Series.dropna() implementation for series of strings with string index"""
        def test_impl(S):
            return S.dropna()
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'cccd', ''], ['a', 'b', 'c', 'd', 'e'])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_str_index3(self):
        def test_impl(S):
            return S.dropna()

        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'cccd', ''], index=[1, 2, 5, 7, 10])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    @unittest.skip('BUG: old-style dropna impl returns series without index, in new-style inplace is unsupported')
    def test_series_dropna_float_inplace_no_index1(self):
        """Verifies Series.dropna() implementation for float series with default index and inplace argument True"""
        def test_impl(S):
            S.dropna(inplace=True)
            return S
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    @unittest.skip('TODO: add reflection support and check method return value')
    def test_series_dropna_float_inplace_no_index2(self):
        """Verifies Series.dropna(inplace=True) results are reflected back in the original float series"""
        def test_impl(S):
            return S.dropna(inplace=True)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0, np.inf])
        S2 = S1.copy()
        self.assertIsNone(hpat_func(S1))
        self.assertIsNone(test_impl(S2))
        pd.testing.assert_series_equal(S1, S2)

    @unittest.skip('BUG: old-style dropna impl returns series without index, in new-style inplace is unsupported')
    def test_series_dropna_str_inplace_no_index1(self):
        """Verifies Series.dropna() implementation for series of strings
           with default index and inplace argument True
        """
        def test_impl(S):
            S.dropna(inplace=True)
            return S
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'cccd', ''])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    @unittest.skip('TODO: add reflection support and check method return value')
    def test_series_dropna_str_inplace_no_index2(self):
        """Verifies Series.dropna(inplace=True) results are reflected back in the original string series"""
        def test_impl(S):
            return S.dropna(inplace=True)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'cccd', ''])
        S2 = S1.copy()
        self.assertIsNone(hpat_func(S1))
        self.assertIsNone(test_impl(S2))
        pd.testing.assert_series_equal(S1, S2)

    @skip_numba_jit
    def test_series_dropna_str_parallel1(self):
        """Verifies Series.dropna() distributed work for series of strings with default index"""
        def test_impl(A):
            B = A.dropna()
            return (B == 'gg').sum()
        hpat_func = self.jit(distributed=['A'])(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'ccc', 'dd', 'gg'])
        start, end = get_start_end(len(S1))
        # TODO: gatherv
        self.assertEqual(hpat_func(S1[start:end]), test_impl(S1))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertTrue(count_array_OneDs() > 0)

    @unittest.skip('AssertionError: Series are different\n'
                   'Series length are different\n'
                   '[left]:  3, Int64Index([0, 1, 2], dtype=\'int64\')\n'
                   '[right]: 2, Int64Index([1, 2], dtype=\'int64\')')
    def test_series_dropna_dt_no_index1(self):
        """Verifies Series.dropna() implementation for datetime series with default index"""
        def test_impl(S):
            return S.dropna()
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([pd.NaT, pd.Timestamp('1970-12-01'), pd.Timestamp('2012-07-25')])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_bool_no_index1(self):
        """Verifies Series.dropna() implementation for bool series with default index"""
        def test_impl(S):
            return S.dropna()
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([True, False, False, True])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_int_no_index1(self):
        """Verifies Series.dropna() implementation for integer series with default index"""
        def test_impl(S):
            return S.dropna()
        hpat_func = self.jit(test_impl)

        n = 11
        S1 = pd.Series(np.arange(n, dtype=np.int64))
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_rename_str_noidx(self):
        def test_impl(A):
            return A.rename('B')
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2.0, np.nan, 1.0], name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_rename_str_noidx_noname(self):
        def test_impl(S):
            return S.rename('Name')
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3])
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    def test_series_rename_str_idx(self):
        def test_impl(S):
            return S.rename('Name')
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    def test_series_rename_no_name_str_noidx(self):
        def test_impl(S):
            return S.rename()
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], name='Name')
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    def test_series_rename_no_name_str_idx(self):
        def test_impl(S):
            return S.rename()
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='Name')
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    def test_series_rename_str_noidx_no_copy(self):
        def test_impl(S):
            return S.rename('Another Name', copy=False)
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], name='Name')
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    def test_series_rename_str_idx_no_copy(self):
        def test_impl(S):
            return S.rename('Another Name', copy=False)
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='Name')
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    @skip_numba_jit("Requires full scalar types (not only str) support as Series name")
    def test_series_rename_int_noidx(self):
        def test_impl(S):
            return S.rename(1)
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], name='Name')
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    @skip_numba_jit("Requires full scalar types (not only str) support as Series name")
    def test_series_rename_int_idx(self):
        def test_impl(S):
            return S.rename(1)
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='Name')
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    @skip_numba_jit("Requires full scalar types (not only str) support as Series name")
    def test_series_rename_float_noidx(self):
        def test_impl(S):
            return S.rename(1.1)
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], name='Name')
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    @skip_numba_jit("Requires full scalar types (not only str) support as Series name")
    def test_series_rename_float_idx(self):
        def test_impl(S):
            return S.rename(1.1)
        jit_func = self.jit(test_impl)

        S = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='Name')
        pd.testing.assert_series_equal(jit_func(S), test_impl(S))

    def test_series_sum_default(self):
        def test_impl(S):
            return S.sum()
        hpat_func = self.jit(test_impl)

        S = pd.Series([1., 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_sum_bool(self):
        def test_impl(S):
            return S.sum()
        hpat_func = self.jit(test_impl)

        S = pd.Series([True, True, False])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_sum_nan(self):
        def test_impl(S):
            return S.sum()
        hpat_func = self.jit(test_impl)

        # column with NA
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

        # all NA case should produce 0
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_sum_skipna_false(self):
        def test_impl(S):
            return S.sum(skipna=False)
        hpat_func = self.jit(test_impl)

        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(np.isnan(hpat_func(S)), np.isnan(test_impl(S)))

    def test_series_sum2(self):
        def test_impl(S):
            return (S + S).sum()
        hpat_func = self.jit(test_impl)

        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

        S = pd.Series([np.nan, np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_prod(self):
        def test_impl(S, skipna):
            return S.prod(skipna=skipna)
        hpat_func = self.jit(test_impl)

        data_samples = [
            [6, 6, 2, 1, 3, 3, 2, 1, 2],
            [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
            [6, 6.1, 2.2, 1, 3, 3, 2.2, 1, 2],
            [6, 6, np.nan, 2, np.nan, 1, 3, 3, np.inf, 2, 1, 2, np.inf],
            [1.1, 0.3, np.nan, 1.0, np.inf, 0.3, 2.1, np.nan, 2.2, np.inf],
            [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.inf],
        ]

        for data in data_samples:
            S = pd.Series(data)

            for skipna_var in [True, False]:
                actual = hpat_func(S, skipna=skipna_var)
                expected = test_impl(S, skipna=skipna_var)

                if np.isnan(actual) or np.isnan(expected):
                    # con not compare Nan != Nan directly
                    self.assertEqual(np.isnan(actual), np.isnan(expected))
                else:
                    self.assertAlmostEqual(actual, expected)

    def test_series_prod_skipna_default(self):
        def test_impl(S):
            return S.prod()
        hpat_func = self.jit(test_impl)

        S = pd.Series([np.nan, 2, 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_count1(self):
        def test_impl(S):
            return S.count()
        hpat_func = self.jit(test_impl)

        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

        S = pd.Series([np.nan, np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))

        S = pd.Series(['aa', 'bb', np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))

    def _mean_data_samples(self):
        yield [6, 6, 2, 1, 3, 3, 2, 1, 2]
        yield [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2]
        yield [6, 6.1, 2.2, 1, 3, 3, 2.2, 1, 2]
        yield [6, 6, np.nan, 2, np.nan, 1, 3, 3, np.inf, 2, 1, 2, np.inf]
        yield [1.1, 0.3, np.nan, 1.0, np.inf, 0.3, 2.1, np.nan, 2.2, np.inf]
        yield [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2]
        yield [np.nan, np.nan, np.nan]
        yield [np.nan, np.nan, np.inf]

    def _check_mean(self, pyfunc, *args):
        cfunc = self.jit(pyfunc)

        actual = cfunc(*args)
        expected = pyfunc(*args)
        if np.isnan(actual) or np.isnan(expected):
            self.assertEqual(np.isnan(actual), np.isnan(expected))
        else:
            np.testing.assert_almost_equal(actual, expected)

    def test_series_mean(self):
        def test_impl(S):
            return S.mean()

        for data in self._mean_data_samples():
            with self.subTest(data=data):
                S = pd.Series(data)
                self._check_mean(test_impl, S)

    def test_series_mean_skipna(self):
        def test_impl(S, skipna):
            return S.mean(skipna=skipna)

        for skipna in [True, False]:
            for data in self._mean_data_samples():
                S = pd.Series(data)
                self._check_mean(test_impl, S, skipna)

    def test_series_var1(self):
        def test_impl(S):
            return S.var()
        hpat_func = self.jit(test_impl)

        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_min(self):
        def test_impl(S):
            return S.min()
        hpat_func = self.jit(test_impl)

        # TODO type_min/type_max
        for input_data in [
                [np.nan, 2., np.nan, 3., np.inf, 1, -1000],
                [8, 31, 1123, -1024],
                [2., 3., 1, -1000, np.inf],
                [np.nan, np.nan, np.inf, np.nan],
            ]:
            S = pd.Series(input_data)

            result_ref = test_impl(S)
            result = hpat_func(S)
            self.assertEqual(result, result_ref)

    def test_series_min_param(self):
        def test_impl(S, param_skipna):
            return S.min(skipna=param_skipna)

        hpat_func = self.jit(test_impl)

        for input_data, param_skipna in [([np.nan, 2., np.nan, 3., 1, -1000, np.inf], True),
                                         ([2., 3., 1, np.inf, -1000], False)]:
            S = pd.Series(input_data)

            result_ref = test_impl(S, param_skipna)
            result = hpat_func(S, param_skipna)
            self.assertEqual(result, result_ref)

    @unittest.expectedFailure
    def test_series_min_param_fail(self):
        def test_impl(S, param_skipna):
            return S.min(skipna=param_skipna)

        hpat_func = self.jit(test_impl)

        cases = [
            ([2., 3., 1, np.inf, -1000, np.nan], False),  # min == np.nan
        ]

        for input_data, param_skipna in cases:
            S = pd.Series(input_data)

            result_ref = test_impl(S, param_skipna)
            result = hpat_func(S, param_skipna)
            self.assertEqual(result, result_ref)

    def test_series_max(self):
        def test_impl(S):
            return S.max()
        hpat_func = self.jit(test_impl)

        # TODO type_min/type_max
        for input_data in [
                [np.nan, 2., np.nan, 3., np.inf, 1, -1000],
                [8, 31, 1123, -1024],
                [2., 3., 1, -1000, np.inf],
                [np.inf, np.inf, np.inf, np.inf],
                [np.inf, np.nan, np.nan, np.nan],

                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1.0, np.nan, np.nan],
                [np.nan, 1.0, 1.0, np.nan],

                [np.nan, np.nan, 1.0, np.nan],
                [np.nan, np.nan, 1.0, np.nan, np.nan],

                [np.nan, np.nan, np.inf, np.nan],
                [np.nan, np.nan, np.inf, np.nan, np.nan],

                [np.nan, np.nan, np.nan, np.inf],
                np.arange(11),
            ]:
            with self.subTest(data=input_data):
                S = pd.Series(input_data)

                result_ref = test_impl(S)
                result = hpat_func(S)
                np.testing.assert_equal(result, result_ref)

    def test_series_max_param(self):
        def test_impl(S, param_skipna):
            return S.max(skipna=param_skipna)

        hpat_func = self.jit(test_impl)

        for input_data, param_skipna in [([np.nan, 2., np.nan, 3., 1, -1000, np.inf], True),
                                         ([2., 3., 1, np.inf, -1000], False)]:
            S = pd.Series(input_data)

            result_ref = test_impl(S, param_skipna)
            result = hpat_func(S, param_skipna)
            self.assertEqual(result, result_ref)

    def test_series_value_counts_number(self):
        def test_impl(S):
            return S.value_counts()

        input_data = [test_global_input_data_integer64, test_global_input_data_float64]
        extras = [[1, 2, 3, 1, 1, 3], [0.1, 0., 0.1, 0.1]]

        hpat_func = self.jit(test_impl)

        for data_to_test, extra in zip(input_data, extras):
            for d in data_to_test:
                data = d + extra
                with self.subTest(series_data=data):
                    S = pd.Series(data)
                    # use sort_index() due to possible different order of values with the same counts in results
                    result_ref = test_impl(S).sort_index()
                    result = hpat_func(S).sort_index()
                    pd.testing.assert_series_equal(result, result_ref)

    def test_series_value_counts_boolean(self):
        def test_impl(S):
            return S.value_counts()

        input_data = [True, False, True, True, False]

        sdc_func = self.jit(test_impl)

        S = pd.Series(input_data)
        result_ref = test_impl(S)
        result = sdc_func(S)
        pd.testing.assert_series_equal(result, result_ref)

    def test_series_value_counts_sort(self):
        def test_impl(S, value):
            return S.value_counts(sort=True, ascending=value)

        hpat_func = self.jit(test_impl)

        data = [1, 0, 0, 1, 1, -1, 0, -1, 0]

        for ascending in (False, True):
            with self.subTest(ascending=ascending):
                S = pd.Series(data)
                # to test sorting of result series works correctly do not use sort_index() on results!
                # instead ensure that there are no elements with the same frequency in the data
                result_ref = test_impl(S, ascending)
                result = hpat_func(S, ascending)
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_value_counts_numeric_dropna_false(self):
        def test_impl(S):
            return S.value_counts(dropna=False)

        data_to_test = [[1, 2, 3, 1, 1, 3],
                        [1, 2, 3, np.nan, 1, 3, np.nan, np.inf],
                        [0.1, 3., np.nan, 3., 0.1, 3., np.nan, np.inf, 0.1, 0.1]]

        hpat_func = self.jit(test_impl)

        for data in data_to_test:
            with self.subTest(series_data=data):
                S = pd.Series(data)
                pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_value_counts_str_dropna_false(self):
        def test_impl(S):
            return S.value_counts(dropna=False)

        data_to_test = [['a', '', 'a', '', 'b', None, 'a', '', None, 'b'],
                        ['dog', None, 'NaN', '', 'cat', None, 'cat', None, 'dog', ''],
                        ['dog', 'NaN', '', 'cat', 'cat', 'dog', '']]

        hpat_func = self.jit(test_impl)

        for data in data_to_test:
            with self.subTest(series_data=data):
                S = pd.Series(data)
                # use sort_index() due to possible different order of values with the same counts in results
                result_ref = test_impl(S).sort_index()
                result = hpat_func(S).sort_index()
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_value_counts_str_sort(self):
        def test_impl(S, ascending):
            return S.value_counts(sort=True, ascending=ascending)

        data_to_test = [['a', '', 'a', '', 'b', None, 'a', '', 'a', 'b'],
                        ['dog', 'cat', 'cat', 'cat', 'dog']]

        hpat_func = self.jit(test_impl)

        for data in data_to_test:
            for ascending in (True, False):
                with self.subTest(series_data=data, ascending=ascending):
                    S = pd.Series(data)
                    # to test sorting of result series works correctly do not use sort_index() on results!
                    # instead ensure that there are no elements with the same frequency in the data
                    result_ref = test_impl(S, ascending)
                    result = hpat_func(S, ascending)
                    pd.testing.assert_series_equal(result, result_ref)

    def test_series_value_counts_index(self):
        def test_impl(S):
            return S.value_counts()

        sdc_func = self.jit(test_impl)

        for data in test_global_input_data_integer64:
            index = np.arange(start=1, stop=len(data) + 1)
            with self.subTest(series_data=data):
                S = pd.Series(data, index=index)
                result = sdc_func(S)
                result_ref = test_impl(S)
                pd.testing.assert_series_equal(result.sort_index(), result_ref.sort_index())

    def test_series_value_counts_no_unboxing(self):
        def test_impl():
            S = pd.Series([1, 2, 3, 1, 1, 3])
            return S.value_counts()

        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl())

    @skip_numba_jit
    def test_series_dist_input1(self):
        """Verify distribution of a Series without index"""
        def test_impl(S):
            return S.max()
        hpat_func = self.jit(distributed={'S'})(test_impl)

        n = 111
        S = pd.Series(np.arange(n))
        start, end = get_start_end(n)
        self.assertEqual(hpat_func(S[start:end]), test_impl(S))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_series_dist_input2(self):
        """Verify distribution of a Series with integer index"""
        def test_impl(S):
            return S.max()
        hpat_func = self.jit(distributed={'S'})(test_impl)

        n = 111
        S = pd.Series(np.arange(n), 1 + np.arange(n))
        start, end = get_start_end(n)
        self.assertEqual(hpat_func(S[start:end]), test_impl(S[start:end]))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip("Passed if run single")
    def test_series_dist_input3(self):
        """Verify distribution of a Series with string index"""
        def test_impl(S):
            return S.max()
        hpat_func = self.jit(distributed={'S'})(test_impl)

        n = 111
        S = pd.Series(np.arange(n), ['abc{}'.format(id) for id in range(n)])
        start, end = get_start_end(n)
        self.assertEqual(hpat_func(S[start:end]), test_impl(S))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_series_tuple_input1(self):
        def test_impl(s_tup):
            return s_tup[0].max()
        hpat_func = self.jit(test_impl)

        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n) + 1.0)
        s_tup = (S, 1, S2)
        self.assertEqual(hpat_func(s_tup), test_impl(s_tup))

    @unittest.skip("pending handling of build_tuple in dist pass")
    def test_series_tuple_input_dist1(self):
        def test_impl(s_tup):
            return s_tup[0].max()
        hpat_func = self.jit(locals={'s_tup:input': 'distributed'})(test_impl)

        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n) + 1.0)
        start, end = get_start_end(n)
        s_tup = (S, 1, S2)
        h_s_tup = (S[start:end], 1, S2[start:end])
        self.assertEqual(hpat_func(h_s_tup), test_impl(s_tup))

    @skip_numba_jit
    def test_series_concat1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2]).values
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6., 7.])
        np.testing.assert_array_equal(hpat_func(S1, S2), test_impl(S1, S2))

    @skip_numba_jit
    def test_series_combine(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6.0, 21., 3.6, 5.])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    @skip_numba_jit
    def test_series_combine_float3264(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([np.float64(1), np.float64(2),
                        np.float64(3), np.float64(4), np.float64(5)])
        S2 = pd.Series([np.float32(1), np.float32(2),
                        np.float32(3), np.float32(4), np.float32(5)])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    @skip_numba_jit
    def test_series_combine_assert1(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1, 2, 3])
        S2 = pd.Series([6., 21., 3., 5.])
        with self.assertRaises(AssertionError):
            hpat_func(S1, S2)

    @skip_numba_jit
    def test_series_combine_assert2(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([6., 21., 3., 5.])
        S2 = pd.Series([1, 2, 3])
        with self.assertRaises(AssertionError):
            hpat_func(S1, S2)

    @skip_numba_jit
    def test_series_combine_integer(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b, 16)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1, 2, 3, 4, 5])
        S2 = pd.Series([6, 21, 3, 5])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    @skip_numba_jit
    def test_series_combine_different_types(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([6.1, 21.2, 3.3, 5.4, 6.7])
        S2 = pd.Series([1, 2, 3, 4, 5])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    @skip_numba_jit
    def test_series_combine_integer_samelen(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1, 2, 3, 4, 5])
        S2 = pd.Series([6, 21, 17, -5, 4])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    @skip_numba_jit
    def test_series_combine_samelen(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6.0, 21., 3.6, 5., 0.0])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    @skip_numba_jit
    def test_series_combine_value(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b, 1237.56)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6.0, 21., 3.6, 5.])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    @skip_numba_jit
    def test_series_combine_value_samelen(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2 * a + b, 1237.56)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6.0, 21., 3.6, 5., 0.0])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_abs1(self):
        def test_impl(S):
            return S.abs()
        hpat_func = self.jit(test_impl)

        S = pd.Series([np.nan, -2., 3., 0.5E-01, 0xFF, 0o7, 0b101])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_corr1(self):
        def test_impl(s1, s2):
            return s1.corr(s2)
        hpat_func = self.jit(test_impl)

        for pair in _cov_corr_series:
            s1, s2 = pair
            with self.subTest(s1=s1.values, s2=s2.values):
                result = hpat_func(s1, s2)
                result_ref = test_impl(s1, s2)
                np.testing.assert_almost_equal(result, result_ref)

    def test_series_str_center_default_fillchar(self):
        def test_impl(series, width):
            return series.str.center(width)

        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind1
        series = pd.Series(data)
        width = max(len(s) for s in data) + 10

        pd.testing.assert_series_equal(hpat_func(series, width),
                                       test_impl(series, width))

    def test_series_str_center(self):
        def test_impl(series, width, fillchar):
            return series.str.center(width, fillchar)

        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind1
        data_lengths = [len(s) for s in data]
        widths = [max(data_lengths) + 10, min(data_lengths)]

        for index in [None, list(range(len(data)))[::-1], data[::-1]]:
            series = pd.Series(data, index, name='A')
            for width, fillchar in product(widths, ['\t']):
                jit_result = hpat_func(series, width, fillchar)
                ref_result = test_impl(series, width, fillchar)
                pd.testing.assert_series_equal(jit_result, ref_result)

    def test_series_str_center_exception_unsupported_fillchar(self):
        def test_impl(series, width, fillchar):
            return series.str.center(width, fillchar)

        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind1
        series = pd.Series(data)
        width = max(len(s) for s in data) + 10

        assert_raises_ty_checker(self,
                                 ['Method center().', 'fillchar', 'int64', 'str'],
                                 hpat_func,
                                 series, width, 10)

    @unittest.expectedFailure  # https://jira.devtools.intel.com/browse/SAT-2348
    def test_series_str_center_exception_unsupported_kind4(self):
        def test_impl(series, width):
            return series.str.center(width)

        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind4
        series = pd.Series(data)
        width = max(len(s) for s in data) + 10

        jit_result = hpat_func(series, width)
        ref_result = test_impl(series, width)
        pd.testing.assert_series_equal(jit_result, ref_result)

    def test_series_str_center_with_none(self):
        def test_impl(series, width, fillchar):
            return series.str.center(width, fillchar)

        cfunc = self.jit(test_impl)
        idx = ['City 1', 'City 2', 'City 3', 'City 4', 'City 5', 'City 6', 'City 7', 'City 8']
        s = pd.Series(['New_York', 'Lisbon', np.nan, 'Tokyo', 'Paris', None, 'Munich', None], index=idx)
        pd.testing.assert_series_equal(cfunc(s, width=13, fillchar='*'), test_impl(s, width=13, fillchar='*'))

    def test_series_str_endswith(self):
        def test_impl(series, pat):
            return series.str.endswith(pat)

        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind4
        pats = [''] + [s[-min(len(s) for s in data):] for s in data] + data
        indices = [None, list(range(len(data)))[::-1], data[::-1]]
        names = [None, 'A']
        for index, name in product(indices, names):
            series = pd.Series(data, index, name=name)
            for pat in pats:
                pd.testing.assert_series_equal(hpat_func(series, pat),
                                               test_impl(series, pat))

    def test_series_str_endswith_exception_unsupported_na(self):
        def test_impl(series, pat, na):
            return series.str.endswith(pat, na)

        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_unicode_kind4)

        assert_raises_ty_checker(self,
                                 ['Method endswith().', 'na', 'unicode_type', 'bool'],
                                 hpat_func,
                                 series, '', 'None')

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, '', False)
        msg = 'Method endswith(). The object na\n expected: None'
        self.assertIn(msg, str(raises.exception))

    def test_series_str_find(self):
        def test_impl(series, sub):
            return series.str.find(sub)
        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind4
        subs = [''] + [s[:min(len(s) for s in data)] for s in data] + data
        indices = [None, list(range(len(data)))[::-1], data[::-1]]
        names = [None, 'A']
        for index, name in product(indices, names):
            series = pd.Series(data, index, name=name)
            for sub in subs:
                pd.testing.assert_series_equal(hpat_func(series, sub),
                                               test_impl(series, sub))

    def test_series_str_find_exception_unsupported_start(self):
        def test_impl(series, sub, start):
            return series.str.find(sub, start)
        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_unicode_kind4)
        self.assertRaisesRegex(TypingError,
                               r'Method find\(\)\. The object start\n'
                               r'\s+given: unicode_type\n'
                               r'\s+expected: None, int',
                               hpat_func,
                               series, '', '0')

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, '', 1)
        msg = 'Method find(). The object start\n expected: 0'
        self.assertIn(msg, str(raises.exception))

    def test_series_str_find_exception_unsupported_end(self):
        def test_impl(series, sub, start, end):
            return series.str.find(sub, start, end)
        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_unicode_kind4)

        self.assertRaisesRegex(TypingError,
                               r'Method find\(\)\. The object end\n'
                               r'\s+given: unicode_type\n'
                               r'\s+expected: None, int',
                               hpat_func,
                               series, '', 0, 'None')

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, '', 0, 0)
        msg = 'Method find(). The object end\n expected: None'
        self.assertIn(msg, str(raises.exception))

    def test_series_str_len1(self):
        def test_impl(S):
            return S.str.len()
        hpat_func = self.jit(test_impl)

        data = ['aa', 'abc', 'c', 'cccd']
        indices = [None, [1, 3, 2, 0], data]
        names = [None, 'A']
        for index, name in product(indices, names):
            S = pd.Series(data, index, name=name)
            pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_str_just_default_fillchar(self):
        data = test_global_input_data_unicode_kind1
        series = pd.Series(data)
        width = max(len(s) for s in data) + 5

        pyfuncs = [ljust_usecase, rjust_usecase]
        for pyfunc in pyfuncs:
            cfunc = self.jit(pyfunc)
            pd.testing.assert_series_equal(cfunc(series, width),
                                           pyfunc(series, width))

    def test_series_str_just(self):
        data = test_global_input_data_unicode_kind1
        data_lengths = [len(s) for s in data]
        widths = [max(data_lengths) + 5, min(data_lengths)]

        pyfuncs = [ljust_with_fillchar_usecase, rjust_with_fillchar_usecase]
        for index in [None, list(range(len(data)))[::-1], data[::-1]]:
            series = pd.Series(data, index, name='A')
            for width, fillchar in product(widths, ['\t']):
                for pyfunc in pyfuncs:
                    cfunc = self.jit(pyfunc)
                    jit_result = cfunc(series, width, fillchar)
                    ref_result = pyfunc(series, width, fillchar)
                    pd.testing.assert_series_equal(jit_result, ref_result)

    def test_series_str_just_exception_unsupported_fillchar(self):
        data = test_global_input_data_unicode_kind1
        series = pd.Series(data)
        width = max(len(s) for s in data) + 5

        pyfuncs = [('ljust', ljust_with_fillchar_usecase),
                   ('rjust', rjust_with_fillchar_usecase)]
        for name, pyfunc in pyfuncs:
            cfunc = self.jit(pyfunc)
            self.assertRaisesRegex(TypingError,
                                   fr'Method {name}\(\)\. The object fillchar\n'
                                   r'\s+given: int64\n'
                                   r'\s+expected: str',
                                   cfunc,
                                   series, width, 5)

    @unittest.expectedFailure  # https://jira.devtools.intel.com/browse/SAT-2348
    def test_series_str_just_exception_unsupported_kind4(self):
        data = test_global_input_data_unicode_kind4
        series = pd.Series(data)
        width = max(len(s) for s in data) + 5

        for pyfunc in [ljust_usecase, rjust_usecase]:
            cfunc = self.jit(pyfunc)
            jit_result = cfunc(series, width)
            ref_result = pyfunc(series, width)
            pd.testing.assert_series_equal(jit_result, ref_result)

    def test_series_str_ljust_with_none(self):
        def test_impl(series, width, fillchar):
            return series.str.ljust(width, fillchar)

        cfunc = self.jit(test_impl)
        idx = ['City 1', 'City 2', 'City 3', 'City 4', 'City 5', 'City 6', 'City 7', 'City 8']
        s = pd.Series(['New_York', 'Lisbon', np.nan, 'Tokyo', 'Paris', None, 'Munich', None], index=idx)
        pd.testing.assert_series_equal(cfunc(s, width=13, fillchar='*'), test_impl(s, width=13, fillchar='*'))


    def test_series_str_rjust_with_none(self):
        def test_impl(series, width, fillchar):
            return series.str.rjust(width, fillchar)

        cfunc = self.jit(test_impl)
        idx = ['City 1', 'City 2', 'City 3', 'City 4', 'City 5', 'City 6', 'City 7', 'City 8']
        s = pd.Series(['New_York', 'Lisbon', np.nan, 'Tokyo', 'Paris', None, 'Munich', None], index=idx)
        pd.testing.assert_series_equal(cfunc(s, width=13, fillchar='*'), test_impl(s, width=13, fillchar='*'))

    def test_series_str_startswith(self):
        def test_impl(series, pat):
            return series.str.startswith(pat)

        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind4
        pats = [''] + [s[:min(len(s) for s in data)] for s in data] + data
        indices = [None, list(range(len(data)))[::-1], data[::-1]]
        names = [None, 'A']
        for index, name in product(indices, names):
            series = pd.Series(data, index, name=name)
            for pat in pats:
                pd.testing.assert_series_equal(hpat_func(series, pat),
                                               test_impl(series, pat))

    def test_series_str_startswith_exception_unsupported_na(self):
        def test_impl(series, pat, na):
            return series.str.startswith(pat, na)

        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_unicode_kind4)

        assert_raises_ty_checker(self,
                                 ['Method startswith().', 'na', 'unicode_type', 'bool'],
                                 hpat_func,
                                 series, '', 'None')

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, '', False)
        msg = 'Method startswith(). The object na\n expected: None'
        self.assertIn(msg, str(raises.exception))

    def test_series_str_zfill(self):
        def test_impl(series, width):
            return series.str.zfill(width)

        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind1
        data_lengths = [len(s) for s in data]

        for index in [None, list(range(len(data)))[::-1], data[::-1]]:
            series = pd.Series(data, index, name='A')
            for width in [max(data_lengths) + 5, min(data_lengths)]:
                jit_result = hpat_func(series, width)
                ref_result = test_impl(series, width)
                pd.testing.assert_series_equal(jit_result, ref_result)

    @unittest.expectedFailure
    def test_series_str_zfill_limitation(self):
        def test_impl(series, width):
            return series.str.zfill(width)

        cfunc = self.jit(test_impl)
        s = pd.Series(['-1', '1', '1000', np.nan])
        jit_result = cfunc(s, 3)
        ref_result = test_impl(s, 3)
        pd.testing.assert_series_equal(jit_result, ref_result)

    def test_series_str_zfill_with_none(self):
        def test_impl(series, width):
            return series.str.zfill(width)

        cfunc = self.jit(test_impl)
        s = pd.Series(['1', '1000', np.nan])
        jit_result = cfunc(s, 3)
        ref_result = test_impl(s, 3)
        pd.testing.assert_series_equal(jit_result, ref_result)

    @unittest.expectedFailure  # https://jira.devtools.intel.com/browse/SAT-2348
    def test_series_str_zfill_exception_unsupported_kind4(self):
        def test_impl(series, width):
            return series.str.zfill(width)

        hpat_func = self.jit(test_impl)

        data = test_global_input_data_unicode_kind4
        series = pd.Series(data)
        width = max(len(s) for s in data) + 5

        jit_result = hpat_func(series, width)
        ref_result = test_impl(series, width)
        pd.testing.assert_series_equal(jit_result, ref_result)

    def test_series_str2str(self):
        common_methods = ['lower', 'upper', 'isupper']
        sdc_methods = ['capitalize', 'swapcase', 'title',
                       'lstrip', 'rstrip', 'strip']
        str2str_methods = common_methods[:]

        data = [' \tbbCD\t ', 'ABC', ' mCDm\t', 'abc']
        indices = [None]
        names = [None, 'A']
        indices += [[1, 3, 2, 0], data]

        for method in str2str_methods:
            func_lines = ['def test_impl(S):',
                          '  return S.str.{}()'.format(method)]
            func_text = '\n'.join(func_lines)
            test_impl = _make_func_from_text(func_text)
            hpat_func = self.jit(test_impl)

            check_names = method in common_methods
            for index, name in product(indices, names):
                S = pd.Series(data, index, name=name)
                pd.testing.assert_series_equal(hpat_func(S), test_impl(S),
                                               check_names=check_names)

    def test_series_capitalize_str(self):
        def test_impl(S):
            return S.str.capitalize()

        sdc_func = self.jit(test_impl)
        test_data = [test_global_input_data_unicode_kind4,
                     ['lower', None, 'CAPITALS', None, 'this is a sentence', 'SwApCaSe', None]]
        for data in test_data:
            with self.subTest(data=data):
                s = pd.Series(data)
                pd.testing.assert_series_equal(sdc_func(s), test_impl(s))

    def test_series_title_str(self):
        def test_impl(S):
            return S.str.title()

        sdc_func = self.jit(test_impl)
        test_data = [test_global_input_data_unicode_kind4,
                     ['lower', None, 'CAPITALS', None, 'this is a sentence', 'SwApCaSe', None]]
        for data in test_data:
            with self.subTest(data=data):
                s = pd.Series(data)
                pd.testing.assert_series_equal(sdc_func(s), test_impl(s))

    def test_series_upper_str(self):
        sdc_func = self.jit(upper_usecase)
        test_data = [test_global_input_data_unicode_kind4,
                     ['lower', None, 'CAPITALS', None, 'this is a sentence', 'SwApCaSe', None]]
        for data in test_data:
            with self.subTest(data=data):
                s = pd.Series(data)
                pd.testing.assert_series_equal(sdc_func(s), upper_usecase(s))

    def test_series_swapcase_str(self):
        def test_impl(S):
            return S.str.swapcase()

        sdc_func = self.jit(test_impl)
        test_data = [test_global_input_data_unicode_kind4,
                     ['lower', None, 'CAPITALS', None, 'this is a sentence', 'SwApCaSe', None]]
        for data in test_data:
            with self.subTest(data=data):
                s = pd.Series(data)
                pd.testing.assert_series_equal(sdc_func(s), test_impl(s))

    def test_series_casefold_str(self):
        def test_impl(S):
            return S.str.casefold()

        sdc_func = self.jit(test_impl)
        test_data = [test_global_input_data_unicode_kind4,
                     ['lower', None, 'CAPITALS', None, 'this is a sentence', 'SwApCaSe', None]]
        for data in test_data:
            with self.subTest(data=data):
                s = pd.Series(data)
                pd.testing.assert_series_equal(sdc_func(s), test_impl(s))

    @sdc_limitation
    def test_series_append_same_names(self):
        """SDC discards name"""
        def test_impl():
            s1 = pd.Series(data=[0, 1, 2], name='A')
            s2 = pd.Series(data=[3, 4, 5], name='A')
            return s1.append(s2)

        sdc_func = self.jit(test_impl)
        pd.testing.assert_series_equal(sdc_func(), test_impl())

    def test_series_append_single_ignore_index(self):
        """Verify Series.append() concatenates Series with other single Series ignoring indexes"""
        def test_impl(S, other):
            return S.append(other, ignore_index=True)
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1, np.nan], [-2., 5.0, np.inf, 0, -1]],
                         'string': [['a', None, 'bbbb', ''], ['dd', None, '', 'e', 'ttt']]}

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2 = [pd.Series(data) for data in data_list]
                pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_append_list_ignore_index(self):
        """Verify Series.append() concatenates Series with list of other Series ignoring indexes"""
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3], ignore_index=True)
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1], [-2., 5.0], [1.0]],
                         'string': [['a', None, ''], ['d', None], ['']]}

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2, S3 = [pd.Series(data) for data in data_list]
                pd.testing.assert_series_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    @unittest.skip('BUG: Pandas 0.25.1 Series.append() doesn\'t support tuple as appending values')
    def test_series_append_tuple_ignore_index(self):
        """Verify Series.append() concatenates Series with tuple of other Series ignoring indexes"""
        def test_impl(S1, S2, S3):
            return S1.append((S2, S3, ), ignore_index=True)
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1], [-2., 5.0], [1.0]]}
        dtype_to_data['string'] = [['a', None, ''], ['d', None], ['']]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2, S3 = [pd.Series(data) for data in data_list]
                pd.testing.assert_series_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    def test_series_append_single_index_default(self):
        """Verify Series.append() concatenates Series with other single Series respecting default indexes"""
        def test_impl(S, other):
            return S.append(other)
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1], [-2., 5.0]]}
        dtype_to_data['string'] = [['a', None, 'bbbb', ''], ['dd', None, '', 'e']]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2 = [pd.Series(data) for data in data_list]
                pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_append_list_index_default(self):
        """Verify Series.append() concatenates Series with list of other Series respecting default indexes"""
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3])
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1], [-2., 5.0], [1.0]]}
        dtype_to_data['string'] = [['a', 'b', 'q'], ['d', 'e'], ['s']]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2, S3 = [pd.Series(data) for data in data_list]
                pd.testing.assert_series_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    @unittest.skip('BUG: Pandas 0.25.1 Series.append() doesn\'t support tuple as appending values')
    def test_series_append_tuple_index_default(self):
        """Verify Series.append() concatenates Series with tuple of other Series respecting default indexes"""
        def test_impl(S1, S2, S3):
            return S1.append((S2, S3, ))
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1], [-2., 5.0], [1.0]]}
        dtype_to_data['string'] = [['a', 'b', 'q'], ['d', 'e'], ['s']]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2, S3 = [pd.Series(data) for data in data_list]
                pd.testing.assert_series_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    def test_series_append_single_index_int(self):
        """Verify Series.append() concatenates Series with other single Series respecting integer indexes"""
        def test_impl(S, other):
            return S.append(other)
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1, np.nan], [-2., 5.0, np.inf, 0, -1]]}
        dtype_to_data['string'] = [['a', None, 'bbbb', ''], ['dd', None, '', 'e', 'ttt']]
        indexes = [[1, 2, 3, 4], [7, 8, 11, 3, 4]]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2 = [pd.Series(data, index=indexes[i]) for i, data in enumerate(data_list)]
                pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_append_list_index_int(self):
        """Verify Series.append() concatenates Series with list of other Series respecting integer indexes"""
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3])
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1, np.nan], [-2., 5.0, np.inf, 0], [-1.0]]}
        dtype_to_data['string'] = [['a', None, 'bbbb', ''], ['dd', None, '', 'e'], ['ttt']]
        indexes = [[1, 2, 3, 4], [7, 8, 11, 3], [4]]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2, S3 = [pd.Series(data, index=indexes[i]) for i, data in enumerate(data_list)]
                pd.testing.assert_series_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    @unittest.skip('BUG: Pandas 0.25.1 Series.append() doesn\'t support tuple as appending values')
    def test_series_append_tuple_index_int(self):
        """Verify Series.append() concatenates Series with tuple of other Series respecting integer indexes"""
        def test_impl(S1, S2, S3):
            return S1.append((S2, S3, ))
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1, np.nan], [-2., 5.0, np.inf, 0], [-1.0]]}
        dtype_to_data['string'] = [['a', None, 'bbbb', ''], ['dd', None, '', 'e'], ['ttt']]
        indexes = [[1, 2, 3, 4], [7, 8, 11, 3], [4]]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2, S3 = [pd.Series(data, index=indexes[i]) for i, data in enumerate(data_list)]
                pd.testing.assert_series_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    def test_series_append_single_index_str(self):
        """Verify Series.append() concatenates Series with other single Series respecting string indexes"""
        def test_impl(S, other):
            return S.append(other)
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1, np.nan], [-2., 5.0, np.inf, 0, -1.0]]}
        dtype_to_data['string'] = [['a', None, 'bbbb', ''], ['dd', None, '', 'e', 'ttt']]
        indexes = [['a', 'bb', 'ccc', 'dddd'], ['a1', 'a2', 'a3', 'a4', 'a5']]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2 = [pd.Series(data, index=indexes[i]) for i, data in enumerate(data_list)]
                pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_append_list_index_str(self):
        """Verify Series.append() concatenates Series with list of other Series respecting string indexes"""
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3])
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1, np.nan], [-2., 5.0, np.inf, 0], [-1.0]]}
        dtype_to_data['string'] = [['a', None, 'bbbb', ''], ['dd', None, '', 'e'], ['ttt']]
        indexes = [['a', 'bb', 'ccc', 'dddd'], ['q', 't', 'a', 'x'], ['dd']]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2, S3 = [pd.Series(data, index=indexes[i]) for i, data in enumerate(data_list)]
                pd.testing.assert_series_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    @unittest.skip('BUG: Pandas 0.25.1 Series.append() doesn\'t support tuple as appending values')
    def test_series_append_tuple_index_str(self):
        """Verify Series.append() concatenates Series with tuple of other Series respecting string indexes"""
        def test_impl(S1, S2, S3):
            return S1.append((S2, S3, ))
        hpat_func = self.jit(test_impl)

        dtype_to_data = {'float': [[-2., 3., 9.1, np.nan], [-2., 5.0, np.inf, 0], [-1.0]]}
        dtype_to_data['string'] = [['a', None, 'bbbb', ''], ['dd', None, '', 'e'], ['ttt']]
        indexes = [['a', 'bb', 'ccc', 'dddd'], ['q', 't', 'a', 'x'], ['dd']]

        for dtype, data_list in dtype_to_data.items():
            with self.subTest(series_dtype=dtype, concatenated_data=data_list):
                S1, S2, S3 = [pd.Series(data, index=indexes[i]) for i, data in enumerate(data_list)]
                pd.testing.assert_series_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    def test_series_append_ignore_index_literal(self):
        """Verify Series.append() implementation handles ignore_index argument as Boolean literal"""
        def test_impl(S, other):
            return S.append(other, ignore_index=False)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([-2., 3., 9.1], ['a1', 'b1', 'c1'])
        S2 = pd.Series([-2., 5.0], ['a2', 'b2'])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_append_ignore_index_non_literal(self):
        """Verify Series.append() implementation raises if ignore_index argument is not a Boolean literal"""
        def test_impl(S, other, param):
            return S.append(other, ignore_index=param)
        hpat_func = self.jit(test_impl)

        ignore_index = True
        S1 = pd.Series([-2., 3., 9.1], ['a1', 'b1', 'c1'])
        S2 = pd.Series([-2., 5.0], ['a2', 'b2'])

        self.assertRaisesRegex(TypingError,
                               r'Method append\(\)\. The object ignore_index\n'
                               r'\s+given: bool\n'
                               r'\s+expected: literal Boolean constant',
                               hpat_func,
                               S1, S2, ignore_index)

    def test_series_append_single_dtype_promotion(self):
        """Verify Series.append() implementation handles appending single Series with different dtypes"""
        def test_impl(S, other):
            return S.append(other)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([-2., 3., 9.1], ['a1', 'b1', 'c1'])
        S2 = pd.Series([-2, 5], ['a2', 'b2'])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_append_list_dtype_promotion(self):
        """Verify Series.append() implementation handles appending list of Series with different dtypes"""
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3])
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([-2, 3, 9])
        S2 = pd.Series([-2., 5.0])
        S3 = pd.Series([1.0])
        pd.testing.assert_series_equal(hpat_func(S1, S2, S3),
                                       test_impl(S1, S2, S3))

    def test_series_isin_list1(self):
        def test_impl(S, values):
            return S.isin(values)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        values = [1, 2, 5, 7, 8]
        pd.testing.assert_series_equal(hpat_func(S, values), test_impl(S, values))

    def test_series_isin_index(self):
        def test_impl(S, values):
            return S.isin(values)
        hpat_func = self.jit(test_impl)

        n = 11
        data = np.arange(n)
        index = [item + 1 for item in data]
        S = pd.Series(data=data, index=index)
        values = [1, 2, 5, 7, 8]
        pd.testing.assert_series_equal(hpat_func(S, values), test_impl(S, values))

    def test_series_isin_name(self):
        def test_impl(S, values):
            return S.isin(values)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n), name='A')
        values = [1, 2, 5, 7, 8]
        pd.testing.assert_series_equal(hpat_func(S, values), test_impl(S, values))

    def test_series_isin_list2(self):
        def test_impl(S, values):
            return S.isin(values)
        hpat_func = self.jit(test_impl)

        n = 11.0
        S = pd.Series(np.arange(n))
        values = [1., 2., 5., 7., 8.]
        pd.testing.assert_series_equal(hpat_func(S, values), test_impl(S, values))

    def test_series_isin_list3(self):
        def test_impl(S, values):
            return S.isin(values)
        hpat_func = self.jit(test_impl)

        S = pd.Series(['a', 'b', 'q', 'w', 'c', 'd', 'e', 'r'])
        values = ['a', 'q', 'c', 'd', 'e']
        pd.testing.assert_series_equal(hpat_func(S, values), test_impl(S, values))

    def test_series_isin_set1(self):
        def test_impl(S, values):
            return S.isin(values)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        values = {1, 2, 5, 7, 8}
        pd.testing.assert_series_equal(hpat_func(S, values), test_impl(S, values))

    def test_series_isin_set2(self):
        def test_impl(S, values):
            return S.isin(values)
        hpat_func = self.jit(test_impl)

        n = 11.0
        S = pd.Series(np.arange(n))
        values = {1., 2., 5., 7., 8.}
        pd.testing.assert_series_equal(hpat_func(S, values), test_impl(S, values))

    @unittest.skip('TODO: requires hashable unicode strings in Numba')
    def test_series_isin_set3(self):
        def test_impl(S, values):
            return S.isin(values)
        hpat_func = self.jit(test_impl)

        S = pd.Series(['a', 'b', 'c', 'd', 'e'] * 2)
        values = {'b', 'c', 'e'}
        pd.testing.assert_series_equal(hpat_func(S, values), test_impl(S, values))

    def test_series_isna(self):
        def test_impl(S):
            return S.isna()

        jit_func = self.jit(test_impl)

        datas = [[0, 1, 2, 3], [0., 1., np.inf, np.nan], ['a', None, 'b', 'c'], [True, True, False, False]]
        indices = [None, [3, 2, 1, 0], ['a', 'b', 'c', 'd']]
        names = [None, 'A']

        for data, index, name in product(datas, indices, names):
            with self.subTest(data=data, index=index, name=name):
                series = pd.Series(data=data, index=index, name=name)
                jit_result = jit_func(series)
                ref_result = test_impl(series)
                pd.testing.assert_series_equal(jit_result, ref_result)

    def test_series_isnull(self):
        def test_impl(S):
            return S.isnull()

        jit_func = self.jit(test_impl)

        datas = [[0, 1, 2, 3], [0., 1., np.inf, np.nan], ['a', None, 'b', 'c'], [True, True, False, False]]
        indices = [None, [3, 2, 1, 0], ['a', 'b', 'c', 'd']]
        names = [None, 'A']

        for data, index, name in product(datas, indices, names):
            with self.subTest(data=data, index=index, name=name):
                series = pd.Series(data=data, index=index, name=name)
                jit_result = jit_func(series)
                ref_result = test_impl(series)
                pd.testing.assert_series_equal(jit_result, ref_result)

    def test_series_isnull1(self):
        def test_impl(S):
            return S.isnull()
        hpat_func = self.jit(test_impl)

        # column with NA
        S = pd.Series([np.nan, 2., 3.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_notna(self):
        def test_impl(S):
            return S.notna()

        jit_func = self.jit(test_impl)

        datas = [[0, 1, 2, 3], [0., 1., np.inf, np.nan], ['a', None, 'b', 'c'], [True, True, False, False]]
        indices = [None, [3, 2, 1, 0], ['a', 'b', 'c', 'd']]
        names = [None, 'A']

        for data, index, name in product(datas, indices, names):
            with self.subTest(data=data, index=index, name=name):
                series = pd.Series(data=data, index=index, name=name)
                jit_result = jit_func(series)
                ref_result = test_impl(series)
                pd.testing.assert_series_equal(jit_result, ref_result)

    @unittest.skip('AssertionError: Series are different')
    def test_series_dt_isna1(self):
        def test_impl(S):
            return S.isna()
        hpat_func = self.jit(test_impl)

        S = pd.Series([pd.NaT, pd.Timestamp('1970-12-01'), pd.Timestamp('2012-07-25')])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_nlargest(self):
        def test_impl():
            series = pd.Series([1., np.nan, -1., 0., min_float64, max_float64])
            return series.nlargest(4)
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(test_impl(), hpat_func())

    def test_series_nlargest_unboxing(self):
        def test_impl(series, n):
            return series.nlargest(n)
        hpat_func = self.jit(test_impl)

        for data in test_global_input_data_numeric + [[]]:
            series = pd.Series(data * 3)
            for n in range(-1, 10):
                ref_result = test_impl(series, n)
                jit_result = hpat_func(series, n)

                pd.testing.assert_series_equal(ref_result, jit_result)

    @skip_numba_jit('Series.nlargest() parallelism unsupported and parquet not supported')
    def test_series_nlargest_parallel(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.nlargest(4)
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(test_impl(), hpat_func())
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertTrue(count_array_OneDs() > 0)

    def test_series_nlargest_full(self):
        def test_impl(series, n, keep):
            return series.nlargest(n, keep)
        hpat_func = self.jit(test_impl)

        keep = 'first'
        for data in test_global_input_data_numeric + [[]]:
            series = pd.Series(data * 3)
            for n in range(-1, 10):
                ref_result = test_impl(series, n, keep)
                jit_result = hpat_func(series, n, keep)
                pd.testing.assert_series_equal(ref_result, jit_result)

    def test_series_nlargest_index(self):
        def test_impl(series, n):
            return series.nlargest(n)
        hpat_func = self.jit(test_impl)

        # TODO: check data == [] after index is fixed
        for data in test_global_input_data_numeric:
            data_duplicated = data * 3
            # TODO: add integer index not equal to range after index is fixed
            indexes = [range(len(data_duplicated))]
            indexes.append(gen_strlist(len(data_duplicated)))

            for index in indexes:
                series = pd.Series(data_duplicated, index)
                for n in range(-1, 10):
                    ref_result = test_impl(series, n)
                    jit_result = hpat_func(series, n)
                    pd.testing.assert_series_equal(ref_result, jit_result)

    def test_series_nlargest_typing(self):

        def test_impl(series, n, keep):
            return series.nlargest(n, keep)
        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_float64[0])
        for n, ntype in [(True, types.boolean), (None, types.none),
                         (0.1, 'float64'), ('n', types.unicode_type)]:
            self.assertRaisesRegex(TypingError,
                                   r'Method nlargest\(\)\. The object n\n'
                                   fr'\s+given: {ntype}\n'
                                   r'\s+expected: int',
                                   hpat_func,
                                   series, n=n, keep='first')

        for keep, dtype in [(True, types.boolean), (None, types.none),
                            (0.1, 'float64'), (1, 'int64')]:
            self.assertRaisesRegex(TypingError,
                                   r'Method nlargest\(\)\. The object keep\n'
                                   fr'\s+given: {dtype}\n'
                                   r'\s+expected: str',
                                   hpat_func,
                                   series, n=5, keep=keep)

    def test_series_nlargest_unsupported(self):
        msg = "Method nlargest(). Unsupported parameter. Given 'keep' != 'first'"

        def test_impl(series, n, keep):
            return series.nlargest(n, keep)
        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_float64[0])
        for keep in ['last', 'all', '']:
            with self.assertRaises(ValueError) as raises:
                hpat_func(series, n=5, keep=keep)
            self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, n=5, keep='last')
        self.assertIn(msg, str(raises.exception))

    def test_series_nsmallest(self):
        def test_impl():
            series = pd.Series([1., np.nan, -1., 0., min_float64, max_float64])
            return series.nsmallest(4)
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(test_impl(), hpat_func())

    def test_series_nsmallest_unboxing(self):
        def test_impl(series, n):
            return series.nsmallest(n)
        hpat_func = self.jit(test_impl)

        for data in test_global_input_data_numeric + [[]]:
            series = pd.Series(data * 3)
            for n in range(-1, 10):
                ref_result = test_impl(series, n)
                jit_result = hpat_func(series, n)
                pd.testing.assert_series_equal(ref_result, jit_result)

    @skip_numba_jit('Series.nsmallest() parallelism unsupported and parquet not supported')
    def test_series_nsmallest_parallel(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.nsmallest(4)
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(test_impl(), hpat_func())
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertTrue(count_array_OneDs() > 0)

    def test_series_nsmallest_full(self):
        def test_impl(series, n, keep):
            return series.nsmallest(n, keep)
        hpat_func = self.jit(test_impl)

        keep = 'first'
        for data in test_global_input_data_numeric + [[]]:
            series = pd.Series(data * 3)
            for n in range(-1, 10):
                ref_result = test_impl(series, n, keep)
                jit_result = hpat_func(series, n, keep)
                pd.testing.assert_series_equal(ref_result, jit_result)

    def test_series_nsmallest_index(self):
        def test_impl(series, n):
            return series.nsmallest(n)
        hpat_func = self.jit(test_impl)

        # TODO: check data == [] after index is fixed
        for data in test_global_input_data_numeric:
            data_duplicated = data * 3
            # TODO: add integer index not equal to range after index is fixed
            indexes = [range(len(data_duplicated))]
            indexes.append(gen_strlist(len(data_duplicated)))

            for index in indexes:
                series = pd.Series(data_duplicated, index)
                for n in range(-1, 10):
                    ref_result = test_impl(series, n)
                    jit_result = hpat_func(series, n)
                    pd.testing.assert_series_equal(ref_result, jit_result)

    def test_series_nsmallest_typing(self):

        def test_impl(series, n, keep):
            return series.nsmallest(n, keep)
        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_float64[0])
        for n, ntype in [(True, types.boolean), (None, types.none),
                         (0.1, 'float64'), ('n', types.unicode_type)]:
            self.assertRaisesRegex(TypingError,
                                   r'Method nsmallest\(\)\. The object n\n'
                                   fr'\s+given: {ntype}\n'
                                   r'\s+expected: int',
                                   hpat_func,
                                   series, n=n, keep='first')

        for keep, dtype in [(True, types.boolean), (None, types.none),
                            (0.1, 'float64'), (1, 'int64')]:
            self.assertRaisesRegex(TypingError,
                                   r'Method nsmallest\(\)\. The object keep\n'
                                   fr'\s+given: {dtype}\n'
                                   r'\s+expected: str',
                                   hpat_func,
                                   series, n=5, keep=keep)

    def test_series_nsmallest_unsupported(self):
        msg = "Method nsmallest(). Unsupported parameter. Given 'keep' != 'first'"

        def test_impl(series, n, keep):
            return series.nsmallest(n, keep)
        hpat_func = self.jit(test_impl)

        series = pd.Series(test_global_input_data_float64[0])
        for keep in ['last', 'all', '']:
            with self.assertRaises(ValueError) as raises:
                hpat_func(series, n=5, keep=keep)
            self.assertIn(msg, str(raises.exception))

        with self.assertRaises(ValueError) as raises:
            hpat_func(series, n=5, keep='last')
        self.assertIn(msg, str(raises.exception))

    def test_series_head1(self):
        def test_impl(S):
            return S.head(4)
        hpat_func = self.jit(test_impl)

        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_head_named(self):
        def test_impl(S):
            return S.head(4)
        hpat_func = self.jit(test_impl)

        m = 100
        np.random.seed(0)
        S = pd.Series(data=np.random.randint(-30, 30, m), name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_head_default1(self):
        """Verifies default head method for non-distributed pass of Series with no index"""
        def test_impl(S):
            return S.head()
        hpat_func = self.jit(test_impl)

        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_head_index1(self):
        """Verifies head method for Series with integer index created inside jitted function"""
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], [8, 1, 6, 0, 9, 1, 3])
            return S.head(3)
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_head_index_named(self):
        """Verifies head method for Series with integer index created inside jitted function"""
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], [8, 1, 6, 0, 9, 1, 3], name='A')
            return S.head(3)
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_head_index2(self):
        """Verifies head method for Series with string index created inside jitted function"""
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], ['a', 'ab', 'abc', 'c', 'f', 'hh', ''])
            return S.head(3)
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_head_index3(self):
        """Verifies head method for non-distributed pass of Series with integer index"""
        def test_impl(S):
            return S.head(3)
        hpat_func = self.jit(test_impl)

        S = pd.Series([6, 9, 2, 3, 6, 4, 5], [8, 1, 6, 0, 9, 1, 3])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_head_index4(self):
        """Verifies head method for non-distributed pass of Series with string index"""
        def test_impl(S):
            return S.head(3)
        hpat_func = self.jit(test_impl)

        S = pd.Series([6, 9, 2, 4, 6, 4, 5], ['a', 'ab', 'abc', 'c', 'f', 'hh', ''])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_numba_jit
    def test_series_head_parallel1(self):
        """Verifies head method for distributed Series with string data and no index"""
        def test_impl(S):
            return S.head(7)

        hpat_func = self.jit(distributed={'S'})(test_impl)

        # need to test different lenghts, as head's size is fixed and implementation
        # depends on relation of size of the data per processor to output data size
        for n in range(1, 5):
            S = pd.Series(['a', 'ab', 'abc', 'c', 'f', 'hh', ''] * n)
            start, end = get_start_end(len(S))
            pd.testing.assert_series_equal(hpat_func(S[start:end]), test_impl(S[start:end]))
            self.assertTrue(count_array_OneDs() > 0)

    @skip_numba_jit
    @unittest.expectedFailure
    def test_series_head_index_parallel1(self):
        """Verifies head method for distributed Series with integer index"""
        def test_impl(S):
            return S.head(3)
        hpat_func = self.jit(distributed={'S'})(test_impl)

        S = pd.Series([6, 9, 2, 3, 6, 4, 5], [8, 1, 6, 0, 9, 1, 3])
        start, end = get_start_end(len(S))
        pd.testing.assert_series_equal(hpat_func(S[start:end]), test_impl(S[start:end]))
        self.assertTrue(count_array_OneDs() > 0)

    @unittest.skip("Passed if run single")
    def test_series_head_index_parallel2(self):
        """Verifies head method for distributed Series with string index"""
        def test_impl(S):
            return S.head(3)
        hpat_func = self.jit(distributed={'S'})(test_impl)

        S = pd.Series([6, 9, 2, 3, 6, 4, 5], ['a', 'ab', 'abc', 'c', 'f', 'hh', ''])
        start, end = get_start_end(len(S))
        pd.testing.assert_series_equal(hpat_func(S[start:end]), test_impl(S))
        self.assertTrue(count_array_OneDs() > 0)

    def test_series_head_noidx_float(self):
        def test_impl(S, n):
            return S.head(n)
        hpat_func = self.jit(test_impl)
        for input_data in test_global_input_data_float64:
            S = pd.Series(input_data)
            for n in [-1, 0, 2, 3]:
                result_ref = test_impl(S, n)
                result_jit = hpat_func(S, n)
                pd.testing.assert_series_equal(result_jit, result_ref)

    def test_series_head_noidx_int(self):
        def test_impl(S, n):
            return S.head(n)
        hpat_func = self.jit(test_impl)
        for input_data in test_global_input_data_integer64:
            S = pd.Series(input_data)
            for n in [-1, 0, 2, 3]:
                result_ref = test_impl(S, n)
                result_jit = hpat_func(S, n)
                pd.testing.assert_series_equal(result_jit, result_ref)

    def test_series_head_noidx_num(self):
        def test_impl(S, n):
            return S.head(n)
        hpat_func = self.jit(test_impl)
        for input_data in test_global_input_data_numeric:
            S = pd.Series(input_data)
            for n in [-1, 0, 2, 3]:
                result_ref = test_impl(S, n)
                result_jit = hpat_func(S, n)
                pd.testing.assert_series_equal(result_jit, result_ref)

    @unittest.skip("Old implementation not work with n negative and data str")
    def test_series_head_noidx_str(self):
        def test_impl(S, n):
            return S.head(n)
        hpat_func = self.jit(test_impl)
        input_data = test_global_input_data_unicode_kind4
        S = pd.Series(input_data)
        for n in [-1, 0, 2, 3]:
            result_ref = test_impl(S, n)
            result_jit = hpat_func(S, n)
            pd.testing.assert_series_equal(result_jit, result_ref)

    @unittest.skip("Broke another three tests")
    def test_series_head_idx(self):
        def test_impl(S):
            return S.head()

        def test_impl_param(S, n):
            return S.head(n)

        hpat_func = self.jit(test_impl)

        data_test = [[6, 6, 2, 1, 3, 3, 2, 1, 2],
                     [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
                     [6, 6.1, 2.2, 1, 3, 0, 2.2, 1, 2],
                     ['as', 'b', 'abb', 'sss', 'ytr65', '', 'qw', 'a', 'b'],
                     [6, 6, 2, 1, 3, np.inf, np.nan, np.nan, np.nan],
                     [3., 5.3, np.nan, np.nan, np.inf, np.inf, 4.4, 3.7, 8.9]
                     ]

        for input_data in data_test:
            for index_data in data_test:
                S = pd.Series(input_data, index_data)

                result_ref = test_impl(S)
                result = hpat_func(S)
                pd.testing.assert_series_equal(result, result_ref)

                hpat_func_param1 = self.jit(test_impl_param)

                for param1 in [1, 3, 7]:
                    result_param1_ref = test_impl_param(S, param1)
                    result_param1 = hpat_func_param1(S, param1)
                    pd.testing.assert_series_equal(result_param1, result_param1_ref)

    def test_series_median1(self):
        """Verifies median implementation for float and integer series of random data"""
        def test_impl(S):
            return S.median()
        hpat_func = self.jit(test_impl)

        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(hpat_func(S), test_impl(S))

        S = pd.Series(np.random.ranf(m))
        self.assertEqual(hpat_func(S), test_impl(S))

        # odd size
        m = 101
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(hpat_func(S), test_impl(S))

        S = pd.Series(np.random.ranf(m))
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_median_skipna_default1(self):
        """Verifies median implementation with default skipna=True argument on a series with NA values"""
        def test_impl(S):
            return S.median()
        hpat_func = self.jit(test_impl)

        S = pd.Series([2., 3., 5., np.nan, 5., 6., 7.])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_median_skipna_false1(self):
        """Verifies median implementation with skipna=False on a series with NA values"""
        def test_impl(S):
            return S.median(skipna=False)
        hpat_func = self.jit(test_impl)

        # np.inf is not NaN, so verify that a correct number is returned
        S1 = pd.Series([2., 3., 5., np.inf, 5., 6., 7.])
        self.assertEqual(hpat_func(S1), test_impl(S1))

        # TODO: both return values are 'nan', but SDC's is not np.nan, hence checking with
        # assertIs() doesn't work - check if it's Numba relatated
        S2 = pd.Series([2., 3., 5., np.nan, 5., 6., 7.])
        self.assertEqual(np.isnan(hpat_func(S2)), np.isnan(test_impl(S2)))

    @skip_numba_jit
    def test_series_median_parallel1(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.median()
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertTrue(count_array_OneDs() > 0)

    @skip_numba_jit
    def test_series_argsort_parallel(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.argsort().values
        hpat_func = self.jit(test_impl)

        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_idxmin1(self):
        def test_impl(a):
            return a.idxmin()
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        s = pd.Series(np.random.ranf(n))
        np.testing.assert_array_equal(hpat_func(s), test_impl(s))

    def test_series_idxmin_str(self):
        def test_impl(s):
            return s.idxmin()
        hpat_func = self.jit(test_impl)

        s = pd.Series([8, 6, 34, np.nan], ['a', 'ab', 'abc', 'c'])
        self.assertEqual(hpat_func(s), test_impl(s))

    @unittest.skip("Skipna is not implemented")
    def test_series_idxmin_str_idx(self):
        def test_impl(s):
            return s.idxmin(skipna=False)

        hpat_func = self.jit(test_impl)

        s = pd.Series([8, 6, 34, np.nan], ['a', 'ab', 'abc', 'c'])
        self.assertEqual(hpat_func(s), test_impl(s))

    def test_series_idxmin_no(self):
        def test_impl(s):
            return s.idxmin()
        hpat_func = self.jit(test_impl)

        s = pd.Series([8, 6, 34, np.nan])
        self.assertEqual(hpat_func(s), test_impl(s))

    def test_series_idxmin_int(self):
        def test_impl(s):
            return s.idxmin()
        hpat_func = self.jit(test_impl)

        s = pd.Series([1, 2, 3], [4, 45, 14])
        self.assertEqual(hpat_func(s), test_impl(s))

    def test_series_idxmin_noidx(self):
        def test_impl(s):
            return s.idxmin()

        hpat_func = self.jit(test_impl)

        data_test = [[6, 6, 2, 1, 3, 3, 2, 1, 2],
                     [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
                     [6, 6.1, 2.2, 1, 3, 0, 2.2, 1, 2],
                     [6, 6, 2, 1, 3, np.inf, np.nan, np.nan, np.nan],
                     [3., 5.3, np.nan, np.nan, np.inf, np.inf, 4.4, 3.7, 8.9]
                     ]

        for input_data in data_test:
            s = pd.Series(input_data)

            result_ref = test_impl(s)
            result = hpat_func(s)
            self.assertEqual(result, result_ref)

    def test_series_idxmin_idx(self):
        def test_impl(s):
            return s.idxmin()

        hpat_func = self.jit(test_impl)

        data_test = [[6, 6, 2, 1, 3, 3, 2, 1, 2],
                     [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
                     [6, 6.1, 2.2, 1, 3, 0, 2.2, 1, 2],
                     [6, 6, 2, 1, 3, -np.inf, np.nan, np.inf, np.nan],
                     [3., 5.3, np.nan, np.nan, np.inf, np.inf, 4.4, 3.7, 8.9]
                     ]

        for input_data in data_test:
            for index_data in data_test:
                s = pd.Series(input_data, index_data)
                result_ref = test_impl(s)
                result = hpat_func(s)
                if np.isnan(result) or np.isnan(result_ref):
                    self.assertEqual(np.isnan(result), np.isnan(result_ref))
                else:
                    self.assertEqual(result, result_ref)

    def test_series_idxmax1(self):
        def test_impl(a):
            return a.idxmax()
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        s = pd.Series(np.random.ranf(n))
        np.testing.assert_array_equal(hpat_func(s), test_impl(s))

    @unittest.skip("Skipna is not implemented")
    def test_series_idxmax_str_idx(self):
        def test_impl(s):
            return s.idxmax(skipna=False)

        hpat_func = self.jit(test_impl)

        s = pd.Series([8, 6, 34, np.nan], ['a', 'ab', 'abc', 'c'])
        self.assertEqual(hpat_func(s), test_impl(s))

    def test_series_idxmax_noidx(self):
        def test_impl(s):
            return s.idxmax()

        hpat_func = self.jit(test_impl)

        data_test = [[6, 6, 2, 1, 3, 3, 2, 1, 2],
                     [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
                     [6, 6.1, 2.2, 1, 3, 0, 2.2, 1, 2],
                     [6, 6, 2, 1, 3, np.inf, np.nan, np.inf, np.nan],
                     [3., 5.3, np.nan, np.nan, np.inf, np.inf, 4.4, 3.7, 8.9]
                     ]

        for input_data in data_test:
            s = pd.Series(input_data)

            result_ref = test_impl(s)
            result = hpat_func(s)
            self.assertEqual(result, result_ref)

    def test_series_idxmax_idx(self):
        def test_impl(s):
            return s.idxmax()

        hpat_func = self.jit(test_impl)

        data_test = [[6, 6, 2, 1, 3, 3, 2, 1, 2],
                     [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
                     [6, 6.1, 2.2, 1, 3, 0, 2.2, 1, 2],
                     [6, 6, 2, 1, 3, np.nan, np.nan, np.nan, np.nan],
                     [3., 5.3, np.nan, np.nan, np.inf, np.inf, 4.4, 3.7, 8.9]
                     ]

        for input_data in data_test:
            for index_data in data_test:
                s = pd.Series(input_data, index_data)
                result_ref = test_impl(s)
                result = hpat_func(s)
                if np.isnan(result) or np.isnan(result_ref):
                    self.assertEqual(np.isnan(result), np.isnan(result_ref))
                else:
                    self.assertEqual(result, result_ref)

    def test_series_sort_values_default(self):
        """Verifies Series.sort_values method with default parameters
            on a named Series of different dtypes and default index"""
        def test_impl(A):
            return A.sort_values()
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        # using sequences of unique values because default sorting algorithm is not stable
        data_to_test = [
            [1, -0., 0.2, -3.7, np.inf, np.nan, -1.0, 2/3, 21.2, -np.inf, 9.99],
            np.arange(-10, 20, 1),
            np.unique(np.random.ranf(n)),
            np.unique(np.random.randint(0, 100, n)),
            ['ac', 'c', 'cb', 'ca', None, 'da', 'cc', 'ddd', 'd']
        ]
        for data in data_to_test:
            with self.subTest(series_data=data):
                S = pd.Series(data, name='A')
                pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_sort_values_ascending(self):
        """Verifies Series.sort_values method handles parameter 'ascending' as a literal and non-literal value"""
        def test_impl(S, param_value):
            return S.sort_values(ascending=param_value)

        def test_impl_literal(S):
            return S.sort_values(ascending=False)

        hpat_func1 = self.jit(test_impl)
        hpat_func2 = self.jit(test_impl_literal)

        S = pd.Series(['ac', 'c', 'cb', 'ca', None, 'da', 'cc', 'ddd', 'd'])
        for ascending in (False, True):
            with self.subTest(literal_value='no', ascending=ascending):
                pd.testing.assert_series_equal(hpat_func1(S, ascending), test_impl(S, ascending))

        with self.subTest(literal_value='yes'):
            pd.testing.assert_series_equal(hpat_func2(S), test_impl_literal(S))

    def test_series_sort_values_invalid_axis(self):
        """Verifies Series.sort_values method raises with invalid value of parameter 'axis'"""
        def test_impl(S, param_value):
            return S.sort_values(axis=param_value)
        hpat_func = self.jit(test_impl)

        S = pd.Series(['ac', 'c', 'cb', 'ca', None, 'da', 'cc', 'ddd', 'd'])
        unsupported_values = [1, 'columns', 'abcde']
        for axis in unsupported_values:
            with self.assertRaises(Exception) as context:
                test_impl(S, axis)
            pandas_exception = context.exception

            self.assertRaises(type(pandas_exception), hpat_func, S, axis)

    @skip_numba_jit('TODO: inplace sorting is not implemented yet')
    def test_series_sort_values_inplace(self):
        def test_impl(S):
            S.sort_values(inplace=True)
            return S
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['ac', 'c', 'cb', 'ca', None, 'da', 'cc', 'ddd', 'd'])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_sort_values_kind(self):
        """Verifies Series.sort_values method support of parameter 'kind'
           on a unnamed Series of different dtypes and default index"""
        def test_impl_literal_kind(A, param_value):
            # FIXME: use literally(kind) because, numpy.argsort is supported by Numba with literal kind value only
            # and literally when applied inside sort_values overload_method impl is not working due to some bug in Numba
            return A.sort_values(kind=literally(param_value))
        hpat_func1 = self.jit(test_impl_literal_kind)

        def test_impl_non_literal_kind(A, param_value):
            return A.sort_values(kind=param_value)
        hpat_func2 = self.jit(test_impl_non_literal_kind)

        # using sequences of unique values because default sorting algorithm is not stable
        n = 11
        np.random.seed(0)
        data_to_test = [
            [1, -0., 0.2, -3.7, np.inf, np.nan, -1.0, 2/3, 21.2, -np.inf, 9.99],
            np.arange(-10, 20, 1),
            np.unique(np.random.ranf(n)),
            np.unique(np.random.randint(0, 100, n)),
            ['ac', 'c', 'cb', 'ca', None, 'da', 'cc', 'ddd', 'd']
        ]
        kind_values = ['quicksort', 'mergesort']
        for data in data_to_test:
            S = pd.Series(data=data)
            for kind in kind_values:
                with self.subTest(series_data=data, kind=kind):
                    pd.testing.assert_series_equal(hpat_func1(S, kind), test_impl_literal_kind(S, kind))

        kind = None
        with self.subTest(series_data=data, kind=kind):
            pd.testing.assert_series_equal(hpat_func2(S, kind), test_impl_non_literal_kind(S, kind))

    def test_series_sort_values_na_position(self):
        """Verifies Series.sort_values method support of parameter 'na_position'
           on a unnamed Series of different dtypes and default index"""
        def test_impl(S, param_value):
            # kind=mergesort is used for sort stability
            return S.sort_values(kind='mergesort', na_position=param_value)
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        data_to_test = [
            [1, -0., 0.2, -3.7, np.inf, np.nan, -1.0, 2/3, 21.2, -np.inf, 9.99],
            np.arange(-10, 20, 1),
            np.random.ranf(n),
            np.random.randint(0, 100, n),
            ['ac', 'c', None, '', 'cb', 'ca', None, 'da', 'cc', 'ddd', '', '', 'd']
        ]
        na_position_values = ['first', 'last']
        for data in data_to_test:
            S = pd.Series(data=data)
            for na_position in na_position_values:
                with self.subTest(series_data=data, na_position=na_position):
                    pd.testing.assert_series_equal(hpat_func(S, na_position), test_impl(S, na_position))

    def test_series_sort_values_index(self):
        """Verifies Series.sort_values method with default parameters
           on an unnamed integer Series and different indexes"""
        def test_impl(S):
            # kind=mergesort is used for sort stability
            return S.sort_values(kind='mergesort')
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        data = np.random.randint(0, 100, n)
        use_indexes = ['id1', None, '', 'abc', 'a', 'dd', 'd1', '23']
        dtype_to_index = {'None': None,
                          'int': np.arange(n, dtype='int'),
                          'float': np.arange(n, dtype='float'),
                          'string': [use_indexes[i] for i in np.random.randint(0, len(use_indexes), n)]}

        for dtype, index_data in dtype_to_index.items():
            with self.subTest(index_dtype=dtype, index=index_data):
                S = pd.Series(data, index=index_data)
                pd.testing.assert_series_equal(hpat_func(S), test_impl(S), check_names=False)

    @skip_parallel
    def test_series_sort_values_full(self):
        def test_impl(series, ascending, kind):
            return series.sort_values(axis=0, ascending=ascending, kind=literally(kind), na_position='last')

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_numeric + [test_global_input_data_unicode_kind1]

        for data in all_data:
            series = pd.Series(data * 3)
            for ascending in [True, False]:
                for kind in ['quicksort', 'mergesort']:
                    ref_result = test_impl(series, ascending, kind=kind)
                    jit_result = hpat_func(series, ascending, kind=kind)
                    ref = restore_series_sort_values(series, ref_result.index, ascending)
                    jit = restore_series_sort_values(series, jit_result.index, ascending)
                    if kind == 'mergesort':
                        pd.testing.assert_series_equal(ref_result, jit_result)
                    else:
                        np.testing.assert_array_equal(ref_result.data, jit_result.data)
                        self.assertEqual(ref, jit)

    def test_series_sort_values_full_unicode4(self):
        def test_impl(series, ascending, kind):
            return series.sort_values(axis=0, ascending=ascending, kind=literally(kind), na_position='last')

        hpat_func = self.jit(test_impl)

        all_data = [test_global_input_data_unicode_kind1]

        for data in all_data:
            series = pd.Series(data * 3)
            for ascending in [True, False]:
                for kind in ['quicksort', 'mergesort']:
                    ref_result = test_impl(series, ascending, kind=kind)
                    jit_result = hpat_func(series, ascending, kind=kind)
                    ref = restore_series_sort_values(series, ref_result.index, ascending)
                    jit = restore_series_sort_values(series, jit_result.index, ascending)
                    if kind == 'mergesort':
                        pd.testing.assert_series_equal(ref_result, jit_result)
                    else:
                        np.testing.assert_array_equal(ref_result.values, jit_result.values)
                        self.assertEqual(ref, jit)

    @skip_parallel
    def test_series_sort_values_full_idx(self):
        def test_impl(series, ascending, kind):
            return series.sort_values(axis=0, ascending=ascending, kind=literally(kind), na_position='last')

        hpat_func = self.jit(test_impl)

        all_data = test_global_input_data_numeric + [test_global_input_data_unicode_kind1]

        for data in all_data:
            data = data * 3
            for index in [gen_srand_array(len(data)), gen_frand_array(len(data)), range(len(data))]:
                for ascending in [True, False]:
                    for kind in ['quicksort', 'mergesort']:
                        series = pd.Series(data, index)
                        ref_result = test_impl(series, ascending, kind=kind)
                        jit_result = hpat_func(series, ascending, kind=kind)
                        ref = restore_series_sort_values(series, ref_result.index, ascending)
                        jit = restore_series_sort_values(series, jit_result.index, ascending)
                        if kind == 'mergesort':
                            pd.testing.assert_series_equal(ref_result, jit_result)
                        else:
                            np.testing.assert_array_equal(ref_result.data, jit_result.data)
                            self.assertEqual(ref, jit)

    @skip_numba_jit
    def test_series_sort_values_parallel1(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.sort_values()
        hpat_func = self.jit(test_impl)

        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_shift(self):
        def pyfunc():
            series = pd.Series([1.0, np.nan, -1.0, 0.0, 5e-324])
            return series.shift()

        cfunc = self.jit(pyfunc)
        pd.testing.assert_series_equal(cfunc(), pyfunc())

    def test_series_shift_name(self):
        def pyfunc():
            series = pd.Series([1.0, np.nan, -1.0, 0.0, 5e-324], name='A')
            return series.shift()

        cfunc = self.jit(pyfunc)
        pd.testing.assert_series_equal(cfunc(), pyfunc())

    def test_series_shift_unboxing(self):
        def pyfunc(series):
            return series.shift()

        cfunc = self.jit(pyfunc)
        for data in test_global_input_data_float64:
            series = pd.Series(data)
            pd.testing.assert_series_equal(cfunc(series), pyfunc(series))

    def test_series_shift_full(self):
        def pyfunc(series, periods, freq, axis, fill_value):
            return series.shift(periods=periods, freq=freq, axis=axis, fill_value=fill_value)

        cfunc = self.jit(pyfunc)
        freq = None
        axis = 0
        datas = test_global_input_data_signed_integer64 + test_global_input_data_float64
        for data in datas:
            for periods in [1, 2, -1]:
                for fill_value in [-1, 0, 9.1, np.nan, -3.3, None]:
                    with self.subTest(data=data, periods=periods, fill_value=fill_value):
                        series = pd.Series(data)
                        jit_result = cfunc(series, periods, freq, axis, fill_value)
                        ref_result = pyfunc(series, periods, freq, axis, fill_value)
                        pd.testing.assert_series_equal(jit_result, ref_result)

    @sdc_limitation
    def test_series_shift_0period(self):
        """SDC implementation always changes dtype to float. Even in case of period = 0"""
        def pyfunc():
            series = pd.Series([6, 4, 3])
            return series.shift(periods=0)

        cfunc = self.jit(pyfunc)
        ref_result = pyfunc()
        pd.testing.assert_series_equal(cfunc(), ref_result)

    def test_series_shift_0period_sdc(self):
        def pyfunc():
            series = pd.Series([6, 4, 3])
            return series.shift(periods=0)

        cfunc = self.jit(pyfunc)
        ref_result = pyfunc()
        pd.testing.assert_series_equal(cfunc(), ref_result, check_dtype=False)

    @sdc_limitation
    def test_series_shift_uint_int(self):
        """SDC assumes fill_value is int and unifies unsigned int and int to float. Even if fill_value is positive"""
        def pyfunc():
            series = pd.Series([max_uint64])
            return series.shift(fill_value=0)

        cfunc = self.jit(pyfunc)
        ref_result = pyfunc()
        pd.testing.assert_series_equal(cfunc(), ref_result)

    def test_series_shift_uint_int_sdc(self):
        def pyfunc():
            series = pd.Series([max_uint64])
            return series.shift(fill_value=0)

        cfunc = self.jit(pyfunc)
        ref_result = pyfunc()
        pd.testing.assert_series_equal(cfunc(), ref_result, check_dtype=False)

    def test_series_shift_str(self):
        def pyfunc(series):
            return series.shift()

        cfunc = self.jit(pyfunc)
        series = pd.Series(test_global_input_data_unicode_kind4)
        self.assertRaisesRegex(TypingError,
                               r'Method shift\(\)\. The object self\.data\.dtype\n'
                               r'\s+given: unicode_type\n'
                               r'\s+expected: number',
                               cfunc,
                               series)

    def test_series_shift_fill_str(self):
        def pyfunc(series, fill_value):
            return series.shift(fill_value=fill_value)

        cfunc = self.jit(pyfunc)
        series = pd.Series(test_global_input_data_float64[0])

        assert_raises_ty_checker(self,
                                 ['Method shift().', 'fill_value', 'unicode_type', 'number'],
                                 cfunc,
                                 series, fill_value='unicode')

    def test_series_shift_unsupported_params(self):
        def pyfunc(series, freq, axis):
            return series.shift(freq=freq, axis=axis)

        cfunc = self.jit(pyfunc)
        series = pd.Series(test_global_input_data_float64[0])
        assert_raises_ty_checker(self,
                                 ['Method shift().', 'freq', 'unicode_type', 'None'],
                                 cfunc,
                                 series, freq='12H', axis=0)

        with self.assertRaises(TypingError) as raises:
            cfunc(series, freq=None, axis=1)
        msg = 'Method shift(). Unsupported parameters. Given axis != 0'
        self.assertIn(msg, str(raises.exception))

    def test_series_shift_index_str(self):
        def test_impl(S):
            return S.shift()
        hpat_func = self.jit(test_impl)

        S = pd.Series([np.nan, 2., 3., 5., np.nan, 6., 7.], index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_shift_index_int(self):
        def test_impl(S):
            return S.shift()

        hpat_func = self.jit(test_impl)

        S = pd.Series([np.nan, 2., 3., 5., np.nan, 6., 7.], index=[1, 2, 3, 4, 5, 6, 7])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_getattr_index1(self):
        def test_impl():
            A = pd.Series([1, 2, 3], index=['A', 'C', 'B'])
            return A.index

        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_getattr_index2(self):
        def test_impl():
            A = pd.Series([1, 2, 3], index=[0, 1, 2])
            return A.index

        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_getattr_index3(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A.index

        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_take_index_default(self):
        def pyfunc():
            series = pd.Series([1.0, 13.0, 9.0, -1.0, 7.0])
            indices = [1, 3]
            return series.take(indices)

        cfunc = self.jit(pyfunc)
        ref_result = pyfunc()
        result = cfunc()
        pd.testing.assert_series_equal(ref_result, result)

    def test_series_take_index_default_unboxing(self):
        def pyfunc(series, indices):
            return series.take(indices)

        cfunc = self.jit(pyfunc)
        series = pd.Series([1.0, 13.0, 9.0, -1.0, 7.0])
        indices = [1, 3]
        ref_result = pyfunc(series, indices)
        result = cfunc(series, indices)
        pd.testing.assert_series_equal(ref_result, result)

    def test_series_take_index_int(self):
        def pyfunc():
            series = pd.Series([1.0, 13.0, 9.0, -1.0, 7.0], index=[3, 0, 4, 2, 1])
            indices = [1, 3]
            return series.take(indices)

        cfunc = self.jit(pyfunc)
        ref_result = pyfunc()
        result = cfunc()
        pd.testing.assert_series_equal(ref_result, result)

    def test_series_take_index_int_unboxing(self):
        def pyfunc(series, indices):
            return series.take(indices)

        cfunc = self.jit(pyfunc)
        series = pd.Series([1.0, 13.0, 9.0, -1.0, 7.0], index=[3, 0, 4, 2, 1])
        indices = [1, 3]
        ref_result = pyfunc(series, indices)
        result = cfunc(series, indices)
        pd.testing.assert_series_equal(ref_result, result)

    def test_series_take_index_str(self):
        def pyfunc():
            series = pd.Series([1.0, 13.0, 9.0, -1.0, 7.0], index=['test', 'series', 'take', 'str', 'index'])
            indices = [1, 3]
            return series.take(indices)

        cfunc = self.jit(pyfunc)
        ref_result = pyfunc()
        result = cfunc()
        pd.testing.assert_series_equal(ref_result, result)

    def test_series_take_index_str_unboxing(self):
        def pyfunc(series, indices):
            return series.take(indices)

        cfunc = self.jit(pyfunc)
        series = pd.Series([1.0, 13.0, 9.0, -1.0, 7.0], index=['test', 'series', 'take', 'str', 'index'])
        indices = [1, 3]
        ref_result = pyfunc(series, indices)
        result = cfunc(series, indices)
        pd.testing.assert_series_equal(ref_result, result)

    def test_series_iterator_int(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([3, 2, 1, 5, 4])
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_float(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([0.3, 0.2222, 0.1756, 0.005, 0.4])
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_boolean(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([True, False])
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_string(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series(['a', 'ab', 'abc', '', 'dddd'])
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_one_value(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([5])
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_no_param(self):
        def test_impl():
            A = pd.Series([3, 2, 1, 5, 4])
            return [i for i in A]

        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_iterator_empty(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([np.int64(x) for x in range(0)])
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_getattr_default_index(self):
        def test_impl():
            A = pd.Series([3, 2, 1, 5, 4])
            return A.index

        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    @unittest.skip("Implement drop_duplicates for Series")
    def test_series_drop_duplicates(self):
        def test_impl():
            A = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])
            return A.drop_duplicates()

        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_quantile(self):
        def test_impl():
            a = pd.Series([1, 2.5, .5, 3, 5])
            return a.quantile()

        hpat_func = self.jit(test_impl)
        np.testing.assert_equal(hpat_func(), test_impl())

    def test_series_quantile_q_vector(self):
        def test_series_quantile_q_vector_impl(S, param1):
            return S.quantile(param1)

        s = pd.Series(np.random.ranf(100))
        hpat_func = self.jit(test_series_quantile_q_vector_impl)

        param1 = [0.0, 0.25, 0.5, 0.75, 1.0]
        result_ref = test_series_quantile_q_vector_impl(s, param1)
        result = hpat_func(s, param1)
        np.testing.assert_equal(result, result_ref)

    @unittest.skip("Implement unique without sorting like in pandas")
    def test_unique(self):
        def test_impl(S):
            return S.unique()

        hpat_func = self.jit(test_impl)
        S = pd.Series([2, 1, 3, 3])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_unique_sorted(self):
        def test_impl(S):
            return S.unique()

        hpat_func = self.jit(test_impl)
        n = 11
        S = pd.Series(np.arange(n))
        S[2] = 0
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_unique_str(self):
        def test_impl():
            data = pd.Series(['aa', 'aa', 'b', 'b', 'cccc', 'dd', 'ddd', 'dd'])
            return data.unique()

        hpat_func = self.jit(test_impl)

        # since the orider of the elements are diffrent - check count of elements only
        ref_result = test_impl().size
        result = hpat_func().size
        np.testing.assert_array_equal(ref_result, result)

    def test_series_std(self):
        def pyfunc():
            series = pd.Series([1.0, np.nan, -1.0, 0.0, 5e-324])
            return series.std()

        cfunc = self.jit(pyfunc)
        ref_result = pyfunc()
        result = cfunc()
        np.testing.assert_equal(ref_result, result)

    def test_series_std_unboxing(self):
        def pyfunc(series, skipna, ddof):
            return series.std(skipna=skipna, ddof=ddof)

        cfunc = self.jit(pyfunc)
        for data in test_global_input_data_numeric + [[]]:
            series = pd.Series(data)
            for ddof in [0, 1]:
                for skipna in [True, False]:
                    ref_result = pyfunc(series, skipna=skipna, ddof=ddof)
                    result = cfunc(series, skipna=skipna, ddof=ddof)
                    np.testing.assert_equal(ref_result, result)

    def test_series_std_str(self):
        def pyfunc(series):
            return series.std()

        cfunc = self.jit(pyfunc)
        series = pd.Series(test_global_input_data_unicode_kind4)
        assert_raises_ty_checker(self,
                                 ['Method std().', 'self.data', 'StringArrayType()', 'number'],
                                 cfunc,
                                 series)

    def test_series_std_unsupported_params(self):
        def pyfunc(series, axis, level, numeric_only):
            return series.std(axis=axis, level=level, numeric_only=numeric_only)

        cfunc = self.jit(pyfunc)
        series = pd.Series(test_global_input_data_float64[0])
        method_name = 'Method std().'
        assert_raises_ty_checker(self,
                                 [method_name, 'axis', 'int64', 'None'],
                                 cfunc,
                                 series, axis=1, level=None, numeric_only=None)

        assert_raises_ty_checker(self,
                                 [method_name, 'level', 'int64', 'None'],
                                 cfunc,
                                 series, axis=None, level=1, numeric_only=None)

        assert_raises_ty_checker(self,
                                 [method_name, 'numeric_only', 'bool', 'None'],
                                 cfunc,
                                 series, axis=None, level=None, numeric_only=True)

    def test_series_nunique(self):
        def test_series_nunique_impl(s):
            return s.nunique()

        def test_series_nunique_param1_impl(s, dropna):
            return s.nunique(dropna)

        hpat_func = self.jit(test_series_nunique_impl)

        the_same_string = "the same string"
        test_input_data = []
        data_simple = [[6, 6, 2, 1, 3, 3, 2, 1, 2],
                       [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
                       [6, 6.1, 2.2, 1, 3, 3, 2.2, 1, 2],
                       ['aa', 'aa', 'b', 'b', 'cccc', 'dd', 'ddd', 'dd'],
                       ['aa', 'copy aa', the_same_string, 'b', 'b', 'cccc', the_same_string,
                        'dd', 'ddd', 'dd', 'copy aa', 'copy aa'],
                       []
                       ]

        data_extra = [[6, 6, np.nan, 2, np.nan, 1, 3, 3, np.inf, 2, 1, 2, np.inf],
                      [1.1, 0.3, np.nan, 1.0, np.inf, 0.3, 2.1, np.nan, 2.2, np.inf],
                      [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2],
                      ['aa', np.nan, 'b', 'b', 'cccc', np.nan, 'ddd', 'dd'],
                      [np.nan, 'copy aa', the_same_string, 'b', 'b', 'cccc', the_same_string,
                       'dd', 'ddd', 'dd', 'copy aa', 'copy aa'],
                      [np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.inf],
                      ]

        test_input_data = data_simple + data_extra

        for input_data in test_input_data:
            s = pd.Series(input_data)

            result_ref = test_series_nunique_impl(s)
            result = hpat_func(s)
            self.assertEqual(result, result_ref)

            """
            SDC pipeline does not support parameter to Series.nunique(dropna=True)
            """

            hpat_func_param1 = self.jit(test_series_nunique_param1_impl)

            for param1 in [True, False]:
                result_param1_ref = test_series_nunique_param1_impl(s, param1)
                result_param1 = hpat_func_param1(s, param1)
                self.assertEqual(result_param1, result_param1_ref)

    def test_series_var(self):
        def pyfunc():
            series = pd.Series([1.0, np.nan, -1.0, 0.0, 5e-324])
            return series.var()

        cfunc = self.jit(pyfunc)
        np.testing.assert_equal(pyfunc(), cfunc())

    def test_series_var_unboxing(self):
        def pyfunc(series):
            return series.var()

        cfunc = self.jit(pyfunc)
        for data in test_global_input_data_numeric + [[]]:
            series = pd.Series(data)
            np.testing.assert_equal(pyfunc(series), cfunc(series))

    def test_series_var_full(self):
        def pyfunc(series, skipna, ddof):
            return series.var(skipna=skipna, ddof=ddof)

        cfunc = self.jit(pyfunc)
        for data in test_global_input_data_numeric + [[]]:
            series = pd.Series(data)
            for ddof in [0, 1]:
                for skipna in [True, False]:
                    ref_result = pyfunc(series, skipna=skipna, ddof=ddof)
                    result = cfunc(series, skipna=skipna, ddof=ddof)
                    np.testing.assert_equal(ref_result, result)

    def test_series_var_str(self):
        def pyfunc(series):
            return series.var()

        cfunc = self.jit(pyfunc)
        series = pd.Series(test_global_input_data_unicode_kind4)

        assert_raises_ty_checker(self,
                                 ['Method var().', 'self.data', 'StringArrayType()', 'number'],
                                 cfunc,
                                 series)

    def test_series_var_unsupported_params(self):
        def pyfunc(series, axis, level, numeric_only):
            return series.var(axis=axis, level=level, numeric_only=numeric_only)

        cfunc = self.jit(pyfunc)
        series = pd.Series(test_global_input_data_float64[0])

        method_name = 'Method var().'
        assert_raises_ty_checker(self,
                                 [method_name, 'axis', 'int64', 'None'],
                                 cfunc,
                                 series, axis=1, level=None, numeric_only=None)

        assert_raises_ty_checker(self,
                                 [method_name, 'level', 'int64', 'None'],
                                 cfunc,
                                 series, axis=None, level=1, numeric_only=None)

        assert_raises_ty_checker(self,
                                 [method_name, 'numeric_only', 'bool', 'None'],
                                 cfunc,
                                 series, axis=None, level=None, numeric_only=True)

    def test_series_count(self):
        def test_series_count_impl(S):
            return S.count()

        hpat_func = self.jit(test_series_count_impl)

        the_same_string = "the same string"
        test_input_data = [[6, 6, 2, 1, 3, 3, 2, 1, 2],
                           [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
                           [6, 6.1, 2.2, 1, 3, 3, 2.2, 1, 2],
                           ['aa', 'aa', 'b', 'b', 'cccc', 'dd', 'ddd', 'dd'],
                           ['aa', None, '', '', None, 'cccc', 'dd', 'ddd', None, 'dd'],
                           ['aa', 'copy aa', the_same_string, 'b', 'b', 'cccc', the_same_string, 'dd', 'ddd', 'dd',
                            'copy aa', 'copy aa'],
                           [],
                           [6, 6, np.nan, 2, np.nan, 1, 3, 3, np.inf, 2, 1, 2, np.inf],
                           [1.1, 0.3, np.nan, 1.0, np.inf, 0.3, 2.1, np.nan, 2.2, np.inf],
                           [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2],
                           [np.nan, np.nan, np.nan],
                           [np.nan, np.nan, np.inf]
                           ]

        for input_data in test_input_data:
            S = pd.Series(input_data)

            result_ref = test_series_count_impl(S)
            result = hpat_func(S)
            self.assertEqual(result, result_ref)

    def test_series_cumsum(self):
        def test_impl():
            series = pd.Series([1.0, np.nan, -1.0, 0.0, 5e-324])
            return series.cumsum()
        hpat_func = self.jit(test_impl)

        result = hpat_func()
        result_ref = test_impl()
        pd.testing.assert_series_equal(result, result_ref)

    def _gen_cumulative_data_skip(self):
        yield [min_int64, max_int64, max_int64, min_int64]
        yield [max_uint64, max_uint64]

    def _gen_cumulative_data(self):
        for case in test_global_input_data_numeric:
            if case not in self._gen_cumulative_data_skip():
                yield case
        yield [1.0, np.nan, -1.0, 0.0, 5e-324]
        yield []

    def _check_cumulative(self, pyfunc, generator, **kwds):
        cfunc = self.jit(pyfunc)

        for data in generator():
            with self.subTest(series_data=data):
                S = pd.Series(data)
                pd.testing.assert_series_equal(cfunc(S, **kwds), pyfunc(S, **kwds))

    def test_series_cumsum_unboxing(self):
        def test_impl(s):
            return s.cumsum()

        self._check_cumulative(test_impl, self._gen_cumulative_data)

    def _cumsum_full_usecase(self):
        def test_impl(s, axis, skipna):
            return s.cumsum(axis=axis, skipna=skipna)
        return test_impl

    def test_series_cumsum_full(self):
        axis = None
        for skipna in [True, False]:
            with self.subTest(skipna=skipna):
                self._check_cumulative(self._cumsum_full_usecase(),
                                       self._gen_cumulative_data, axis=axis, skipna=skipna)

    @unittest.expectedFailure
    def test_series_cumsum_expectedFailure(self):
        axis = None
        for skipna in [True, False]:
            with self.subTest(skipna=skipna):
                self._check_cumulative(self._cumsum_full_usecase(),
                                       self._gen_cumulative_data_skip, axis=axis, skipna=skipna)

    def test_series_cumsum_str(self):
        def test_impl(s):
            return s.cumsum()
        hpat_func = self.jit(test_impl)

        S = pd.Series(test_global_input_data_unicode_kind4)
        assert_raises_ty_checker(self,
                                 ['Method cumsum().', 'self.data.dtype', 'unicode_type', 'numeric'],
                                 hpat_func,
                                 S)

    def test_series_cumsum_unsupported_axis(self):
        def test_impl(s, axis):
            return s.cumsum(axis=axis)
        hpat_func = self.jit(test_impl)

        S = pd.Series(test_global_input_data_float64[0])
        for axis in [0, 1]:
            with self.subTest(axis=axis):
                assert_raises_ty_checker(self,
                                         ['Method cumsum().', 'axis', 'int64', 'None'],
                                         hpat_func,
                                         S, axis=axis)

    def test_series_cov1(self):
        def test_impl(s1, s2):
            return s1.cov(s2)
        hpat_func = self.jit(test_impl)

        for pair in _cov_corr_series:
            s1, s2 = pair
            np.testing.assert_almost_equal(
                hpat_func(s1, s2), test_impl(s1, s2),
                err_msg='s1={}\ns2={}'.format(s1, s2))

    def test_series_cov(self):
        def test_series_cov_impl(s1, s2, min_periods=None):
            return s1.cov(s2, min_periods)

        hpat_func = self.jit(test_series_cov_impl)
        test_input_data1 = [[.2, .0, .6, .2],
                            [.2, .0, .6, .2, .5, .6, .7, .8],
                            [],
                            [2, 0, 6, 2],
                            [.2, .1, np.nan, .5, .3],
                            [-1, np.nan, 1, np.inf]]
        test_input_data2 = [[.3, .6, .0, .1],
                            [.3, .6, .0, .1, .8],
                            [],
                            [3, 6, 0, 1],
                            [.3, .2, .9, .6, np.nan],
                            [np.nan, np.nan, np.inf, np.nan]]
        for input_data1 in test_input_data1:
            for input_data2 in test_input_data2:
                s1 = pd.Series(input_data1)
                s2 = pd.Series(input_data2)
                for period in [None, 2, 1, 8, -4]:
                    with self.subTest(input_data1=input_data1, input_data2=input_data2, min_periods=period):
                        result_ref = test_series_cov_impl(s1, s2, min_periods=period)
                        result = hpat_func(s1, s2, min_periods=period)
                        np.testing.assert_allclose(result, result_ref)

    def test_series_cov_unsupported_dtype(self):
        def test_series_cov_impl(s1, s2, min_periods=None):
            return s1.cov(s2, min_periods=min_periods)

        hpat_func = self.jit(test_series_cov_impl)
        s1 = pd.Series([.2, .0, .6, .2])
        s2 = pd.Series(['abcdefgh', 'a', 'abcdefg', 'ab', 'abcdef', 'abc'])
        s3 = pd.Series(['aaaaa', 'bbbb', 'ccc', 'dd', 'e'])
        s4 = pd.Series([.3, .6, .0, .1])

        with self.assertRaises(TypingError) as raises:
            hpat_func(s1, s2, min_periods=5)
        msg = 'Method cov(). The object other.data'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(s3, s4, min_periods=5)
        msg = 'Method cov(). The object self.data'
        self.assertIn(msg, str(raises.exception))

    def test_series_cov_unsupported_period(self):
        def test_series_cov_impl(s1, s2, min_periods=None):
            return s1.cov(s2, min_periods)

        hpat_func = self.jit(test_series_cov_impl)
        s1 = pd.Series([.2, .0, .6, .2])
        s2 = pd.Series([.3, .6, .0, .1])

        with self.assertRaises(TypingError) as raises:
            hpat_func(s1, s2, min_periods='aaaa')
        msg = 'Method cov(). The object min_periods'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(s1, s2, min_periods=0.5)
        msg = 'Method cov(). The object min_periods'
        self.assertIn(msg, str(raises.exception))

    @skip_numba_jit
    def test_series_pct_change(self):
        def test_series_pct_change_impl(S, periods, method):
            return S.pct_change(periods=periods, fill_method=method, limit=None, freq=None)

        hpat_func = self.jit(test_series_pct_change_impl)
        test_input_data = [
            [],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.inf],
            [0] * 8,
            [0, 0, 0, np.nan, np.nan, 0, 0, np.nan, np.inf, 0, 0, np.inf, np.inf],
            [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2],
            [1, 2, 3, 4, np.nan, np.inf, 0, 0, np.nan, np.nan]
        ]
        for input_data in test_input_data:
            S = pd.Series(input_data)
            for periods in [0, 1, 2, 5, 10, -1, -2, -5]:
                for method in [None, 'pad', 'ffill', 'backfill', 'bfill']:
                    result_ref = test_series_pct_change_impl(S, periods, method)
                    result = hpat_func(S, periods, method)
                    pd.testing.assert_series_equal(result, result_ref)

    def test_series_pct_change_str(self):
        def test_series_pct_change_impl(S):
            return S.pct_change(periods=1, fill_method='pad', limit=None, freq=None)

        hpat_func = self.jit(test_series_pct_change_impl)
        S = pd.Series(test_global_input_data_unicode_kind4)

        with self.assertRaises(TypingError) as raises:
            hpat_func(S)
        msg = 'Method pct_change(). The object self.data'
        self.assertIn(msg, str(raises.exception))

    def test_series_pct_change_not_supported(self):
        def test_series_pct_change_impl(S, periods=1, fill_method='pad', limit=None, freq=None):
            return S.pct_change(periods=periods, fill_method=fill_method, limit=limit, freq=freq)

        hpat_func = self.jit(test_series_pct_change_impl)
        S = pd.Series([0, 0, 0, np.nan, np.nan, 0, 0, np.nan, np.inf, 0, 0, np.inf, np.inf])
        with self.assertRaises(ValueError) as raises:
            hpat_func(S, fill_method='ababa')
        msg = 'Method pct_change(). Unsupported parameter. The function uses fill_method pad (ffill) or backfill (bfill) or None.'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(S, limit=5)
        msg = 'Method pct_change(). The object limit'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(S, freq=5)
        msg = 'Method pct_change(). The object freq'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(S, fill_method=1.6)
        msg = 'Method pct_change(). The object fill_method'
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            hpat_func(S, periods=1.6)
        msg = 'Method pct_change(). The object periods'
        self.assertIn(msg, str(raises.exception))

    def test_series_setitem_value_int(self):
        def test_impl(S, val):
            S[3] = val
            return S

        hpat_func = self.jit(test_impl)
        data_to_test = [[0, 1, 2, 3, 4]]

        for data in data_to_test:
            S1 = pd.Series(data)
            S2 = S1.copy(deep=True)
            value = 50
            result_ref = test_impl(S1, value)
            result = hpat_func(S2, value)
            pd.testing.assert_series_equal(result_ref, result)

    def test_series_setitem_value_float(self):
        def test_impl(S, val):
            S[3] = val
            return S

        hpat_func = self.jit(test_impl)
        data_to_test = [[0, 0, 0, np.nan, np.nan, 0, 0, np.nan, np.inf, 0, 0, np.inf, np.inf],
                        [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2],
                        [1, 2, 3, 4, np.nan, np.inf, 0, 0, np.nan, np.nan]]

        for data in data_to_test:
            S1 = pd.Series(data)
            S2 = S1.copy(deep=True)
            value = np.nan
            result_ref = test_impl(S1, value)
            result = hpat_func(S2, value)
            pd.testing.assert_series_equal(result_ref, result)

    @skip_numba_jit('Requires fully functional StringArray setitem')
    def test_series_setitem_value_string(self):
        def test_impl(S, val):
            S[2] = val
            return S
        hpat_func = self.jit(test_impl)

        data_to_test = [['a', '', 'abc', '', 'b', None, 'a', '', None, 'b'],
                        ['dog', None, 'NaN', '', 'cat', None, 'cat', None, 'dog', ''],
                        ['dog', 'NaN', 'cat', '', 'cat', 'dog', '']]

        for data in data_to_test:
            S1 = pd.Series(data)
            S2 = S1.copy(deep=True)
            # the length of value is greater than S[idx] length!
            value = 'Hello, world!'
            result_ref = test_impl(S1, value)
            result = hpat_func(S2, value)
            pd.testing.assert_series_equal(result, result_ref)

    def test_series_setitem_idx_int_slice_all_dtypes(self):
        def test_impl(S, idx, val):
            S[idx] = val
            return S
        hpat_func = self.jit(test_impl)

        dtype_to_data = {
            'int':    [[0, 1, 2, 3, 4]],
            'float':  [[0, 0, 0, np.nan, np.nan, 0, 0, np.nan, np.inf, 0, 0, np.inf, np.inf],
                       [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2],
                       [1, 2, 3, 4, np.nan, np.inf, 0, 0, np.nan, np.nan]],
            'string': [['a', '', 'a', '', 'b', None, 'a', '', None, 'b'],
                       ['dog', None, 'NaN', '', 'cat', None, 'cat', None, 'dog', ''],
                       ['dog', 'NaN', '', 'cat', 'cat', 'dog', '']]
        }
        dtype_to_values = {
            'int':    50,
            'float':  np.nan,
            'string': 'bird'
        }

        idx = slice(2, None)
        for dtype, all_data in dtype_to_data.items():
            # FIXME: setitem for StringArray type has no overload for slice idx
            if dtype == 'string':
                continue
            value = dtype_to_values[dtype]
            for series_data in all_data:
                S1 = pd.Series(series_data)
                S2 = S1.copy(deep=True)
                with self.subTest(series=S1, value=value):
                    hpat_func(S1, idx, value)
                    test_impl(S2, idx, value)
                    pd.testing.assert_series_equal(S1, S2)

    def test_series_setitem_idx_int_series_all_dtypes(self):
        def test_impl(S, idx, val):
            S[idx] = val
            return S
        hpat_func = self.jit(test_impl)

        dtype_to_data = {
            'int':    [[0, 1, 2, 3, 4]],
            'float':  [[0, 0, 0, np.nan, np.nan, 0, 0, np.nan, np.inf, 0, 0, np.inf, np.inf],
                       [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2],
                       [1, 2, 3, 4, np.nan, np.inf, 0, 0, np.nan, np.nan]],
            'string': [['a', '', 'a', '', 'b', None, 'a', '', None, 'b'],
                       ['dog', None, 'NaN', '', 'cat', None, 'cat', None, 'dog', ''],
                       ['dog', 'NaN', '', 'cat', 'cat', 'dog', '']]
        }
        dtype_to_values = {
            'int':    50,
            'float':  np.nan,
            'string': 'bird'
        }

        idx = pd.Series([0, 2, 4])
        for dtype, all_data in dtype_to_data.items():
            # FIXME: setitem for StringArray type has no overload for slice idx
            if dtype == 'string':
                continue
            value = dtype_to_values[dtype]
            for series_data in all_data:
                S1 = pd.Series(series_data)
                S2 = S1.copy(deep=True)
                with self.subTest(series=S1, value=value):
                    hpat_func(S1, idx, value)
                    test_impl(S2, idx, value)
                    pd.testing.assert_series_equal(S1, S2)

    def test_series_setitem_unsupported(self):
        def test_impl(S, idx, value):
            S[idx] = value
            return S
        hpat_func = self.jit(test_impl)

        S = pd.Series([0, 1, 2, 3, 4, 5])

        with self.subTest(subtest="series data and value type mismatch"):
            idx, value = 5, 'ababa'
            msg_tmpl = 'Operator setitem(). The value and Series data must be comparable. ' \
                       'Given: self.dtype={}, value={}'
            with self.assertRaises(TypingError) as raises:
                hpat_func(S, idx, value)
            msg = msg_tmpl.format(S.dtype, 'unicode_type')
            self.assertIn(msg, str(raises.exception))

        with self.subTest(subtest="series index and indexer type mismatch"):
            idx, value = '3', 101
            msg_tmpl = 'Operator setitem(). The idx is not comparable to Series index, ' \
                       'not a Boolean or integer indexer or a Slice. Given: self.index={}, idx={}'
            with self.assertRaises(TypingError) as raises:
                hpat_func(S, idx, value)
            msg = msg_tmpl.format('none', 'unicode_type')
            self.assertIn(msg, str(raises.exception))

    def test_series_istitle_str(self):
        series = pd.Series(['Cat', 'dog', 'Bird'])

        cfunc = self.jit(istitle_usecase)
        pd.testing.assert_series_equal(cfunc(series), istitle_usecase(series))

    @skip_numba_jit("Not work with None and np.nan")
    def test_series_istitle_str_fixme(self):
        series = pd.Series(['Cat', 'dog', 'Bird', None, np.nan])

        cfunc = self.jit(istitle_usecase)
        pd.testing.assert_series_equal(cfunc(series), istitle_usecase(series))

    def test_series_isspace_str(self):
        series = [['', '  ', '    ', '           '],
                  ['', ' c ', '  b ', '     a     '],
                  ['aaaaaa', 'bb', 'c', '  d']
                  ]

        cfunc = self.jit(isspace_usecase)
        for ser in series:
            S = pd.Series(ser)
            pd.testing.assert_series_equal(cfunc(S), isspace_usecase(S))

    def test_series_isalpha_str(self):
        series = [['leopard', 'Golden Eagle', 'SNAKE', ''],
                  ['Hello world!', 'hello 123', 'mynameisPeter'],
                  ['one', 'one1', '1', '']
                  ]

        cfunc = self.jit(isalpha_usecase)
        for ser in series:
            S = pd.Series(ser)
            pd.testing.assert_series_equal(cfunc(S), isalpha_usecase(S))

    def test_series_islower_str(self):
        series = [['leopard', 'Golden Eagle', 'SNAKE', ''],
                  ['Hello world!', 'hello 123', 'mynameisPeter']
                  ]

        cfunc = self.jit(islower_usecase)
        for ser in series:
            S = pd.Series(ser)
            pd.testing.assert_series_equal(cfunc(S), islower_usecase(S))

    def test_series_lower_str(self):
        all_data = [['leopard', None, 'Golden Eagle', np.nan, 'SNAKE', ''],
                    ['Hello world!', np.nan, 'hello 123', None, 'mynameisPeter']
                    ]

        cfunc = self.jit(lower_usecase)
        for data in all_data:
            s = pd.Series(data)
            pd.testing.assert_series_equal(cfunc(s), lower_usecase(s))

    def test_series_strip_str(self):
        s = pd.Series(['1. Ant.  ', None, '2. Bee!\n', np.nan, '3. Cat?\t'])
        cfunc = self.jit(strip_usecase)
        for to_strip in [None, '123.', '.!? \n\t', '123.!? \n\t']:
            pd.testing.assert_series_equal(cfunc(s, to_strip), strip_usecase(s, to_strip))

    def test_series_lstrip_str(self):
        s = pd.Series(['1. Ant.  ', None, '2. Bee!\n', np.nan, '3. Cat?\t'])
        cfunc = self.jit(lstrip_usecase)
        for to_strip in [None, '123.', '.!? \n\t', '123.!? \n\t']:
            pd.testing.assert_series_equal(cfunc(s, to_strip), lstrip_usecase(s, to_strip))

    def test_series_rstrip_str(self):
        s = pd.Series(['1. Ant.  ', None, '2. Bee!\n', np.nan, '3. Cat?\t'])
        cfunc = self.jit(rstrip_usecase)
        for to_strip in [None, '123.', '.!? \n\t', '123.!? \n\t']:
            pd.testing.assert_series_equal(cfunc(s, to_strip), rstrip_usecase(s, to_strip))

    def test_series_isalnum_str(self):
        cfunc = self.jit(isalnum_usecase)
        test_data = [test_global_input_data_unicode_kind1, test_global_input_data_unicode_kind4]
        for data in test_data:
            S = pd.Series(data)
            pd.testing.assert_series_equal(cfunc(S), isalnum_usecase(S))

    def test_series_isnumeric_str(self):
        cfunc = self.jit(isnumeric_usecase)
        test_data = [test_global_input_data_unicode_kind1, test_global_input_data_unicode_kind4]
        for data in test_data:
            S = pd.Series(data)
            pd.testing.assert_series_equal(cfunc(S), isnumeric_usecase(S))

    def test_series_isdigit_str(self):
        cfunc = self.jit(isdigit_usecase)
        test_data = [test_global_input_data_unicode_kind1, test_global_input_data_unicode_kind4]
        for data in test_data:
            S = pd.Series(data)
            pd.testing.assert_series_equal(cfunc(S), isdigit_usecase(S))

    def test_series_isdecimal_str(self):
        cfunc = self.jit(isdecimal_usecase)
        test_data = [test_global_input_data_unicode_kind1, test_global_input_data_unicode_kind4]
        for data in test_data:
            S = pd.Series(data)
            pd.testing.assert_series_equal(cfunc(S), isdecimal_usecase(S))

    def test_series_isupper_str(self):
        cfunc = self.jit(isupper_usecase)
        test_data = [test_global_input_data_unicode_kind1, test_global_input_data_unicode_kind4]
        for data in test_data:
            s = pd.Series(data)
            pd.testing.assert_series_equal(cfunc(s), isupper_usecase(s))

    def test_series_contains(self):
        hpat_func = self.jit(contains_usecase)
        s = pd.Series(['Mouse', 'dog', 'house and parrot', '23'])
        for pat in ['og', 'Og', 'OG', 'o']:
            for case in [True, False]:
                with self.subTest(pat=pat, case=case):
                    pd.testing.assert_series_equal(hpat_func(s, pat, case), contains_usecase(s, pat, case))

    def test_series_contains_with_na_flags_regex(self):
        hpat_func = self.jit(contains_usecase)
        s = pd.Series(['Mouse', 'dog', 'house and parrot', '23'])
        pat = 'og'
        pd.testing.assert_series_equal(hpat_func(s, pat, flags=0, na=None, regex=True),
                                       contains_usecase(s, pat, flags=0, na=None, regex=True))

    def test_series_contains_unsupported(self):
        hpat_func = self.jit(contains_usecase)
        s = pd.Series(['Mouse', 'dog', 'house and parrot', '23'])
        pat = 'og'

        with self.assertRaises(SDCLimitation) as raises:
            hpat_func(s, pat, flags=1)
        msg = "Method contains(). Unsupported parameter. Given 'flags' != 0"
        self.assertIn(msg, str(raises.exception))

        assert_raises_ty_checker(self,
                                 ['Method contains().', 'na', 'int64', 'none'],
                                 hpat_func,
                                 s, pat, na=0)

        with self.assertRaises(SDCLimitation) as raises:
            hpat_func(s, pat, regex=False)
        msg = "Method contains(). Unsupported parameter. Given 'regex' is False"
        self.assertIn(msg, str(raises.exception))

    def test_series_describe_numeric(self):
        def test_impl(A):
            return A.describe()
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_describe_numeric_percentiles(self):
        def test_impl(A, values):
            return A.describe(percentiles=values)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        supported_values = [
            [0.323, 0.778, 0.1, 0.01, 0.2],
            [0.001, 0.002],
            [0.001, 0.5, 0.002],
            [0.9999, 0.0001],
            (0.323, 0.778, 0.1, 0.01, 0.2),
            np.array([0, 1.0]),
            np.array([0.323, 0.778, 0.1, 0.01, 0.2]),
            None,
        ]
        for percentiles in supported_values:
            with self.subTest(percentiles=percentiles):
                pd.testing.assert_series_equal(hpat_func(S, percentiles), test_impl(S, percentiles))

    def test_series_describe_str(self):
        def test_impl(A):
            return A.describe()
        hpat_func = self.jit(test_impl)

        S = pd.Series(['a', 'dd', None, 'bbbb', 'dd', '', 'dd', '', 'dd'])
        # SDC implementation returns series of string, hence conversion of reference result is needed
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S).astype(str))

    @skip_numba_jit('Series.describe is not implemented for datatime Series due to Numba limitations\n'
                    'Requires dropna for pd.Timestamp (depends on Numba isnat) to be implemented')
    def test_series_describe_dt(self):
        def test_impl(A):
            return A.describe()
        hpat_func = self.jit(test_impl)

        S = pd.Series([pd.Timestamp('1970-12-01 03:02:35'),
                       pd.NaT,
                       pd.Timestamp('1970-03-03 12:34:59'),
                       pd.Timestamp('1970-12-01 03:02:35'),
                       pd.Timestamp('2012-07-25'),
                       None])
        # SDC implementation returns series of string, hence conversion of reference result is needed
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S).astype(str))

    def test_series_describe_unsupported_percentiles(self):
        def test_impl(A, values):
            return A.describe(percentiles=values)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        unsupported_values = [0.5, '0.77', True, ('a', 'b'), ['0.5', '0.7'], np.arange(0.1, 0.5, 0.1).astype(str)]
        for percentiles in unsupported_values:
            with self.assertRaises(TypingError) as raises:
                hpat_func(S, percentiles)
            msg = 'Method describe(). The object percentiles'
            self.assertIn(msg, str(raises.exception))

    def test_series_describe_invalid_percentiles(self):
        def test_impl(A, values):
            return A.describe(percentiles=values)
        hpat_func = self.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        unsupported_values = [
            [0.5, 0.7, 1.1],
            [-0.5, 0.7, 1.1],
            [0.5, 0.7, 0.2, 0.7]
        ]
        for percentiles in unsupported_values:
            with self.assertRaises(Exception) as context:
                test_impl(S, percentiles)
            pandas_exception = context.exception

            self.assertRaises(type(pandas_exception), hpat_func, S, percentiles)

    @skip_numba_jit("TODO: support StringArray reflection")
    def test_series_setitem_str_reflection(self):
        """ Verifies that changes made to string Series passed into a jitted function
            are propagated back to the native python object.
        """
        def test_impl(S, idx, val):
            S[idx] = val
            return S
        hpat_func = self.jit(test_impl)

        S = pd.Series(['cat', '', 'bbb', '', 'a', None, 'a', '', None, 'b'])
        idx, value = 0, 'dog'
        result = hpat_func(S, idx, value)
        pd.testing.assert_series_equal(result, S)

    def _test_series_setitem(self, all_data, indexes, idxs, values, dtype=None):
        """ Common function used by setitem tests to compile and run setitem on provided data"""
        def test_impl(A, i, value):
            A[i] = value
        hpat_func = self.jit(test_impl)

        for series_data in all_data:
            for series_index in indexes:
                S = pd.Series(series_data, series_index, dtype=dtype, name='A')
                for idx, value in product(idxs, values):
                    with self.subTest(series=S, idx=idx, value=value):
                        S1 = S.copy(deep=True)
                        S2 = S.copy(deep=True)
                        hpat_func(S1, idx, value)
                        test_impl(S2, idx, value)
                        pd.testing.assert_series_equal(S1, S2)

    @skip_numba_jit('Requires StringArray support of operator.eq')
    def test_series_setitem_idx_str_scalar(self):
        """ Verifies Series.setitem for scalar string idx operand and integer Series with index of matching dtype"""

        series_data = np.arange(5)
        series_index = ['a', 'a', 'c', 'd', 'a']
        idx = 'a'
        values_to_test = [-100,
                          np.array([5, 55, 555]),
                          pd.Series([5, 55, 555])]

        self._test_series_setitem([series_data], [series_index], [idx], values_to_test)

    def test_series_setitem_idx_int_scalar1(self):
        """ Verifies Series.setitem for scalar integer idx operand and integer Series with index of matching dtype"""

        series_data = np.arange(5)
        series_index = [8, 8, 9, 8, 5]
        idx = 8
        values_to_test = [-100,
                          np.array([5, 55, 555]),
                          pd.Series([5, 55, 555])]

        self._test_series_setitem([series_data], [series_index], [idx], values_to_test)

    def test_series_setitem_idx_int_scalar2(self):
        """ Verifies Series.setitem for scalar integer idx operand and integer Series with
            index of non matching dtype (i.e. set along positions, not index)"""

        n = 11
        series_data = np.arange(n)
        series_indexes = [
                            None,
                            np.arange(n, dtype=np.float),
                            gen_strlist(n, 2, 'abcd123 ')
        ]
        idx = 8
        value = -100
        self._test_series_setitem([series_data], series_indexes, [idx], [value])

    @skip_numba_jit('TODO: support replacing data in Series with new array')
    def test_series_setitem_idx_int_scalar_non_existing(self):
        """ Verifies adding new element to an integer Series by using Series.setitem with
            scalar integer idx not present in index """
        def test_impl(A, i, value):
            A[i] = value
        hpat_func = self.jit(test_impl)

        idx, value = 0, -100
        S1 = pd.Series(np.arange(5), index=[8, 8, 9, 8, 5])
        S2 = S1.copy(deep=True)
        hpat_func(S1, idx, value)
        test_impl(S2, idx, value)
        pd.testing.assert_series_equal(S1, S2)

    def test_series_setitem_idx_str_series(self):
        """ Verifies Series.setitem for idx operand of type pandas.Series and string dtype called on
            integer Series with index of matching dtype and scalar and non scalar assigned values """

        n, k = 11, 4
        series_data = np.arange(n)
        series_index = gen_strlist(n, 2, 'abcd123 ')

        idx = create_series_from_values(k, series_index, seed=0)
        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [-100,
                          np.array(assigned_values),
                          pd.Series(assigned_values)]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test, np.intp)

    def test_series_setitem_idx_float_series(self):
        """ Verifies Series.setitem for idx operand of type pandas.Series and float dtype called on
            integer Series with index of matching dtype and scalar and non scalar assigned values """

        n, k = 11, 4
        series_data = np.arange(n)
        series_index = np.arange(n, dtype=np.float)

        idx = create_series_from_values(k, series_index, seed=0)
        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [
                            -100,
                            np.array(assigned_values),
                            pd.Series(assigned_values)
        ]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test)

    def test_series_setitem_idx_int_series1(self):
        """ Verifies Series.setitem for idx operand of type pandas.Series and integer dtype called on
            integer Series with index of matching dtype and scalar and non scalar assigned values """
        def test_impl(A, i, value):
            A[i] = value
        hpat_func = self.jit(test_impl)

        n, k = 11, 4
        series_data = np.arange(n)
        series_index = np.arange(n)

        idx = create_series_from_values(k, series_index, seed=0)
        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [-100,
                          np.array(assigned_values),
                          pd.Series(assigned_values)]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test)

    def test_series_setitem_idx_int_series2(self):
        """ Verifies Series.setitem for idx operand of type pandas.Series and integer dtype called on
            integer Series with index of non-matching dtype and scalar and non scalar assigned values """

        n, k = 11, 4
        series_data = np.arange(n)
        series_index = gen_strlist(n, 2, 'abcd123 ')

        idx = create_series_from_values(k, np.arange(n), seed=0)
        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [-100,
                          np.array(assigned_values),
                          pd.Series(assigned_values)]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test)

    def test_series_setitem_idx_int_series3(self):
        """ Verifies negative case of using Series.setitem with idx operand of type pandas.Series
            and integer dtype called on integer Series with index that has duplicate values """
        def test_impl(A, i, value):
            A[i] = value
        hpat_func = self.jit(test_impl)

        value = pd.Series(-10 + np.arange(5) * (-1))
        idx = pd.Series([8, 5, 9])
        S = pd.Series(np.arange(5), index=[8, 8, 9, 8, 5])

        # pandas raises it's own exception - pandas.core.indexes.base.InvalidIndexError
        # SDC implementation currently raises ValueError, so assert for the correct message only
        with self.assertRaises(ValueError) as raises:
            hpat_func(S, idx, value)
        msg = 'Reindexing only valid with uniquely valued Index objects'
        self.assertIn(msg, str(raises.exception))

    def test_series_setitem_idx_int_series4(self):
        """ Verifies negative case of using Series.setitem with idx operand of type pandas.Series
            and integer dtype called on integer Series with index not containg some values in idx """
        def test_impl(A, i, value):
            A[i] = value
        hpat_func = self.jit(test_impl)

        value = -100
        idx = pd.Series([1, 5, 77])
        S1 = pd.Series(np.arange(5), index=[1, 4, 3, 2, 5])
        S2 = S1.copy(deep=True)

        with self.assertRaises(Exception) as context:
            test_impl(S2, idx, value)
        pandas_exception = context.exception

        self.assertRaises(type(pandas_exception), hpat_func, S1, idx, value)

    def test_series_setitem_idx_int_array1(self):
        """ Verifies Series.setitem for idx operand of type numpy.ndarray and integer dtype called on
            integer Series with integer index and scalar and non scalar assigned values """

        n, k = 11, 4
        series_data = np.arange(n)
        series_index = np.arange(n)

        np.random.seed(0)
        idx = take_k_elements(k, series_index, seed=0)
        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [
                            -100,
                            np.array(assigned_values),
                            pd.Series(assigned_values)
        ]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test)

    def test_series_setitem_idx_int_array2(self):
        """ Verifies Series.setitem for idx operand of type numpy.ndarray and integer dtype called on
            integer Series with string index and scalar and non scalar assigned values """

        n, k = 11, 4
        series_data = np.arange(n)
        series_index = gen_strlist(n, 2, 'abcd123 ')

        idx = take_k_elements(k, np.arange(n), seed=0)
        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [
                            -100,
                            np.array(assigned_values),
                            pd.Series(assigned_values)
        ]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test)

    def test_series_setitem_idx_int_slice1(self):
        """ Verifies that Series.setitem for int slice as idx operand called on integer Series
            with index of matching dtype assigns vector value along positions (but not along index) """
        def test_impl(A, i, value):
            A[i] = value
        hpat_func = self.jit(test_impl)

        S = pd.Series(np.arange(7), index=[2, 5, 1, 3, 6, 0, 4])
        slices_to_test = [(4, ), (None, 4), (1, 4), (2, 7, 3), (None, -4), (-4, ), (None, ), (None, None, 2)]

        for slice_members in slices_to_test:
            idx = slice(*slice_members)
            k = len(np.arange(len(S))[idx])
            assigned_values = -10 + np.arange(k) * (-1)
            value = pd.Series(assigned_values)
            with self.subTest(value=value):
                S1 = S.copy(deep=True)
                S2 = S.copy(deep=True)
                hpat_func(S1, idx, value)
                test_impl(S2, idx, value)
                pd.testing.assert_series_equal(S1, S2)

    def test_series_setitem_idx_int_slice2(self):
        """ Verifies that Series.setitem for int slice as idx operand called on integer Series
            with index of matching dtype assigns scalar value along positions (but not along index) """
        def test_impl(A, idx, value):
            A[idx] = value
        hpat_func = self.jit(test_impl)

        n = 11
        dtype_to_data = {
            'int':   np.arange(n),
            'float': np.arange(n, dtype=np.float_)
        }
        dtype_to_values = {
            'int':   [100, -100],
            'float': [np.nan, np.inf, 1.25, np.PZERO, -2]
        }
        series_indexes = [
             None,
             np.arange(n, dtype='int'),
             gen_strlist(n, 2, 'abcd123 ')
        ]

        slices_to_test = [(4, ), (None, 4), (1, 4), (2, 7, 3), (None, -4), (-4, ), (None, ), (None, None, 2)]
        for dtype, series_data in dtype_to_data.items():
            for value in dtype_to_values[dtype]:
                for index in series_indexes:
                    S = pd.Series(series_data, index=index)
                    for slice_members in slices_to_test:
                        idx = slice(*slice_members)
                        S1 = S.copy(deep=True)
                        S2 = S.copy(deep=True)
                        with self.subTest(series=S1, idx=idx, value=value):
                            hpat_func(S1, idx, value)
                            test_impl(S2, idx, value)
                            pd.testing.assert_series_equal(S1, S2)

    @unittest.expectedFailure   # Fails due to incorrect behavior of pandas (doesn't set anything)
    def test_series_setitem_idx_int_slice2_fixme(self):
        """ The same as test_series_setitem_idx_int_slice2, but for float series index.
            Fails because pandas doesn't make assignment.
        """
        def test_impl(A, idx, value):
            A[idx] = value
        hpat_func = self.jit(test_impl)

        n = 11
        value = 1.25
        series_indexes = [np.arange(n, dtype='float')]
        slices_to_test = [(4, ), (None, 4), (1, 4), (2, 7, 3), (None, -4), (-4, ), (None, ), (None, None, 2)]

        for index in series_indexes:
            S = pd.Series(np.arange(n, dtype=np.float_), index=index)
            for slice_members in slices_to_test:
                idx = slice(*slice_members)
                S1 = S.copy(deep=True)
                S2 = S.copy(deep=True)
                with self.subTest(series=S1, idx=idx, value=value):
                    hpat_func(S1, idx, value)
                    test_impl(S2, idx, value)
                    pd.testing.assert_series_equal(S1, S2)

    def test_series_setitem_idx_int_scalar_no_dtype_change(self):
        """ Verifies that setting float value to an element of integer Series via scalar integer index
            converts the value and keeps origin Series dtype unchanged """
        def test_impl(A, i, value):
            A[i] = value
        hpat_func = self.jit(test_impl)

        idx, value = 3, -100.25
        S1 = pd.Series(np.arange(5), index=[5, 3, 1, 3, 2], dtype='int')
        S2 = S1.copy(deep=True)
        hpat_func(S1, idx, value)
        test_impl(S2, idx, value)
        pd.testing.assert_series_equal(S1, S2)

    @skip_numba_jit('TODO: support changing Series.dtype')
    def test_series_setitem_idx_int_slice_dtype_change(self):
        """ Verifies that setting float value to an element of integer Series with default index via integer slice
            does not trim the value but promotes Series dtype """
        def test_impl(A, i, value):
            A[i] = value
        hpat_func = self.jit(test_impl)

        idx, value = slice(1, 3), -100.25
        S1 = pd.Series(np.arange(5), index=[5, 3, 1, 3, 2], dtype='int')
        S2 = S1.copy(deep=True)
        hpat_func(S1, idx, value)
        test_impl(S2, idx, value)
        pd.testing.assert_series_equal(S1, S2)

    def test_series_setitem_idx_bool_series1(self):
        """ Verifies Series.setitem assigning scalar and non scalar values
            via mask indicated by a Boolean pandas.Series with integer index """

        n, k = 11, 4
        np.random.seed(0)
        series_data = np.arange(n)
        series_index = np.arange(n)

        # create a bool Series with the same len as S and True values at exactly k positions
        idx = pd.Series(np.zeros(n, dtype=np.bool))
        idx[take_k_elements(k, np.arange(n))] = True
        values_index = take_k_elements(k, series_index)

        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [
                            -100,
                            np.array(assigned_values),
                            pd.Series(assigned_values),
                            pd.Series(assigned_values, index=values_index),
                            pd.Series(assigned_values, index=values_index.astype('float'))
        ]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test, dtype=np.float)

    def test_series_setitem_idx_bool_series2(self):
        """ Verifies Series.setitem assigning scalar and non scalar values
            via mask indicated by a Boolean pandas.Series with string index """

        n, k = 11, 4
        np.random.seed(0)
        series_data = np.arange(n)
        series_index = gen_strlist(n, 2, 'abcd123 ')

        # create a bool Series with the same len as S, reordered values from original series index
        # as index and True values at exactly k positions
        idx = pd.Series(np.zeros(n, dtype=np.bool), index=take_k_elements(n, series_index))
        idx[take_k_elements(k, np.arange(n))] = True
        values_index = take_k_elements(k, series_index)

        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [
                            -100,
                            np.array(assigned_values),
                            pd.Series(assigned_values, index=values_index),
        ]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test, dtype=np.float)

    def test_series_setitem_idx_bool_array1(self):
        """ Verifies Series.setitem for idx operand of type numpy.ndarray and Boolean dtype called on
            integer Series with default index and scalar and non scalar assigned values. Due to no duplicates
            in idx.index or S.index the assignment is made along provided indexes. """

        n, k = 11, 4
        np.random.seed(0)
        series_data = np.arange(n)
        series_index = np.arange(n)

        # create a bool array with the same len as S and True values at exactly k positions
        idx = np.zeros(n, dtype=np.bool)
        idx[take_k_elements(k, np.arange(n))] = True
        values_index = take_k_elements(k, series_index)

        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [
                            -100,
                            np.array(assigned_values),
                            pd.Series(assigned_values),
                            pd.Series(assigned_values, index=values_index),
                            pd.Series(assigned_values, index=values_index.astype('float'))
        ]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test, dtype=np.float)

    def test_series_setitem_idx_bool_array2(self):
        """ Verifies Series.setitem for idx operand of type numpy.ndarray and Boolean dtype called on
            integer Series with default index and scalar and non scalar assigned values. Due to duplicates
            in idx.index and S.index the assignment is made along Series positions (but not index). """

        n, k = 7, 4
        series_data = np.arange(n)
        series_index = [6, 2, 3, 2, 7, 5, 1]

        idx = np.asarray([False, True, True, False, False, True, True])
        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [
                            pd.Series(assigned_values),
                            pd.Series(assigned_values, index=[1, 6, 3, 5])
        ]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test, dtype=np.float)

    def test_series_setitem_idx_bool_array3(self):
        """ Verifies Series.setitem for idx operand of type numpy.ndarray and Boolean dtype called on
            integer Series with string index and scalar and non scalar assigned values. Due to no duplicates
            in idx.index or S.index the assignment is made along provided indexes. """

        n, k = 11, 4
        np.random.seed(0)
        series_data = np.arange(n)
        series_index = gen_strlist(n, 2, 'abcd123 ')

        # create a bool array with the same len as S and True values at exactly k positions
        idx = np.zeros(n, dtype=np.bool)
        idx[take_k_elements(k, np.arange(n))] = True
        values_index = take_k_elements(k, series_index)

        assigned_values = -10 + np.arange(k) * (-1)
        values_to_test = [
                            -100,
                            np.array(assigned_values),
                            pd.Series(assigned_values, index=values_index)
        ]
        self._test_series_setitem([series_data], [series_index], [idx], values_to_test, dtype=np.float)

    def _test_series_getitem(self, all_data, indexes, idxs, dtype=None):
        """ Common function used by getitem tests to compile and run getitem on provided data"""
        def test_impl(A, idx):
            return A[idx]
        hpat_func = self.jit(test_impl)

        for series_data in all_data:
            for series_index in indexes:
                S = pd.Series(series_data, series_index, dtype=dtype, name='A')
                for idx in idxs:
                    with self.subTest(series=S, idx=idx):
                        result = hpat_func(S, idx)
                        result_ref = test_impl(S, idx)
                        pd.testing.assert_series_equal(result, result_ref)

    def test_series_getitem_idx_bool_array1(self):
        """ Verifies Series.getitem by mask indicated by a Boolean array on Series of various dtypes and indexes """

        n = 11
        np.random.seed(0)
        data_to_test = [
            np.arange(n),
            np.arange(n, dtype='float'),
            np.random.choice([True, False], n),
            gen_strlist(n, 2, 'abcd123 ')
        ]
        idxs_to_test = [
            None,
            np.arange(n),
            np.arange(n, dtype='float'),
            gen_strlist(n, 2, 'abcd123 ')
        ]

        idx = np.random.choice([True, False], n)
        self._test_series_getitem(data_to_test, idxs_to_test, [idx])

    def test_series_getitem_idx_bool_array2(self):
        """ Verifies negative case of using Series.getitem by Boolean array indexer
        on a Series with different length than the indexer """
        def test_impl(A, idx):
            return A[idx]
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        S = pd.Series(np.arange(n))
        idxs_to_test = [
            np.random.choice([True, False], n - 3),
            np.random.choice([True, False], n + 3)
        ]

        for idx in idxs_to_test:
            with self.subTest(idx=idx):
                with self.assertRaises(Exception) as context:
                    test_impl(S, idx)
                pandas_exception = context.exception

                with self.assertRaises(type(pandas_exception)) as context:
                    hpat_func(S, idx)
                sdc_exception = context.exception
                self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_series_getitem_idx_bool_list(self):
        """ Verifies Series.getitem by mask indicated by a Boolean list on Series of various dtypes and indexes """

        n = 11
        np.random.seed(0)

        data_to_test = [
            np.arange(n),
            np.arange(n, dtype='float'),
            np.random.choice([True, False], n),
            gen_strlist(n, 2, 'abcd123 ')
        ]
        idxs_to_test = [
            None,
            np.arange(n),
            np.arange(n, dtype='float'),
            gen_strlist(n, 2, 'abcd123 ')
        ]

        idx = list(np.random.choice([True, False], n))
        self._test_series_getitem(data_to_test, idxs_to_test, [idx])

    def test_series_getitem_idx_bool_series1(self):
        """ Verifies Series.getitem by mask indicated by a Boolean Series on Series of various dtypes
        when both Series and indexer have default indexes """

        n, k = 21, 13
        np.random.seed(0)

        data_to_test = [
            np.arange(n),
            np.arange(n, dtype='float'),
            np.random.choice([True, False], n),
            gen_strlist(n, 2, 'abcd123 ')
        ]

        idxs_to_test = []
        for s in (n, 2*n):
            idx = pd.Series(np.zeros(s, dtype=np.bool), index=None)
            idx[take_k_elements(k, np.arange(s))] = True
            idxs_to_test.append(idx)

        self._test_series_getitem(data_to_test, [None], idxs_to_test)

    def test_series_getitem_idx_bool_series2(self):
        """ Verifies negative case of using Series.getitem with Boolean Series indexer idx with default index
        on a Series with default index but wider range of index values """
        def test_impl(A, idx):
            return A[idx]
        hpat_func = self.jit(test_impl)

        n, k = 21, 13
        np.random.seed(0)

        S = pd.Series(np.arange(n))
        idx = pd.Series(np.zeros(n - 3, dtype=np.bool), index=None)
        idx[take_k_elements(k, np.arange(k))] = True

        with self.assertRaises(Exception) as context:
            test_impl(S, idx)
        pandas_exception = context.exception

        with self.assertRaises(type(pandas_exception)) as context:
            hpat_func(S, idx)
        sdc_exception = context.exception
        self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_series_getitem_idx_bool_series3(self):
        """ Verifies Series.getitem by mask indicated by a Boolean Series with the same object as index """
        def test_impl(A, mask, index):
            S = pd.Series(A, index)
            idx = pd.Series(mask, S.index)
            return S[idx]
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)

        idxs_to_test = [
            np.arange(n),
            np.arange(n, dtype='float'),
            gen_strlist(n, 2, 'abcd123 ')
        ]
        series_data = np.arange(n)
        mask = np.random.choice([True, False], n)
        for index in idxs_to_test:
            with self.subTest(series_index=index):
                result = hpat_func(series_data, mask, index)
                result_ref = test_impl(series_data, mask, index)
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_getitem_idx_bool_series_reindex(self):
        """ Verifies Series.getitem with reindexing by mask indicated by a Boolean Series
        on Series with various types of indexes """

        def test_impl(A, idx):
            return A[idx]
        hpat_func = self.jit(test_impl)

        n, k = 21, 13
        np.random.seed(0)

        idx_indexes_to_test = {
            'default': None,
            'int': np.arange(n),
            'float': np.arange(n, dtype='float'),
            'str': gen_strlist(n, 2, 'abcd123 ')
        }

        idx_data = np.random.choice([True, False], n)
        for idx_index in idx_indexes_to_test.values():
            idx = pd.Series(idx_data, idx_index)
            # create a series with index values in idx_index
            idx_values = idx_index if idx_index is not None else np.arange(k)
            series_index = np.random.choice(idx_values, k)
            S = pd.Series(np.arange(k), index=series_index)
            with self.subTest(series=S, idx=idx):
                result = hpat_func(S, idx)
                result_ref = test_impl(S, idx)
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_getitem_idx_bool_series_restrictions1(self):
        """ Verifies negative case of using Series.getitem with Boolean indexer with duplicate index values """
        def test_impl(A, idx):
            return A[idx]
        hpat_func = self.jit(test_impl)

        n = 7
        np.random.seed(0)

        S = pd.Series(np.arange(n))
        idx_data = [True, False, False, True, False, False, True]
        idx = pd.Series(idx_data, index=[0, 1, 2, 3, 4, 5, 0])

        with self.assertRaises(Exception) as context:
            test_impl(S, idx)
        pandas_exception = context.exception

        with self.assertRaises(type(pandas_exception)) as context:
            hpat_func(S, idx)
        sdc_exception = context.exception
        self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_series_getitem_idx_bool_series_restrictions2(self):
        """ Verifies negative case of using Series.getitem with Boolean indexer
        on a Series with some indices not present in the indexer (reindexing failure) """
        def test_impl(A, idx):
            return A[idx]
        hpat_func = self.jit(test_impl)

        n = 7
        np.random.seed(0)

        S = pd.Series(np.arange(n), index=[5, 3, 1, 2, 6, 4, 0])
        idx_data = [True, False, True, True, False]
        idx = pd.Series(idx_data, index=[4, 3, 2, 1, 0])

        with self.assertRaises(Exception) as context:
            test_impl(S, idx)
        pandas_exception = context.exception

        with self.assertRaises(type(pandas_exception)) as context:
            hpat_func(S, idx)
        sdc_exception = context.exception
        self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_series_getitem_idx_bool_series_restrictions3(self):
        """ Verifies negative case of using Series.getitem with Boolean indexer
        on a Series with index of different type that in the indexer """
        def test_impl(A, idx):
            return A[idx]
        hpat_func = self.jit(test_impl)

        n = 7
        np.random.seed(0)

        incompatible_indexes = [
            np.arange(n),
            gen_strlist(n, 2, 'abcd123 ')
        ]
        for index1, index2 in combinations(incompatible_indexes, 2):
            S = pd.Series(np.arange(n), index=index1)
            idx = pd.Series(np.random.choice([True, False], n), index=index2)
            with self.subTest(series_index=index1, idx_index=index2):
                with self.assertRaises(TypingError) as raises:
                    hpat_func(S, idx)
                msg = 'The index of boolean indexer is not comparable to Series index.'
                self.assertIn(msg, str(raises.exception))

    def test_series_skew(self):
        def test_impl(series, axis, skipna):
            return series.skew(axis=axis, skipna=skipna)

        hpat_func = self.jit(test_impl)
        test_data = [[6, 6, 2, 1, 3, 3, 2, 1, 2],
                     [1.1, 0.3, 2.1, 1, 3, 0.3, 2.1, 1.1, 2.2],
                     [6, 6.1, 2.2, 1, 3, 3, 2.2, 1, 2],
                     [],
                     [6, 6, np.nan, 2, np.nan, 1, 3, 3, np.inf, 2, 1, 2, np.inf],
                     [1.1, 0.3, np.nan, 1.0, np.inf, 0.3, 2.1, np.nan, 2.2, np.inf],
                     [1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2],
                     [np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.inf],
                     [np.inf, 0, np.inf, 1, 2, 3, 4, 5]
                     ]
        all_test_data = test_data + test_global_input_data_float64
        for data in all_test_data:
            with self.subTest(data=data):
                s = pd.Series(data)
                for axis in [0, None]:
                    with self.subTest(axis=axis):
                        for skipna in [None, False, True]:
                            with self.subTest(skipna=skipna):
                                res1 = test_impl(s, axis, skipna)
                                res2 = hpat_func(s, axis, skipna)
                                np.testing.assert_allclose(res1, res2)

    def test_series_skew_default(self):
        def test_impl():
            s = pd.Series([np.nan, -2., 3., 9.1])
            return s.skew()

        hpat_func = self.jit(test_impl)
        np.testing.assert_allclose(test_impl(), hpat_func())

    def test_series_skew_not_supported(self):
        def test_impl(series, axis=None, skipna=None, level=None, numeric_only=None):
            return series.skew(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only)

        hpat_func = self.jit(test_impl)
        s = pd.Series([1.1, 0.3, np.nan, 1, np.inf, 0, 1.1, np.nan, 2.2, np.inf, 2, 2])
        method_name = 'Method Series.skew().'
        assert_raises_ty_checker(self,
                                 [method_name, 'axis', 'float64', 'int64'],
                                 hpat_func,
                                 s, axis=0.75)

        assert_raises_ty_checker(self,
                                 [method_name, 'skipna', 'int64', 'bool'],
                                 hpat_func,
                                 s, skipna=0)

        assert_raises_ty_checker(self,
                                 [method_name, 'level', 'int64', 'None'],
                                 hpat_func,
                                 s, level=0)

        assert_raises_ty_checker(self,
                                 [method_name, 'numeric_only', 'int64', 'None'],
                                 hpat_func,
                                 s, numeric_only=0)

        with self.assertRaises(ValueError) as raises:
            hpat_func(s, axis=5)
        msg = 'Parameter axis must be only 0 or None.'
        self.assertIn(msg, str(raises.exception))


if __name__ == "__main__":
    unittest.main()

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

import numba
import numpy as np
import pandas as pd
import platform
import pyarrow.parquet as pq
import random
import string
import unittest
from pandas.api.types import CategoricalDtype

import sdc
from sdc.str_arr_ext import StringArray
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_OneDs,
                                  count_array_REPs,
                                  count_parfor_OneDs,
                                  count_parfor_REPs,
                                  dist_IR_contains,
                                  get_start_end,
                                  skip_numba_jit)


class TestJoin(TestCase):

    @skip_numba_jit
    def test_join1(self):
        def test_impl(n):
            df1 = pd.DataFrame({'key1': np.arange(n) + 3, 'A': np.arange(n) + 1.0})
            df2 = pd.DataFrame({'key2': 2 * np.arange(n) + 1, 'B': n + np.arange(n) + 1.0})
            df3 = pd.merge(df1, df2, left_on='key1', right_on='key2')
            return df3.B.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        n = 11111
        self.assertEqual(hpat_func(n), test_impl(n))

    @skip_numba_jit
    def test_join1_seq(self):
        def test_impl(df1, df2):
            df3 = df1.merge(df2, left_on='key1', right_on='key2')
            return df3

        hpat_func = self.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'key1': np.arange(n) + 3, 'A': np.arange(n) + 1.0})
        df2 = pd.DataFrame({'key2': 2 * np.arange(n) + 1, 'B': n + np.arange(n) + 1.0})
        pd.testing.assert_frame_equal(hpat_func(df1, df2), test_impl(df1, df2))
        n = 11111
        df1 = pd.DataFrame({'key1': np.arange(n) + 3, 'A': np.arange(n) + 1.0})
        df2 = pd.DataFrame({'key2': 2 * np.arange(n) + 1, 'B': n + np.arange(n) + 1.0})
        pd.testing.assert_frame_equal(hpat_func(df1, df2), test_impl(df1, df2))

    @skip_numba_jit
    def test_join1_seq_str(self):
        def test_impl():
            df1 = pd.DataFrame({'key1': ['foo', 'bar', 'baz']})
            df2 = pd.DataFrame({'key2': ['baz', 'bar', 'baz'], 'B': ['b', 'zzz', 'ss']})
            df3 = pd.merge(df1, df2, left_on='key1', right_on='key2')
            return df3.B

        hpat_func = self.jit(test_impl)
        self.assertEqual(set(hpat_func()), set(test_impl()))

    @skip_numba_jit
    def test_join1_seq_str_na(self):
        # test setting NA in string data column
        def test_impl():
            df1 = pd.DataFrame({'key1': ['foo', 'bar', 'baz']})
            df2 = pd.DataFrame({'key2': ['baz', 'bar', 'baz'], 'B': ['b', 'zzz', 'ss']})
            df3 = df1.merge(df2, left_on='key1', right_on='key2', how='left')
            return df3.B

        hpat_func = self.jit(test_impl)
        self.assertEqual(set(hpat_func()), set(test_impl()))

    @skip_numba_jit
    def test_join_mutil_seq1(self):
        def test_impl(df1, df2):
            return df1.merge(df2, on=['A', 'B'])

        hpat_func = self.jit(test_impl)
        df1 = pd.DataFrame({'A': [3, 1, 1, 3, 4],
                            'B': [1, 2, 3, 2, 3],
                            'C': [7, 8, 9, 4, 5]})

        df2 = pd.DataFrame({'A': [2, 1, 4, 4, 3],
                            'B': [1, 3, 2, 3, 2],
                            'D': [1, 2, 3, 4, 8]})

        pd.testing.assert_frame_equal(hpat_func(df1, df2), test_impl(df1, df2))

    @skip_numba_jit
    def test_join_mutil_parallel1(self):
        def test_impl(A1, B1, C1, A2, B2, D2):
            df1 = pd.DataFrame({'A': A1, 'B': B1, 'C': C1})
            df2 = pd.DataFrame({'A': A2, 'B': B2, 'D': D2})
            df3 = df1.merge(df2, on=['A', 'B'])
            return df3.C.sum() + df3.D.sum()

        hpat_func = self.jit(locals={
            'A1:input': 'distributed',
            'B1:input': 'distributed',
            'C1:input': 'distributed',
            'A2:input': 'distributed',
            'B2:input': 'distributed',
            'D2:input': 'distributed', })(test_impl)
        df1 = pd.DataFrame({'A': [3, 1, 1, 3, 4],
                            'B': [1, 2, 3, 2, 3],
                            'C': [7, 8, 9, 4, 5]})

        df2 = pd.DataFrame({'A': [2, 1, 4, 4, 3],
                            'B': [1, 3, 2, 3, 2],
                            'D': [1, 2, 3, 4, 8]})

        start, end = get_start_end(len(df1))
        h_A1 = df1.A.values[start:end]
        h_B1 = df1.B.values[start:end]
        h_C1 = df1.C.values[start:end]
        h_A2 = df2.A.values[start:end]
        h_B2 = df2.B.values[start:end]
        h_D2 = df2.D.values[start:end]
        p_A1 = df1.A.values
        p_B1 = df1.B.values
        p_C1 = df1.C.values
        p_A2 = df2.A.values
        p_B2 = df2.B.values
        p_D2 = df2.D.values
        h_res = hpat_func(h_A1, h_B1, h_C1, h_A2, h_B2, h_D2)
        p_res = test_impl(p_A1, p_B1, p_C1, p_A2, p_B2, p_D2)
        self.assertEqual(h_res, p_res)

    @skip_numba_jit
    def test_join_left_parallel1(self):
        """
        """
        def test_impl(A1, B1, C1, A2, B2, D2):
            df1 = pd.DataFrame({'A': A1, 'B': B1, 'C': C1})
            df2 = pd.DataFrame({'A': A2, 'B': B2, 'D': D2})
            df3 = df1.merge(df2, on=('A', 'B'))
            return df3.C.sum() + df3.D.sum()

        hpat_func = self.jit(locals={
            'A1:input': 'distributed',
            'B1:input': 'distributed',
            'C1:input': 'distributed', })(test_impl)
        df1 = pd.DataFrame({'A': [3, 1, 1, 3, 4],
                            'B': [1, 2, 3, 2, 3],
                            'C': [7, 8, 9, 4, 5]})

        df2 = pd.DataFrame({'A': [2, 1, 4, 4, 3],
                            'B': [1, 3, 2, 3, 2],
                            'D': [1, 2, 3, 4, 8]})

        start, end = get_start_end(len(df1))
        h_A1 = df1.A.values[start:end]
        h_B1 = df1.B.values[start:end]
        h_C1 = df1.C.values[start:end]
        h_A2 = df2.A.values
        h_B2 = df2.B.values
        h_D2 = df2.D.values
        p_A1 = df1.A.values
        p_B1 = df1.B.values
        p_C1 = df1.C.values
        p_A2 = df2.A.values
        p_B2 = df2.B.values
        p_D2 = df2.D.values
        h_res = hpat_func(h_A1, h_B1, h_C1, h_A2, h_B2, h_D2)
        p_res = test_impl(p_A1, p_B1, p_C1, p_A2, p_B2, p_D2)
        self.assertEqual(h_res, p_res)
        self.assertEqual(count_array_OneDs(), 3)

    @skip_numba_jit
    def test_join_datetime_seq1(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, on='time')

        hpat_func = self.jit(test_impl)
        df1 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-03', '2017-01-06', '2017-02-21']), 'B': [4, 5, 6]})
        df2 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-01', '2017-01-06', '2017-01-03']), 'A': [7, 8, 9]})
        pd.testing.assert_frame_equal(hpat_func(df1, df2), test_impl(df1, df2))

    @unittest.skip("Method max(). Currently function supports only numeric values. Given data type: datetime64[ns]")
    def test_join_datetime_parallel1(self):
        def test_impl(df1, df2):
            df3 = pd.merge(df1, df2, on='time')
            return (df3.A.sum(), df3.time.max(), df3.B.sum())

        hpat_func = self.jit(distributed=['df1', 'df2'])(test_impl)
        df1 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-03', '2017-01-06', '2017-02-21']), 'B': [4, 5, 6]})
        df2 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-01', '2017-01-06', '2017-01-03']), 'A': [7, 8, 9]})
        start1, end1 = get_start_end(len(df1))
        start2, end2 = get_start_end(len(df2))
        self.assertEqual(
            hpat_func(df1.iloc[start1:end1], df2.iloc[start2:end2]),
            test_impl(df1, df2))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_merge_asof_seq1(self):
        def test_impl(df1, df2):
            return pd.merge_asof(df1, df2, on='time')

        hpat_func = self.jit(test_impl)
        df1 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-03', '2017-01-06', '2017-02-21']), 'B': [4, 5, 6]})
        df2 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-01', '2017-01-02', '2017-01-04', '2017-02-23',
                 '2017-02-25']), 'A': [2, 3, 7, 8, 9]})
        pd.testing.assert_frame_equal(hpat_func(df1, df2), test_impl(df1, df2))

    @unittest.skip("Method max(). Currently function supports only numeric values. Given data type: datetime64[ns]")
    def test_merge_asof_parallel1(self):
        def test_impl():
            df1 = pd.read_parquet('asof1.pq')
            df2 = pd.read_parquet('asof2.pq')
            df3 = pd.merge_asof(df1, df2, on='time')
            return (df3.A.sum(), df3.time.max(), df3.B.sum())

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    @skip_numba_jit
    def test_join_left_seq1(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, how='left', on='key')

        hpat_func = self.jit(test_impl)
        df1 = pd.DataFrame(
            {'key': [2, 3, 5, 1, 2, 8], 'A': np.array([4, 6, 3, 9, 9, -1], np.float)})
        df2 = pd.DataFrame(
            {'key': [1, 2, 9, 3, 2], 'B': np.array([1, 7, 2, 6, 5], np.float)})
        h_res = hpat_func(df1, df2)
        res = test_impl(df1, df2)
        np.testing.assert_array_equal(h_res.key.values, res.key.values)
        # converting arrays to sets since order of values can be different
        self.assertEqual(set(h_res.A.values), set(res.A.values))
        self.assertEqual(
            set(h_res.B.dropna().values), set(res.B.dropna().values))

    @skip_numba_jit
    def test_join_left_seq2(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, how='left', on='key')

        hpat_func = self.jit(test_impl)
        # test left run where a key is repeated on left but not right side
        df1 = pd.DataFrame(
            {'key': [2, 3, 5, 3, 2, 8], 'A': np.array([4, 6, 3, 9, 9, -1], np.float)})
        df2 = pd.DataFrame(
            {'key': [1, 2, 9, 3, 10], 'B': np.array([1, 7, 2, 6, 5], np.float)})
        h_res = hpat_func(df1, df2)
        res = test_impl(df1, df2)
        np.testing.assert_array_equal(h_res.key.values, res.key.values)
        # converting arrays to sets since order of values can be different
        self.assertEqual(set(h_res.A.values), set(res.A.values))
        self.assertEqual(
            set(h_res.B.dropna().values), set(res.B.dropna().values))

    @skip_numba_jit
    def test_join_right_seq1(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, how='right', on='key')

        hpat_func = self.jit(test_impl)
        df1 = pd.DataFrame(
            {'key': [2, 3, 5, 1, 2, 8], 'A': np.array([4, 6, 3, 9, 9, -1], np.float)})
        df2 = pd.DataFrame(
            {'key': [1, 2, 9, 3, 2], 'B': np.array([1, 7, 2, 6, 5], np.float)})
        h_res = hpat_func(df1, df2)
        res = test_impl(df1, df2)
        self.assertEqual(set(h_res.key.values), set(res.key.values))
        # converting arrays to sets since order of values can be different
        self.assertEqual(set(h_res.B.values), set(res.B.values))
        self.assertEqual(
            set(h_res.A.dropna().values), set(res.A.dropna().values))

    @skip_numba_jit
    def test_join_outer_seq1(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, how='outer', on='key')

        hpat_func = self.jit(test_impl)
        df1 = pd.DataFrame(
            {'key': [2, 3, 5, 1, 2, 8], 'A': np.array([4, 6, 3, 9, 9, -1], np.float)})
        df2 = pd.DataFrame(
            {'key': [1, 2, 9, 3, 2], 'B': np.array([1, 7, 2, 6, 5], np.float)})
        h_res = hpat_func(df1, df2)
        res = test_impl(df1, df2)
        self.assertEqual(set(h_res.key.values), set(res.key.values))
        # converting arrays to sets since order of values can be different
        self.assertEqual(
            set(h_res.B.dropna().values), set(res.B.dropna().values))
        self.assertEqual(
            set(h_res.A.dropna().values), set(res.A.dropna().values))

    @skip_numba_jit
    def test_join1_seq_key_change1(self):
        # make sure const list typing doesn't replace const key values
        def test_impl(df1, df2, df3, df4):
            o1 = df1.merge(df2, on=['A'])
            o2 = df3.merge(df4, on=['B'])
            return o1, o2

        hpat_func = self.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.arange(n) + 3, 'AA': np.arange(n) + 1.0})
        df2 = pd.DataFrame({'A': 2 * np.arange(n) + 1, 'AAA': n + np.arange(n) + 1.0})
        df3 = pd.DataFrame({'B': 2 * np.arange(n) + 1, 'BB': n + np.arange(n) + 1.0})
        df4 = pd.DataFrame({'B': 2 * np.arange(n) + 1, 'BBB': n + np.arange(n) + 1.0})
        pd.testing.assert_frame_equal(hpat_func(df1, df2, df3, df4)[1], test_impl(df1, df2, df3, df4)[1])

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows', "error on windows")
    def test_join_cat1(self):
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1': np.int, 'C2': ct_dtype, 'C3': str}
            df1 = pd.read_csv("csv_data_cat1.csv",
                              names=['C1', 'C2', 'C3'],
                              dtype=dtypes,
                              )
            n = len(df1)
            df2 = pd.DataFrame({'C1': 2 * np.arange(n) + 1, 'AAA': n + np.arange(n) + 1.0})
            df3 = df1.merge(df2, on='C1')
            return df3

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows', "error on windows")
    def test_join_cat2(self):
        # test setting NaN in categorical array
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1': np.int, 'C2': ct_dtype, 'C3': str}
            df1 = pd.read_csv("csv_data_cat1.csv",
                              names=['C1', 'C2', 'C3'],
                              dtype=dtypes,
                              )
            n = len(df1)
            df2 = pd.DataFrame({'C1': 2 * np.arange(n) + 1, 'AAA': n + np.arange(n) + 1.0})
            df3 = df1.merge(df2, on='C1', how='right')
            return df3

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(
            hpat_func().sort_values('C1').reset_index(drop=True),
            test_impl().sort_values('C1').reset_index(drop=True))

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows', "error on windows")
    def test_join_cat_parallel1(self):
        # TODO: cat as keys
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1': np.int, 'C2': ct_dtype, 'C3': str}
            df1 = pd.read_csv("csv_data_cat1.csv",
                              names=['C1', 'C2', 'C3'],
                              dtype=dtypes,
                              )
            n = len(df1)
            df2 = pd.DataFrame({'C1': 2 * np.arange(n) + 1, 'AAA': n + np.arange(n) + 1.0})
            df3 = df1.merge(df2, on='C1')
            return df3

        hpat_func = self.jit(distributed=['df3'])(test_impl)
        # TODO: check results
        self.assertTrue((hpat_func().columns == test_impl().columns).all())


if __name__ == "__main__":
    unittest.main()

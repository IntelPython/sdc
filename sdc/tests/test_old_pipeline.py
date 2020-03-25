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
import numba
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
import platform
import random
import string
import unittest

from numba import types
from numba.config import IS_32BITS

import sdc
from sdc import hiframes
from sdc.str_arr_ext import StringArray
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_OneDs,
                                  count_array_REPs,
                                  count_parfor_OneDs,
                                  count_parfor_REPs,
                                  dist_IR_contains,
                                  get_rank,
                                  get_start_end,
                                  skip_numba_jit,
                                  skip_sdc_jit)
from sdc.tests.gen_test_data import ParquetGenerator
from sdc.tests.test_rolling import LONG_TEST, test_funcs
from sdc.tests.test_io import TestIO
from sdc.tests.test_basic import get_np_state_ptr

kde_file = 'kde.parquet'


class TestOldPipeline(TestCase):
    pass


class TestOldHiFrames(TestOldPipeline):

    @skip_numba_jit
    def test_pd_DataFrame_from_series_par(self):
        def test_impl(n):
            S1 = pd.Series(np.ones(n))
            S2 = pd.Series(np.random.ranf(n))
            df = pd.DataFrame({'A': S1, 'B': S2})
            return df.A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_cumsum(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df.A.cumsum()
            return Ac.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_array_OneDs(), 2)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 2)
        self.assertTrue(dist_IR_contains('dist_cumsum'))

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_column_distribution(self):
        # make sure all column calls are distributed
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            df.A.fillna(5.0, inplace=True)
            DF = df.A.fillna(5.0)
            s = DF.sum()
            m = df.A.mean()
            v = df.A.var()
            t = df.A.std()
            Ac = df.A.cumsum()
            return Ac.sum() + s + m + v + t

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertTrue(dist_IR_contains('dist_cumsum'))

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_quantile_parallel(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float64)})
            return df.A.quantile(.25)

        hpat_func = self.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_quantile_parallel_float_nan(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float32)})
            df.A[0:100] = np.nan
            df.A[200:331] = np.nan
            return df.A.quantile(.25)

        hpat_func = self.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_quantile_parallel_int(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.int32)})
            return df.A.quantile(.25)

        hpat_func = self.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_nunique_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.four.nunique()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        # test compile again for overload related issues
        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)

    @skip_numba_jit
    def test_nunique_str(self):
        def test_impl(n):
            df = pd.DataFrame({'A': ['aa', 'bb', 'aa', 'cc', 'cc']})
            return df.A.nunique()

        hpat_func = self.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        # test compile again for overload related issues
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    @unittest.skip('AssertionError - fix needed\n'
                   '5 != 3\n')
    def test_nunique_str_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.two.nunique()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        # test compile again for overload related issues
        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)

    @skip_numba_jit
    def test_unique_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return (df.four.unique() == 3.0).sum()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)

    @unittest.skip('AssertionError - fix needed\n'
                   '2 != 1\n')
    def test_unique_str_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return (df.two.unique() == 'foo').sum()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_describe(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float64)})
            return df.A.describe()

        hpat_func = self.jit(test_impl)
        n = 1001
        hpat_func(n)
        # XXX: test actual output
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_str_contains_regex(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'ADEF'])
            df = pd.DataFrame({'A': A})
            B = df.A.str.contains('AB*', regex=True)
            return B.sum()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), 2)

    @skip_numba_jit
    def test_str_contains_noregex(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'ADEF'])
            df = pd.DataFrame({'A': A})
            B = df.A.str.contains('BB', regex=False)
            return B.sum()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), 1)

    @skip_numba_jit
    def test_str_replace_regex(self):
        def test_impl(df):
            return df.A.str.replace('AB*', 'EE', regex=True)

        df = pd.DataFrame({'A': ['ABCC', 'CABBD']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_str_replace_noregex(self):
        def test_impl(df):
            return df.A.str.replace('AB', 'EE', regex=False)

        df = pd.DataFrame({'A': ['ABCC', 'CABBD']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_str_replace_regex_parallel(self):
        def test_impl(df):
            B = df.A.str.replace('AB*', 'EE', regex=True)
            return B

        n = 5
        A = ['ABCC', 'CABBD', 'CCD', 'CCDAABB', 'ED']
        start, end = get_start_end(n)
        df = pd.DataFrame({'A': A[start:end]})
        hpat_func = self.jit(distributed={'df', 'B'})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)
        self.assertEqual(count_array_REPs(), 3)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_str_split(self):
        def test_impl(df):
            return df.A.str.split(',')

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D', 'G', '', 'g,f']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_str_split_default(self):
        def test_impl(df):
            return df.A.str.split()

        df = pd.DataFrame({'A': ['AB CC', 'C ABB D', 'G ', ' ', 'g\t f']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_str_split_filter(self):
        def test_impl(df):
            B = df.A.str.split(',')
            df2 = pd.DataFrame({'B': B})
            return df2[df2.B.str.len() > 1]

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D', 'G', '', 'g,f']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(
            hpat_func(df), test_impl(df).reset_index(drop=True))

    @skip_numba_jit
    def test_str_split_box_df(self):
        def test_impl(df):
            return pd.DataFrame({'B': df.A.str.split(',')})

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df).B, test_impl(df).B, check_names=False)

    @skip_numba_jit
    def test_str_split_unbox_df(self):
        def test_impl(df):
            return df.A.iloc[0]

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D']})
        df2 = pd.DataFrame({'A': df.A.str.split(',')})
        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(df2), test_impl(df2))

    @unittest.skip('Getitem Series with list values not implement')
    def test_str_split_bool_index(self):
        def test_impl(df):
            C = df.A.str.split(',')
            return C[df.B == 'aa']

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D'], 'B': ['aa', 'bb']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_str_split_parallel(self):
        def test_impl(df):
            B = df.A.str.split(',')
            return B

        n = 5
        start, end = get_start_end(n)
        A = ['AB,CC', 'C,ABB,D', 'CAD', 'CA,D', 'AA,,D']
        df = pd.DataFrame({'A': A[start:end]})
        hpat_func = self.jit(distributed={'df', 'B'})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)
        self.assertEqual(count_array_REPs(), 3)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_str_get(self):
        def test_impl(df):
            B = df.A.str.split(',')
            return B.str.get(1)

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_str_get_parallel(self):
        def test_impl(df):
            A = df.A.str.split(',')
            B = A.str.get(1)
            return B

        n = 5
        start, end = get_start_end(n)
        A = ['AB,CC', 'C,ABB,D', 'CAD,F', 'CA,D', 'AA,,D']
        df = pd.DataFrame({'A': A[start:end]})
        hpat_func = self.jit(distributed={'df', 'B'})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)
        self.assertEqual(count_array_REPs(), 3)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_str_get_to_numeric(self):
        def test_impl(df):
            B = df.A.str.split(',')
            C = pd.to_numeric(B.str.get(1), errors='coerce')
            return C

        df = pd.DataFrame({'A': ['AB,12', 'C,321,D']})
        hpat_func = self.jit(locals={'C': types.int64[:]})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_str_flatten(self):
        def test_impl(df):
            A = df.A.str.split(',')
            return pd.Series(list(itertools.chain(*A)))

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_str_flatten_parallel(self):
        def test_impl(df):
            A = df.A.str.split(',')
            B = pd.Series(list(itertools.chain(*A)))
            return B

        n = 5
        start, end = get_start_end(n)
        A = ['AB,CC', 'C,ABB,D', 'CAD', 'CA,D', 'AA,,D']
        df = pd.DataFrame({'A': A[start:end]})
        hpat_func = self.jit(distributed={'df', 'B'})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)
        self.assertEqual(count_array_REPs(), 3)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_to_numeric(self):
        def test_impl(df):
            B = pd.to_numeric(df.A, errors='coerce')
            return B

        df = pd.DataFrame({'A': ['123.1', '331.2']})
        hpat_func = self.jit(locals={'B': types.float64[:]})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    @skip_numba_jit
    def test_1D_Var_len(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n) + 1.0})
            df1 = df[df.A > 5]
            return len(df1.B)

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_rolling1(self):
        # size 3 without unroll
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
            Ac = df.A.rolling(3).sum()
            return Ac.sum()

        hpat_func = self.jit(test_impl)
        n = 121
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        # size 7 with unroll

        def test_impl_2(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.random.ranf(n)})
            Ac = df.A.rolling(7).sum()
            return Ac.sum()

        hpat_func = self.jit(test_impl)
        n = 121
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_rolling2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            df['moving average'] = df.A.rolling(window=5, center=True).mean()
            return df['moving average'].sum()

        hpat_func = self.jit(test_impl)
        n = 121
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_rolling3(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df.A.rolling(3, center=True).apply(lambda a: a[0] + 2 * a[1] + a[2])
            return Ac.sum()

        hpat_func = self.jit(test_impl)
        n = 121
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_shift1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.random.ranf(n)})
            Ac = df.A.shift(1)
            return Ac.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_shift2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.random.ranf(n)})
            Ac = df.A.pct_change(1)
            return Ac.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_df_input(self):
        def test_impl(df):
            return df.B.sum()

        n = 121
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_df_input2(self):
        def test_impl(df):
            C = df.B == 'two'
            return C.sum()

        n = 11
        df = pd.DataFrame({'A': np.random.ranf(3 * n), 'B': ['one', 'two', 'three'] * n})
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_df_input_dist1(self):
        def test_impl(df):
            return df.B.sum()

        n = 121
        A = [3, 4, 5, 6, 1]
        B = [5, 6, 2, 1, 3]
        n = 5
        start, end = get_start_end(n)
        df = pd.DataFrame({'A': A, 'B': B})
        df_h = pd.DataFrame({'A': A[start:end], 'B': B[start:end]})
        hpat_func = self.jit(distributed={'df'})(test_impl)
        np.testing.assert_almost_equal(hpat_func(df_h), test_impl(df))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_concat(self):
        def test_impl(n):
            df1 = pd.DataFrame({'key1': np.arange(n), 'A': np.arange(n) + 1.0})
            df2 = pd.DataFrame({'key2': n - np.arange(n), 'A': n + np.arange(n) + 1.0})
            df3 = pd.concat([df1, df2])
            return df3.A.sum() + df3.key2.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        n = 11111
        self.assertEqual(hpat_func(n), test_impl(n))

    @skip_numba_jit
    def test_concat_str(self):
        def test_impl():
            df1 = pq.read_table('example.parquet').to_pandas()
            df2 = pq.read_table('example.parquet').to_pandas()
            A3 = pd.concat([df1, df2])
            return (A3.two == 'foo').sum()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_concat_series(self):
        def test_impl(n):
            df1 = pd.DataFrame({'key1': np.arange(n), 'A': np.arange(n) + 1.0})
            df2 = pd.DataFrame({'key2': n - np.arange(n), 'A': n + np.arange(n) + 1.0})
            A3 = pd.concat([df1.A, df2.A])
            return A3.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        n = 11111
        self.assertEqual(hpat_func(n), test_impl(n))

    @skip_numba_jit
    def test_concat_series_str(self):
        def test_impl():
            df1 = pq.read_table('example.parquet').to_pandas()
            df2 = pq.read_table('example.parquet').to_pandas()
            A3 = pd.concat([df1.two, df2.two])
            return (A3 == 'foo').sum()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    @unittest.skipIf(int(os.getenv('SDC_NP_MPI', '0')) > 1, 'Test hangs on NP=2 and NP=3 on all platforms')
    def test_intraday(self):
        def test_impl(nsyms):
            max_num_days = 100
            all_res = 0.0
            for i in sdc.prange(nsyms):
                s_open = 20 * np.ones(max_num_days)
                s_low = 28 * np.ones(max_num_days)
                s_close = 19 * np.ones(max_num_days)
                df = pd.DataFrame({'Open': s_open, 'Low': s_low, 'Close': s_close})
                df['Stdev'] = df['Close'].rolling(window=90).std()
                df['Moving Average'] = df['Close'].rolling(window=20).mean()
                df['Criteria1'] = (df['Open'] - df['Low'].shift(1)) < -df['Stdev']
                df['Criteria2'] = df['Open'] > df['Moving Average']
                df['BUY'] = df['Criteria1'] & df['Criteria2']
                df['Pct Change'] = (df['Close'] - df['Open']) / df['Open']
                df['Rets'] = df['Pct Change'][df['BUY']]
                all_res += df['Rets'].mean()
            return all_res

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_OneDs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

    @skip_numba_jit
    def test_var_dist1(self):
        def test_impl(A, B):
            df = pd.DataFrame({'A': A, 'B': B})
            df2 = df.groupby('A', as_index=False)['B'].sum()
            # TODO: fix handling of df setitem to force match of array dists
            # probably with a new node that is appended to the end of basic block
            # df2['C'] = np.full(len(df2.B), 3, np.int8)
            # TODO: full_like for Series
            df2['C'] = np.full_like(df2.B.values, 3, np.int8)
            return df2

        A = np.array([1, 1, 2, 3])
        B = np.array([3, 4, 5, 6])
        hpat_func = self.jit(locals={'A:input': 'distributed',
                                     'B:input': 'distributed', 'df2:return': 'distributed'})(test_impl)
        start, end = get_start_end(len(A))
        df2 = hpat_func(A[start:end], B[start:end])
        # TODO:
        # pd.testing.assert_frame_equal(
        #     hpat_func(A[start:end], B[start:end]), test_impl(A, B))


class TestOldDataFrame(TestOldPipeline):

    @unittest.expectedFailure  # https://github.com/numba/numba/issues/4690
    def test_box_dist_return(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        hpat_func = self.jit(distributed={'df'})(test_impl)
        n = 11
        hres, res = hpat_func(n), test_impl(n)
        self.assertEqual(count_array_OneDs(), 3)
        self.assertEqual(count_parfor_OneDs(), 2)
        dist_sum = self.jit(
            lambda a: sdc.distributed_api.dist_reduce(
                a, np.int32(sdc.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1)  # run to compile
        np.testing.assert_allclose(dist_sum(hres.A.sum()), res.A.sum())
        np.testing.assert_allclose(dist_sum(hres.B.sum()), res.B.sum())

    @skip_numba_jit
    def test_len1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            return len(df)

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_shape1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            return df.shape

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_column_getitem1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df['A'].values
            return Ac.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

    @skip_numba_jit
    def test_df_values_parallel1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.values.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        np.testing.assert_array_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_df_describe(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float32),
                               'B': np.arange(n)})
            # df.A[0:1] = np.nan
            return df.describe()

        hpat_func = self.jit(test_impl)
        n = 1001
        hpat_func(n)
        # XXX: test actual output
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_sort_parallel_single_col(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        # TODO: better parallel sort test
        def test_impl():
            df = pd.read_parquet('kde.parquet')
            df.sort_values('points', inplace=True)
            res = df.points.values
            return res

        hpat_func = self.jit(locals={'res:return': 'distributed'})(test_impl)

        save_min_samples = sdc.hiframes.sort.MIN_SAMPLES
        try:
            sdc.hiframes.sort.MIN_SAMPLES = 10
            res = hpat_func()
            self.assertTrue((np.diff(res) >= 0).all())
        finally:
            # restore global val
            sdc.hiframes.sort.MIN_SAMPLES = save_min_samples

    @skip_numba_jit
    def test_sort_parallel(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        # TODO: better parallel sort test
        def test_impl():
            df = pd.read_parquet('kde.parquet')
            df['A'] = df.points.astype(np.float64)
            df.sort_values('points', inplace=True)
            res = df.A.values
            return res

        hpat_func = self.jit(locals={'res:return': 'distributed'})(test_impl)

        save_min_samples = sdc.hiframes.sort.MIN_SAMPLES
        try:
            sdc.hiframes.sort.MIN_SAMPLES = 10
            res = hpat_func()
            self.assertTrue((np.diff(res) >= 0).all())
        finally:
            # restore global val
            sdc.hiframes.sort.MIN_SAMPLES = save_min_samples

    def test_filter1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + n, 'B': np.arange(n)**2})
            df1 = df[df.A > .5]
            return df1.B.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit('np.sum of Series unsupported')
    def test_filter2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + n, 'B': np.arange(n)**2})
            df1 = df.loc[df.A > .5]
            return np.sum(df1.B)

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit('np.sum of Series unsupported')
    def test_filter3(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + n, 'B': np.arange(n)**2})
            df1 = df.iloc[(df.A > .5).values]
            return np.sum(df1.B)

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)


class TestOldGroupBy(TestOldPipeline):

    @skip_sdc_jit
    @skip_numba_jit
    def test_agg_parallel(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].agg(lambda x: x.max() - x.min())
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_agg_parallel_sum(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].sum()
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_agg_parallel_count(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].count()
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_agg_parallel_mean(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].mean()
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_agg_parallel_min(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].min()
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_agg_parallel_max(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].max()
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_sdc_jit
    @skip_numba_jit
    def test_agg_parallel_var(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].var()
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_sdc_jit
    @skip_numba_jit
    def test_agg_parallel_std(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].std()
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('AssertionError - fix needed\n'
                   '16 != 20\n')
    def test_agg_parallel_str(self):
        def test_impl():
            df = pq.read_table("groupby3.pq").to_pandas()
            A = df.groupby('A')['B'].agg(lambda x: x.max() - x.min())
            return A.sum()

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_agg_parallel_all_col(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            df2 = df.groupby('A').max()
            return df2.B.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_agg_parallel_as_index(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            df2 = df.groupby('A', as_index=False).max()
            return df2.A.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_sdc_jit
    @skip_numba_jit
    def test_agg_multikey_parallel(self):
        def test_impl(in_A, in_B, in_C):
            df = pd.DataFrame({'A': in_A, 'B': in_B, 'C': in_C})
            A = df.groupby(['A', 'C'])['B'].sum()
            return A.sum()

        hpat_func = self.jit(locals={'in_A:input': 'distributed',
                                     'in_B:input': 'distributed',
                                     'in_C:input': 'distributed'})(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7],
                           'C': [3, 5, 6, 5, 4, 4, 3]})
        start, end = get_start_end(len(df))
        h_A = df.A.values[start:end]
        h_B = df.B.values[start:end]
        h_C = df.C.values[start:end]
        p_A = df.A.values
        p_B = df.B.values
        p_C = df.C.values
        h_res = hpat_func(h_A, h_B, h_C)
        p_res = test_impl(p_A, p_B, p_C)
        self.assertEqual(h_res, p_res)

    @skip_sdc_jit
    @skip_numba_jit
    def test_muti_hiframes_node_filter_agg(self):
        def test_impl(df, cond):
            df2 = df[cond]
            c = df2.groupby('A')['B'].count()
            return df2.C, c

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7], 'C': [2, 3, -1, 1, 2, 3, -1]})
        cond = df.A > 1
        res = test_impl(df, cond)
        h_res = hpat_func(df, cond)
        self.assertEqual(set(res[1]), set(h_res[1]))
        np.testing.assert_array_equal(res[0], h_res[0])

    @skip_numba_jit
    def test_pivot_parallel(self):
        def test_impl():
            df = pd.read_parquet("pivot2.pq")
            pt = df.pivot_table(index='A', columns='C', values='D', aggfunc='sum')
            res = pt.small.values.sum()
            return res

        hpat_func = self.jit(
            pivots={'pt': ['small', 'large']})(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    @skip_numba_jit
    def test_crosstab_parallel1(self):
        def test_impl():
            df = pd.read_parquet("pivot2.pq")
            pt = pd.crosstab(df.A, df.C)
            res = pt.small.values.sum()
            return res

        hpat_func = self.jit(
            pivots={'pt': ['small', 'large']})(test_impl)
        self.assertEqual(hpat_func(), test_impl())


class TestOldJoin(TestOldPipeline):

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


class TestOldRolling(TestOldPipeline):

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


class TestOldSeries(TestOldPipeline):

    @skip_numba_jit
    def test_series_fusion1(self):
        def test_impl(A, B):
            return A + B + 1
        hpat_func = self.jit(test_impl)

        n = 11
        if platform.system() == 'Windows' and not IS_32BITS:
            A = pd.Series(np.arange(n), dtype=np.int64)
            B = pd.Series(np.arange(n)**2, dtype=np.int64)
        else:
            A = pd.Series(np.arange(n))
            B = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 1)

    @skip_numba_jit
    def test_series_fusion2(self):
        # make sure getting data var avoids incorrect single def assumption
        def test_impl(A, B):
            S = B + 2
            if A[0] == 0:
                S = A + 1
            return S + B
        hpat_func = self.jit(test_impl)

        n = 11
        if platform.system() == 'Windows' and not IS_32BITS:
            A = pd.Series(np.arange(n), dtype=np.int64)
            B = pd.Series(np.arange(n)**2, dtype=np.int64)
        else:
            A = pd.Series(np.arange(n))
            B = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 3)

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

    @skip_sdc_jit("Fails to compile with latest Numba")
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

    @skip_numba_jit('Series.nlargest() parallelism unsupported and parquet not supported')
    def test_series_nlargest_parallel(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.nlargest(4)
        hpat_func = self.jit(test_impl)

        if sdc.config.config_pipeline_hpat_default:
            np.testing.assert_array_equal(test_impl(), hpat_func())
        else:
            pd.testing.assert_series_equal(test_impl(), hpat_func())
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertTrue(count_array_OneDs() > 0)

    @skip_numba_jit('Series.nsmallest() parallelism unsupported and parquet not supported')
    def test_series_nsmallest_parallel(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.nsmallest(4)
        hpat_func = self.jit(test_impl)

        if sdc.config.config_pipeline_hpat_default:
            np.testing.assert_array_equal(test_impl(), hpat_func())
        else:
            pd.testing.assert_series_equal(test_impl(), hpat_func())
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertTrue(count_array_OneDs() > 0)

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


class TestOldIO(TestOldPipeline, TestIO):

    @skip_numba_jit
    @skip_sdc_jit('AttributeError: Failed in hpat mode pipeline (step: convert to distributed)\n'
                  'module \'sdc.hio\' has no attribute \'file_write_parallel\'')
    def test_write_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({'A': np.arange(n)})
            df.to_csv(fname)

        hpat_func = self.jit(test_impl)
        n = 111
        hp_fname = 'test_write_csv1_hpat_par.csv'
        pd_fname = 'test_write_csv1_pd_par.csv'
        hpat_func(n, hp_fname)
        test_impl(n, pd_fname)
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        # TODO: delete files
        if get_rank() == 0:
            pd.testing.assert_frame_equal(
                pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_np_io_parallel(self):
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_read(self):
        def test_impl():
            t = pq.read_table('kde.parquet')
            df = t.to_pandas()
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_read_global_str1(self):
        def test_impl():
            df = pd.read_parquet(kde_file)
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_read_freevar_str1(self):
        kde_file2 = 'kde.parquet'

        def test_impl():
            df = pd.read_parquet(kde_file2)
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pd_read_parquet(self):
        def test_impl():
            df = pd.read_parquet('kde.parquet')
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_str(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.two.values == 'foo'
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_str_with_nan_seq(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values == 'foo'
            return A

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    @skip_numba_jit
    def test_pq_str_with_nan_par(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values == 'foo'
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_str_with_nan_par_multigroup(self):
        def test_impl():
            df = pq.read_table('example2.parquet').to_pandas()
            A = df.five.values == 'foo'
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_bool(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.three.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.one.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_float_no_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.four.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_pandas_date(self):
        def test_impl():
            df = pd.read_parquet('pandas_dt.pq')
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_sdc_jit('Error: Attribute "dtype" are different\n'
                  '[left]:  datetime64[ns]\n'
                  '[right]: object')
    @skip_numba_jit
    def test_pq_spark_date(self):
        def test_impl():
            df = pd.read_parquet('sdf_dt.pq')
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())


class TestOldBasic(TestOldPipeline):

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_getitem(self):
        def test_impl(N):
            A = np.ones(N)
            B = np.ones(N) > .5
            C = A[B]
            return C.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_setitem1(self):
        def test_impl(N):
            A = np.arange(10) + 1.0
            A[0] = 30
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_setitem2(self):
        def test_impl(N):
            A = np.arange(10) + 1.0
            A[0:4] = 30
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_astype(self):
        def test_impl(N):
            return np.ones(N).astype(np.int32).sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shape(self):
        def test_impl(N):
            return np.ones(N).shape[0]

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

        # def test_impl(N):
        #     return np.ones((N, 3, 4)).shape
        #
        # hpat_func = self.jit(test_impl)
        # n = 128
        # np.testing.assert_allclose(hpat_func(n), test_impl(n))
        # self.assertEqual(count_array_REPs(), 0)
        # self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_inplace_binop(self):
        def test_impl(N):
            A = np.ones(N)
            B = np.ones(N)
            B += A
            return B.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_getitem_multidim(self):
        def test_impl(N):
            A = np.ones((N, 3))
            B = np.ones(N) > .5
            C = A[B, 2]
            return C.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_whole_slice(self):
        def test_impl(N):
            X = np.ones((N, 4))
            X[:, 3] = (X[:, 3]) / (np.max(X[:, 3]) - np.min(X[:, 3]))
            return X.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_strided_getitem(self):
        def test_impl(N):
            A = np.ones(N)
            B = A[::7]
            return B.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_reduce(self):
        import sys
        dtypes = ['float32', 'float64', 'int32', 'int64']
        funcs = ['sum', 'prod', 'min', 'max', 'argmin', 'argmax']
        for (dtype, func) in itertools.product(dtypes, funcs):
            # loc allreduce doesn't support int64 on windows
            if (sys.platform.startswith('win')
                    and dtype == 'int64'
                    and func in ['argmin', 'argmax']):
                continue
            func_text = """def f(n):
                A = np.arange(0, n, 1, np.{})
                return A.{}()
            """.format(dtype, func)
            loc_vars = {}
            exec(func_text, {'np': np}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = self.jit(test_impl)
            n = 21  # XXX arange() on float32 has overflow issues on large n
            np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_reduce2(self):
        import sys
        dtypes = ['float32', 'float64', 'int32', 'int64']
        funcs = ['sum', 'prod', 'min', 'max', 'argmin', 'argmax']
        for (dtype, func) in itertools.product(dtypes, funcs):
            # loc allreduce doesn't support int64 on windows
            if (sys.platform.startswith('win')
                    and dtype == 'int64'
                    and func in ['argmin', 'argmax']):
                continue
            func_text = """def f(A):
                return A.{}()
            """.format(func)
            loc_vars = {}
            exec(func_text, {'np': np}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = self.jit(locals={'A:input': 'distributed'})(test_impl)
            n = 21
            start, end = get_start_end(n)
            np.random.seed(0)
            A = np.random.randint(0, 10, n).astype(dtype)
            np.testing.assert_almost_equal(
                hpat_func(A[start:end]), test_impl(A), decimal=3)
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_reduce_filter1(self):
        import sys
        dtypes = ['float32', 'float64', 'int32', 'int64']
        funcs = ['sum', 'prod', 'min', 'max', 'argmin', 'argmax']
        for (dtype, func) in itertools.product(dtypes, funcs):
            # loc allreduce doesn't support int64 on windows
            if (sys.platform.startswith('win')
                    and dtype == 'int64'
                    and func in ['argmin', 'argmax']):
                continue
            func_text = """def f(A):
                A = A[A>5]
                return A.{}()
            """.format(func)
            loc_vars = {}
            exec(func_text, {'np': np}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = self.jit(locals={'A:input': 'distributed'})(test_impl)
            n = 21
            start, end = get_start_end(n)
            np.random.seed(0)
            A = np.random.randint(0, 10, n).astype(dtype)
            np.testing.assert_almost_equal(
                hpat_func(A[start:end]), test_impl(A), decimal=3,
                err_msg="{} on {}".format(func, dtype))
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
    def test_array_reduce(self):
        binops = ['+=', '*=', '+=', '*=', '|=', '|=']
        dtypes = ['np.float32', 'np.float32', 'np.float64', 'np.float64', 'np.int32', 'np.int64']
        for (op, typ) in zip(binops, dtypes):
            func_text = """def f(n):
                  A = np.arange(0, 10, 1, {})
                  B = np.arange(0 +  3, 10 + 3, 1, {})
                  for i in numba.prange(n):
                      A {} B
                  return A
            """.format(typ, typ, op)
            loc_vars = {}
            exec(func_text, {'np': np, 'numba': numba}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = self.jit(test_impl)
            n = 128
            np.testing.assert_allclose(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_OneDs(), 0)
            self.assertEqual(count_parfor_OneDs(), 1)

    @unittest.expectedFailure  # https://github.com/numba/numba/issues/4690
    def test_dist_return(self):
        def test_impl(N):
            A = np.arange(N)
            return A

        hpat_func = self.jit(locals={'A:return': 'distributed'})(test_impl)
        n = 128
        dist_sum = self.jit(
            lambda a: sdc.distributed_api.dist_reduce(
                a, np.int32(sdc.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1)  # run to compile
        np.testing.assert_allclose(
            dist_sum(hpat_func(n).sum()), test_impl(n).sum())
        self.assertEqual(count_array_OneDs(), 1)
        self.assertEqual(count_parfor_OneDs(), 1)

    @unittest.expectedFailure  # https://github.com/numba/numba/issues/4690
    def test_dist_return_tuple(self):
        def test_impl(N):
            A = np.arange(N)
            B = np.arange(N) + 1.5
            return A, B

        hpat_func = self.jit(locals={'A:return': 'distributed',
                                     'B:return': 'distributed'})(test_impl)
        n = 128
        dist_sum = self.jit(
            lambda a: sdc.distributed_api.dist_reduce(
                a, np.int32(sdc.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1.0)  # run to compile
        np.testing.assert_allclose(
            dist_sum((hpat_func(n)[0] + hpat_func(n)[1]).sum()), (test_impl(n)[0] + test_impl(n)[1]).sum())
        self.assertEqual(count_array_OneDs(), 2)
        self.assertEqual(count_parfor_OneDs(), 2)

    @skip_numba_jit
    def test_dist_input(self):
        def test_impl(A):
            return len(A)

        hpat_func = self.jit(distributed=['A'])(test_impl)
        n = 128
        arr = np.ones(n)
        np.testing.assert_allclose(hpat_func(arr) / self.num_ranks, test_impl(arr))
        self.assertEqual(count_array_OneDs(), 1)

    @skip_sdc_jit('Not implemented in sequential transport layer')
    @unittest.expectedFailure  # https://github.com/numba/numba/issues/4690
    def test_rebalance(self):
        def test_impl(N):
            A = np.arange(n)
            B = A[A > 10]
            C = sdc.distributed_api.rebalance_array(B)
            return C.sum()

        try:
            sdc.distributed_analysis.auto_rebalance = True
            hpat_func = self.jit(test_impl)
            n = 128
            np.testing.assert_allclose(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_OneDs(), 3)
            self.assertEqual(count_parfor_OneDs(), 2)
        finally:
            sdc.distributed_analysis.auto_rebalance = False

    @skip_sdc_jit('Not implemented in sequential transport layer')
    @unittest.expectedFailure  # https://github.com/numba/numba/issues/4690
    def test_rebalance_loop(self):
        def test_impl(N):
            A = np.arange(n)
            B = A[A > 10]
            s = 0
            for i in range(3):
                s += B.sum()
            return s

        try:
            sdc.distributed_analysis.auto_rebalance = True
            hpat_func = self.jit(test_impl)
            n = 128
            np.testing.assert_allclose(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_OneDs(), 4)
            self.assertEqual(count_parfor_OneDs(), 2)
            self.assertIn('allgather', list(hpat_func.inspect_llvm().values())[0])
        finally:
            sdc.distributed_analysis.auto_rebalance = False

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_transpose(self):
        def test_impl(n):
            A = np.ones((30, 40, 50))
            B = A.transpose((0, 2, 1))
            C = A.transpose(0, 2, 1)
            return B.sum() + C.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip("Numba's perfmute generation needs to use np seed properly")
    def test_permuted_array_indexing(self):

        # Since Numba uses Python's PRNG for producing random numbers in NumPy,
        # we cannot compare against NumPy.  Therefore, we implement permutation
        # in Python.
        def python_permutation(n, r):
            arr = np.arange(n)
            r.shuffle(arr)
            return arr

        def test_one_dim(arr_len):
            A = np.arange(arr_len)
            B = np.copy(A)
            P = np.random.permutation(arr_len)
            A, B = A[P], B[P]
            return A, B

        # Implementation that uses Python's PRNG for producing a permutation.
        # We test against this function.
        def python_one_dim(arr_len, r):
            A = np.arange(arr_len)
            B = np.copy(A)
            P = python_permutation(arr_len, r)
            A, B = A[P], B[P]
            return A, B

        # Ideally, in above *_impl functions we should just call
        # np.random.seed() and they should produce the same sequence of random
        # numbers.  However, since Numba's PRNG uses NumPy's initialization
        # method for initializing PRNG, we cannot just set seed.  Instead, we
        # resort to this hack that generates a Python Random object with a fixed
        # seed and copies the state to Numba's internal NumPy PRNG state.  For
        # details please see https://github.com/numba/numba/issues/2782.
        r = self._follow_cpython(get_np_state_ptr())

        hpat_func1 = self.jit(locals={'A:return': 'distributed',
                                      'B:return': 'distributed'})(test_one_dim)

        # Test one-dimensional array indexing.
        for arr_len in [11, 111, 128, 120]:
            hpat_A, hpat_B = hpat_func1(arr_len)
            python_A, python_B = python_one_dim(arr_len, r)
            rank_bounds = self._rank_bounds(arr_len)
            np.testing.assert_allclose(hpat_A, python_A[slice(*rank_bounds)])
            np.testing.assert_allclose(hpat_B, python_B[slice(*rank_bounds)])

        # Test two-dimensional array indexing.  Like in one-dimensional case
        # above, in addition to NumPy version that is compiled by Numba, we
        # implement a Python version.
        def test_two_dim(arr_len):
            first_dim = arr_len // 2
            A = np.arange(arr_len).reshape(first_dim, 2)
            B = np.copy(A)
            P = np.random.permutation(first_dim)
            A, B = A[P], B[P]
            return A, B

        def python_two_dim(arr_len, r):
            first_dim = arr_len // 2
            A = np.arange(arr_len).reshape(first_dim, 2)
            B = np.copy(A)
            P = python_permutation(first_dim, r)
            A, B = A[P], B[P]
            return A, B

        hpat_func2 = self.jit(locals={'A:return': 'distributed',
                                      'B:return': 'distributed'})(test_two_dim)

        for arr_len in [18, 66, 128]:
            hpat_A, hpat_B = hpat_func2(arr_len)
            python_A, python_B = python_two_dim(arr_len, r)
            rank_bounds = self._rank_bounds(arr_len // 2)
            np.testing.assert_allclose(hpat_A, python_A[slice(*rank_bounds)])
            np.testing.assert_allclose(hpat_B, python_B[slice(*rank_bounds)])

        # Test that the indexed array is not modified if it is not being
        # assigned to.
        def test_rhs(arr_len):
            A = np.arange(arr_len)
            B = np.copy(A)
            P = np.random.permutation(arr_len)
            C = A[P]
            return A, B, C

        hpat_func3 = self.jit(locals={'A:return': 'distributed',
                                      'B:return': 'distributed',
                                      'C:return': 'distributed'})(test_rhs)

        for arr_len in [15, 23, 26]:
            A, B, _ = hpat_func3(arr_len)
            np.testing.assert_allclose(A, B)


if __name__ == "__main__":
    unittest.main()

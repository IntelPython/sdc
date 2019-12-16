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
import platform
import random
import string
import platform
import pandas as pd
import numpy as np

import numba
import sdc
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_REPs, count_parfor_REPs, count_parfor_OneDs,
                                  count_array_OneDs, dist_IR_contains, get_start_end, check_numba_version,
                                  skip_numba_jit)

from sdc.tests.gen_test_data import ParquetGenerator
from sdc.tests.test_utils import (min_float64, max_float64, test_global_input_data_float64,
                                  test_global_input_data_unicode_kind4, test_datatime,
                                  min_int64, max_int64, test_global_input_data_int64)
from numba.config import IS_32BITS


@sdc.jit
def inner_get_column(df):
    # df2 = df[['A', 'C']]
    # df2['D'] = np.ones(3)
    return df.A


COL_IND = 0


class TestDataFrame(TestCase):

    def test_create1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            return df.A

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_create_cond1(self):
        def test_impl(A, B, c):
            if c:
                df = pd.DataFrame({'A': A})
            else:
                df = pd.DataFrame({'A': B})
            return df.A

        hpat_func = self.jit(test_impl)
        n = 11
        A = np.ones(n)
        B = np.arange(n) + 1.0
        c = 0
        pd.testing.assert_series_equal(hpat_func(A, B, c), test_impl(A, B, c))
        c = 2
        pd.testing.assert_series_equal(hpat_func(A, B, c), test_impl(A, B, c))

    @unittest.skip('Implement feature to create DataFrame without column names')
    def test_create_without_column_names(self):
        def test_impl():
            df = pd.DataFrame([100, 200, 300, 400, 200, 100])
            return df

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_unbox1(self):
        def test_impl(df):
            return df.A

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @unittest.skip("needs properly refcounted dataframes")
    def test_unbox2(self):
        def test_impl(df, cond):
            n = len(df)
            if cond:
                df['A'] = np.arange(n) + 2.0
            return df.A

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        pd.testing.assert_series_equal(hpat_func(df.copy(), True), test_impl(df.copy(), True))
        pd.testing.assert_series_equal(hpat_func(df.copy(), False), test_impl(df.copy(), False))

    @unittest.skip('Implement feature to create DataFrame without column names')
    def test_unbox_without_column_names(self):
        def test_impl(df):
            return df

        df = pd.DataFrame([100, 200, 300, 400, 200, 100])
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @unittest.skip('returned NULL without setting an error')
    def test_box1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        hpat_func = self.jit(test_impl)
        n = 11
        do_check = False if platform.system() == 'Windows' and not IS_32BITS else True
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n), check_dtype=do_check)

    def test_box2(self):
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'bb', 'ccc']})
            return df

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip("pending df filter support")
    def test_box3(self):
        def test_impl(df):
            df = df[df.A != 'dd']
            return df

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': ['aa', 'bb', 'cc']})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_box_categorical(self):
        def test_impl(df):
            df['A'] = df['A'] + 1
            return df

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': pd.Series(['N', 'Y', 'Y'],
                                          dtype=pd.api.types.CategoricalDtype(['N', 'Y']))})
        pd.testing.assert_frame_equal(hpat_func(df.copy(deep=True)), test_impl(df))

    @unittest.skip('does not support option: "distributed"')
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

    def test_len1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            return len(df)

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

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

    def test_column_list_getitem1(self):
        def test_impl(df):
            return df[['A', 'C']]

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame(
            {'A': np.arange(n), 'B': np.ones(n), 'C': np.random.ranf(n)})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
    def test_iloc1(self):
        def test_impl(df, n):
            return df.iloc[1:n].B.values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df, n), test_impl(df, n))

    @skip_numba_jit
    def test_iloc2(self):
        def test_impl(df, n):
            return df.iloc[np.array([1, 4, 9])].B.values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df, n), test_impl(df, n))

    def test_iloc3(self):
        def test_impl(df):
            return df.iloc[:, 1].values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @unittest.skip("TODO: support A[[1,2,3]] in Numba")
    def test_iloc4(self):
        def test_impl(df, n):
            return df.iloc[[1, 4, 9]].B.values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df, n), test_impl(df, n))

    def test_iloc5(self):
        # test iloc with global value
        def test_impl(df):
            return df.iloc[:, COL_IND].values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_loc1(self):
        def test_impl(df):
            return df.loc[:, 'B'].values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_iat1(self):
        def test_impl(n):
            df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
            return df.iat[3, 1]
        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_iat2(self):
        def test_impl(df):
            return df.iat[3, 1]
        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_iat3(self):
        def test_impl(df, n):
            return df.iat[n - 1, 1]
        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        self.assertEqual(hpat_func(df, n), test_impl(df, n))

    def test_iat_set1(self):
        def test_impl(df, n):
            df.iat[n - 1, 1] = n**2
            return df.A  # return the column to check column aliasing
        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        df2 = df.copy()
        pd.testing.assert_series_equal(hpat_func(df, n), test_impl(df2, n))

    def test_iat_set2(self):
        def test_impl(df, n):
            df.iat[n - 1, 1] = n**2
            return df  # check df aliasing/boxing
        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        df2 = df.copy()
        pd.testing.assert_frame_equal(hpat_func(df, n), test_impl(df2, n))

    def test_set_column1(self):
        # set existing column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
            df['A'] = np.arange(n)
            return df

        hpat_func = self.jit(test_impl)
        n = 11
        do_check = False if platform.system() == 'Windows' and not IS_32BITS else True
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n), check_dtype=do_check)

    def test_set_column_reflect4(self):
        # set existing column
        def test_impl(df, n):
            df['A'] = np.arange(n)

        hpat_func = self.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
        df2 = df1.copy()
        hpat_func(df1, n)
        test_impl(df2, n)
        do_check = False if platform.system() == 'Windows' and not IS_32BITS else True
        pd.testing.assert_frame_equal(df1, df2, check_dtype=do_check)

    def test_set_column_new_type1(self):
        # set existing column with a new type
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n) + 3.0})
            df['A'] = np.arange(n)
            return df

        hpat_func = self.jit(test_impl)
        n = 11
        do_check = False if platform.system() == 'Windows' and not IS_32BITS else True
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n), check_dtype=do_check)

    def test_set_column2(self):
        # create new column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n) + 1.0})
            df['C'] = np.arange(n)
            return df

        hpat_func = self.jit(test_impl)
        n = 11
        do_check = False if platform.system() == 'Windows' and not IS_32BITS else True
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n), check_dtype=do_check)

    def test_set_column_reflect3(self):
        # create new column
        def test_impl(df, n):
            df['C'] = np.arange(n)

        hpat_func = self.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
        df2 = df1.copy()
        hpat_func(df1, n)
        test_impl(df2, n)
        do_check = False if platform.system() == 'Windows' and not IS_32BITS else True
        pd.testing.assert_frame_equal(df1, df2, check_dtype=do_check)

    @skip_numba_jit
    def test_set_column_bool1(self):
        def test_impl(df):
            df['C'] = df['A'][df['B']]

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [True, False, True]})
        df2 = df.copy()
        test_impl(df2)
        hpat_func(df)
        pd.testing.assert_series_equal(df.C, df2.C)

    def test_set_column_reflect1(self):
        def test_impl(df, arr):
            df['C'] = arr
            return df.C.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        hpat_func(df, arr)
        self.assertIn('C', df)
        np.testing.assert_almost_equal(df.C.values, arr)

    def test_set_column_reflect2(self):
        def test_impl(df, arr):
            df['C'] = arr
            return df.C.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        df2 = df.copy()
        np.testing.assert_almost_equal(hpat_func(df, arr), test_impl(df2, arr))

    def test_df_values1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.values

        hpat_func = self.jit(test_impl)
        n = 11
        np.testing.assert_array_equal(hpat_func(n), test_impl(n))

    def test_df_values2(self):
        def test_impl(df):
            return df.values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

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
    def test_df_apply(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A + r.B, axis=1)
            return df.B.sum()

        n = 121
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    @skip_numba_jit
    def test_df_apply_branch(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A < 10 and r.B > 20, axis=1)
            return df.B.sum()

        n = 121
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    def test_df_describe(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float32),
                               'B': np.arange(n)})
            #df.A[0:1] = np.nan
            return df.describe()

        hpat_func = self.jit(test_impl)
        n = 1001
        hpat_func(n)
        # XXX: test actual output
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_sort_values(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.B.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n), 'B': np.arange(n), 'C': np.random.ranf(n)})
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df.copy()), test_impl(df))

    @skip_numba_jit
    def test_sort_values_copy(self):
        def test_impl(df):
            df2 = df.sort_values('A')
            return df2.B.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n), 'B': np.arange(n), 'C': np.random.ranf(n)})
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df.copy()), test_impl(df))

    @skip_numba_jit
    def test_sort_values_single_col(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.A.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n)})
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df.copy()), test_impl(df))

    @skip_numba_jit
    def test_sort_values_single_col_str(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.A.values

        n = 1211
        random.seed(2)
        str_vals = []

        for _ in range(n):
            k = random.randint(1, 30)
            val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)
        df = pd.DataFrame({'A': str_vals})
        hpat_func = self.jit(test_impl)
        self.assertTrue((hpat_func(df.copy()) == test_impl(df)).all())

    @skip_numba_jit
    def test_sort_values_str(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.B.values

        n = 1211
        random.seed(2)
        str_vals = []
        str_vals2 = []

        for i in range(n):
            k = random.randint(1, 30)
            val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)
            val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals2.append(val)

        df = pd.DataFrame({'A': str_vals, 'B': str_vals2})
        # use mergesort for stability, in str generation equal keys are more probable
        sorted_df = df.sort_values('A', inplace=False, kind='mergesort')
        hpat_func = self.jit(test_impl)
        self.assertTrue((hpat_func(df) == sorted_df.B.values).all())

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

    def test_df_isna1(self):
        '''Verify DataFrame.isna implementation for various types of data'''
        def test_impl(df):
            return df.isna()
        hpat_func = self.jit(test_impl)

        # TODO: add column with datetime values when test_series_datetime_isna1 is fixed
        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0],
                           'B': [np.inf, 5, np.nan, 6],
                           'C': ['aa', 'b', None, 'ccc'],
                           'D': [None, 'dd', '', None]})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_astype_str1(self):
        '''Verifies DataFrame.astype implementation converting various types to string'''
        def test_impl(df):
            return df.astype(str)
        hpat_func = self.jit(test_impl)

        # TODO: add column with float values when test_series_astype_float_to_str1 is fixed
        df = pd.DataFrame({'A': [-1, 2, 11, 5, 0, -7],
                           'B': ['aa', 'bb', 'cc', 'dd', '', 'fff']
                           })
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_astype_float1(self):
        '''Verifies DataFrame.astype implementation converting various types to float'''
        def test_impl(df):
            return df.astype(np.float64)
        hpat_func = self.jit(test_impl)

        # TODO: uncomment column with string values when test_series_astype_str_to_float64 is fixed
        df = pd.DataFrame({'A': [-1, 2, 11, 5, 0, -7],
                           #                   'B': ['3.24', '1E+05', '-1', '-1.3E-01', 'nan', 'inf'],
                           'C': [3.24, 1E+05, -1, -1.3E-01, np.nan, np.inf]
                           })
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_astype_int1(self):
        '''Verifies DataFrame.astype implementation converting various types to int'''
        def test_impl(df):
            return df.astype(np.int32)
        hpat_func = self.jit(test_impl)

        n = 6
        # TODO: uncomment column with string values when test_series_astype_str_to_int32 is fixed
        df = pd.DataFrame({'A': np.ones(n, dtype=np.int64),
                           'B': np.arange(n, dtype=np.int32),
                           #                   'C': ['-1', '2', '3', '0', '-7', '99'],
                           'D': np.arange(float(n), dtype=np.float32)
                           })
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

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

    def test_itertuples(self):
        def test_impl(df):
            res = 0.0
            for r in df.itertuples():
                res += r[1]
            return res

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.ones(n, np.int64)})
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_itertuples_str(self):
        def test_impl(df):
            res = ""
            for r in df.itertuples():
                res += r[1]
            return res

        hpat_func = self.jit(test_impl)
        n = 3
        df = pd.DataFrame({'A': ['aa', 'bb', 'cc'], 'B': np.ones(n, np.int64)})
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_itertuples_order(self):
        def test_impl(n):
            res = 0.0
            df = pd.DataFrame({'B': np.arange(n), 'A': np.ones(n, np.int64)})
            for r in df.itertuples():
                res += r[1]
            return res

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_itertuples_analysis(self):
        """tests array analysis handling of generated tuples, shapes going
        through blocks and getting used in an array dimension
        """
        def test_impl(n):
            res = 0
            df = pd.DataFrame({'B': np.arange(n), 'A': np.ones(n, np.int64)})
            for r in df.itertuples():
                if r[1] == 2:
                    A = np.ones(r[1])
                    res += len(A)
            return res

        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    @unittest.skipIf(platform.system() == 'Windows', "Attribute 'dtype' are different int64 and int32")
    def test_df_head1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.head(3)

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @skip_numba_jit
    def test_pct_change1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.pct_change(3)

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    def test_mean1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.mean()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_median1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': 2 ** np.arange(n), 'B': np.arange(n) + 1.0})
            return df.median()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_std1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.std()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_var1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.var()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_max1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.max()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_min1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.min()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    @unittest.skipIf(not sdc.config.config_pipeline_hpat_default, "DataFrame.sum() not implemented in new style")
    def test_sum1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.sum()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_prod1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.prod()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_count(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            return df.count()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_count1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.count()

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_df_fillna1(self):
        def test_impl(df):
            return df.fillna(0.)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_fillna_str1(self):
        def test_impl(df):
            return df.fillna("dd")

        df = pd.DataFrame({'A': ['aa', 'b', None, 'ccc']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_fillna_inplace1(self):
        def test_impl(A):
            A.fillna(11.0, inplace=True)
            return A

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        df2 = df.copy()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df2))

    def test_df_reset_index1(self):
        def test_impl(df):
            return df.reset_index(drop=True)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_reset_index_inplace1(self):
        def test_impl():
            df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
            df.reset_index(drop=True, inplace=True)
            return df

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_numba_jit
    def test_df_dropna1(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        hpat_func = self.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = hpat_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    @skip_numba_jit
    def test_df_dropna2(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        hpat_func = self.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = hpat_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    @skip_numba_jit
    def test_df_dropna_inplace1(self):
        # TODO: fix error when no df is returned
        def test_impl(df):
            df.dropna(inplace=True)
            return df

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        df2 = df.copy()
        hpat_func = self.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = hpat_func(df2)
        pd.testing.assert_frame_equal(out, h_out)

    @skip_numba_jit
    def test_df_dropna_str1(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, 4.0, 1.0], 'B': ['aa', 'b', None, 'ccc']})
        hpat_func = self.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = hpat_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    @skip_numba_jit
    def test_df_drop1(self):
        def test_impl(df):
            return df.drop(columns=['A'])

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_df_drop_inplace2(self):
        # test droping after setting the column
        def test_impl(df):
            df2 = df[['A', 'B']]
            df2['D'] = np.ones(3)
            df2.drop(columns=['D'], inplace=True)
            return df2

        df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_drop_inplace1(self):
        def test_impl(df):
            df.drop('A', axis=1, inplace=True)
            return df

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        df2 = df.copy()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df2))

    def test_isin_df1(self):
        def test_impl(df, df2):
            return df.isin(df2)

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'C': np.arange(n)**2})
        df2.A[n // 2:] = n
        pd.testing.assert_frame_equal(hpat_func(df, df2), test_impl(df, df2))

    @unittest.skip("needs dict typing in Numba")
    def test_isin_dict1(self):
        def test_impl(df):
            vals = {'A': [2, 3, 4], 'C': [4, 5, 6]}
            return df.isin(vals)

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_isin_list1(self):
        def test_impl(df):
            vals = [2, 3, 4]
            return df.isin(vals)

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_append1(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=True)

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'C': np.arange(n)**2})
        df2.A[n // 2:] = n
        pd.testing.assert_frame_equal(hpat_func(df, df2), test_impl(df, df2))

    @skip_numba_jit
    def test_append2(self):
        def test_impl(df, df2, df3):
            return df.append([df2, df3], ignore_index=True)

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2.A[n // 2:] = n
        df3 = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(
            hpat_func(df, df2, df3), test_impl(df, df2, df3))

    @skip_numba_jit
    def test_concat_columns1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2], axis=1)

        hpat_func = self.jit(test_impl)
        S1 = pd.Series([4, 5])
        S2 = pd.Series([6., 7.])
        # TODO: support int as column name
        pd.testing.assert_frame_equal(
            hpat_func(S1, S2),
            test_impl(S1, S2).rename(columns={0: '0', 1: '1'}))

    def test_var_rename(self):
        # tests df variable replacement in hiframes_untyped where inlining
        # can cause extra assignments and definition handling errors
        # TODO: inline freevar
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
            # TODO: df['C'] = [5,6,7]
            df['C'] = np.ones(3)
            return inner_get_column(df)

        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl(), check_names=False)

    @unittest.skip("Implement getting columns attribute")
    def test_dataframe_columns_attribute(self):
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
            return df.columns

        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    @unittest.skip("Implement getting columns attribute")
    def test_dataframe_columns_iterator(self):
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
            return [column for column in df.columns]

        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    @unittest.skip("Implement set_index for DataFrame")
    def test_dataframe_set_index(self):
        def test_impl():
            df = pd.DataFrame({'month': [1, 4, 7, 10],
                               'year': [2012, 2014, 2013, 2014],
                               'sale': [55, 40, 84, 31]})
            return df.set_index('month')

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip("Implement sort_index for DataFrame")
    def test_dataframe_sort_index(self):
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=[100, 29, 234, 1, 150])
            return df.sort_index()

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip("Implement iterrows for DataFrame")
    def test_dataframe_iterrows(self):
        def test_impl(df):
            return [row for _, row in df.iterrows()]

        df = pd.DataFrame({'A': [1, 2, 3], 'B': [0.2, 0.5, 0.001], 'C': ['a', 'bb', 'ccc']})
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @unittest.skip("Support parameter axis=1")
    def test_dataframe_axis_param(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            return df.sum(axis=1)

        n = 100
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_dataframe_head(self):
        def test_impl(df):
            return df.head()
        sdc_func = sdc.jit(test_impl)
        df = pd.DataFrame({"FLOAT": test_global_input_data_float64[0][:5],
                           "DATATIME": test_datatime,
                           "INT": test_global_input_data_int64[:5],
                           "STRING": ['a', 'dd', 'c', '12', 'ddf']})
        pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    def test_dataframe_head1(self):
        def test_impl(df, n):
            return df.head(n)
        sdc_func = sdc.jit(test_impl)
        df = pd.DataFrame({"FLOAT": test_global_input_data_float64[0][:5],
                           "DATATIME": test_datatime,
                           "INT": test_global_input_data_int64[:5],
                           "STRING": ['a', 'dd', 'c', '12', 'ddf']})
        for n in [-1, 0, 2, 5]:
            pd.testing.assert_frame_equal(sdc_func(df, n), test_impl(df, n))

    @unittest.skip('Dataframe.index not support')
    def test_dataframe_head1_index(self):
        def test_impl(df, n):
            return df.head(n)
        sdc_func = sdc.jit(test_impl)
        df = pd.DataFrame({"FLOAT": test_global_input_data_float64[0][:5],
                           "DATATIME": test_datatime,
                           "INT": test_global_input_data_int64[:5],
                           "STRING": ['a', 'dd', 'c', '12', 'ddf']},
                           index=[32, 3, 6, 17, 23])
        for n in [-1, 0, 2, 5]:
            pd.testing.assert_frame_equal(sdc_func(df, n), test_impl(df, n))

    def test_dataframe_head2(self):
        def test_impl(df, n):
            return df.head(n)
        sdc_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [12, 4, 5, 1, 6, 8],
                           "B": [5, 2, 54, 3, 6, 4],
                           "C": [20, 16, 3, 8, 2, 3],
                           "D": [14, 3, 2, 6, 4, 5]})
        for n in [-1, 0, 2, 5]:
            pd.testing.assert_frame_equal(sdc_func(df, n), test_impl(df, n))

if __name__ == "__main__":
    unittest.main()

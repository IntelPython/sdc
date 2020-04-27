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
import random
import string
import unittest
from itertools import permutations, product
from numba import types
from numba.config import IS_32BITS
from numba.special import literal_unroll
from numba.errors import TypingError
from pandas.core.indexing import IndexingError

import sdc
from sdc.datatypes.common_functions import SDCLimitation
from sdc.tests.gen_test_data import ParquetGenerator
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (check_numba_version,
                                  count_array_OneDs,
                                  count_array_REPs,
                                  count_parfor_OneDs,
                                  count_parfor_REPs,
                                  dist_IR_contains,
                                  gen_df, gen_df_int_cols,
                                  get_start_end,
                                  skip_numba_jit,
                                  skip_sdc_jit,
                                  test_global_input_data_float64,
                                  test_global_input_data_unicode_kind4)


@sdc.jit
def inner_get_column(df):
    # df2 = df[['A', 'C']]
    # df2['D'] = np.ones(3)
    return df.A


COL_IND = 0


class TestDataFrame(TestCase):

    # TODO: Data generator for DataFrames

    @unittest.skip('New DataFrame structure: implement boxing')
    def test_create1(self):
        def test_impl(A, B):
            df = pd.DataFrame({'A': A, 'B': B})
            return df

        hpat_func = self.jit(test_impl)
        n = 11
        A = np.ones(n)
        B = np.random.ranf(n)
        pd.testing.assert_frame_equal(hpat_func(A, B), test_impl(A, B))

    @unittest.skip('New DataFrame structure: implement getitem')
    def test_create2(self):
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3]})
            return (df.A == 1).sum()
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('New DataFrame structure: implement getitem')
    def test_create3(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)})
            return (df.A == 2).sum()
        hpat_func = self.jit(test_impl)

        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    @unittest.skip('New DataFrame structure: implement getitem')
    def test_create_str(self):
        def test_impl():
            df = pd.DataFrame({'A': ['a', 'b', 'c']})
            return (df.A == 'a').sum()
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('New DataFrame structure: implement boxing')
    def test_create_with_series1(self):
        def test_impl(n):
            A = pd.Series(np.ones(n, dtype=np.int64))
            B = pd.Series(np.zeros(n, dtype=np.float64))
            df = pd.DataFrame({'A': A, 'B': B})
            return df

        hpat_func = sdc.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('New DataFrame structure: implement getitem')
    def test_create_with_series2(self):
        # test creating dataframe from passed series
        def test_impl(A):
            df = pd.DataFrame({'A': A})
            return (df.A == 2).sum()
        hpat_func = self.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement boxing')
    def test_create_string_index(self):
        def test_impl(a):
            data = {'A': ['a', 'b'], 'B': [2, 3]}
            df = pd.DataFrame(data=data, index=['A', 'B'])
            return df

        hpat_func = sdc.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(True), test_impl(True))

    @unittest.skip('New DataFrame structure: implement boxing')
    def test_create_cond1(self):
        def test_impl(A, B, c):
            if c:
                df = pd.DataFrame({'A': A})
            else:
                df = pd.DataFrame({'A': B})
            return df

        hpat_func = self.jit(test_impl)
        n = 11
        A = np.ones(n)
        B = np.arange(n) + 1.0
        c = 0
        pd.testing.assert_frame_equal(hpat_func(A, B, c), test_impl(A, B, c))
        c = 2
        pd.testing.assert_frame_equal(hpat_func(A, B, c), test_impl(A, B, c))

    @unittest.skip('Implement feature to create DataFrame without column names')
    def test_create_without_column_names(self):
        def test_impl():
            df = pd.DataFrame([100, 200, 300, 400, 200, 100])
            return df

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('New DataFrame structure: implement unboxing + getitem')
    def test_pass_df1(self):
        def test_impl(df):
            return (df.A == 2).sum()
        hpat_func = self.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing + getitem')
    def test_pass_df_str(self):
        def test_impl(df):
            return (df.A == 'a').sum()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame({'A': ['a', 'b', 'c']})
        self.assertEqual(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_unbox1(self):
        def test_impl(df):
            return df

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

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

    @skip_numba_jit
    def test_box1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        hpat_func = self.jit(test_impl)
        n = 11
        do_check = False if platform.system() == 'Windows' and not IS_32BITS else True
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n), check_dtype=do_check)

    @unittest.skip('New DataFrame structure: implement boxing')
    def test_box2(self):
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'bb', 'ccc']})
            return df

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @skip_sdc_jit("pending df filter support")
    @unittest.skip('New DataFrame structure: implement unboxing + boxing')
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
    def test_column_list_getitem1(self):
        def test_impl(df):
            return df[['A', 'C']]

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame(
            {'A': np.arange(n), 'B': np.ones(n), 'C': np.random.ranf(n)})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement getitem')
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

    @skip_numba_jit
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

    @skip_numba_jit
    def test_iloc5(self):
        # test iloc with global value
        def test_impl(df):
            return df.iloc[:, COL_IND].values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_loc1(self):
        def test_impl(df):
            return df.loc[:, 'B'].values

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_iat1(self):
        def test_impl(n):
            df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
            return df.iat[3, 1]
        hpat_func = self.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    @skip_numba_jit
    def test_iat2(self):
        def test_impl(df):
            return df.iat[3, 1]
        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        self.assertEqual(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_iat3(self):
        def test_impl(df, n):
            return df.iat[n - 1, 1]
        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        self.assertEqual(hpat_func(df, n), test_impl(df, n))

    @skip_numba_jit
    def test_iat_set1(self):
        def test_impl(df, n):
            df.iat[n - 1, 1] = n**2
            return df.A  # return the column to check column aliasing
        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        df2 = df.copy()
        pd.testing.assert_series_equal(hpat_func(df, n), test_impl(df2, n))

    @skip_numba_jit
    def test_iat_set2(self):
        def test_impl(df, n):
            df.iat[n - 1, 1] = n**2
            return df  # check df aliasing/boxing
        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        df2 = df.copy()
        pd.testing.assert_frame_equal(hpat_func(df, n), test_impl(df2, n))

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    def _test_df_set_column(self, all_data, key, value):
        def gen_test_impl(value, do_jit=False):
            if isinstance(value, pd.Series):
                def test_impl(df, key, value):
                    if do_jit == True:  # noqa
                        return df._set_column(key, value.values)
                    else:
                        df[key] = value.values
            else:
                def test_impl(df, key, value):
                    if do_jit == True:  # noqa
                        return df._set_column(key, value)
                    else:
                        df[key] = value

            return test_impl

        test_impl = gen_test_impl(value)
        sdc_func = self.jit(gen_test_impl(value, do_jit=True))

        for data in all_data:
            with self.subTest(data=data):
                df1 = pd.DataFrame(data)
                df2 = df1.copy(deep=True)
                test_impl(df1, key, value)
                result_ref = df1  # in pandas setitem modifies original DF
                result_jit = sdc_func(df2, key, value)
                pd.testing.assert_frame_equal(result_jit, result_ref)

    def _test_df_set_column_exception_invalid_length(self, df, key, value):
        def test_impl(df, key, value):
            return df._set_column(key, value)

        sdc_func = self.jit(test_impl)

        with self.assertRaises(ValueError) as raises:
            sdc_func(df, key, value)
        msg = 'Length of values does not match length of index'
        self.assertIn(msg, str(raises.exception))

    def _test_df_set_column_exception_empty_columns(self, df, key, value):
        def test_impl(df, key, value):
            return df._set_column(key, value)

        sdc_func = self.jit(test_impl)

        with self.assertRaises(SDCLimitation) as raises:
            sdc_func(df, key, value)
        msg = 'Could not set item for DataFrame with empty columns'
        self.assertIn(msg, str(raises.exception))

    @unittest.skip('New DataFrame structure: implement setitem')
    def test_df_add_column(self):
        all_data = [{'A': [0, 1, 2], 'C': [0., np.nan, np.inf]}, {}]
        key, value = 'B', np.array([1., -1., 0.])

        self._test_df_set_column(all_data, key, value)

    @unittest.skip('New DataFrame structure: implement setitem')
    def test_df_add_column_str(self):
        all_data = [{'A': [0, 1, 2], 'C': [0., np.nan, np.inf]}, {}]
        key, value = 'B', pd.Series(test_global_input_data_unicode_kind4)

        self._test_df_set_column(all_data, key, value)

    @unittest.skip('New DataFrame structure: implement setitem')
    def test_df_add_column_exception_invalid_length(self):
        df = pd.DataFrame({'A': [0, 1, 2], 'C': [3., 4., 5.]})
        key, value = 'B', np.array([1., np.nan, -1., 0.])
        self._test_df_set_column_exception_invalid_length(df, key, value)

        df = pd.DataFrame({'A': []})
        self._test_df_set_column_exception_empty_columns(df, key, value)

    @unittest.skip('New DataFrame structure: implement setitem')
    def test_df_replace_column(self):
        all_data = [{'A': [0, 1, 2], 'C': [0., np.nan, np.inf]}]
        key, value = 'A', np.array([1., -1., 0.])

        self._test_df_set_column(all_data, key, value)

    @unittest.skip('New DataFrame structure: implement setitem')
    def test_df_replace_column_str(self):
        all_data = [{'A': [0, 1, 2], 'C': [0., np.nan, np.inf]}]
        key, value = 'A', pd.Series(test_global_input_data_unicode_kind4)

        self._test_df_set_column(all_data, key, value)

    @unittest.skip('New DataFrame structure: implement setitem')
    def test_df_replace_column_exception_invalid_length(self):
        df = pd.DataFrame({'A': [0, 1, 2], 'C': [3., 4., 5.]})
        key, value = 'A', np.array([1., np.nan, -1., 0.])
        self._test_df_set_column_exception_invalid_length(df, key, value)

        df = pd.DataFrame({'A': []})
        self._test_df_set_column_exception_empty_columns(df, key, value)

    def _test_df_values_unboxing(self, df):
        def test_impl(df):
            return df.values

        sdc_func = self.jit(test_impl)
        np.testing.assert_array_equal(sdc_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.values()')
    def test_df_values_unboxing(self):
        values_to_test = [[1, 2, 3, 4, 5],
                          [.1, .2, .3, .4, .5],
                          [np.nan, np.inf, .0, .1, -1.]]
        n = 5
        np.random.seed(0)
        A = np.ones(n)
        B = np.random.ranf(n)

        for values in values_to_test:
            with self.subTest(values=values):
                df = pd.DataFrame({'A': A, 'B': B, 'C D E': values})
                self._test_df_values_unboxing(df)

    @unittest.skip('New DataFrame structure: implement df.values()')
    def test_df_values(self):
        def test_impl(n, values):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n), 'C': values})
            return df.values

        sdc_func = self.jit(test_impl)
        n = 5
        values_to_test = [[1, 2, 3, 4, 5],
                          [.1, .2, .3, .4, .5],
                          [np.nan, np.inf, .0, .1, -1.]]

        for values in values_to_test:
            with self.subTest(values=values):
                np.testing.assert_array_equal(sdc_func(n, values), test_impl(n, values))

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

    def _test_df_index(self, df):
        def test_impl(df):
            return df.index

        sdc_func = self.jit(test_impl)
        np.testing.assert_array_equal(sdc_func(df), test_impl(df))

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_index_attribute(self):
        index_to_test = [[1, 2, 3, 4, 5],
                         [.1, .2, .3, .4, .5],
                         ['a', 'b', 'c', 'd', 'e']]
        n = 5
        np.random.seed(0)
        A = np.ones(n)
        B = np.random.ranf(n)

        for index in index_to_test:
            with self.subTest(index=index):
                df = pd.DataFrame({'A': A, 'B': B}, index=index)
                self._test_df_index(df)

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_index_attribute_empty(self):
        n = 5
        np.random.seed(0)
        A = np.ones(n)
        B = np.random.ranf(n)
        df = pd.DataFrame({'A': A, 'B': B})

        self._test_df_index(df)

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_index_attribute_empty_df(self):
        df = pd.DataFrame()
        self._test_df_index(df)

    @skip_sdc_jit
    @skip_numba_jit
    def test_df_apply(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A + r.B, axis=1)
            return df.B.sum()

        n = 121
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    @skip_sdc_jit
    @skip_numba_jit
    def test_df_apply_branch(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A < 10 and r.B > 20, axis=1)
            return df.B.sum()

        n = 121
        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    @skip_numba_jit
    @skip_sdc_jit('Not implemented in sequential transport layer')
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

    @skip_numba_jit
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

    @unittest.skip('New DataFrame structure: implement df.isna()')
    def test_df_isna(self):
        def test_impl(df):
            return df.isna()

        sdc_func = sdc.jit(test_impl)
        indexes = [[3, 4, 2, 6, 1], ['a', 'b', 'c', 'd', 'e'], None]

        for idx in indexes:
            df = pd.DataFrame({"A": [3.2, np.nan, 7.0, 3.3, np.nan],
                               "B": [3, 4, 1, 0, 222],
                               "C": [True, True, False, False, True],
                               "D": ['a', 'dd', 'c', '12', None]}, index=idx)
            with self.subTest(index=idx):
                pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    @unittest.skip('DF with column named "bool" Segmentation fault')
    def test_df_bool(self):
        def test_impl(df):
            return df.isna()

        sdc_func = sdc.jit(test_impl)
        df = pd.DataFrame({"bool": [True, True, False, False, True]}, index=None)
        pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
    @unittest.skipIf(platform.system() == 'Windows', "Attribute 'dtype' are different int64 and int32")
    def test_df_head1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.head(3)

        hpat_func = self.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('New DataFrame structure: implement df.head()')
    def test_df_head_unbox(self):
        def test_impl(df, n):
            return df.head(n)
        sdc_func = sdc.jit(test_impl)
        for n in [-3, 0, 3, 5, None]:
            for idx in [[3, 4, 2, 6, 1], None]:
                df = pd.DataFrame({"float": [3.2, 4.4, 7.0, 3.3, 1.0],
                                   "int": [3, 4, 1, 0, 222],
                                   "string": ['a', 'dd', 'c', '12', 'ddf']}, index=idx)
                with self.subTest(n=n, index=idx):
                    pd.testing.assert_frame_equal(sdc_func(df, n), test_impl(df, n))

    @unittest.skip('New DataFrame structure: implement df.iloc')
    def test_df_iloc_slice(self):
        def test_impl(df, n, k):
            return df.iloc[n:k]
        sdc_func = sdc.jit(test_impl)
        cases_idx = [[3, 4, 2, 6, 1], None]
        cases_n = [-10, 0, 8, None]
        for idx in cases_idx:
            df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                               "B": [5.5, np.nan, 3, 0, 7.7],
                               "C": [3, 4, 1, 0, 222]}, index=idx)
            for n, k in product(cases_n, cases_n[::-1]):
                with self.subTest(index=idx, n=n, k=k):
                    pd.testing.assert_frame_equal(sdc_func(df, n, k), test_impl(df, n, k))

    @unittest.skip('New DataFrame structure: implement df.iloc')
    def test_df_iloc_values(self):
        def test_impl(df, n):
            return df.iloc[n, 1]
        sdc_func = sdc.jit(test_impl)
        cases_idx = [[3, 4, 2, 6, 1], None]
        cases_n = [1, 0, 2]
        for idx in cases_idx:
            df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                               "B": [5.5, np.nan, 3, 0, 7.7],
                               "C": [3, 4, 1, 0, 222]}, index=idx)
            for n in cases_n:
                with self.subTest(index=idx, n=n):
                    if not (np.isnan(sdc_func(df, n)) and np.isnan(test_impl(df, n))):
                        self.assertEqual(sdc_func(df, n), test_impl(df, n))

    @unittest.skip('New DataFrame structure: implement df.iloc')
    def test_df_iloc_value_error(self):
        def int_impl(df):
            return df.iloc[11]

        def list_impl(df):
            return df.iloc[[7, 14]]

        def list_bool_impl(df):
            return df.iloc[[True, False]]

        msg1 = 'Index is out of bounds for axis'
        msg2 = 'Item wrong length'
        df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                           "B": [5.5, np.nan, 3, 0, 7.7],
                           "C": [3, 4, 1, 0, 222]})

        impls = [(int_impl, msg1), (list_impl, msg1), (list_bool_impl, msg2)]
        for impl, msg in impls:
            with self.subTest(case=impl, msg=msg):
                func = self.jit(impl)
                with self.assertRaises(IndexingError) as raises:
                    func(df)
                self.assertIn(msg, str(raises.exception))

    @unittest.skip('New DataFrame structure: implement df.iloc')
    def test_df_iloc_int(self):
        def test_impl(df, n):
            return df.iloc[n]
        sdc_func = sdc.jit(test_impl)
        cases_idx = [[3, 4, 2, 6, 1], None]
        cases_n = [0, 1, 2]
        for idx in cases_idx:
            df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                               "B": [5.5, np.nan, 3, 0, 7.7],
                               "C": [3, 4, 1, 0, 222]}, index=idx)
            for n in cases_n:
                with self.subTest(index=idx, n=n):
                    pd.testing.assert_series_equal(sdc_func(df, n), test_impl(df, n), check_names=False)

    @unittest.skip('New DataFrame structure: implement df.iloc')
    def test_df_iloc_list(self):
        def test_impl(df, n):
            return df.iloc[n]
        sdc_func = sdc.jit(test_impl)
        cases_idx = [[3, 4, 2, 6, 1], None]
        cases_n = [[0, 1], [2, 0]]
        for idx in cases_idx:
            df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                               "B": [5.5, np.nan, 3, 0, 7.7],
                               "C": [3, 4, 1, 0, 222]}, index=idx)
            for n in cases_n:
                with self.subTest(index=idx, n=n):
                    pd.testing.assert_frame_equal(sdc_func(df, n), test_impl(df, n))

    @unittest.skip('New DataFrame structure: implement df.iloc')
    def test_df_iloc_list_bool(self):
        def test_impl(df, n):
            return df.iloc[n]
        sdc_func = sdc.jit(test_impl)
        cases_idx = [[3, 4, 2, 6, 1], None]
        cases_n = [[True, False, True, False, True]]
        for idx in cases_idx:
            df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                               "B": [5.5, np.nan, 3, 0, 7.7],
                               "C": [3, 4, 1, 0, 222]}, index=idx)
            for n in cases_n:
                with self.subTest(index=idx, n=n):
                    pd.testing.assert_frame_equal(sdc_func(df, n), test_impl(df, n))

    @unittest.skip('New DataFrame structure: implement df.iat')
    def test_df_iat(self):
        def test_impl(df):
            return df.iat[0, 1]
        sdc_func = sdc.jit(test_impl)
        idx = [3, 4, 2, 6, 1]
        df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                           "B": [3, 4, 1, 0, 222],
                           "C": ['a', 'dd', 'c', '12', 'ddf']}, index=idx)
        self.assertEqual(sdc_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.iat')
    def test_df_iat_value_error(self):
        def test_impl(df):
            return df.iat[1, 22]
        sdc_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                           "B": [3, 4, 1, 0, 222],
                           "C": ['a', 'dd', 'c', '12', 'ddf']})

        with self.assertRaises(TypingError) as raises:
            sdc_func(df)
        msg = 'Index is out of bounds for axis'
        self.assertIn(msg, str(raises.exception))

    @unittest.skip('New DataFrame structure: implement df.loc')
    def test_df_loc(self):
        def test_impl(df):
            return df.loc[4]

        sdc_func = sdc.jit(test_impl)
        idx = [3, 4, 1, 4, 0]
        df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                           "B": [3, 4, 1, 0, 222],
                           "C": [3.1, 8.4, 7.1, 3.2, 1]}, index=idx)
        pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    @unittest.skip("SDC Dataframe.loc[] always return Dataframe")
    def test_df_loc_str(self):
        def test_impl(df):
            return df.loc['c']

        sdc_func = sdc.jit(test_impl)
        idx = ['a', 'b', 'c', 'с', 'e']
        df = pd.DataFrame({"A": ['3.2', '4.4', '7.0', '3.3', '1.0'],
                           "B": ['3', '4', '1', '0', '222'],
                           "C": ['3.1', '8.4', '7.1', '3.2', '1']}, index=idx)
        pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    @unittest.skip("SDC Dataframe.loc[] always return Dataframe")
    def test_df_loc_no_idx(self):
        def test_impl(df):
            return df.loc[2]

        sdc_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [3.2, 4.4, 7.0, 3.3, 1.0],
                           "B": [3, 4, 1, 0, 222],
                           "C": [3.1, 8.4, 7.1, 3.2, 1]})
        pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.head()')
    def test_df_head(self):
        def get_func(n):
            def impl(a):
                return a.head(n)

            return impl

        cases_n = [-3, 0, 3, 5, None]
        cases_index = [[3, 4, 2, 6, 1], None]
        for n in cases_n:
            for idx in cases_index:
                ref_impl = get_func(n)
                sdc_impl = get_func(n)
                sdc_func = self.jit(sdc_impl)
                with self.subTest(n=n, index=idx):
                    df = pd.DataFrame(
                        {"float": [3.2, 4.4, 7.0, 3.3, 1.0],
                         "int": [3, 4, 1, 0, 222],
                         "string": ['a', 'dd', 'c', '12', 'ddf']},
                        index=[3, 4, 2, 6, 1]
                    )
                    pd.testing.assert_frame_equal(sdc_func(df), ref_impl(df))

    @unittest.skip('New DataFrame structure: implement df.copy()')
    def test_df_copy(self):
        def test_impl(df, deep):
            return df.copy(deep=deep)

        sdc_func = sdc.jit(test_impl)
        indexes = [[3, 4, 2, 6, 1], ['a', 'b', 'c', 'd', 'e'], None]
        cases_deep = [None, True, False]

        for idx in indexes:
            df = pd.DataFrame({"A": [3.2, np.nan, 7.0, 3.3, np.nan],
                               "B": [3, 4, 1, 0, 222],
                               "C": [True, True, False, False, True],
                               "D": ['a', 'dd', 'c', '12', None]}, index=idx)
            for deep in cases_deep:
                with self.subTest(index=idx, deep=deep):
                    pd.testing.assert_frame_equal(sdc_func(df, deep), test_impl(df, deep))

    @unittest.skip('New DataFrame structure: implement df.pct_change()')
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
            df = pd.DataFrame({'A': 2 ** np.arange(n), 'B D': np.arange(n) + 1.0})
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

    @skip_numba_jit
    def test_df_fillna1(self):
        def test_impl(df):
            return df.fillna(5.0)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_df_fillna_str1(self):
        def test_impl(df):
            return df.fillna("dd")

        df = pd.DataFrame({'A': ['aa', 'b', None, 'ccc']})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_df_fillna_inplace1(self):
        def test_impl(A):
            A.fillna(11.0, inplace=True)
            return A

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        df2 = df.copy()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df2))

    def test_df_reset_index_drop(self):
        def test_impl(df, drop):
            return df.reset_index(drop=drop)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': np.arange(4.0)})
        hpat_func = self.jit(test_impl)

        for drop in [True, False]:
            with self.subTest(drop=drop):
                with self.assertRaises(Exception) as raises:
                    hpat_func(df, drop)
                msg = 'only work with Boolean literals drop'
                self.assertIn(msg.format(types.bool_), str(raises.exception))

    @unittest.skip('New DataFrame structure: implement df.reset_index()')
    def test_df_reset_index_drop_false_index_int(self):
        def test_impl(df):
            return df.reset_index(drop=False)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0],
                           'B': np.arange(4.0)}, index=[5, 8, 4, 6])
        hpat_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.reset_index()')
    def test_df_reset_index_drop_true_index_int(self):
        def test_impl(df):
            return df.reset_index(drop=True)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0],
                           'B': np.arange(4.0)}, index=[5, 8, 4, 6])
        hpat_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.reset_index()')
    def test_df_reset_index_drop_default_index_int(self):
        def test_impl(df):
            return df.reset_index()

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0],
                           'B': np.arange(4.0)}, index=[5, 8, 4, 6])
        hpat_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_df_reset_index_empty_df(self):
        def test_impl(df):
            return df.reset_index()

        df = pd.DataFrame({})
        hpat_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
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

    @unittest.skip('New DataFrame structure: implement df.drop()')
    def test_df_drop_one_column_unboxing(self):
        def test_impl(df):
            return df.drop(columns='C D')

        index_to_test = [[1, 2, 3, 4],
                         [.1, .2, .3, .4],
                         None,
                         ['a', 'b', 'c', 'd']]

        sdc_func = self.jit(test_impl)

        for index in index_to_test:
            with self.subTest(index=index):
                df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7], 'C D': [1.0, 2.0, np.nan, 1.0]},
                                  index=index)
                pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.drop()')
    def test_df_drop_one_column(self):
        def test_impl(index):
            df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7], 'C': [1.0, 2.0, np.nan, 1.0]},
                              index=index)
            return df.drop(columns='A')

        index_to_test = [[1, 2, 3, 4],
                         [.1, .2, .3, .4],
                         ['a', 'b', 'c', 'd']]

        sdc_func = self.jit(test_impl)

        for index in index_to_test:
            with self.subTest(index=index):
                pd.testing.assert_frame_equal(sdc_func(index), test_impl(index))

    @unittest.skip('New DataFrame structure: implement df.drop()')
    def test_df_drop_tuple_column_unboxing(self):
        def gen_test_impl(do_jit=False):
            def test_impl(df):
                if do_jit == True:  # noqa
                    return df.drop(columns=('A', 'C'))
                else:
                    return df.drop(columns=['A', 'C'])

            return test_impl

        index_to_test = [[1, 2, 3, 4],
                         [.1, .2, .3, .4],
                         None,
                         ['a', 'b', 'c', 'd']]

        test_impl = gen_test_impl()
        sdc_func = self.jit(gen_test_impl(do_jit=True))

        for index in index_to_test:
            with self.subTest(index=index):
                df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7], 'C': [1.0, 2.0, np.nan, 1.0]},
                                  index=index)
                pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.drop()')
    def test_df_drop_tuple_column(self):
        def gen_test_impl(do_jit=False):
            def test_impl(index):
                df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7], 'C': [1.0, 2.0, np.nan, 1.0]},
                                  index=index)
                if do_jit == True:  # noqa
                    return df.drop(columns=('A', 'C'))
                else:
                    return df.drop(columns=['A', 'C'])

            return test_impl

        index_to_test = [[1, 2, 3, 4],
                         [.1, .2, .3, .4],
                         ['a', 'b', 'c', 'd']]

        test_impl = gen_test_impl()
        sdc_func = self.jit(gen_test_impl(do_jit=True))

        for index in index_to_test:
            with self.subTest(index=index):
                pd.testing.assert_frame_equal(sdc_func(index), test_impl(index))

    @unittest.skip("ValueError when return empty dataframe")
    def test_df_drop_tuple_columns_all(self):
        def gen_test_impl(do_jit=False):
            def test_impl(df):
                if do_jit == True:  # noqa
                    return df.drop(columns=('A', 'B', 'C'))
                else:
                    return df.drop(columns=['A', 'B', 'C'])

            return test_impl

        index_to_test = [[1, 2, 3, 4],
                         [.1, .2, .3, .4],
                         None,
                         ['a', 'b', 'c', 'd']]

        test_impl = gen_test_impl()
        sdc_func = self.jit(gen_test_impl(do_jit=True))

        for index in index_to_test:
            with self.subTest(index=index):
                df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7], 'C': [1.0, 2.0, np.nan, 1.0]},
                                  index=index)

                pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.drop()')
    def test_df_drop_by_column_errors_ignore(self):
        def test_impl(df):
            return df.drop(columns='M', errors='ignore')

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7], 'C': [1.0, 2.0, np.nan, 1.0]})
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

    @skip_numba_jit
    def test_df_drop_inplace1(self):
        def test_impl(df):
            df.drop('A', axis=1, inplace=True)
            return df

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        df2 = df.copy()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df2))

    def _test_df_getitem_str_literal_idx(self, df):
        def test_impl(df):
            return df['A']

        sdc_func = self.jit(test_impl)
        pd.testing.assert_series_equal(sdc_func(df), test_impl(df))

    def _test_df_getitem_unicode_idx(self, df, idx):
        def test_impl(df, idx):
            return df[idx]

        sdc_func = self.jit(test_impl)
        pd.testing.assert_series_equal(sdc_func(df, idx), test_impl(df, idx))

    def _test_df_getitem_slice_idx(self, df):
        def test_impl(df):
            return df[1:3]

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    def _test_df_getitem_unbox_slice_idx(self, df, start, end):
        def test_impl(df, start, end):
            return df[start:end]

        sdc_func = self.jit(test_impl)
        jit_result = sdc_func(df, start, end)
        ref_result = test_impl(df, start, end)
        pd.testing.assert_frame_equal(jit_result, ref_result)

    def _test_df_getitem_tuple_idx(self, df):
        def gen_test_impl(do_jit=False):
            def test_impl(df):
                if do_jit == True:  # noqa
                    return df[('A', 'C')]
                else:
                    return df[['A', 'C']]

            return test_impl

        test_impl = gen_test_impl()
        sdc_func = self.jit(gen_test_impl(do_jit=True))

        pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    def _test_df_getitem_bool_series_idx(self, df):
        def test_impl(df):
            return df[df['A'] == -1.]

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(df), test_impl(df))

    def _test_df_getitem_bool_series_even_idx(self, df):
        def test_impl(df, series):
            return df[series]

        s = pd.Series([False, True] * 5)

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(df, s), test_impl(df, s))

    def _test_df_getitem_bool_array_even_idx(self, df):
        def test_impl(df, arr):
            return df[arr]

        arr = np.array([i % 2 for i in range(len(df))], dtype=np.bool_)

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(df, arr), test_impl(df, arr))

    @skip_sdc_jit('DF.getitem unsupported exceptions')
    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_str_literal_idx_exception_key_error(self):
        def test_impl(df):
            return df['ABC']

        sdc_func = self.jit(test_impl)

        for df in [gen_df(test_global_input_data_float64), pd.DataFrame()]:
            with self.subTest(df=df):
                with self.assertRaises(KeyError):
                    sdc_func(df)

    @skip_sdc_jit('DF.getitem unsupported exceptions')
    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_unicode_idx_exception_key_error(self):
        def test_impl(df, idx):
            return df[idx]

        sdc_func = self.jit(test_impl)

        for df in [gen_df(test_global_input_data_float64), pd.DataFrame()]:
            with self.subTest(df=df):
                with self.assertRaises(KeyError):
                    sdc_func(df, 'ABC')

    @skip_sdc_jit('DF.getitem unsupported exceptions')
    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_tuple_idx_exception_key_error(self):
        sdc_func = self.jit(lambda df: df[('A', 'Z')])

        for df in [gen_df(test_global_input_data_float64), pd.DataFrame()]:
            with self.subTest(df=df):
                with self.assertRaises(KeyError):
                    sdc_func(df)

    @skip_sdc_jit('DF.getitem unsupported exceptions')
    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_bool_array_idx_exception_value_error(self):
        sdc_func = self.jit(lambda df, arr: df[arr])

        for df in [gen_df(test_global_input_data_float64), pd.DataFrame()]:
            arr = np.array([i % 2 for i in range(len(df) + 1)], dtype=np.bool_)
            with self.subTest(df=df, arr=arr):
                with self.assertRaises(ValueError) as raises:
                    sdc_func(df, arr)
                self.assertIn('Item wrong length', str(raises.exception))

    @skip_sdc_jit('DF.getitem unsupported Series name')
    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_idx(self):
        dfs = [gen_df(test_global_input_data_float64),
               gen_df(test_global_input_data_float64, with_index=True),
               pd.DataFrame({'A': [], 'B': [], 'C': []})]
        for df in dfs:
            with self.subTest(df=df):
                self._test_df_getitem_str_literal_idx(df)
                self._test_df_getitem_unicode_idx(df, 'A')
                self._test_df_getitem_slice_idx(df)
                self._test_df_getitem_unbox_slice_idx(df, 1, 3)
                self._test_df_getitem_tuple_idx(df)
                self._test_df_getitem_bool_series_idx(df)

    @skip_sdc_jit('DF.getitem unsupported Series name')
    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_idx_no_index(self):
        dfs = [gen_df(test_global_input_data_float64), pd.DataFrame({'A': []})]
        for df in dfs:
            with self.subTest(df=df):
                self._test_df_getitem_bool_series_even_idx(df)
                self._test_df_getitem_bool_array_even_idx(df)

    @skip_sdc_jit('DF.getitem unsupported Series name')
    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_idx_multiple_types(self):
        int_data = [-1, 1, 0]
        float_data = [0.1, 0., -0.1]
        str_data = ['ascii', '12345', '1234567890']
        for a, b, c in permutations([int_data, float_data, str_data], 3):
            df = pd.DataFrame({'A': a, 'B': b, 'C': c})
            with self.subTest(df=df):
                self._test_df_getitem_str_literal_idx(df)
                self._test_df_getitem_unicode_idx(df, 'A')
                self._test_df_getitem_slice_idx(df)
                self._test_df_getitem_unbox_slice_idx(df, 1, 3)
                self._test_df_getitem_tuple_idx(df)
                self._test_df_getitem_bool_series_even_idx(df)
                self._test_df_getitem_bool_array_even_idx(df)

    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_bool_series_even_idx_with_index(self):
        df = gen_df(test_global_input_data_float64, with_index=True)
        self._test_df_getitem_bool_series_even_idx(df)

    @unittest.skip('DF.getitem unsupported integer columns')
    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_int_literal_idx(self):
        def test_impl(df):
            return df[1]

        sdc_func = self.jit(test_impl)
        df = gen_df_int_cols(test_global_input_data_float64)

        pd.testing.assert_series_equal(sdc_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.getitem')
    def test_df_getitem_attr(self):
        def test_impl(df):
            return df.A

        sdc_func = self.jit(test_impl)
        df = gen_df(test_global_input_data_float64)

        pd.testing.assert_series_equal(sdc_func(df), test_impl(df))

    @skip_numba_jit
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

    @skip_numba_jit
    def test_isin_list1(self):
        def test_impl(df):
            vals = [2, 3, 4]
            return df.isin(vals)

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement df.append()')
    def test_append_df_same_cols_no_index(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=True)

        sdc_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2.A[n // 2:] = n
        pd.testing.assert_frame_equal(sdc_func(df, df2), test_impl(df, df2))

    @unittest.skip('New DataFrame structure: implement df.append()')
    def test_append_df_same_cols_index_default(self):
        def test_impl(df, df2):
            return df.append(df2)

        sdc_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n) ** 2}, index=np.arange(n) ** 4)
        df2 = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n) ** 2}, index=np.arange(n) ** 8)
        df2.A[n // 2:] = n

        pd.testing.assert_frame_equal(sdc_func(df, df2), test_impl(df, df2))

    @unittest.skip('New DataFrame structure: implement df.append()')
    def test_append_df_diff_cols_index_ignore_false(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=False)

        sdc_func = self.jit(test_impl)
        n1 = 11
        n2 = n1 * 2
        df = pd.DataFrame({'A': np.arange(n1), 'B': np.arange(n1)**2}, index=np.arange(n1) ** 4)
        df2 = pd.DataFrame({'C': np.arange(n2), 'D': np.arange(n2)**2, 'E S D': np.arange(n2) + 100},
                           index=np.arange(n2) ** 8)

        pd.testing.assert_frame_equal(sdc_func(df, df2), test_impl(df, df2))

    @unittest.skip('New DataFrame structure: implement df.append()')
    def test_append_df_diff_cols_index_ignore_index(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=True)

        sdc_func = self.jit(test_impl)
        n1 = 11
        n2 = n1 * 2
        df = pd.DataFrame({'A': np.arange(n1), 'B': np.arange(n1)**2}, index=np.arange(n1) ** 4)
        df2 = pd.DataFrame({'C': np.arange(n2), 'D': np.arange(n2)**2, 'E S D': np.arange(n2) + 100},
                           index=np.arange(n2) ** 8)

        pd.testing.assert_frame_equal(sdc_func(df, df2), test_impl(df, df2))

    @unittest.skip('New DataFrame structure: implement df.append()')
    def test_append_df_diff_cols_no_index(self):
        def test_impl(df, df2):
            return df.append(df2)

        sdc_func = self.jit(test_impl)
        n1 = 4
        n2 = n1 * 2
        df = pd.DataFrame({'A': np.arange(n1), 'B': np.arange(n1)**2})
        df2 = pd.DataFrame({'C': np.arange(n2), 'D': np.arange(n2)**2, 'E S D': np.arange(n2) + 100})

        pd.testing.assert_frame_equal(sdc_func(df, df2), test_impl(df, df2))

    @unittest.skip('New DataFrame structure: implement df.append()')
    def test_append_df_cross_cols_no_index(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=True)

        sdc_func = self.jit(test_impl)
        n1 = 11
        n2 = n1 * 2
        df = pd.DataFrame({'A': np.arange(n1), 'B': np.arange(n1)**2})
        df2 = pd.DataFrame({'A': np.arange(n2), 'D': np.arange(n2)**2, 'E S D': np.arange(n2) + 100})

        pd.testing.assert_frame_equal(sdc_func(df, df2), test_impl(df, df2))

    @unittest.skip('New DataFrame structure: implement df.append()')
    def test_append_df_exception_incomparable_index_type(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=False)

        sdc_func = self.jit(test_impl)

        n1 = 2
        n2 = n1 * 2
        df = pd.DataFrame({'A': np.arange(n1), 'B': np.arange(n1) ** 2}, index=['a', 'b'])
        df2 = pd.DataFrame({'A': np.arange(n2), 'D': np.arange(n2) ** 2, 'E S D': np.arange(n2) + 100},
                           index=np.arange(n2))

        with self.assertRaises(SDCLimitation) as raises:
            sdc_func(df, df2)

        msg = "Indexes of dataframes are expected to have comparable (both Numeric or String) types " \
              "if parameter ignore_index is set to False."
        self.assertIn(msg, str(raises.exception))

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement df.append()')
    def test_append_df_diff_types_no_index(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=True)

        hpat_func = self.jit(test_impl)

        df = pd.DataFrame({'A': ['cat', 'dog', np.nan], 'B': [.2, .3, np.nan]})
        df2 = pd.DataFrame({'C': [5, 6, 7, 8]*64, 'D': ['a', 'b', np.nan, '']*64})

        pd.testing.assert_frame_equal(hpat_func(df, df2), test_impl(df, df2))

    @skip_numba_jit('Unsupported functionality df.append([df2, df3])')
    def test_append_no_index(self):
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

    @skip_numba_jit
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

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_min_dataframe_default(self):
        def test_impl(df):
            return df.min()

        sdc_func = sdc.jit(test_impl)
        df = pd.DataFrame({
            "A": [12, 4, 5, 44, 1],
            "B": [5.0, np.nan, 9, 2, -1],
            # unsupported
            # "C": ['a', 'aa', 'd', 'cc', None],
            # "D": [True, True, False, True, True]
        })
        pd.testing.assert_series_equal(sdc_func(df), test_impl(df))

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_median_default(self):
        def test_impl(df):
            return df.median()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_mean_default(self):
        def test_impl(df):
            return df.mean()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F H": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_std_default(self):
        def test_impl(df):
            return df.std()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F H": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_var_default(self):
        def test_impl(df):
            return df.var()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F H": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_max_default(self):
        def test_impl(df):
            return df.max()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F H": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_min_default(self):
        def test_impl(df):
            return df.min()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F H": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_sum_default(self):
        def test_impl(df):
            return df.sum()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F H": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_prod_default(self):
        def test_impl(df):
            return df.prod()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F H": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_count2_default(self):
        def test_impl(df):
            return df.count()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F H": [np.nan, np.nan, np.inf, np.nan]})
        pd.testing.assert_series_equal(hpat_func(df), test_impl(df))

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_pct_change(self):
        def test_impl(df):
            return df.pct_change()

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [14, 4, 5, 4, 1, 55],
                           "B": [5, 2, None, 3, 2, 32],
                           "C": [20, 20, 7, 21, 8, None],
                           "D": [14, None, 6, 2, 6, 4]})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_pct_change_with_parameters_limit_and_freq(self):
        def test_impl(df, limit, freq):
            return df.pct_change(limit=limit, freq=freq)

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [14, 4, 5, 4, 1, 55],
                           "B": [5, 2, None, 3, 2, 32],
                           "C": [20, 20, 7, 21, 8, None],
                           "D": [14, None, 6, 2, 6, 4]})
        pd.testing.assert_frame_equal(hpat_func(df, None, None), test_impl(df, None, None))

    @skip_sdc_jit
    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_pct_change_with_parametrs(self):
        def test_impl(df, periods, method):
            return df.pct_change(periods=periods, fill_method=method, limit=None, freq=None)

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({"A": [.2, .0, .6, .2],
                           "B": [.5, .6, .7, .8],
                           "C": [2, 0, 6, 2],
                           "D": [.2, .1, np.nan, .5],
                           "E": [-1, np.nan, 1, np.inf],
                           "F": [np.nan, np.nan, np.inf, np.nan]})
        all_periods = [0, 1, 2, 5, 10, -1, -2, -5]
        methods = [None, 'pad', 'ffill', 'backfill', 'bfill']
        for periods, method in product(all_periods, methods):
            with self.subTest(periods=periods, method=method):
                result_ref = test_impl(df, periods, method)
                result = hpat_func(df, periods, method)
                pd.testing.assert_frame_equal(result, result_ref)

    @unittest.skip('New DataFrame structure: implement getitem')
    def test_list_convert(self):
        def test_impl():
            df = pd.DataFrame({'one': np.array([-1, np.nan, 2.5]),
                               'two': ['foo', 'bar', 'baz'],
                               'three': [True, False, True]})
            return df.one.values, df.two.values, df.three.values
        hpat_func = self.jit(test_impl)

        one, two, three = hpat_func()
        self.assertTrue(isinstance(one, np.ndarray))
        self.assertTrue(isinstance(two, np.ndarray))
        self.assertTrue(isinstance(three, np.ndarray))

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_df_len(self):
        def test_impl(df):
            return len(df)

        hpat_func = self.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n) ** 2})
        self.assertEqual(hpat_func(df), test_impl(df))

    @unittest.skip("Literal unrol is broken by inline get_dataframe_data")
    def test_df_iterate_over_columns1(self):
        """ Verifies iteration over df columns using literal tuple of column indices. """
        from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
        from sdc.hiframes.api import get_nan_mask

        @self.jit
        def jitted_func():
            df = pd.DataFrame({
                        'A': ['a', 'b', None, 'a', '', None, 'b'],
                        'B': ['a', 'b', 'd', 'a', '', 'c', 'b'],
                        'C': [np.nan, 1, 2, 1, np.nan, 2, 1],
                        'D': [1, 2, 9, 5, 2, 1, 0]
            })

            # tuple of literals has to be created in a jitted function, otherwise
            # col_id won't be literal and unboxing in get_dataframe_data won't compile
            column_ids = (0, 1, 2, 3)
            res_nan_mask = np.zeros(len(df), dtype=np.bool_)
            for col_id in literal_unroll(column_ids):
                res_nan_mask += get_nan_mask(get_dataframe_data(df, col_id))
            return res_nan_mask

        # expected is a boolean mask of df rows that have None values
        expected = np.asarray([True, False, True, False, True, True, False])
        result = jitted_func()
        np.testing.assert_array_equal(result, expected)

    @unittest.skip('New DataFrame structure: implement getitem')
    def test_df_create_str_with_none(self):
        """ Verifies creation of a dataframe with a string column from a list of Optional values. """
        def test_impl():
            df = pd.DataFrame({
                        'A': ['a', 'b', None, 'a', '', None, 'b'],
                        'B': ['a', 'b', 'd', 'a', '', 'c', 'b'],
                        'C': [np.nan, 1, 2, 1, np.nan, 2, 1]
            })

            return df['A'].isna()
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    @unittest.skip('New DataFrame structure: implement unboxing')
    def test_df_iterate_over_columns2(self):
        """ Verifies iteration over unboxed df columns using literal unroll. """
        from sdc.hiframes.api import get_nan_mask

        @self.jit
        def jitted_func(df):
            res_nan_mask = np.zeros(len(df), dtype=np.bool_)
            for col in literal_unroll(df._data):
                res_nan_mask += get_nan_mask(col)
            return res_nan_mask

        df = pd.DataFrame({
                    'A': ['a', 'b', None, 'a', '', None, 'b'],
                    'B': ['a', 'b', 'd', 'a', '', 'c', 'b'],
                    'C': [np.nan, 1, 2, 1, np.nan, 2, 1],
                    'D': [1, 2, 9, 5, 2, 1, 0]
        })
        # expected is a boolean mask of df rows that have None values
        expected = np.asarray([True, False, True, False, True, True, False])
        result = jitted_func(df)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

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
import unittest
from itertools import product

import sdc
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_OneDs,
                                  count_array_REPs,
                                  count_parfor_OneDs,
                                  count_parfor_REPs,
                                  dist_IR_contains,
                                  get_start_end,
                                  skip_numba_jit,
                                  sdc_limitation)
from sdc.tests.test_series import gen_frand_array


_pivot_df1 = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                                 "bar", "bar", "bar", "bar"],
                           "B": ["one", "one", "one", "two", "two",
                                 "one", "one", "two", "two"],
                           "C": ["small", "large", "large", "small",
                                 "small", "large", "small", "small",
                                 "large"],
                           "D": [1, 2, 2, 6, 3, 4, 5, 6, 9]})

_default_df_numeric_data = {
                    'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                    'B': np.arange(11, dtype=np.intp),
                    'C': np.arange(11, dtype=np.float_),
                    'D': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                    'E': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
    }

class TestGroupBy(TestCase):

    @sdc_limitation
    def test_dataframe_groupby_index_name(self):
        """SDC indexes do not have names, so index created from a named Series looses it's name."""
        def test_impl(df):
            return df.groupby('A').min()
        hpat_func = self.jit(test_impl)

        n = 11
        df = pd.DataFrame({
                    'A': [2, 1, 1, 1, 2, 2, 1, 0, 3, 1, 3],
                    'B': np.arange(n, dtype=np.intp)
        })
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_frame_equal(result, result_ref)

    def test_dataframe_groupby_by_all_dtypes(self):
        def test_impl(df):
            return df.groupby('A').count()
        hpat_func = self.jit(test_impl)

        dtype_to_column_data = {
                'int': [2, 1, 1, 1, 2, 2, 1, 0, 3, 1, 3],
                'float': [2, 1, 1, 1, 2, 2, 1, 3, np.nan, 1, np.nan],
                'string': ['b', 'a', 'a', 'a', 'b', 'b', 'a', ' ', None, 'a', None]
        }
        df = pd.DataFrame(_default_df_numeric_data)
        for dtype, col_data in dtype_to_column_data.items():
            with self.subTest(by_dtype=dtype, by_data=col_data):
                df['A'] = col_data
                result = hpat_func(df)
                result_ref = test_impl(df)
                # TODO: implement index classes, as current indexes do not have names
                pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_sort(self):
        def test_impl(df, param):
            return df.groupby('A', sort=param).min()
        hpat_func = self.jit(test_impl)

        n, m = 1000, 20
        np.random.seed(0)
        df = pd.DataFrame({
                    'A': np.random.choice(np.arange(m), n),
                    'B': np.arange(n, dtype=np.intp),
                    'C': np.arange(n, dtype=np.float_),
                    'D': gen_frand_array(n, nancount=n // 2),
        })

        for value in [True, False]:
            with self.subTest(sort=value):
                result = hpat_func(df, value) if value else hpat_func(df, value).sort_index()
                result_ref = test_impl(df, value) if value else hpat_func(df, value).sort_index()
                # TODO: implement index classes, as current indexes do not have names
                pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_count(self):
        def test_impl(df):
            return df.groupby('A').count()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_count_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').count()

        sdc_impl = self.jit(test_impl)

        result_jit = sdc_impl()
        result_ref = test_impl()
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result_jit, result_ref, check_names=False)


    def test_dataframe_groupby_max(self):
        def test_impl(df):
            return df.groupby('A').max()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_max_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').max()

        sdc_impl = self.jit(test_impl)

        # TODO: implement index classes, as current indexes do not have names
        kwargs = {'check_names': False}
        if platform.system() == 'Windows':
            # Attribute "dtype" are different on windows int64 vs int32
            kwargs['check_dtype'] = False

        pd.testing.assert_frame_equal(sdc_impl(), test_impl(), **kwargs)

    def test_dataframe_groupby_min(self):
        def test_impl(df):
            return df.groupby('A').min()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_min_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').min()

        sdc_impl = self.jit(test_impl)

        # TODO: implement index classes, as current indexes do not have names
        kwargs = {'check_names': False}
        if platform.system() == 'Windows':
            # Attribute "dtype" are different on windows int64 vs int32
            kwargs['check_dtype'] = False

        pd.testing.assert_frame_equal(sdc_impl(), test_impl(), **kwargs)

    @unittest.expectedFailure  # FIXME_pandas#43292: pandas groupby.sum impl broken
    def test_dataframe_groupby_mean(self):
        def test_impl(df):
            return df.groupby('A').mean()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    @unittest.expectedFailure  # FIXME_pandas#43292: pandas groupby.sum impl broken
    def test_dataframe_groupby_mean_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').mean()

        sdc_impl = self.jit(test_impl)

        result_jit = sdc_impl()
        result_ref = test_impl()
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result_jit, result_ref, check_names=False)

    def test_dataframe_groupby_median(self):
        def test_impl(df):
            return df.groupby('A').median()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_median_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').median()

        sdc_impl = self.jit(test_impl)

        result_jit = sdc_impl()
        result_ref = test_impl()
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result_jit, result_ref, check_names=False)

    def test_dataframe_groupby_median_result_dtype(self):
        def test_impl(df):
            return df.groupby('A').median()
        hpat_func = self.jit(test_impl)

        n = 11
        df = pd.DataFrame({
                    'A': [2, 1, 1, 1, 2, 2, 1, 0, 3, 1, 3],
                    'B': np.arange(n, dtype=np.intp)
        })
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_prod(self):
        def test_impl(df):
            return df.groupby('A').prod()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_prod_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').prod()

        sdc_impl = self.jit(test_impl)

        # TODO: implement index classes, as current indexes do not have names
        kwargs = {'check_names': False}
        if platform.system() == 'Windows':
            # Attribute "dtype" are different on windows int64 vs int32
            kwargs['check_dtype'] = False

        pd.testing.assert_frame_equal(sdc_impl(), test_impl(), **kwargs)

    @skip_numba_jit("BUG: SDC impl of Series.sum returns float64 on as series of ints")
    def test_dataframe_groupby_sum(self):
        def test_impl(df):
            return df.groupby('A').sum()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    @unittest.expectedFailure  # FIXME_pandas#43292: pandas groupby.sum impl broken
    def test_dataframe_groupby_sum_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').sum()

        sdc_impl = self.jit(test_impl)

        # TODO: implement index classes, as current indexes do not have names
        # Attribute "dtype" are different int64 vs int32
        kwargs = {'check_names': False, 'check_dtype': False}
        pd.testing.assert_frame_equal(sdc_impl(), test_impl(), **kwargs)

    def test_dataframe_groupby_std(self):
        def test_impl(df):
            return df.groupby('A').std()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_std_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').std()

        sdc_impl = self.jit(test_impl)

        result_jit = sdc_impl()
        result_ref = test_impl()
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result_jit, result_ref, check_names=False)

    def test_dataframe_groupby_var(self):
        def test_impl(df):
            return df.groupby('A').var()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_var_no_unboxing(self):
        def test_impl():
            df = pd.DataFrame({
                'A': [2, 1, 2, 1, 2, 2, 1, 0, 3, 1, 3],
                'B': np.arange(11),
                'C': [np.nan, 2., -1.3, np.nan, 3.5, 0, 10, 0.42, np.nan, -2.5, 23],
                'D': [np.inf, 2., -1.3, -np.inf, 3.5, 0, 10, 0.42, np.nan, -2.5, 23]
            })
            return df.groupby('A').var()

        sdc_impl = self.jit(test_impl)

        result_jit = sdc_impl()
        result_ref = test_impl()
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result_jit, result_ref, check_names=False)

    @skip_numba_jit
    def test_agg_seq(self):
        def test_impl(df):
            A = df.groupby('A')['B'].agg(lambda x: x.max() - x.min())
            return A.values

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        # np.testing.assert_array_equal(hpat_func(df), test_impl(df))
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    @skip_numba_jit("BUG: SDC impl of Series.sum returns float64 on as series of ints")
    def test_agg_seq_sum(self):
        def test_impl(df):
            return df.groupby('A')['B'].sum()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_agg_seq_count(self):
        def test_impl(df):
            return df.groupby('A')['B'].count()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    def test_agg_seq_mean(self):
        def test_impl(df):
            return df.groupby('A')['B'].mean()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    def test_agg_seq_median(self):
        def test_impl(df):
            return df.groupby('A')['B'].median()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    def test_agg_seq_min(self):
        def test_impl(df):
            return df.groupby('A')['B'].min()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    @skip_numba_jit
    def test_agg_seq_min_date(self):
        def test_impl(df):
            df2 = df.groupby('A', as_index=False).min()
            return df2

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': pd.date_range('2019-1-3', '2019-1-9')})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_max(self):
        def test_impl(df):
            return df.groupby('A')['B'].max()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    @skip_numba_jit
    def test_agg_seq_as_index(self):
        def test_impl(df):
            df2 = df.groupby('A', as_index=False).mean()
            return df2.A.values

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_prod(self):
        def test_impl(df):
            return df.groupby('A')['B'].prod()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    def test_agg_seq_var(self):
        def test_impl(df):
            return df.groupby('A')['B'].var()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    def test_agg_seq_std(self):
        def test_impl(df):
            return df.groupby('A')['B'].std()

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        result = hpat_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    @skip_numba_jit
    def test_agg_multikey_seq(self):
        def test_impl(df):
            A = df.groupby(['A', 'C'])['B'].sum()
            return A.values

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7],
                           'C': [3, 5, 6, 5, 4, 4, 3]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

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
    def test_agg_seq_str(self):
        def test_impl(df):
            A = df.groupby('A')['B'].agg(lambda x: (x == 'aa').sum())
            return A.values

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': ['aa', 'b', 'b', 'b', 'aa', 'aa', 'b'],
                           'B': ['ccc', 'a', 'bb', 'aa', 'dd', 'ggg', 'rr']})
        # np.testing.assert_array_equal(hpat_func(df), test_impl(df))
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    @skip_numba_jit
    def test_agg_seq_count_str(self):
        def test_impl(df):
            A = df.groupby('A')['B'].count()
            return A.values

        hpat_func = self.jit(test_impl)
        df = pd.DataFrame({'A': ['aa', 'b', 'b', 'b', 'aa', 'aa', 'b'],
                           'B': ['ccc', 'a', 'bb', 'aa', 'dd', 'ggg', 'rr']})
        # np.testing.assert_array_equal(hpat_func(df), test_impl(df))
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    @skip_numba_jit
    def test_pivot(self):
        def test_impl(df):
            pt = df.pivot_table(index='A', columns='C', values='D', aggfunc='sum')
            return (pt.small.values, pt.large.values)

        hpat_func = self.jit(pivots={'pt': ['small', 'large']})(test_impl)
        self.assertEqual(
            set(hpat_func(_pivot_df1)[0]), set(test_impl(_pivot_df1)[0]))
        self.assertEqual(
            set(hpat_func(_pivot_df1)[1]), set(test_impl(_pivot_df1)[1]))

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
    def test_crosstab1(self):
        def test_impl(df):
            pt = pd.crosstab(df.A, df.C)
            return (pt.small.values, pt.large.values)

        hpat_func = self.jit(pivots={'pt': ['small', 'large']})(test_impl)
        self.assertEqual(
            set(hpat_func(_pivot_df1)[0]), set(test_impl(_pivot_df1)[0]))
        self.assertEqual(
            set(hpat_func(_pivot_df1)[1]), set(test_impl(_pivot_df1)[1]))

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

    @unittest.skip("Implement groupby(lambda) for DataFrame")
    def test_groupby_lambda(self):
        def test_impl(df):
            group = df.groupby(lambda x: x % 2 == 0)
            return group.count()

        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_dataframe_groupby_getitem_literal_tuple(self):
        def test_impl(df):
            return df.groupby('A')['B', 'C'].count()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_getitem_literal_str(self):
        def test_impl(df):
            return df.groupby('C')['B'].count()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        result = hpat_func(df)
        result_ref = test_impl(df)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_getitem_unicode_str(self):
        def test_impl(df, col_name):
            return df.groupby('A')[col_name].count()
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        col_name = 'C'
        # pandas returns groupby.generic.SeriesGroupBy object in this case, hence align result_ref
        result = hpat_func(df, col_name)
        result_ref = test_impl(df, col_name)
        # TODO: implement index classes, as current indexes do not have names
        pd.testing.assert_series_equal(result, result_ref, check_names=False)

    def test_dataframe_groupby_getitem_repeated(self):
        def test_impl(df):
            return df.groupby('A')['B', 'C']['D']
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(_default_df_numeric_data)
        with self.assertRaises(Exception) as context:
            test_impl(df)
        pandas_exception = context.exception

        self.assertRaises(type(pandas_exception), hpat_func, df)

    def test_series_groupby_by_array(self):
        def test_impl(A, data):
            return A.groupby(data).count()
        hpat_func = self.jit(test_impl)

        data_to_test = [
                    [True, False, False, True, False, False, True, False, True, True, False],
                    [2, 1, 1, 1, 2, 2, 1, 0, 3, 1, 3],
                    [2, 1, 1, 1, 2, 2, 1, 3, np.nan, 1, np.nan],
                    ['b', 'a', 'a', 'a', 'b', 'b', 'a', ' ', None, 'a', None]
        ]
        for series_data, arr_data in product(data_to_test, data_to_test):
            S = pd.Series(series_data)
            by_arr = np.asarray(arr_data)

            # arrays of dtype object cannot be jitted, so skip group by string data for now
            if by_arr.dtype.name == 'object':
                continue
            with self.subTest(series_data=series_data, by_arr=by_arr):
                result = hpat_func(S, by_arr)
                result_ref = test_impl(S, by_arr)
                pd.testing.assert_series_equal(result, result_ref)

    @unittest.skip("getiter for this type is not implemented yet")
    def test_series_groupby_iterator_int(self):
        def test_impl():
            A = pd.Series([13, 11, 21, 13, 13, 51, 42, 21])
            grouped = A.groupby(A)
            return [i for i in grouped]

        hpat_func = self.jit(test_impl)

        ref_result = test_impl()
        result = hpat_func()
        np.testing.assert_array_equal(result, ref_result)


if __name__ == "__main__":
    unittest.main()

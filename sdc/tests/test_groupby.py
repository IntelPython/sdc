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
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import numba
import sdc
from sdc.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                   count_parfor_OneDs, count_array_OneDs, dist_IR_contains,
                                   get_start_end,
                                   TestCase)


_pivot_df1 = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                                 "bar", "bar", "bar", "bar"],
                           "B": ["one", "one", "one", "two", "two",
                                 "one", "one", "two", "two"],
                           "C": ["small", "large", "large", "small",
                                 "small", "large", "small", "small",
                                 "large"],
                           "D": [1, 2, 2, 6, 3, 4, 5, 6, 9]})


class TestGroupBy(TestCase):
    def test_agg_seq(self):
        def test_impl(df):
            A = df.groupby('A')['B'].agg(lambda x: x.max() - x.min())
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        # np.testing.assert_array_equal(hpat_func(df), test_impl(df))
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_sum(self):
        def test_impl(df):
            A = df.groupby('A')['B'].sum()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_count(self):
        def test_impl(df):
            A = df.groupby('A')['B'].count()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_mean(self):
        def test_impl(df):
            A = df.groupby('A')['B'].mean()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_min(self):
        def test_impl(df):
            A = df.groupby('A')['B'].min()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_min_date(self):
        def test_impl(df):
            df2 = df.groupby('A', as_index=False).min()
            return df2

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': pd.date_range('2019-1-3', '2019-1-9')})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_max(self):
        def test_impl(df):
            A = df.groupby('A')['B'].max()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_all_col(self):
        def test_impl(df):
            df2 = df.groupby('A').mean()
            return df2.B.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_as_index(self):
        def test_impl(df):
            df2 = df.groupby('A', as_index=False).mean()
            return df2.A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_prod(self):
        def test_impl(df):
            A = df.groupby('A')['B'].prod()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_var(self):
        def test_impl(df):
            A = df.groupby('A')['B'].var()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_std(self):
        def test_impl(df):
            A = df.groupby('A')['B'].std()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_multiselect(self):
        def test_impl(df):
            df2 = df.groupby('A')['B', 'C'].sum()
            return df2.C.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7],
                           'C': [3, 5, 6, 5, 4, 4, 3]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_multikey_seq(self):
        def test_impl(df):
            A = df.groupby(['A', 'C'])['B'].sum()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7],
                           'C': [3, 5, 6, 5, 4, 4, 3]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_multikey_parallel(self):
        def test_impl(in_A, in_B, in_C):
            df = pd.DataFrame({'A': in_A, 'B': in_B, 'C': in_C})
            A = df.groupby(['A', 'C'])['B'].sum()
            return A.sum()

        hpat_func = sdc.jit(locals={'in_A:input': 'distributed',
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

    def test_agg_parallel(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].agg(lambda x: x.max() - x.min())
            return A.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_sum(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].sum()
            return A.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_count(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].count()
            return A.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_mean(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].mean()
            return A.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_min(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].min()
            return A.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_max(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].max()
            return A.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_var(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].var()
            return A.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_std(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].std()
            return A.sum()

        hpat_func = sdc.jit(test_impl)
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

        hpat_func = sdc.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_all_col(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            df2 = df.groupby('A').max()
            return df2.B.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_as_index(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            df2 = df.groupby('A', as_index=False).max()
            return df2.A.sum()

        hpat_func = sdc.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_muti_hiframes_node_filter_agg(self):
        def test_impl(df, cond):
            df2 = df[cond]
            c = df2.groupby('A')['B'].count()
            return df2.C, c

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7], 'C': [2, 3, -1, 1, 2, 3, -1]})
        cond = df.A > 1
        res = test_impl(df, cond)
        h_res = hpat_func(df, cond)
        self.assertEqual(set(res[1]), set(h_res[1]))
        np.testing.assert_array_equal(res[0], h_res[0])

    def test_agg_seq_str(self):
        def test_impl(df):
            A = df.groupby('A')['B'].agg(lambda x: (x == 'aa').sum())
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': ['aa', 'b', 'b', 'b', 'aa', 'aa', 'b'],
                           'B': ['ccc', 'a', 'bb', 'aa', 'dd', 'ggg', 'rr']})
        # np.testing.assert_array_equal(hpat_func(df), test_impl(df))
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_count_str(self):
        def test_impl(df):
            A = df.groupby('A')['B'].count()
            return A.values

        hpat_func = sdc.jit(test_impl)
        df = pd.DataFrame({'A': ['aa', 'b', 'b', 'b', 'aa', 'aa', 'b'],
                           'B': ['ccc', 'a', 'bb', 'aa', 'dd', 'ggg', 'rr']})
        # np.testing.assert_array_equal(hpat_func(df), test_impl(df))
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_pivot(self):
        def test_impl(df):
            pt = df.pivot_table(index='A', columns='C', values='D', aggfunc='sum')
            return (pt.small.values, pt.large.values)

        hpat_func = sdc.jit(pivots={'pt': ['small', 'large']})(test_impl)
        self.assertEqual(
            set(hpat_func(_pivot_df1)[0]), set(test_impl(_pivot_df1)[0]))
        self.assertEqual(
            set(hpat_func(_pivot_df1)[1]), set(test_impl(_pivot_df1)[1]))

    def test_pivot_parallel(self):
        def test_impl():
            df = pd.read_parquet("pivot2.pq")
            pt = df.pivot_table(index='A', columns='C', values='D', aggfunc='sum')
            res = pt.small.values.sum()
            return res

        hpat_func = sdc.jit(
            pivots={'pt': ['small', 'large']})(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_crosstab1(self):
        def test_impl(df):
            pt = pd.crosstab(df.A, df.C)
            return (pt.small.values, pt.large.values)

        hpat_func = sdc.jit(pivots={'pt': ['small', 'large']})(test_impl)
        self.assertEqual(
            set(hpat_func(_pivot_df1)[0]), set(test_impl(_pivot_df1)[0]))
        self.assertEqual(
            set(hpat_func(_pivot_df1)[1]), set(test_impl(_pivot_df1)[1]))

    def test_crosstab_parallel1(self):
        def test_impl():
            df = pd.read_parquet("pivot2.pq")
            pt = pd.crosstab(df.A, df.C)
            res = pt.small.values.sum()
            return res

        hpat_func = sdc.jit(
            pivots={'pt': ['small', 'large']})(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip("Implement groupby(lambda) for DataFrame")
    def test_groupby_lambda(self):
        def test_impl(df):
            group = df.groupby(lambda x: x % 2 == 0)
            return group.count()

        df = pd.DataFrame({'A': [2, 1, 1, 1, 2, 2, 1], 'B': [-8, 2, 3, 1, 5, 6, 7]})
        hpat_func = sdc.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))


if __name__ == "__main__":
    unittest.main()

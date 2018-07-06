import unittest
import pandas as pd
import numpy as np
import random
import string
import pyarrow.parquet as pq
import numba
import hpat
from hpat import hiframes_sort
from hpat.str_arr_ext import StringArray
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs, dist_IR_contains)

class TestHiFrames(unittest.TestCase):
    def test_basics(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df['A'].values
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

    def test_set_column1(self):
        # set existing column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            df['A'] = np.arange(n)
            return df.A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 2)

    def test_set_column2(self):
        # create new column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            df['C'] = np.arange(n)
            return df.C.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 2)

    def test_len_df(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            return len(df)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_set_column_reflect(self):
        def test_impl(df, arr):
            df['C'] = arr
            return

        hpat_func = hpat.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        hpat_func(df, arr)
        self.assertIn('C', df)
        np.testing.assert_almost_equal(df.C, arr)

    def test_fillna(self):
        def test_impl():
            A = np.array([1., 2., 3.])
            A[0] = np.nan
            df = pd.DataFrame({'A': A})
            B = df.A.fillna(5.0)
            return B.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_fillna_inplace(self):
        def test_impl():
            A = np.array([1., 2., 3.])
            A[0] = np.nan
            df = pd.DataFrame({'A': A})
            df.A.fillna(5.0, inplace=True)
            return df.A.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_column_sum(self):
        def test_impl():
            A = np.array([1., 2., 3.])
            A[0] = np.nan
            df = pd.DataFrame({'A': A})
            return df.A.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_column_mean(self):
        def test_impl():
            A = np.array([1., 2., 3.])
            A[0] = np.nan
            df = pd.DataFrame({'A': A})
            return df.A.mean()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_column_var(self):
        def test_impl():
            A = np.array([1., 2., 3.])
            A[0] = 4.0
            df = pd.DataFrame({'A': A})
            return df.A.var()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    def test_column_std(self):
        def test_impl():
            A = np.array([1., 2., 3.])
            A[0] = 4.0
            df = pd.DataFrame({'A': A})
            return df.A.std()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    def test_column_map(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)})
            df['B'] = df.A.map(lambda a: 2*a)
            return df.B.sum()

        n = 121
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    def test_column_map_arg(self):
        def test_impl(df):
            df['B'] = df.A.map(lambda a: 2*a)
            return

        n = 121
        df1 = pd.DataFrame({'A': np.arange(n)})
        df2 = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        hpat_func(df1)
        self.assertTrue(hasattr(df1, 'B'))
        test_impl(df2)
        np.testing.assert_equal(df1.B.values, df2.B.values)

    def test_df_apply(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A + r.B, axis=1)
            return df.B.sum()

        n = 121
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    # TODO: enable when namedtuple analysis patch is merged (#2984)
    # def test_df_apply_branch(self):
    #     def test_impl(n):
    #         df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
    #         B = df.apply(lambda r: r.A < 10 and r.B > 20, axis=1)
    #         return df.B.sum()
    #
    #     n = 121
    #     hpat_func = hpat.jit(test_impl)
    #     np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    def test_cumsum(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df.A.cumsum()
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_array_OneDs(), 2)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 2)
        self.assertTrue(dist_IR_contains('dist_cumsum'))

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

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertTrue(dist_IR_contains('dist_cumsum'))

    def test_quantile_parallel(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float64)})
            return df.A.quantile(.25)

        hpat_func = hpat.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_quantile_parallel_float_nan(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float32)})
            df.A[0:100] = np.nan
            df.A[200:331] = np.nan
            return df.A.quantile(.25)

        hpat_func = hpat.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_quantile_parallel_int(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.int32)})
            return df.A.quantile(.25)

        hpat_func = hpat.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_quantile_sequential(self):
        def test_impl(A):
            df = pd.DataFrame({'A': A})
            return df.A.quantile(.25)

        hpat_func = hpat.jit(test_impl)
        n = 1001
        A = np.arange(0, n, 1, np.float64)
        np.testing.assert_almost_equal(hpat_func(A), test_impl(A))

    def test_nunique(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)})
            df.A[2] = 0
            return df.A.nunique()

        hpat_func = hpat.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    def test_nunique_str(self):
        def test_impl(n):
            df = pd.DataFrame({'A': ['aa', 'bb', 'aa', 'cc', 'cc']})
            return df.A.nunique()

        hpat_func = hpat.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    def test_nunique_str_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.two.nunique()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)

    def test_describe(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float64)})
            return df.A.describe()

        hpat_func = hpat.jit(test_impl)
        n = 1001
        hpat_func(n)
        # XXX: test actual output
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_df_describe(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float32),
                               'B': np.arange(n)})
            #df.A[0:1] = np.nan
            return df.describe()

        hpat_func = hpat.jit(test_impl)
        n = 1001
        hpat_func(n)
        # XXX: test actual output
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_str_contains_regex(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'ADEF'])
            df = pd.DataFrame({'A': A})
            B = df.A.str.contains('AB*', regex=True)
            return B.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), 2)

    def test_str_contains_noregex(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'ADEF'])
            df = pd.DataFrame({'A': A})
            B = df.A.str.contains('BB', regex=False)
            return B.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), 1)

    def test_filter1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.ones(n)})
            df1 = df[df.A > .5]
            return np.sum(df1.B)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.ones(n)})
            df1 = df.loc[df.A > .5]
            return np.sum(df1.B)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_1D_Var_len(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)+1.0})
            df1 = df[df.A > 5]
            return len(df1.B)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_rolling1(self):
        # size 3 without unroll
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
            Ac = df.A.rolling(3).sum()
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 121
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        # size 7 with unroll
        def test_impl_2(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.random.ranf(n)})
            Ac = df.A.rolling(7).sum()
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 121
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_rolling2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            df['moving average'] = df.A.rolling(window=5, center=True).mean()
            return df['moving average'].sum()

        hpat_func = hpat.jit(test_impl)
        n = 121
        self.assertEqual(hpat_func(n), test_impl(n))
        # small input array to mean is REP
        self.assertEqual(count_array_REPs(), 1)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_rolling3(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df.A.rolling(3, center=True).apply(lambda a: a[0]+2*a[1]+a[2])
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 121
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shift1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df.A.shift(1)
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shift2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df.A.pct_change()
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_list_convert(self):
        def test_impl():
            df = pd.DataFrame({'one': np.array([-1, np.nan, 2.5]),
                        'two': ['foo', 'bar', 'baz'],
                        'three': [True, False, True]})
            return df.one.values, df.two.values, df.three.values

        hpat_func = hpat.jit(test_impl)
        one, two, three = hpat_func()
        self.assertTrue(isinstance(one, np.ndarray))
        self.assertTrue(isinstance(two,  np.ndarray))
        self.assertTrue(isinstance(three, np.ndarray))

    def test_df_input(self):
        def test_impl(df):
            return df.B.sum()

        n = 121
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df), test_impl(df))

    def test_df_input2(self):
        def test_impl(df):
            C = df.B == 'two'
            return C.sum()

        n = 11
        df = pd.DataFrame({'A': np.random.ranf(3*n), 'B': ['one', 'two', 'three']*n})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df), test_impl(df))

    def test_join1(self):
        def test_impl(n):
            df1 = pd.DataFrame({'key1': np.arange(n)+3, 'A': np.arange(n)+1.0})
            df2 = pd.DataFrame({'key2': 2*np.arange(n)+1, 'B': n+np.arange(n)+1.0})
            df3 = pd.merge(df1, df2, left_on='key1', right_on='key2')
            return df3.B.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        n = 11111
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_join1_seq(self):
        def test_impl(n):
            df1 = pd.DataFrame({'key1': np.arange(n)+3, 'A': np.arange(n)+1.0})
            df2 = pd.DataFrame({'key2': 2*np.arange(n)+1, 'B': n+np.arange(n)+1.0})
            df3 = pd.merge(df1, df2, left_on='key1', right_on='key2')
            return df3.B

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n).sum(), test_impl(n).sum())
        self.assertEqual(count_array_OneDs(), 0)
        self.assertEqual(count_parfor_OneDs(), 0)
        n = 11111
        self.assertEqual(hpat_func(n).sum(), test_impl(n).sum())

    def test_concat(self):
        def test_impl(n):
            df1 = pd.DataFrame({'key1': np.arange(n), 'A': np.arange(n)+1.0})
            df2 = pd.DataFrame({'key2': n-np.arange(n), 'A': n+np.arange(n)+1.0})
            df3 = pd.concat([df1, df2])
            return df3.A.sum() + df3.key2.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        n = 11111
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_concat_str(self):
        def test_impl():
            df1 = pq.read_table('example.parquet').to_pandas()
            df2 = pq.read_table('example.parquet').to_pandas()
            A3 = pd.concat([df1, df2])
            return (A3.two=='foo').sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_concat_series(self):
        def test_impl(n):
            df1 = pd.DataFrame({'key1': np.arange(n), 'A': np.arange(n)+1.0})
            df2 = pd.DataFrame({'key2': n-np.arange(n), 'A': n+np.arange(n)+1.0})
            A3 = pd.concat([df1.A, df2.A])
            return A3.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        n = 11111
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_concat_series_str(self):
        def test_impl():
            df1 = pq.read_table('example.parquet').to_pandas()
            df2 = pq.read_table('example.parquet').to_pandas()
            A3 = pd.concat([df1.two, df2.two])
            return (A3=='foo').sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_seq(self):
        def test_impl(df):
            A = df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
            return A.values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [2,1,1,1,2,2,1], 'B': [-8,2,3,1,5,6,7]})
        # np.testing.assert_array_equal(hpat_func(df), test_impl(df))
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_sum(self):
        def test_impl(df):
            A = df.groupby('A')['B'].sum()
            return A.values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [2,1,1,1,2,2,1], 'B': [-8,2,3,1,5,6,7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_count(self):
        def test_impl(df):
            A = df.groupby('A')['B'].count()
            return A.values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [2,1,1,1,2,2,1], 'B': [-8,2,3,1,5,6,7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_mean(self):
        def test_impl(df):
            A = df.groupby('A')['B'].mean()
            return A.values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [2,1,1,1,2,2,1], 'B': [-8,2,3,1,5,6,7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_min(self):
        def test_impl(df):
            A = df.groupby('A')['B'].min()
            return A.values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [2,1,1,1,2,2,1], 'B': [-8,2,3,1,5,6,7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_max(self):
        def test_impl(df):
            A = df.groupby('A')['B'].max()
            return A.values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [2,1,1,1,2,2,1], 'B': [-8,2,3,1,5,6,7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_all_col(self):
        def test_impl(df):
            df2 = df.groupby('A').mean()
            return df2.B.values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [2,1,1,1,2,2,1], 'B': [-8,2,3,1,5,6,7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_seq_as_index(self):
        def test_impl(df):
            df2 = df.groupby('A', as_index=False).mean()
            return df2.A.values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [2,1,1,1,2,2,1], 'B': [-8,2,3,1,5,6,7]})
        self.assertEqual(set(hpat_func(df)), set(test_impl(df)))

    def test_agg_parallel(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_sum(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].sum()
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_count(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].count()
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_mean(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].mean()
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_min(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].min()
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_max(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            A = df.groupby('A')['B'].max()
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_str(self):
        def test_impl():
            df = pq.read_table("groupby3.pq").to_pandas()
            A = df.groupby('A')['B'].agg(lambda x: x.max()-x.min())
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_all_col(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            df2 = df.groupby('A').max()
            return df2.B.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_agg_parallel_as_index(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)})
            df2 = df.groupby('A', as_index=False).max()
            return df2.A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_itertuples(self):
        def test_impl(df):
            res = 0.0
            for r in df.itertuples():
                res += r[1]
            return res

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.ones(n, np.int64)})
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_itertuples_str(self):
        def test_impl(df):
            res = ""
            for r in df.itertuples():
                res += r[1]
            return res

        hpat_func = hpat.jit(test_impl)
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

        hpat_func = hpat.jit(test_impl)
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

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_sort_values(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.B.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n), 'B': np.arange(n), 'C': np.random.ranf(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df.copy()), test_impl(df))


    def test_sort_values_single_col(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.A.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df.copy()), test_impl(df))

    def test_sort_values_single_col_str(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.A.values

        n = 1211
        random.seed(2)
        str_vals = []

        for i in range(n):
            k = random.randint(1, 30)
            val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)
        df = pd.DataFrame({'A': str_vals})
        hpat_func = hpat.jit(test_impl)
        self.assertTrue((hpat_func(df.copy()) == test_impl(df)).all())

    def test_sort_parallel_single_col(self):
        # TODO: better parallel sort test
        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            df.sort_values('points', inplace=True)
            res = df.points.values
            return res

        hpat_func = hpat.jit(locals={'res:return': 'distributed'})(test_impl)

        save_min_samples = hiframes_sort.MIN_SAMPLES
        try:
            hiframes_sort.MIN_SAMPLES = 10
            res = hpat_func()
            self.assertTrue((np.diff(res)>=0).all())
        finally:
            hiframes_sort.MIN_SAMPLES = save_min_samples  # restore global val


    def test_intraday(self):
        def test_impl(nsyms):
            max_num_days = 100
            all_res = 0.0
            for i in hpat.prange(nsyms):
                s_open = 20 * np.ones(max_num_days)
                s_low = 28 * np.ones(max_num_days)
                s_close = 19 * np.ones(max_num_days)
                df = pd.DataFrame({'Open': s_open, 'Low': s_low,
                                    'Close': s_close})
                df['Stdev'] = df['Close'].rolling(window=90).std()
                df['Moving Average'] = df['Close'].rolling(window=20).mean()
                df['Criteria1'] = (df['Open'] - df['Low'].shift(1)) < -df['Stdev']
                df['Criteria2'] = df['Open'] > df['Moving Average']
                df['BUY'] = df['Criteria1'] & df['Criteria2']
                df['Pct Change'] = (df['Close'] - df['Open']) / df['Open']
                df['Rets'] = df['Pct Change'][df['BUY'] == True]
                all_res += df['Rets'].mean()
            return all_res

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_OneDs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)


if __name__ == "__main__":
    unittest.main()

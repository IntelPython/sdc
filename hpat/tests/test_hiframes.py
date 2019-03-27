import unittest
import itertools
import pandas as pd
import numpy as np
import random
import string
import pyarrow.parquet as pq
import numba
import hpat
from hpat import hiframes
from hpat.str_arr_ext import StringArray
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs, dist_IR_contains,
                            get_start_end)


class TestHiFrames(unittest.TestCase):

    def test_column_list_select2(self):
        # make sure HPAT copies the columns like Pandas does
        def test_impl(df):
            df2 = df[['A']]
            df2['A'] += 10
            return df2.A, df.A

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame(
            {'A': np.arange(n), 'B': np.ones(n), 'C': np.random.ranf(n)})
        np.testing.assert_array_equal(hpat_func(df.copy())[1], test_impl(df)[1])

    def test_pd_DataFrame_from_series_par(self):
        def test_impl(n):
            S1 = pd.Series(np.ones(n))
            S2 = pd.Series(np.random.ranf(n))
            df = pd.DataFrame({'A': S1, 'B': S2})
            return df.A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

    def test_getitem_bool_series(self):
        def test_impl(df):
            return df['A'][df['B']].values

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [1,2,3], 'B': [True, False, True]})
        np.testing.assert_array_equal(test_impl(df), hpat_func(df))

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
        # test compile again for overload related issues
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    def test_nunique_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.four.nunique()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        # test compile again for overload related issues
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)

    def test_nunique_str(self):
        def test_impl(n):
            df = pd.DataFrame({'A': ['aa', 'bb', 'aa', 'cc', 'cc']})
            return df.A.nunique()

        hpat_func = hpat.jit(test_impl)
        n = 1001
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        # test compile again for overload related issues
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    def test_nunique_str_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.two.nunique()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        # test compile again for overload related issues
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)

    def test_unique(self):
        def test_impl(S):
            return S.unique()

        hpat_func = hpat.jit(test_impl)
        n = 1001
        S = pd.Series(np.arange(n))
        S[2] = 0
        self.assertEqual(set(hpat_func(S)), set(test_impl(S)))

    def test_unique_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return (df.four.unique() == 3.0).sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)

    def test_unique_str(self):
        def test_impl(n):
            df = pd.DataFrame({'A': ['aa', 'bb', 'aa', 'cc', 'cc']})
            return df.A.unique()

        hpat_func = hpat.jit(test_impl)
        n = 1001
        self.assertEqual(set(hpat_func(n)), set(test_impl(n)))

    def test_unique_str_parallel(self):
        # TODO: test without file
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return (df.two.unique() == 'foo').sum()

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

    def test_str_replace_regex(self):
        def test_impl(df):
            return df.A.str.replace('AB*', 'EE', regex=True)

        df = pd.DataFrame({'A': ['ABCC', 'CABBD']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    def test_str_replace_noregex(self):
        def test_impl(df):
            return df.A.str.replace('AB', 'EE', regex=False)

        df = pd.DataFrame({'A': ['ABCC', 'CABBD']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    def test_str_replace_regex_parallel(self):
        def test_impl(df):
            B = df.A.str.replace('AB*', 'EE', regex=True)
            return B

        n = 5
        A = ['ABCC', 'CABBD', 'CCD', 'CCDAABB', 'ED']
        start, end = get_start_end(n)
        df = pd.DataFrame({'A': A[start:end]})
        hpat_func = hpat.jit(distributed={'df', 'B'})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)
        self.assertEqual(count_array_REPs(), 3)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_str_split(self):
        def test_impl(df):
            return df.A.str.split(',')

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D', 'G', '', 'g,f']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    def test_str_split_filter(self):
        def test_impl(df):
            B = df.A.str.split(',')
            df2 = pd.DataFrame({'B': B})
            return df2[df2.B.str.len()>1]

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D', 'G', '', 'g,f']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_str_split_box_df(self):
        def test_impl(df):
            return pd.DataFrame({'B': df.A.str.split(',')})

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df).B, test_impl(df).B, check_names=False)

    def test_str_split_unbox_df(self):
        def test_impl(df):
            return df.A.iloc[0]

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D']})
        df2 = pd.DataFrame({'A': df.A.str.split(',')})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df2), test_impl(df2))

    def test_str_split_bool_index(self):
        def test_impl(df):
            C = df.A.str.split(',')
            return C[df.B == 'aa']

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D'], 'B': ['aa', 'bb']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    def test_str_split_parallel(self):
        def test_impl(df):
            B = df.A.str.split(',')
            return B

        n = 5
        start, end = get_start_end(n)
        A = ['AB,CC', 'C,ABB,D', 'CAD', 'CA,D', 'AA,,D']
        df = pd.DataFrame({'A': A[start:end]})
        hpat_func = hpat.jit(distributed={'df', 'B'})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)
        self.assertEqual(count_array_REPs(), 3)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_str_get(self):
        def test_impl(df):
            B = df.A.str.split(',')
            return B.str.get(1)

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    def test_str_get_parallel(self):
        def test_impl(df):
            A = df.A.str.split(',')
            B = A.str.get(1)
            return B

        n = 5
        start, end = get_start_end(n)
        A = ['AB,CC', 'C,ABB,D', 'CAD,F', 'CA,D', 'AA,,D']
        df = pd.DataFrame({'A': A[start:end]})
        hpat_func = hpat.jit(distributed={'df', 'B'})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)
        self.assertEqual(count_array_REPs(), 3)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_str_flatten(self):
        def test_impl(df):
            A = df.A.str.split(',')
            return pd.Series(list(itertools.chain(*A)))

        df = pd.DataFrame({'A': ['AB,CC', 'C,ABB,D']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

    def test_str_flatten_parallel(self):
        def test_impl(df):
            A = df.A.str.split(',')
            B = pd.Series(list(itertools.chain(*A)))
            return B

        n = 5
        start, end = get_start_end(n)
        A = ['AB,CC', 'C,ABB,D', 'CAD', 'CA,D', 'AA,,D']
        df = pd.DataFrame({'A': A[start:end]})
        hpat_func = hpat.jit(distributed={'df', 'B'})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)
        self.assertEqual(count_array_REPs(), 3)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_to_numeric(self):
        def test_impl(df):
            B = pd.to_numeric(df.A, errors='coerce')
            return B

        df = pd.DataFrame({'A': ['123', '331']})
        hpat_func = hpat.jit(locals={'B': hpat.int64[:]})(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(df), test_impl(df), check_names=False)

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
        self.assertEqual(count_array_REPs(), 0)
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
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.random.ranf(n)})
            Ac = df.A.shift(1)
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shift2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.random.ranf(n)})
            Ac = df.A.pct_change(1)
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

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

    def test_df_input_dist1(self):
        def test_impl(df):
            return df.B.sum()

        n = 121
        A = [3,4,5,6,1]
        B = [5,6,2,1,3]
        n = 5
        start, end = get_start_end(n)
        df = pd.DataFrame({'A': A, 'B': B})
        df_h = pd.DataFrame({'A': A[start:end], 'B': B[start:end]})
        hpat_func = hpat.jit(distributed={'df'})(test_impl)
        np.testing.assert_almost_equal(hpat_func(df_h), test_impl(df))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

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

        A = np.array([1,1,2,3])
        B = np.array([3,4,5,6])
        hpat_func = hpat.jit(locals={'A:input': 'distributed',
            'B:input': 'distributed', 'df2:return': 'distributed'})(test_impl)
        start, end = get_start_end(len(A))
        df2 = hpat_func(A[start:end], B[start:end])
        # TODO:
        # pd.testing.assert_frame_equal(
        #     hpat_func(A[start:end], B[start:end]), test_impl(A, B))

if __name__ == "__main__":
    unittest.main()

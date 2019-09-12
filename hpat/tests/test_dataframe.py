import unittest
import random
import string
import pandas as pd
import numpy as np

import numba
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs, count_parfor_OneDs,
                                   count_array_OneDs, dist_IR_contains, get_start_end)

from hpat.tests.gen_test_data import ParquetGenerator


@hpat.jit
def inner_get_column(df):
    # df2 = df[['A', 'C']]
    # df2['D'] = np.ones(3)
    return df.A


COL_IND = 0


class TestDataFrame(unittest.TestCase):

    def test_create1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            return df.A

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_create_cond1(self):
        def test_impl(A, B, c):
            if c:
                df = pd.DataFrame({'A': A})
            else:
                df = pd.DataFrame({'A': B})
            return df.A

        hpat_func = hpat.jit(test_impl)
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

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())
        
    def test_unbox1(self):
        def test_impl(df):
            return df.A

        hpat_func = hpat.jit(test_impl)
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

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        pd.testing.assert_series_equal(hpat_func(df.copy(), True), test_impl(df.copy(), True))
        pd.testing.assert_series_equal(hpat_func(df.copy(), False), test_impl(df.copy(), False))

    @unittest.skip('Implement feature to create DataFrame without column names')
    def test_unbox_without_column_names(self):
        def test_impl(df):
            return df

        df = pd.DataFrame([100, 200, 300, 400, 200, 100])
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_box1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    def test_box2(self):
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'bb', 'ccc']})
            return df

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip("pending df filter support")
    def test_box3(self):
        def test_impl(df):
            df = df[df.A != 'dd']
            return df

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': ['aa', 'bb', 'cc']})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_box_dist_return(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        hpat_func = hpat.jit(distributed={'df'})(test_impl)
        n = 11
        hres, res = hpat_func(n), test_impl(n)
        self.assertEqual(count_array_OneDs(), 3)
        self.assertEqual(count_parfor_OneDs(), 2)
        dist_sum = hpat.jit(
            lambda a: hpat.distributed_api.dist_reduce(
                a, np.int32(hpat.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1)  # run to compile
        np.testing.assert_allclose(dist_sum(hres.A.sum()), res.A.sum())
        np.testing.assert_allclose(dist_sum(hres.B.sum()), res.B.sum())

    def test_len1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            return len(df)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shape1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            return df.shape

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_column_getitem1(self):
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

    def test_column_list_getitem1(self):
        def test_impl(df):
            return df[['A', 'C']]

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame(
            {'A': np.arange(n), 'B': np.ones(n), 'C': np.random.ranf(n)})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_filter1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + n, 'B': np.arange(n)**2})
            df1 = df[df.A > .5]
            return df1.B.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + n, 'B': np.arange(n)**2})
            df1 = df.loc[df.A > .5]
            return np.sum(df1.B)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter3(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + n, 'B': np.arange(n)**2})
            df1 = df.iloc[(df.A > .5).values]
            return np.sum(df1.B)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_iloc1(self):
        def test_impl(df, n):
            return df.iloc[1:n].B.values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df, n), test_impl(df, n))

    def test_iloc2(self):
        def test_impl(df, n):
            return df.iloc[np.array([1, 4, 9])].B.values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df, n), test_impl(df, n))

    def test_iloc3(self):
        def test_impl(df):
            return df.iloc[:, 1].values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @unittest.skip("TODO: support A[[1,2,3]] in Numba")
    def test_iloc4(self):
        def test_impl(df, n):
            return df.iloc[[1, 4, 9]].B.values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df, n), test_impl(df, n))

    def test_iloc5(self):
        # test iloc with global value
        def test_impl(df):
            return df.iloc[:, COL_IND].values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_loc1(self):
        def test_impl(df):
            return df.loc[:, 'B'].values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_iat1(self):
        def test_impl(n):
            df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
            return df.iat[3, 1]
        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_iat2(self):
        def test_impl(df):
            return df.iat[3, 1]
        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_iat3(self):
        def test_impl(df, n):
            return df.iat[n - 1, 1]
        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        self.assertEqual(hpat_func(df, n), test_impl(df, n))

    def test_iat_set1(self):
        def test_impl(df, n):
            df.iat[n - 1, 1] = n**2
            return df.A  # return the column to check column aliasing
        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        df2 = df.copy()
        pd.testing.assert_series_equal(hpat_func(df, n), test_impl(df2, n))

    def test_iat_set2(self):
        def test_impl(df, n):
            df.iat[n - 1, 1] = n**2
            return df  # check df aliasing/boxing
        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n) + n})
        df2 = df.copy()
        pd.testing.assert_frame_equal(hpat_func(df, n), test_impl(df2, n))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column1(self):
        # set existing column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
            df['A'] = np.arange(n)
            return df

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column_reflect4(self):
        # set existing column
        def test_impl(df, n):
            df['A'] = np.arange(n)

        hpat_func = hpat.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
        df2 = df1.copy()
        hpat_func(df1, n)
        test_impl(df2, n)
        pd.testing.assert_frame_equal(df1, df2)

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column_new_type1(self):
        # set existing column with a new type
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n) + 3.0})
            df['A'] = np.arange(n)
            return df

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column2(self):
        # create new column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n) + 1.0})
            df['C'] = np.arange(n)
            return df

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    @unittest.skip('AssertionError - fix needed\n'
                   'Attribute "dtype" are different\n'
                   '[left]:  int64\n'
                   '[right]: int32\n')
    def test_set_column_reflect3(self):
        # create new column
        def test_impl(df, n):
            df['C'] = np.arange(n)

        hpat_func = hpat.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n) + 3.0})
        df2 = df1.copy()
        hpat_func(df1, n)
        test_impl(df2, n)
        pd.testing.assert_frame_equal(df1, df2)

    def test_set_column_bool1(self):
        def test_impl(df):
            df['C'] = df['A'][df['B']]

        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [True, False, True]})
        df2 = df.copy()
        test_impl(df2)
        hpat_func(df)
        pd.testing.assert_series_equal(df.C, df2.C)

    def test_set_column_reflect1(self):
        def test_impl(df, arr):
            df['C'] = arr
            return df.C.sum()

        hpat_func = hpat.jit(test_impl)
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

        hpat_func = hpat.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        df2 = df.copy()
        np.testing.assert_almost_equal(hpat_func(df, arr), test_impl(df2, arr))

    def test_df_values1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.values

        hpat_func = hpat.jit(test_impl)
        n = 11
        np.testing.assert_array_equal(hpat_func(n), test_impl(n))

    def test_df_values2(self):
        def test_impl(df):
            return df.values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_df_values_parallel1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.values.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        np.testing.assert_array_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_df_apply(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A + r.B, axis=1)
            return df.B.sum()

        n = 121
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

    def test_df_apply_branch(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A < 10 and r.B > 20, axis=1)
            return df.B.sum()

        n = 121
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(n), test_impl(n))

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

    def test_sort_values(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.B.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n), 'B': np.arange(n), 'C': np.random.ranf(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(df.copy()), test_impl(df))

    def test_sort_values_copy(self):
        def test_impl(df):
            df2 = df.sort_values('A')
            return df2.B.values

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

        for _ in range(n):
            k = random.randint(1, 30)
            val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)
        df = pd.DataFrame({'A': str_vals})
        hpat_func = hpat.jit(test_impl)
        self.assertTrue((hpat_func(df.copy()) == test_impl(df)).all())

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
        hpat_func = hpat.jit(test_impl)
        self.assertTrue((hpat_func(df) == sorted_df.B.values).all())

    @unittest.skip('Error - fix needed; issue is related to __pycache__\n')
    def test_sort_parallel_single_col(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        # TODO: better parallel sort test
        def test_impl():
            df = pd.read_parquet('kde.parquet')
            df.sort_values('points', inplace=True)
            res = df.points.values
            return res

        hpat_func = hpat.jit(locals={'res:return': 'distributed'})(test_impl)

        save_min_samples = hpat.hiframes.sort.MIN_SAMPLES
        try:
            hpat.hiframes.sort.MIN_SAMPLES = 10
            res = hpat_func()
            self.assertTrue((np.diff(res) >= 0).all())
        finally:
            # restore global val
            hpat.hiframes.sort.MIN_SAMPLES = save_min_samples

    @unittest.skip('Error - fix needed; issue is related to __pycache__\n')
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

        hpat_func = hpat.jit(locals={'res:return': 'distributed'})(test_impl)

        save_min_samples = hpat.hiframes.sort.MIN_SAMPLES
        try:
            hpat.hiframes.sort.MIN_SAMPLES = 10
            res = hpat_func()
            self.assertTrue((np.diff(res) >= 0).all())
        finally:
            # restore global val
            hpat.hiframes.sort.MIN_SAMPLES = save_min_samples

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

    def test_df_head1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.head(3)

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    def test_pct_change1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.pct_change(3)

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    def test_mean1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.mean()

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_std1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.std()

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_var1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.var()

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_max1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.max()

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_min1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.min()

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_sum1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_prod1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.prod()

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_count1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n) + 1.0, 'B': np.arange(n) + 1})
            return df.count()

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(hpat_func(n), test_impl(n))

    def test_df_fillna1(self):
        def test_impl(df):
            return df.fillna(5.0)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_fillna_str1(self):
        def test_impl(df):
            return df.fillna("dd")

        df = pd.DataFrame({'A': ['aa', 'b', None, 'ccc']})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_fillna_inplace1(self):
        def test_impl(A):
            A.fillna(11.0, inplace=True)
            return A

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        df2 = df.copy()
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df2))

    def test_df_reset_index1(self):
        def test_impl(df):
            return df.reset_index(drop=True)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_reset_index_inplace1(self):
        def test_impl():
            df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
            df.reset_index(drop=True, inplace=True)
            return df

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_df_dropna1(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        hpat_func = hpat.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = hpat_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna2(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        hpat_func = hpat.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = hpat_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna_inplace1(self):
        # TODO: fix error when no df is returned
        def test_impl(df):
            df.dropna(inplace=True)
            return df

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        df2 = df.copy()
        hpat_func = hpat.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = hpat_func(df2)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna_str1(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, 4.0, 1.0], 'B': ['aa', 'b', None, 'ccc']})
        hpat_func = hpat.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = hpat_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_drop1(self):
        def test_impl(df):
            return df.drop(columns=['A'])

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_drop_inplace2(self):
        # test droping after setting the column
        def test_impl(df):
            df2 = df[['A', 'B']]
            df2['D'] = np.ones(3)
            df2.drop(columns=['D'], inplace=True)
            return df2

        df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_df_drop_inplace1(self):
        def test_impl(df):
            df.drop('A', axis=1, inplace=True)
            return df

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        df2 = df.copy()
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df2))

    def test_isin_df1(self):
        def test_impl(df, df2):
            return df.isin(df2)

        hpat_func = hpat.jit(test_impl)
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

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_isin_list1(self):
        def test_impl(df):
            vals = [2, 3, 4]
            return df.isin(vals)

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(hpat_func(df), test_impl(df))

    def test_append1(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=True)

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'C': np.arange(n)**2})
        df2.A[n // 2:] = n
        pd.testing.assert_frame_equal(hpat_func(df, df2), test_impl(df, df2))

    def test_append2(self):
        def test_impl(df, df2, df3):
            return df.append([df2, df3], ignore_index=True)

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2.A[n // 2:] = n
        df3 = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(
            hpat_func(df, df2, df3), test_impl(df, df2, df3))

    def test_concat_columns1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2], axis=1)

        hpat_func = hpat.jit(test_impl)
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

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl(), check_names=False)


if __name__ == "__main__":
    unittest.main()

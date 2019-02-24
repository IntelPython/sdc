import unittest
import pandas as pd
import numpy as np

import numba
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_start_end)


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


    def test_box1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        hpat_func = hpat.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(hpat_func(n), test_impl(n))

    def test_box2(self):
        def test_impl():
            df = pd.DataFrame({'A': [1,2,3], 'B': ['a', 'bb', 'ccc']})
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
            df = pd.DataFrame({'A': np.arange(n)+n, 'B': np.arange(n)**2})
            df1 = df[df.A > .5]
            return df1.B.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+n, 'B': np.arange(n)**2})
            df1 = df.loc[df.A > .5]
            return np.sum(df1.B)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter3(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+n, 'B': np.arange(n)**2})
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
            return df.iloc[np.array([1,4,9])].B.values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df, n), test_impl(df, n))

    def test_iloc3(self):
        def test_impl(df):
            return df.iloc[:,1].values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @unittest.skip("TODO: support A[[1,2,3]] in Numba")
    def test_iloc4(self):
        def test_impl(df, n):
            return df.iloc[[1,4,9]].B.values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df, n), test_impl(df, n))

    def test_loc1(self):
        def test_impl(df):
            return df.loc[:,'B'].values

        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_iat1(self):
        def test_impl(n):
            df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
            return df.iat[3, 1]
        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_iat2(self):
        def test_impl(df):
            return df.iat[3, 1]
        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_iat3(self):
        def test_impl(df, n):
            return df.iat[n-1, 1]
        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
        self.assertEqual(hpat_func(df, n), test_impl(df, n))

    def test_iat_set1(self):
        def test_impl(df, n):
            df.iat[n-1, 1] = n**2
            return df
        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
        pd.testing.assert_frame_equal(hpat_func(df, n), test_impl(df, n))


if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
import numpy as np
import random
import string
import pyarrow.parquet as pq
import numba
import hpat
from hpat.str_arr_ext import StringArray
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_start_end)

_cov_corr_series = [(pd.Series(x), pd.Series(y)) for x, y in [
    (
        [np.nan, -2., 3., 9.1],
        [np.nan, -2., 3., 5.0],
    ),
    # TODO(quasilyte): more intricate data for complex-typed series.
    # Some arguments make assert_almost_equal fail.
    # Functions that yield mismaching results: _column_corr_impl and _column_cov_impl.
    (
        [complex(-2., 1.0), complex(3.0, 1.0)],
        [complex(-3., 1.0), complex(2.0, 1.0)],
    ),
    (
        [complex(-2.0, 1.0), complex(3.0, 1.0)],
        [1.0, -2.0],
    ),
    (
        [1.0, -4.5],
        [complex(-4.5, 1.0), complex(3.0, 1.0)],
    ),
]]

class TestSeries(unittest.TestCase):
    def test_create1(self):
        def test_impl():
            df = pd.DataFrame({'A': [1,2,3]})
            return (df.A == 1).sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_create2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)})
            return (df.A == 2).sum()

        n = 11
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_create_series1(self):
        def test_impl():
            A = pd.Series([1,2,3])
            return A.values

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_create_series_index1(self):
        # create and box an indexed Series
        def test_impl():
            A = pd.Series([1,2,3], ['A', 'C', 'B'])
            return A

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_create_series_index2(self):
        def test_impl():
            A = pd.Series([1,2,3], index=['A', 'C', 'B'])
            return A

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_create_series_index3(self):
        def test_impl():
            A = pd.Series([1,2,3], index=['A', 'C', 'B'], name='A')
            return A

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_create_series_index4(self):
        def test_impl(name):
            A = pd.Series([1,2,3], index=['A', 'C', 'B'], name=name)
            return A

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func('A'), test_impl('A'))

    def test_create_str(self):
        def test_impl():
            df = pd.DataFrame({'A': ['a', 'b', 'c']})
            return (df.A == 'a').sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_pass_df1(self):
        def test_impl(df):
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_pass_df_str(self):
        def test_impl(df):
            return (df.A == 'a').sum()

        df = pd.DataFrame({'A': ['a', 'b', 'c']})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_pass_series1(self):
        # TODO: check to make sure it is series type
        def test_impl(A):
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_pass_series2(self):
        # test creating dataframe from passed series
        def test_impl(A):
            df = pd.DataFrame({'A': A})
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_pass_series_str(self):
        def test_impl(A):
            return (A == 'a').sum()

        df = pd.DataFrame({'A': ['a', 'b', 'c']})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_series_attr1(self):
        def test_impl(A):
            return A.size

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_series_attr2(self):
        def test_impl(A):
            return A.copy().values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr3(self):
        def test_impl(A):
            return A.min()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_series_attr4(self):
        def test_impl(A):
            return A.cumsum().values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr5(self):
        def test_impl(A):
            return A.argsort().values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr6(self):
        def test_impl(A):
            return A.take([2,3]).values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr7(self):
        def test_impl(A):
            return A.astype(np.float64)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_copy_str1(self):
        def test_impl(A):
            return A.copy()

        n = 11
        S = pd.Series(['aa', 'bb', 'cc'])
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_astype_str1(self):
        def test_impl(A):
            return A.astype(str)

        n = 11
        S = pd.Series(np.arange(n))
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_astype_str2(self):
        def test_impl(A):
            return A.astype(str)

        S = pd.Series(['aa', 'bb', 'cc'])
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_np_call_on_series1(self):
        def test_impl(A):
            return np.min(A)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_values1(self):
        def test_impl(A):
            return (A == 2).values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_static_setitem_series1(self):
        def test_impl(A):
            A[0] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_setitem_series1(self):
        def test_impl(A, i):
            A[i] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_setitem_series2(self):
        def test_impl(A, i):
            A[i] = 100
            # TODO: remove return after aliasing fix
            return A

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        hpat_func = hpat.jit(test_impl)
        hpat_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_static_getitem_series1(self):
        def test_impl(A):
            return A[0]

        n = 11
        A = pd.Series(np.arange(n))
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(A), test_impl(A))

    def test_getitem_series1(self):
        def test_impl(A, i):
            return A[i]

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_getitem_series_str1(self):
        def test_impl(A, i):
            return A[i]

        df = pd.DataFrame({'A': ['aa', 'bb', 'cc']})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_series_iat1(self):
        def test_impl(A):
            return A.iat[3]

        n = 11
        S = pd.Series(np.arange(n)**2)
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_iat2(self):
        def test_impl(A):
            A.iat[3] = 1
            return A

        n = 11
        S = pd.Series(np.arange(n)**2)
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_iloc1(self):
        def test_impl(A):
            return A.iloc[3]

        n = 11
        S = pd.Series(np.arange(n)**2)
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_iloc2(self):
        def test_impl(A):
            return A.iloc[3:8]

        n = 11
        S = pd.Series(np.arange(n)**2)
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(S), test_impl(S).reset_index(drop=True))

    def test_series_op1(self):
        def test_impl(A, i):
            return A+A

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_series_op2(self):
        def test_impl(A, i):
            return A+i

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A, 1), test_impl(df.A, 1))

    def test_series_op3(self):
        def test_impl(A, i):
            A += i
            return A

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A.copy(), 1), test_impl(df.A, 1))

    def test_series_op4(self):
        def test_impl(A):
            return A.add(A)

        n = 11
        A = pd.Series(np.arange(n))
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_op5(self):
        def test_impl(A):
            return A.pow(A)

        n = 11
        A = pd.Series(np.arange(n))
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_op6(self):
        def test_impl(A, B):
            return A.eq(B)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n)**2)
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A, B), test_impl(A, B))

    def test_series_len(self):
        def test_impl(A, i):
            return len(A)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_series_box(self):
        def test_impl():
            A = pd.Series([1,2,3])
            return A

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_box2(self):
        def test_impl():
            A = pd.Series(['1','2','3'])
            return A

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_np_typ_call_replace(self):
        # calltype replacement is tricky for np.typ() calls since variable
        # type can't provide calltype
        def test_impl(i):
            return np.int32(i)

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(1), test_impl(1))

    def test_series_ufunc1(self):
        def test_impl(A, i):
            return np.isinf(A).values

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A, 1), test_impl(df.A, 1))

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

    @unittest.skip("needs empty_like typing fix in npydecl.py")
    def test_series_empty_like(self):
        def test_impl(A):
            return np.empty_like(A)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        hpat_func = hpat.jit(test_impl)
        self.assertTrue(isinstance(hpat_func(df.A), np.ndarray))

    def test_series_fillna1(self):
        def test_impl(A):
            return A.fillna(5.0)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_fillna_str1(self):
        def test_impl(A):
            return A.fillna("dd")

        df = pd.DataFrame({'A': ['aa', 'b', None, 'ccc']})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_fillna_str_inplace1(self):
        def test_impl(A):
            A.fillna("dd", inplace=True)
            return A

        S1 = pd.Series(['aa', 'b', None, 'ccc'])
        S2 = S1.copy()
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))
        # TODO: handle string array reflection
        # hpat_func(S1)
        # test_impl(S2)
        # np.testing.assert_array_equal(S1, S2)

    def test_series_fillna_str_inplace_empty1(self):
        def test_impl(A):
            A.fillna("", inplace=True)
            return A

        S1 = pd.Series(['aa', 'b', None, 'ccc'])
        S2 = S1.copy()
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_float1(self):
        def test_impl(A):
            return A.dropna().values

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0])
        S2 = S1.copy()
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_str1(self):
        def test_impl(A):
            return A.dropna().values

        S1 = pd.Series(['aa', 'b', None, 'ccc'])
        S2 = S1.copy()
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_float_inplace1(self):
        def test_impl(A):
            A.dropna(inplace=True)
            return A.values

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0])
        S2 = S1.copy()
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_str_inplace1(self):
        def test_impl(A):
            A.dropna(inplace=True)
            return A.values

        S1 = pd.Series(['aa', 'b', None, 'ccc'])
        S2 = S1.copy()
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S1), test_impl(S2))

    def test_series_sum1(self):
        def test_impl(S):
            return S.sum()

        hpat_func = hpat.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))
        # all NA case should produce 0
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_sum2(self):
        def test_impl(S):
            return (S+S).sum()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_prod1(self):
        def test_impl(S):
            return S.prod()

        hpat_func = hpat.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))
        # all NA case should produce 1
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_count1(self):
        def test_impl(S):
            return S.count()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))
        S = pd.Series(['aa', 'bb', np.nan])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_mean1(self):
        def test_impl(S):
            return S.mean()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_var1(self):
        def test_impl(S):
            return S.var()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_min1(self):
        def test_impl(S):
            return S.min()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_max1(self):
        def test_impl(S):
            return S.max()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([np.nan, 2., 3.])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_dist_input1(self):
        def test_impl(S):
            return S.max()

        hpat_func = hpat.jit(locals={'S:input': 'distributed'})(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        start, end = get_start_end(n)
        self.assertEqual(hpat_func(S[start:end]), test_impl(S))
        self.assertEqual(count_array_REPs(), 2)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_series_tuple_input1(self):
        def test_impl(s_tup):
            return s_tup[0].max()

        hpat_func = hpat.jit(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n)+1.0)
        s_tup = (S, 1, S2)
        self.assertEqual(hpat_func(s_tup), test_impl(s_tup))

    @unittest.skip("pending handling of build_tuple in dist pass")
    def test_series_tuple_input_dist1(self):
        def test_impl(s_tup):
            return s_tup[0].max()

        hpat_func = hpat.jit(locals={'s_tup:input': 'distributed'})(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n)+1.0)
        start, end = get_start_end(n)
        s_tup = (S, 1, S2)
        h_s_tup = (S[start:end], 1, S2[start:end])
        self.assertEqual(hpat_func(h_s_tup), test_impl(s_tup))

    def test_series_rolling1(self):
        def test_impl(S):
            return S.rolling(3).sum()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_concat1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2]).values

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6., 7.])
        np.testing.assert_array_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_map1(self):
        def test_impl(S):
            return S.map(lambda a: 2*a)

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_map_tup1(self):
        def test_impl(S):
            return S.map(lambda a: (a, 2*a))

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_combine(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6.0, 21., 3.6, 5.])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_combine_float3264(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([np.float64(1), np.float64(2), np.float64(3), np.float64(4), np.float64(5)])
        S2 = pd.Series([np.float32(1), np.float32(2), np.float32(3), np.float32(4), np.float32(5)])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_combine_assert1(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([1, 2, 3])
        S2 = pd.Series([6., 21., 3., 5.])
        with self.assertRaises(AssertionError):
            hpat_func(S1, S2)

    def test_series_combine_assert2(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([6., 21., 3., 5.])
        S2 = pd.Series([1, 2, 3])
        with self.assertRaises(AssertionError):
            hpat_func(S1, S2)

    def test_series_combine_integer(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b, 16)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([1, 2, 3, 4, 5])
        S2 = pd.Series([6, 21, 3, 5])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_combine_different_types(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([6.1, 21.2, 3.3, 5.4, 6.7])
        S2 = pd.Series([1, 2, 3, 4, 5])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_combine_integer_samelen(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([1, 2, 3, 4, 5])
        S2 = pd.Series([6, 21, 17, -5, 4])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_combine_samelen(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6.0, 21., 3.6, 5., 0.0])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_combine_value(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b, 1237.56)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6.0, 21., 3.6, 5.])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_combine_value_samelen(self):
        def test_impl(S1, S2):
            return S1.combine(S2, lambda a, b: 2*a + b, 1237.56)

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([1.0, 2., 3., 4., 5.])
        S2 = pd.Series([6.0, 21., 3.6, 5., 0.0])
        pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_apply1(self):
        def test_impl(S):
            return S.apply(lambda a: 2*a)

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_abs1(self):
        def test_impl(S):
            return S.abs()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([np.nan, -2., 3.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_cov1(self):
        def test_impl(S1, S2):
            return S1.cov(S2)

        hpat_func = hpat.jit(test_impl)
        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(hpat_func(S1, S2), test_impl(S1, S2),
                                           err_msg='S1={}\nS2={}'.format(S1, S2))

    def test_series_corr1(self):
        def test_impl(S1, S2):
            return S1.corr(S2)

        hpat_func = hpat.jit(test_impl)
        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(hpat_func(S1, S2), test_impl(S1, S2),
                                           err_msg='S1={}\nS2={}'.format(S1, S2))

    def test_series_str_len1(self):
        def test_impl(S):
            return S.str.len()

        S = pd.Series(['aa', 'abc', 'c', 'cccd'])
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_append1(self):
        def test_impl(S, other):
            return S.append(other).values

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([-2., 3., 9.1])
        S2 = pd.Series([-2., 5.0])
        # Test single series
        np.testing.assert_array_equal(hpat_func(S1, S2), test_impl(S1, S2))

    def test_series_append2(self):
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3]).values

        hpat_func = hpat.jit(test_impl)
        S1 = pd.Series([-2., 3., 9.1])
        S2 = pd.Series([-2., 5.0])
        S3 = pd.Series([1.0])
        # Test series tuple
        np.testing.assert_array_equal(hpat_func(S1, S2, S3), test_impl(S1, S2, S3))

    def test_series_isna1(self):
        def test_impl(S):
            return S.isna()

        hpat_func = hpat.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2., 3.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_isnull1(self):
        def test_impl(S):
            return S.isnull()

        hpat_func = hpat.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2., 3.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_notna1(self):
        def test_impl(S):
            return S.notna()

        hpat_func = hpat.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2., 3.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_str_isna1(self):
        def test_impl(S):
            return S.isna()

        S = pd.Series(['aa', None, 'c', 'cccd'])
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_nlargest1(self):
        def test_impl(S):
            return S.nlargest(4)

        hpat_func = hpat.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_nlargest_default1(self):
        def test_impl(S):
            return S.nlargest()

        hpat_func = hpat.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_nlargest_nan1(self):
        def test_impl(S):
            return S.nlargest(4)

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_nlargest_parallel1(self):
        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.nlargest(4)

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func().values, test_impl().values)

    def test_series_nsmallest1(self):
        def test_impl(S):
            return S.nsmallest(4)

        hpat_func = hpat.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_nsmallest_default1(self):
        def test_impl(S):
            return S.nsmallest()

        hpat_func = hpat.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_nsmallest_nan1(self):
        def test_impl(S):
            return S.nsmallest(4)

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_nsmallest_parallel1(self):
        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.nsmallest(4)

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func().values, test_impl().values)

    def test_series_head1(self):
        def test_impl(S):
            return S.head(4)

        hpat_func = hpat.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_head_default1(self):
        def test_impl(S):
            return S.head()

        hpat_func = hpat.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_median1(self):
        def test_impl(S):
            return S.median()

        hpat_func = hpat.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(hpat_func(S), test_impl(S))
        S = pd.Series(np.random.ranf(m))
        self.assertEqual(hpat_func(S), test_impl(S))
        # odd size
        m = 101
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(hpat_func(S), test_impl(S))
        S = pd.Series(np.random.ranf(m))
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_median_parallel1(self):
        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.median()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_series_argsort_parallel(self):
        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.argsort().values

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_idxmin1(self):
        def test_impl(A):
            return A.idxmin()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_idxmax1(self):
        def test_impl(A):
            return A.idxmax()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_sort_values1(self):
        def test_impl(A):
            return A.sort_values()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_sort_values_parallel1(self):
        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.sort_values()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_shift_default1(self):
        def test_impl(S):
            return S.shift()

        hpat_func = hpat.jit(test_impl)
        S = pd.Series([np.nan, 2., 3., 5., np.nan, 6., 7.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

if __name__ == "__main__":
    unittest.main()

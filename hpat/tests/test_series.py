import unittest
import platform
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import hpat
from hpat.tests.test_utils import (
    count_array_REPs, count_parfor_REPs, count_array_OneDs, get_start_end)

from hpat.tests.gen_test_data import ParquetGenerator

_cov_corr_series = [(pd.Series(x), pd.Series(y)) for x, y in [
    (
        [np.nan, -2., 3., 9.1],
        [np.nan, -2., 3., 5.0],
    ),
    # TODO(quasilyte): more intricate data for complex-typed series.
    # Some arguments make assert_almost_equal fail.
    # Functions that yield mismaching results:
    # _column_corr_impl and _column_cov_impl.
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


def _make_func_from_text(func_text, func_name='test_impl'):
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars[func_name]
    return test_impl


def _make_func_use_binop1(operator):
    func_text = "def test_impl(A, B):\n"
    func_text += "   return A {} B\n".format(operator)
    return _make_func_from_text(func_text)


def _make_func_use_binop2(operator):
    func_text = "def test_impl(A, B):\n"
    func_text += "   A {} B\n".format(operator)
    func_text += "   return A\n"
    return _make_func_from_text(func_text)


def _make_func_use_method_arg1(method):
    func_text = "def test_impl(A, B):\n"
    func_text += "   return A.{}(B)\n".format(method)
    return _make_func_from_text(func_text)


GLOBAL_VAL = 2


class TestSeries(unittest.TestCase):

    @unittest.skip('AssertionError - fix needed\n'
                   '122 != 1\n'
                   'NUMA_PES=3 build')
    def test_create1(self):
        def test_impl():
            df = pd.DataFrame({'A': [1, 2, 3]})
            return (df.A == 1).sum()
        hpat_func = hpat.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())

    @unittest.skip('AssertionError - fix needed\n'
                   '122 != 1\n'
                   'NUMA_PES=3 build')
    def test_create2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)})
            return (df.A == 2).sum()
        hpat_func = hpat.jit(test_impl)

        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

    def test_create_series1(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A.values
        hpat_func = hpat.jit(test_impl)

        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_create_series_index1(self):
        # create and box an indexed Series
        def test_impl():
            A = pd.Series([1, 2, 3], ['A', 'C', 'B'])
            return A
        hpat_func = hpat.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_create_series_index2(self):
        def test_impl():
            A = pd.Series([1, 2, 3], index=['A', 'C', 'B'])
            return A
        hpat_func = hpat.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_create_series_index3(self):
        def test_impl():
            A = pd.Series([1, 2, 3], index=['A', 'C', 'B'], name='A')
            return A
        hpat_func = hpat.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_create_series_index4(self):
        def test_impl(name):
            A = pd.Series([1, 2, 3], index=['A', 'C', 'B'], name=name)
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
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_pass_df_str(self):
        def test_impl(df):
            return (df.A == 'a').sum()
        hpat_func = hpat.jit(test_impl)

        df = pd.DataFrame({'A': ['a', 'b', 'c']})
        self.assertEqual(hpat_func(df), test_impl(df))

    def test_pass_series1(self):
        # TODO: check to make sure it is series type
        def test_impl(A):
            return (A == 2).sum()
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_pass_series2(self):
        # test creating dataframe from passed series
        def test_impl(A):
            df = pd.DataFrame({'A': A})
            return (df.A == 2).sum()
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_pass_series_str(self):
        def test_impl(A):
            return (A == 'a').sum()
        hpat_func = hpat.jit(test_impl)

        df = pd.DataFrame({'A': ['a', 'b', 'c']})
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_pass_series_index1(self):
        def test_impl(A):
            return A
        hpat_func = hpat.jit(test_impl)

        S = pd.Series([3, 5, 6], ['a', 'b', 'c'], name='A')
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_attr1(self):
        def test_impl(A):
            return A.size
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_series_attr2(self):
        def test_impl(A):
            return A.copy().values
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr3(self):
        def test_impl(A):
            return A.min()
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_series_attr4(self):
        def test_impl(A):
            return A.cumsum().values
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_argsort1(self):
        def test_impl(A):
            return A.argsort()
        hpat_func = hpat.jit(test_impl)

        n = 11
        A = pd.Series(np.random.ranf(n))
        pd.testing.assert_series_equal(hpat_func(A), test_impl(A))

    def test_series_attr6(self):
        def test_impl(A):
            return A.take([2, 3]).values
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_attr7(self):
        def test_impl(A):
            return A.astype(np.float64)
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_copy_str1(self):
        def test_impl(A):
            return A.copy()
        hpat_func = hpat.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'])
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_astype_str1(self):
        def test_impl(A):
            return A.astype(str)
        hpat_func = hpat.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n))
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_astype_str2(self):
        def test_impl(A):
            return A.astype(str)
        hpat_func = hpat.jit(test_impl)

        S = pd.Series(['aa', 'bb', 'cc'])
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_np_call_on_series1(self):
        def test_impl(A):
            return np.min(A)
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_values1(self):
        def test_impl(A):
            return (A == 2).values
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        np.testing.assert_array_equal(hpat_func(df.A), test_impl(df.A))

    def test_series_shape1(self):
        def test_impl(A):
            return A.shape
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_static_setitem_series1(self):
        def test_impl(A):
            A[0] = 2
            return (A == 2).sum()
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A), test_impl(df.A))

    def test_setitem_series1(self):
        def test_impl(A, i):
            A[i] = 2
            return (A == 2).sum()
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A.copy(), 0), test_impl(df.A.copy(), 0))

    def test_setitem_series2(self):
        def test_impl(A, i):
            A[i] = 100
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        hpat_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1.values, A2.values)

    @unittest.skip("enable after remove dead in hiframes is removed")
    def test_setitem_series3(self):
        def test_impl(A, i):
            S = pd.Series(A)
            S[i] = 100
        hpat_func = hpat.jit(test_impl)

        n = 11
        A = np.arange(n)
        A1 = A.copy()
        A2 = A
        hpat_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1, A2)

    def test_setitem_series_bool1(self):
        def test_impl(A):
            A[A > 3] = 100
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        hpat_func(A1)
        test_impl(A2)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_setitem_series_bool2(self):
        def test_impl(A, B):
            A[A > 3] = B[A > 3]
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        A1 = df.A.copy()
        A2 = df.A
        hpat_func(A1, df.B)
        test_impl(A2, df.B)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_static_getitem_series1(self):
        def test_impl(A):
            return A[0]
        hpat_func = hpat.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        self.assertEqual(hpat_func(A), test_impl(A))

    def test_getitem_series1(self):
        def test_impl(A, i):
            return A[i]
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_getitem_series_str1(self):
        def test_impl(A, i):
            return A[i]
        hpat_func = hpat.jit(test_impl)

        df = pd.DataFrame({'A': ['aa', 'bb', 'cc']})
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_series_iat1(self):
        def test_impl(A):
            return A.iat[3]
        hpat_func = hpat.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n)**2)
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_iat2(self):
        def test_impl(A):
            A.iat[3] = 1
            return A
        hpat_func = hpat.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_iloc1(self):
        def test_impl(A):
            return A.iloc[3]
        hpat_func = hpat.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n)**2)
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_series_iloc2(self):
        def test_impl(A):
            return A.iloc[3:8]
        hpat_func = hpat.jit(test_impl)

        n = 11
        S = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(
            hpat_func(S), test_impl(S).reset_index(drop=True))

    def test_series_op1(self):
        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')
        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = hpat.jit(test_impl)

            n = 11
            df = pd.DataFrame({'A': np.arange(1, n), 'B': np.ones(n - 1)})
            pd.testing.assert_series_equal(hpat_func(df.A, df.B), test_impl(df.A, df.B), check_names=False)

    @unittest.skipIf(platform.system() == 'Windows', "error on windows")
    def test_series_op2(self):
        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')

        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = hpat.jit(test_impl)

            n = 11
            df = pd.DataFrame({'A': np.arange(1, n)})
            pd.testing.assert_series_equal(hpat_func(df.A, 1), test_impl(df.A, 1), check_names=False)

    def test_series_op3(self):
        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')

        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop2(operator)
            hpat_func = hpat.jit(test_impl)

            n = 11
            df = pd.DataFrame({'A': np.arange(1, n), 'B': np.ones(n - 1)})
            pd.testing.assert_series_equal(hpat_func(df.A, df.B), test_impl(df.A, df.B), check_names=False)

    def test_series_op4(self):
        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')

        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop2(operator)
            hpat_func = hpat.jit(test_impl)

            n = 11
            df = pd.DataFrame({'A': np.arange(1, n)})
            pd.testing.assert_series_equal(hpat_func(df.A, 1), test_impl(df.A, 1), check_names=False)

    def test_series_op5(self):
        arithmetic_methods = ('add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow')

        for method in arithmetic_methods:
            test_impl = _make_func_use_method_arg1(method)
            hpat_func = hpat.jit(test_impl)

            n = 11
            df = pd.DataFrame({'A': np.arange(1, n), 'B': np.ones(n - 1)})
            pd.testing.assert_series_equal(hpat_func(df.A, df.B), test_impl(df.A, df.B), check_names=False)

    def test_series_op6(self):
        def test_impl(A):
            return -A
        hpat_func = hpat.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        pd.testing.assert_series_equal(hpat_func(A), test_impl(A))

    def test_series_op7(self):
        comparison_binops = ('<', '>', '<=', '>=', '!=', '==')

        for operator in comparison_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = hpat.jit(test_impl)

            n = 11
            A = pd.Series(np.arange(n))
            B = pd.Series(np.arange(n)**2)
            pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_names=False)

    def test_series_op8(self):
        comparison_methods = ('lt', 'gt', 'le', 'ge', 'ne', 'eq')

        for method in comparison_methods:
            test_impl = _make_func_use_method_arg1(method)
            hpat_func = hpat.jit(test_impl)

            n = 11
            A = pd.Series(np.arange(n))
            B = pd.Series(np.arange(n)**2)
            pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_names=False)

    def test_series_inplace_binop_array(self):
        def test_impl(A, B):
            A += B
            return A
        hpat_func = hpat.jit(test_impl)

        n = 11
        A = np.arange(n)**2.0  # TODO: use 2 for test int casting
        B = pd.Series(np.ones(n))
        np.testing.assert_array_equal(hpat_func(A.copy(), B), test_impl(A, B))

    @unittest.skipIf(platform.system() == 'Windows', "error on windows")
    def test_series_fusion1(self):
        def test_impl(A, B):
            return A + B + 1
        hpat_func = hpat.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 1)

    @unittest.skipIf(platform.system() == 'Windows', "error on windows")
    def test_series_fusion2(self):
        # make sure getting data var avoids incorrect single def assumption
        def test_impl(A, B):
            S = B + 2
            if A[0] == 0:
                S = A + 1
            return S + B
        hpat_func = hpat.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 3)

    def test_series_len(self):
        def test_impl(A, i):
            return len(A)
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertEqual(hpat_func(df.A, 0), test_impl(df.A, 0))

    def test_series_box(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A
        hpat_func = hpat.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_box2(self):
        def test_impl():
            A = pd.Series(['1', '2', '3'])
            return A
        hpat_func = hpat.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    @unittest.skip("ERROR: Segmentation fault on the second launch (using HPAT_REPEAT_TEST_NUMBER=2)")
    def test_series_list_str_unbox1(self):
        def test_impl(A):
            return A.iloc[0]
        hpat_func = hpat.jit(test_impl)

        S = pd.Series([['aa', 'b'], ['ccc'], []])
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

        # call twice to test potential refcount errors
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

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
        hpat_func = hpat.jit(test_impl)

        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
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
        hpat_func = hpat.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n)})
        self.assertTrue(isinstance(hpat_func(df.A), np.ndarray))

    def test_series_fillna1(self):
        def test_impl(A):
            return A.fillna(5.0)
        hpat_func = hpat.jit(test_impl)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        pd.testing.assert_series_equal(hpat_func(df.A),
                                       test_impl(df.A), check_names=False)

    # test inplace fillna for named numeric series (obtained from DataFrame)
    def test_series_fillna_inplace1(self):
        def test_impl(A):
            A.fillna(5.0, inplace=True)
            return A
        hpat_func = hpat.jit(test_impl)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        pd.testing.assert_series_equal(hpat_func(df.A),
                                       test_impl(df.A), check_names=False)

    def test_series_fillna_str1(self):
        def test_impl(A):
            return A.fillna("dd")
        hpat_func = hpat.jit(test_impl)

        df = pd.DataFrame({'A': ['aa', 'b', None, 'ccc']})
        pd.testing.assert_series_equal(hpat_func(df.A),
                                       test_impl(df.A), check_names=False)

    def test_series_fillna_str_inplace1(self):
        def test_impl(A):
            A.fillna("dd", inplace=True)
            return A
        hpat_func = hpat.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'ccc'])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))
        # TODO: handle string array reflection
        # hpat_func(S1)
        # test_impl(S2)
        # np.testing.assert_array_equal(S1, S2)

    def test_series_fillna_str_inplace_empty1(self):
        def test_impl(A):
            A.fillna("", inplace=True)
            return A
        hpat_func = hpat.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'ccc'])
        S2 = S1.copy()
        pd.testing.assert_series_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_float1(self):
        def test_impl(A):
            return A.dropna().values
        hpat_func = hpat.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0])
        S2 = S1.copy()
        np.testing.assert_array_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_str1(self):
        def test_impl(A):
            return A.dropna().values
        hpat_func = hpat.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'ccc'])
        S2 = S1.copy()
        np.testing.assert_array_equal(hpat_func(S1), test_impl(S2))

    @unittest.skip("TODO: fix result")
    def test_series_dropna_str_parallel1(self):
        def test_impl(A):
            B = A.dropna()
            return (B == 'gg').sum()
        hpat_func = hpat.jit(distributed=['A'])(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'ccc', 'dd', 'gg'])
        start, end = get_start_end(len(S1))
        # TODO: gatherv
        self.assertEqual(hpat_func(S1[start:end]), test_impl(S1))

    def test_series_dropna_float_inplace1(self):
        def test_impl(A):
            A.dropna(inplace=True)
            return A.values
        hpat_func = hpat.jit(test_impl)

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0])
        S2 = S1.copy()
        np.testing.assert_array_equal(hpat_func(S1), test_impl(S2))

    def test_series_dropna_str_inplace1(self):
        def test_impl(A):
            A.dropna(inplace=True)
            return A.values
        hpat_func = hpat.jit(test_impl)

        S1 = pd.Series(['aa', 'b', None, 'ccc'])
        S2 = S1.copy()
        np.testing.assert_array_equal(hpat_func(S1), test_impl(S2))

    @unittest.skip('numba.errors.TypingError - fix needed\n'
                   'Failed in hpat mode pipeline'
                   '(step: convert to distributed)\n'
                   'Invalid use of Function(<built-in function len>)'
                   'with argument(s) of type(s): (none)\n')
    def test_series_rename1(self):
        def test_impl(A):
            return A.rename('B')
        hpat_func = hpat.jit(test_impl)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        pd.testing.assert_series_equal(hpat_func(df.A), test_impl(df.A))

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

    @unittest.skip('AssertionError - fix needed\n'
                   '5 != 2\n'
                   'NUMA_PES=3 build')
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

    def test_series_value_counts(self):
        def test_impl(S):
            return S.value_counts()
        hpat_func = hpat.jit(test_impl)

        S = pd.Series(['AA', 'BB', 'C', 'AA', 'C', 'AA'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @unittest.skip('AssertionError - fix needed\n'
                   '61 != 110\n'
                   'NUMA_PES=3 build')
    def test_series_dist_input1(self):
        def test_impl(S):
            return S.max()
        hpat_func = hpat.jit(distributed={'S'})(test_impl)

        n = 111
        S = pd.Series(np.arange(n))
        start, end = get_start_end(n)
        self.assertEqual(hpat_func(S[start:end]), test_impl(S))
        self.assertEqual(count_array_REPs(), 0)
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

    def test_series_map_global1(self):
        def test_impl(S):
            return S.map(lambda a: a + GLOBAL_VAL)
        hpat_func = hpat.jit(test_impl)

        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_map_tup1(self):
        def test_impl(S):
            return S.map(lambda a: (a, 2*a))
        hpat_func = hpat.jit(test_impl)

        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_map_tup_map1(self):
        def test_impl(S):
            A = S.map(lambda a: (a, 2*a))
            return A.map(lambda a: a[1])
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

        S1 = pd.Series([np.float64(1), np.float64(2),
                        np.float64(3), np.float64(4), np.float64(5)])
        S2 = pd.Series([np.float32(1), np.float32(2),
                        np.float32(3), np.float32(4), np.float32(5)])
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

        S = pd.Series([np.nan, -2., 3., 0.5E-01, 0xFF, 0o7, 0b101])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @unittest.skip('AssertionError - fix needed\n'
                   'Arrays are not almost equal to 7 decimals\n'
                   'ACTUAL: 4.166666666666667\n'
                   'DESIRED: 12.5\n'
                   'NUMA_PES=3 build')
    def test_series_cov1(self):
        def test_impl(S1, S2):
            return S1.cov(S2)
        hpat_func = hpat.jit(test_impl)

        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(
                hpat_func(S1, S2), test_impl(S1, S2),
                err_msg='S1={}\nS2={}'.format(S1, S2))

    @unittest.skip('AssertionError - fix needed\n'
                   'Arrays are not almost equal to 7 decimals\n'
                   'ACTUAL: 0.9539980920057239\n'
                   'DESIRED: 1.0\n'
                   'NUMA_PES=3 build')
    def test_series_corr1(self):
        def test_impl(S1, S2):
            return S1.corr(S2)
        hpat_func = hpat.jit(test_impl)

        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(
                hpat_func(S1, S2), test_impl(S1, S2),
                err_msg='S1={}\nS2={}'.format(S1, S2))

    def test_series_str_len1(self):
        def test_impl(S):
            return S.str.len()
        hpat_func = hpat.jit(test_impl)

        S = pd.Series(['aa', 'abc', 'c', 'cccd'])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_str2str(self):
        str2str_methods = ('capitalize', 'lower', 'lstrip', 'rstrip',
                           'strip', 'swapcase', 'title', 'upper')
        for method in str2str_methods:
            func_text = "def test_impl(S):\n"
            func_text += "  return S.str.{}()\n".format(method)
            test_impl = _make_func_from_text(func_text)
            hpat_func = hpat.jit(test_impl)

            S = pd.Series([' \tbbCD\t ', 'ABC', ' mCDm\t', 'abc'])
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
        np.testing.assert_array_equal(hpat_func(S1, S2, S3),
                                      test_impl(S1, S2, S3))

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
        hpat_func = hpat.jit(test_impl)

        S = pd.Series(['aa', None, 'c', 'cccd'])
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

    @unittest.skip('AssertionError - fix needed\n'
                   'Arrays are not equal\n'
                   'Mismatch: 100%\n'
                   'Max absolute difference: 0.04361003\n'
                   'Max relative difference: 9.04840049\n'
                   'x: array([0.04843 , 0.05106 , 0.057625, 0.0671  ])\n'
                   'y: array([0.00482 , 0.04843 , 0.05106 , 0.057625])\n'
                   'NUMA_PES=3 build')
    def test_series_nlargest_parallel1(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

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

    @unittest.skip('AssertionError - fix needed\n'
                   'Arrays are not equal\n'
                   'Mismatch: 50%\n'
                   'Max absolute difference: 0.01813261\n'
                   'Max relative difference: 0.50757593\n'
                   'x: array([0.007431, 0.024095, 0.035724, 0.053857])\n'
                   'y: array([0.007431, 0.024095, 0.031374, 0.035724])\n'
                   'NUMA_PES=3 build')
    def test_series_nsmallest_parallel1(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.nsmallest(4)
        hpat_func = hpat.jit(test_impl)

        np.testing.assert_array_equal(hpat_func().values, test_impl().values)

    @unittest.skip('numba.errors.TypingError - fix needed\n'
                   'Failed in hpat mode pipeline'
                   '(step: convert to distributed)\n'
                   'Invalid use of Function(<built-in function len>)'
                   'with argument(s) of type(s): (none)\n')
    def test_series_head1(self):
        def test_impl(S):
            return S.head(4)
        hpat_func = hpat.jit(test_impl)

        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    @unittest.skip('numba.errors.TypingError - fix needed\n'
                   'Failed in hpat mode pipeline'
                   '(step: convert to distributed)\n'
                   'Invalid use of Function(<built-in function len>)'
                   'with argument(s) of type(s): (none)\n')
    def test_series_head_default1(self):
        def test_impl(S):
            return S.head()
        hpat_func = hpat.jit(test_impl)

        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(hpat_func(S).values, test_impl(S).values)

    def test_series_head_index1(self):
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], [8, 1, 6, 0, 9, 1, 3])
            return S.head(3)
        hpat_func = hpat.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    def test_series_head_index2(self):
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5],
                          ['a', 'ab', 'abc', 'c', 'f', 'hh', ''])
            return S.head(3)
        hpat_func = hpat.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    @unittest.skip(
    '''Skipped as it corrupts memmory and causes failures of other tests
    while running with NUM_PES=3 and at least TestSeries and TestBasic suites together.
    Exact commands to reproduce:
        mpiexec -n 3 python -W ignore -u -m unittest -v $SUITES $SUITES
        where SUITES="hpat.tests.TestBasic hpat.tests.TestSeries"
    Test failures occur on the second suite run only.
    Exact errors:
         1. Segmentation fault in TestBasic.test_rebalance
         2. FAIL in TestBasic.test_astype with following error message:
             test_astype (hpat.tests.test_basic.TestBasic) ...
             Fatal error in MPI_Allreduce: Message truncated, error stack:
             MPI_Allreduce(907)..................: MPI_Allreduce(sbuf=0x7ffe3b734128, rbuf=0x7ffe3b734120, count=1,
                MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD) failed
             MPIR_Allreduce_impl(764)............:
             MPIR_Allreduce_intra(238)...........:
             MPIR_Reduce_impl(1070)..............:
             MPIR_Reduce_intra(878)..............:
             MPIR_Reduce_binomial(186)...........:
             MPIC_Recv(353)......................:
             MPIDI_CH3U_Request_unpack_uebuf(568): Message truncated; 40 bytes received but buffer size is 8
             MPIR_Allreduce_intra(268)...........:
             MPIR_Bcast_impl(1452)...............:
             MPIR_Bcast(1476)....................:
             MPIR_Bcast_intra(1287)..............:
             MPIR_Bcast_binomial(310)............: Failure during collective
             Fatal error in MPI_Allreduce: Other MPI error, error stack'''
    )
    def test_series_head_index_parallel1(self):
        def test_impl(S):
            return S.head(3)
        hpat_func = hpat.jit(distributed={'S'})(test_impl)

        S = pd.Series([6, 9, 2, 3, 6, 4, 5],
                      ['a', 'ab', 'abc', 'c', 'f', 'hh', ''])
        start, end = get_start_end(len(S))
        pd.testing.assert_series_equal(hpat_func(S[start:end]), test_impl(S))
        self.assertTrue(count_array_OneDs() > 0)

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

    @unittest.skip('AssertionError - fix needed\n'
                   'nan != 0.45894510159707225\n'
                   'NUMA_PES=3 build')
    def test_series_median_parallel1(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.median()
        hpat_func = hpat.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())

    def test_series_argsort_parallel(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

        def test_impl():
            df = pq.read_table('kde.parquet').to_pandas()
            S = df.points
            return S.argsort().values
        hpat_func = hpat.jit(test_impl)

        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_idxmin1(self):
        def test_impl(A):
            return A.idxmin()
        hpat_func = hpat.jit(test_impl)

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_idxmax1(self):
        def test_impl(A):
            return A.idxmax()
        hpat_func = hpat.jit(test_impl)

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        np.testing.assert_array_equal(hpat_func(S), test_impl(S))

    def test_series_sort_values1(self):
        def test_impl(A):
            return A.sort_values()
        hpat_func = hpat.jit(test_impl)

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_sort_values_index1(self):
        def test_impl(A, B):
            S = pd.Series(A, B)
            return S.sort_values()
        hpat_func = hpat.jit(test_impl)

        n = 11
        np.random.seed(0)
        # TODO: support passing Series with Index
        # S = pd.Series(np.random.ranf(n), np.random.randint(0, 100, n))
        A = np.random.ranf(n)
        B = np.random.ranf(n)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))

    def test_series_sort_values_parallel1(self):
        # create `kde.parquet` file
        ParquetGenerator.gen_kde_pq()

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

    def test_series_index1(self):
        def test_impl():
            A = pd.Series([1, 2, 3], index=['A', 'C', 'B'])
            return A.index

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_index2(self):
        def test_impl():
            A = pd.Series([1, 2, 3], index=[0, 1, 2])
            return A.index

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    @unittest.skip("Enable after fixing distributed for get_series_index")
    def test_series_index3(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A.index

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_iterator_int(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([3, 2, 1, 5, 4])
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_float(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([0.3, 0.2222, 0.1756, 0.005, 0.4])
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_boolean(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([True, False])
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_string(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series(['a', 'ab', 'abc', '', 'dddd'])
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    def test_series_iterator_one_value(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([5])
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    @unittest.skip("Fails when NUMA_PES>=2 due to unimplemented sync of such construction after distribution")
    def test_series_iterator_no_param(self):
        def test_impl():
            A = pd.Series([3, 2, 1, 5, 4])
            return [i for i in A]

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_series_iterator_empty(self):
        def test_impl(A):
            return [i for i in A]

        A = pd.Series([np.int64(x) for x in range(0)])
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))


if __name__ == "__main__":
    unittest.main()

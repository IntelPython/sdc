import unittest
import pandas as pd
import numpy as np
import itertools
import numba
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                        count_parfor_OneDs, count_array_OneDs, dist_IR_contains)


class TestBasic(unittest.TestCase):
    def test_getitem(self):
        def test_impl(N):
            A = np.ones(N)
            B = np.ones(N) > .5
            C = A[B]
            return C.sum()

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_getitem_multidim(self):
        def test_impl(N):
            A = np.ones((N, 3))
            B = np.ones(N) > .5
            C = A[B, 2]
            return C.sum()

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_assert(self):
        # make sure assert in an inlined function works
        def g(a):
            assert a==0

        hpat_g = hpat.jit(g)
        def f():
            hpat_g(0)

        hpat_f = hpat.jit(f)
        hpat_f()

    def test_reduce(self):
        dtypes = ['float32', 'float64', 'int32', 'int64']
        funcs = ['sum', 'prod', 'min', 'max']
        for (dtype, func) in itertools.product(dtypes, funcs):
            func_text = """def f(n):
                A = np.ones(n, dtype=np.{})
                return A.{}()
            """.format(dtype, func)
            loc_vars = {}
            exec(func_text, {'np': np}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = hpat.jit(test_impl)
            n = 128
            np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)

if __name__ == "__main__":
    unittest.main()

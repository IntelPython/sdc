import unittest
import pandas as pd
import numpy as np
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs, dist_IR_contains)

class TestHiFrames(unittest.TestCase):
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

    def test_filter(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.ones(n)})
            df1 = df[df.A > .5]
            return np.sum(df1.B)

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_rolling1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df.A.rolling(5).sum()
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_rolling2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            df['moving average'] = df.A.rolling(window=5, center=True).mean()
            return df['moving average'].sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
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
        n = 11
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

if __name__ == "__main__":
    unittest.main()

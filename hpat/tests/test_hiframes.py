import unittest
import pandas as pd
import numpy as np
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs)

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

if __name__ == "__main__":
    unittest.main()

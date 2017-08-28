import unittest
import pandas as pd
import numpy as np
import hpat

class TestHiFrames(unittest.TestCase):
    def test_cumsum(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df.A.cumsum()
            return Ac.sum()

        hpat_func = hpat.jit(test_impl)
        n = 11
        self.assertEqual(hpat_func(n), test_impl(n))

if __name__ == "__main__":
    unittest.main()

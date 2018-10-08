import unittest
import pandas as pd
import numpy as np
import hpat
import warnings

class TestDataFrameErrors(unittest.TestCase):
    def test_get_unknown_attr(self):
        def test_impl():
            df = pd.DataFrame({'A': [0, 0]})
            A = df['A'] # OK, known column
            field = df.loc # OK, known field
            method = df.apply # OK, known method
            something = df.non_existing # Should warn

        with warnings.catch_warnings(record=True) as w:
            hpat.jit(test_impl)()
            self.assertIn(
                "unknown attribute df.non_existing accessed",
                str(w[0].message))

    def test_get_unimplemented_attr(self):
        def test_impl():
            df = pd.DataFrame({'A': [1.0, 2.0]})
            ndim = df.ndim

        with self.assertRaises(NotImplementedError) as err:
            hpat.jit(test_impl)()
        self.assertIn("data frame attribute ndim not implemented yet", str(err.exception))

    def test_call_unimplemented_method(self):
        def test_impl():
            df = pd.DataFrame({'A': [0.1, 0.2], 'B': [0.3]})
            return df.filter(items=['A'])

        with self.assertRaises(NotImplementedError) as err:
            hpat.jit(test_impl)()
        self.assertIn("data frame function filter not implemented yet", str(err.exception))

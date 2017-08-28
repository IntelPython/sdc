import unittest
import pandas as pd
import numpy as np
import numba
import h5py
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs, dist_IR_contains)


class TestML(unittest.TestCase):
    def test_logistic_regression(self):
        def test_impl(n, d):
            iterations = 3
            X = np.ones((n,d))+.5
            Y = np.ones(n)
            D = X.shape[1]
            w = np.ones(D)-0.5
            for i in range(iterations):
                w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X,w))) - 1.0) * Y), X)
            return w

        hpat_func = hpat.jit(test_impl)
        n = 11
        d = 4
        np.testing.assert_allclose(hpat_func(n, d), test_impl(n, d))
        self.assertEqual(count_array_OneDs(), 3)
        self.assertEqual(count_parfor_OneDs(), 3)

    def test_logistic_regression_acc(self):
        def test_impl(N, D):
            iterations = 3
            g = 2 * np.ones(D) - 1
            X = 2 * np.ones((N, D)) - 1
            Y = (np.dot(X, g) > 0.0) == (np.ones(N) > .90)

            w = 2 * np.ones(D) - 1
            for i in range(iterations):
                w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
                #R = np.dot(X,w) > 0.0
                #accuracy = np.sum(R == Y) / N
            return w

        hpat_func = hpat.jit(test_impl)
        n = 11
        d = 4
        #np.testing.assert_allclose(hpat_func(n, d), test_impl(n, d))
        #self.assertEqual(count_array_OneDs(), 3)
        #self.assertEqual(count_parfor_OneDs(), 3)

if __name__ == "__main__":
    unittest.main()

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

    def test_linear_regression(self):
        def test_impl(N, D):
            p = 2
            iterations = 3
            X = np.ones((N, D)) + .5
            Y = np.ones((N, p))
            alphaN = 0.01 / N
            w = np.zeros((D, p))
            for i in range(iterations):
                w -= alphaN * np.dot(X.T, np.dot(X, w) - Y)
            return w

        hpat_func = hpat.jit(test_impl)
        n = 11
        d = 4
        np.testing.assert_allclose(hpat_func(n, d), test_impl(n, d))
        self.assertEqual(count_array_OneDs(), 5)
        self.assertEqual(count_parfor_OneDs(), 3)

    def test_kde(self):
        def test_impl(n):
            X = np.ones(n)
            b = 0.5
            points = np.array([-1.0, 2.0, 5.0])
            N = points.shape[0]
            exps = 0
            for i in hpat.prange(n):
                p = X[i]
                d = (-(p - points)**2) / (2 * b**2)
                m = np.min(d)
                exps += m - np.log(b * N)+np.log(np.sum(np.exp(d - m)))
            return exps

        hpat_func = hpat.jit(test_impl)
        n = 11
        np.testing.assert_approx_equal(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_OneDs(), 1)
        self.assertEqual(count_parfor_OneDs(), 2)

if __name__ == "__main__":
    unittest.main()

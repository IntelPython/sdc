import unittest
import pandas as pd
import numpy as np
import itertools
import numba
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                   count_parfor_OneDs, count_array_OneDs,
                                   count_array_OneD_Vars, dist_IR_contains)


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

    def test_setitem1(self):
        def test_impl(N):
            A = np.arange(10)+1.0
            A[0] = 30
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_setitem2(self):
        def test_impl(N):
            A = np.arange(10)+1.0
            A[0:4] = 30
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_astype(self):
        def test_impl(N):
            return np.ones(N).astype(np.int32).sum()

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shape(self):
        def test_impl(N):
            return np.ones(N).shape[0]

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)


    def test_inplace_binop(self):
        def test_impl(N):
            A = np.ones(N)
            B = np.ones(N)
            B += A
            return B.sum()

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

    def test_whole_slice(self):
        def test_impl(N):
            X = np.ones((N, 4))
            X[:,3] = (X[:,3]) / (np.max(X[:,3]) - np.min(X[:,3]))
            return X.sum()

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
        import sys
        dtypes = ['float32', 'float64', 'int32', 'int64']
        funcs = ['sum', 'prod', 'min', 'max', 'argmin', 'argmax']
        for (dtype, func) in itertools.product(dtypes, funcs):
            # loc allreduce doesn't support int64 on windows
            if (sys.platform.startswith('win') and dtype=='int64'
                                            and func in ['argmin', 'argmax']):
                continue
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

    def test_array_reduce(self):
        def test_impl(N):
            A = np.ones(3);
            B = np.ones(3);
            for i in numba.prange(N):
                A += B
            return A

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_OneDs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

    def test_dist_return(self):
        def test_impl(N):
            A = np.arange(N);
            return A

        hpat_func = hpat.jit(locals={'A:return': 'distributed'})(test_impl)
        n = 128
        dist_sum = hpat.jit(
            lambda a: hpat.distributed_api.dist_reduce(
                a, np.int32(hpat.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1)  # run to compile
        np.testing.assert_allclose(
            dist_sum(hpat_func(n).sum()), test_impl(n).sum())
        self.assertEqual(count_array_OneDs(), 1)
        self.assertEqual(count_parfor_OneDs(), 1)

    def test_dist_return_tuple(self):
        def test_impl(N):
            A = np.arange(N);
            B = np.arange(N)+1.5;
            return A, B

        hpat_func = hpat.jit(locals={'A:return': 'distributed',
                                     'B:return': 'distributed'})(test_impl)
        n = 128
        dist_sum = hpat.jit(
            lambda a: hpat.distributed_api.dist_reduce(
                a, np.int32(hpat.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1.0)  # run to compile
        np.testing.assert_allclose(
            dist_sum((hpat_func(n)[0] + hpat_func(n)[1]).sum()),
                    (test_impl(n)[0] + test_impl(n)[1]).sum())
        self.assertEqual(count_array_OneDs(), 2)
        self.assertEqual(count_parfor_OneDs(), 2)

    def test_dist_input(self):
        def test_impl(A):
            return len(A)

        hpat_func = hpat.jit(locals={'A:input': 'distributed'})(test_impl)
        n = 128
        arr = np.ones(n)
        n_pes = hpat.jit(lambda: hpat.distributed_api.get_size())()
        np.testing.assert_allclose(hpat_func(arr) / n_pes, test_impl(arr))
        self.assertEqual(count_array_OneD_Vars(), 1)

    def test_rebalance(self):
        def test_impl(N):
            A = np.arange(n)
            B = A[A>10]
            C = hpat.distributed_api.rebalance_array(B)
            return C.sum()

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_OneDs(), 3)
        self.assertEqual(count_parfor_OneDs(), 2)

    def test_rebalance_loop(self):
        def test_impl(N):
            A = np.arange(n)
            B = A[A>10]
            s = 0
            for i in range(3):
                s += B.sum()
            return s

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_OneDs(), 4)
        self.assertEqual(count_parfor_OneDs(), 2)
        self.assertIn('allgather', list(hpat_func.inspect_llvm().values())[0])

if __name__ == "__main__":
    unittest.main()

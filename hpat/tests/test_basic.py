import unittest
import pandas as pd
import numpy as np
import itertools
import numba
import hpat
import random
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                   count_parfor_OneDs, count_array_OneDs,
                                   count_array_OneD_Vars, dist_IR_contains)

def get_np_state_ptr():
    return numba._helperlib.rnd_get_np_state_ptr()

def _copy_py_state(r, ptr):
    """
    Copy state of Python random *r* to Numba state *ptr*.
    """
    mt = r.getstate()[1]
    ints, index = mt[:-1], mt[-1]
    numba._helperlib.rnd_set_state(ptr, (index, list(ints)))
    return ints, index

class BaseTest(unittest.TestCase):

    def _follow_cpython(self, ptr, seed=2):
        r = random.Random(seed)
        _copy_py_state(r, ptr)
        return r


class TestBasic(BaseTest):
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

        # def test_impl(N):
        #     return np.ones((N, 3, 4)).shape
        #
        # hpat_func = hpat.jit(test_impl)
        # n = 128
        # np.testing.assert_allclose(hpat_func(n), test_impl(n))
        # self.assertEqual(count_array_REPs(), 0)
        # self.assertEqual(count_parfor_REPs(), 0)


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

    def test_strided_getitem(self):
        def test_impl(N):
            A = np.ones(N)
            B = A[::7]
            return B.sum()

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
        self.assertEqual(count_array_OneDs(), 1)

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

    def test_transpose(self):
        def test_impl(n):
            A = np.ones((30, 40, 50))
            B = A.transpose((0, 2, 1))
            C = A.transpose(0, 2, 1)
            return B.sum() + C.sum()

        hpat_func = hpat.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_permuted_array_indexing(self):

        # Since Numba uses Python's PRNG for producing random numbers in NumPy,
        # we cannot compare against NumPy.  Therefore, we implement permutation
        # in Python.
        def python_permutation(n, r):
            arr = np.arange(n)
            r.shuffle(arr)
            return arr

        def test_impl(n):
            A, B = np.arange(n), np.arange(n)
            P = np.random.permutation(n)
            A, B = A[P], B[P]
            return A, B

        def python_impl(n, r):
            A, B = np.arange(n), np.arange(n)
            P = python_permutation(n, r)
            A, B = A[P], B[P]
            return A, B

        # Ideally, in above functions we should just call np.random.seed() and
        # they should be producing the same sequence of random numbers.
        # However, since Numba's PRNG uses NumPy's initialization method for
        # initializing PRNG, we cannot just call seed.  Instead, we resort to
        # this hack that generates a Python Random object with a fixed seed and
        # copies the state to Numba's internal NumPy PRNG state.  For details of
        # this mess, see https://github.com/numba/numba/issues/2782.
        r = self._follow_cpython(get_np_state_ptr())

        hpat_func = hpat.jit(locals={'A:return': 'distributed',
                                     'B:return': 'distributed'})(test_impl)
        rank = hpat.jit(lambda: hpat.distributed_api.get_rank())()
        num_ranks = hpat.jit(lambda: hpat.distributed_api.get_size())()

        rank_begin_func = hpat.jit(
            lambda arr_len, num_ranks, rank: hpat.distributed_api.get_start(
                arr_len, np.int32(num_ranks), np.int32(rank)))

        rank_end_func = hpat.jit(
            lambda arr_len, num_ranks, rank: hpat.distributed_api.get_end(
                arr_len, np.int32(num_ranks), np.int32(rank)))

        for arr_len in [11, 111, 128, 120]:
            begin = rank_begin_func(arr_len, num_ranks, rank)
            end = rank_end_func(arr_len, num_ranks, rank)
            hpat_A, hpat_B = hpat_func(arr_len)
            python_A, python_B = python_impl(arr_len, r)
            np.testing.assert_allclose(hpat_A, python_A[begin:end])
            np.testing.assert_allclose(hpat_B, python_B[begin:end])

if __name__ == "__main__":
    unittest.main()

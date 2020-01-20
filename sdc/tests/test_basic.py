# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


import itertools
import numpy as np
import pandas as pd
import random
import unittest

import numba
from numba import types

import sdc
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                   count_parfor_OneDs, count_array_OneDs, count_array_OneD_Vars,
                                   dist_IR_contains, get_rank, get_start_end, check_numba_version,
                                   skip_numba_jit, skip_sdc_jit)


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


class BaseTest(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = self.jit(lambda: sdc.distributed_api.get_rank())()
        self.num_ranks = self.jit(lambda: sdc.distributed_api.get_size())()

    def _rank_begin(self, arr_len):
        f = self.jit(
            lambda arr_len, num_ranks, rank: sdc.distributed_api.get_start(
                arr_len, np.int32(num_ranks), np.int32(rank)))
        return f(arr_len, self.num_ranks, self.rank)

    def _rank_end(self, arr_len):
        f = self.jit(
            lambda arr_len, num_ranks, rank: sdc.distributed_api.get_end(
                arr_len, np.int32(num_ranks), np.int32(rank)))
        return f(arr_len, self.num_ranks, self.rank)

    def _rank_bounds(self, arr_len):
        return self._rank_begin(arr_len), self._rank_end(arr_len)

    def _follow_cpython(self, ptr, seed=2):
        r = random.Random(seed)
        _copy_py_state(r, ptr)
        return r


class TestBasic(BaseTest):

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_getitem(self):
        def test_impl(N):
            A = np.ones(N)
            B = np.ones(N) > .5
            C = A[B]
            return C.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_setitem1(self):
        def test_impl(N):
            A = np.arange(10) + 1.0
            A[0] = 30
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_setitem2(self):
        def test_impl(N):
            A = np.arange(10) + 1.0
            A[0:4] = 30
            return A.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_astype(self):
        def test_impl(N):
            return np.ones(N).astype(np.int32).sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shape(self):
        def test_impl(N):
            return np.ones(N).shape[0]

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

        # def test_impl(N):
        #     return np.ones((N, 3, 4)).shape
        #
        # hpat_func = self.jit(test_impl)
        # n = 128
        # np.testing.assert_allclose(hpat_func(n), test_impl(n))
        # self.assertEqual(count_array_REPs(), 0)
        # self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_inplace_binop(self):
        def test_impl(N):
            A = np.ones(N)
            B = np.ones(N)
            B += A
            return B.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_getitem_multidim(self):
        def test_impl(N):
            A = np.ones((N, 3))
            B = np.ones(N) > .5
            C = A[B, 2]
            return C.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_whole_slice(self):
        def test_impl(N):
            X = np.ones((N, 4))
            X[:, 3] = (X[:, 3]) / (np.max(X[:, 3]) - np.min(X[:, 3]))
            return X.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit("hang with numba.jit. ok with sdc.jit")
    def test_strided_getitem(self):
        def test_impl(N):
            A = np.ones(N)
            B = A[::7]
            return B.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_assert(self):
        # make sure assert in an inlined function works

        def g(a):
            assert a == 0

        hpat_g = self.jit(g)

        def f():
            hpat_g(0)

        hpat_f = self.jit(f)
        hpat_f()

    @skip_numba_jit
    def test_inline_locals(self):
        # make sure locals in inlined function works
        @self.jit(locals={'B': types.float64[:]})
        def g(S):
            B = pd.to_numeric(S, errors='coerce')
            return B

        def f():
            return g(pd.Series(['1.2']))

        pd.testing.assert_series_equal(self.jit(f)(), f())

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_reduce(self):
        import sys
        dtypes = ['float32', 'float64', 'int32', 'int64']
        funcs = ['sum', 'prod', 'min', 'max', 'argmin', 'argmax']
        for (dtype, func) in itertools.product(dtypes, funcs):
            # loc allreduce doesn't support int64 on windows
            if (sys.platform.startswith('win')
                    and dtype == 'int64'
                    and func in ['argmin', 'argmax']):
                continue
            func_text = """def f(n):
                A = np.arange(0, n, 1, np.{})
                return A.{}()
            """.format(dtype, func)
            loc_vars = {}
            exec(func_text, {'np': np}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = self.jit(test_impl)
            n = 21  # XXX arange() on float32 has overflow issues on large n
            np.testing.assert_almost_equal(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_reduce2(self):
        import sys
        dtypes = ['float32', 'float64', 'int32', 'int64']
        funcs = ['sum', 'prod', 'min', 'max', 'argmin', 'argmax']
        for (dtype, func) in itertools.product(dtypes, funcs):
            # loc allreduce doesn't support int64 on windows
            if (sys.platform.startswith('win')
                    and dtype == 'int64'
                    and func in ['argmin', 'argmax']):
                continue
            func_text = """def f(A):
                return A.{}()
            """.format(func)
            loc_vars = {}
            exec(func_text, {'np': np}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = self.jit(locals={'A:input': 'distributed'})(test_impl)
            n = 21
            start, end = get_start_end(n)
            np.random.seed(0)
            A = np.random.randint(0, 10, n).astype(dtype)
            np.testing.assert_almost_equal(
                hpat_func(A[start:end]), test_impl(A), decimal=3)
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_reduce_filter1(self):
        import sys
        dtypes = ['float32', 'float64', 'int32', 'int64']
        funcs = ['sum', 'prod', 'min', 'max', 'argmin', 'argmax']
        for (dtype, func) in itertools.product(dtypes, funcs):
            # loc allreduce doesn't support int64 on windows
            if (sys.platform.startswith('win')
                    and dtype == 'int64'
                    and func in ['argmin', 'argmax']):
                continue
            func_text = """def f(A):
                A = A[A>5]
                return A.{}()
            """.format(func)
            loc_vars = {}
            exec(func_text, {'np': np}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = self.jit(locals={'A:input': 'distributed'})(test_impl)
            n = 21
            start, end = get_start_end(n)
            np.random.seed(0)
            A = np.random.randint(0, 10, n).astype(dtype)
            np.testing.assert_almost_equal(
                hpat_func(A[start:end]), test_impl(A), decimal=3,
                err_msg="{} on {}".format(func, dtype))
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    @skip_sdc_jit
    def test_array_reduce(self):
        binops = ['+=', '*=', '+=', '*=', '|=', '|=']
        dtypes = ['np.float32', 'np.float32', 'np.float64', 'np.float64', 'np.int32', 'np.int64']
        for (op, typ) in zip(binops, dtypes):
            func_text = """def f(n):
                  A = np.arange(0, 10, 1, {})
                  B = np.arange(0 +  3, 10 + 3, 1, {})
                  for i in numba.prange(n):
                      A {} B
                  return A
            """.format(typ, typ, op)
            loc_vars = {}
            exec(func_text, {'np': np, 'numba': numba}, loc_vars)
            test_impl = loc_vars['f']

            hpat_func = self.jit(test_impl)
            n = 128
            np.testing.assert_allclose(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_OneDs(), 0)
            self.assertEqual(count_parfor_OneDs(), 1)

    @unittest.expectedFailure  # https://github.com/numba/numba/issues/4690
    def test_dist_return(self):
        def test_impl(N):
            A = np.arange(N)
            return A

        hpat_func = self.jit(locals={'A:return': 'distributed'})(test_impl)
        n = 128
        dist_sum = self.jit(
            lambda a: sdc.distributed_api.dist_reduce(
                a, np.int32(sdc.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1)  # run to compile
        np.testing.assert_allclose(
            dist_sum(hpat_func(n).sum()), test_impl(n).sum())
        self.assertEqual(count_array_OneDs(), 1)
        self.assertEqual(count_parfor_OneDs(), 1)

    @unittest.expectedFailure # https://github.com/numba/numba/issues/4690
    def test_dist_return_tuple(self):
        def test_impl(N):
            A = np.arange(N)
            B = np.arange(N) + 1.5
            return A, B

        hpat_func = self.jit(locals={'A:return': 'distributed',
                                     'B:return': 'distributed'})(test_impl)
        n = 128
        dist_sum = self.jit(
            lambda a: sdc.distributed_api.dist_reduce(
                a, np.int32(sdc.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1.0)  # run to compile
        np.testing.assert_allclose(
            dist_sum((hpat_func(n)[0] + hpat_func(n)[1]).sum()), (test_impl(n)[0] + test_impl(n)[1]).sum())
        self.assertEqual(count_array_OneDs(), 2)
        self.assertEqual(count_parfor_OneDs(), 2)

    @skip_numba_jit
    def test_dist_input(self):
        def test_impl(A):
            return len(A)

        hpat_func = self.jit(distributed=['A'])(test_impl)
        n = 128
        arr = np.ones(n)
        np.testing.assert_allclose(hpat_func(arr) / self.num_ranks, test_impl(arr))
        self.assertEqual(count_array_OneDs(), 1)

    @unittest.expectedFailure  # https://github.com/numba/numba/issues/4690
    def test_rebalance(self):
        def test_impl(N):
            A = np.arange(n)
            B = A[A > 10]
            C = sdc.distributed_api.rebalance_array(B)
            return C.sum()

        try:
            sdc.distributed_analysis.auto_rebalance = True
            hpat_func = self.jit(test_impl)
            n = 128
            np.testing.assert_allclose(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_OneDs(), 3)
            self.assertEqual(count_parfor_OneDs(), 2)
        finally:
            sdc.distributed_analysis.auto_rebalance = False

    @unittest.expectedFailure  # https://github.com/numba/numba/issues/4690
    def test_rebalance_loop(self):
        def test_impl(N):
            A = np.arange(n)
            B = A[A > 10]
            s = 0
            for i in range(3):
                s += B.sum()
            return s

        try:
            sdc.distributed_analysis.auto_rebalance = True
            hpat_func = self.jit(test_impl)
            n = 128
            np.testing.assert_allclose(hpat_func(n), test_impl(n))
            self.assertEqual(count_array_OneDs(), 4)
            self.assertEqual(count_parfor_OneDs(), 2)
            self.assertIn('allgather', list(hpat_func.inspect_llvm().values())[0])
        finally:
            sdc.distributed_analysis.auto_rebalance = False

    @skip_numba_jit("Failed in nopython mode pipeline (step: Preprocessing for parfors)")
    def test_transpose(self):
        def test_impl(n):
            A = np.ones((30, 40, 50))
            B = A.transpose((0, 2, 1))
            C = A.transpose(0, 2, 1)
            return B.sum() + C.sum()

        hpat_func = self.jit(test_impl)
        n = 128
        np.testing.assert_allclose(hpat_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip("Numba's perfmute generation needs to use np seed properly")
    def test_permuted_array_indexing(self):

        # Since Numba uses Python's PRNG for producing random numbers in NumPy,
        # we cannot compare against NumPy.  Therefore, we implement permutation
        # in Python.
        def python_permutation(n, r):
            arr = np.arange(n)
            r.shuffle(arr)
            return arr

        def test_one_dim(arr_len):
            A = np.arange(arr_len)
            B = np.copy(A)
            P = np.random.permutation(arr_len)
            A, B = A[P], B[P]
            return A, B

        # Implementation that uses Python's PRNG for producing a permutation.
        # We test against this function.
        def python_one_dim(arr_len, r):
            A = np.arange(arr_len)
            B = np.copy(A)
            P = python_permutation(arr_len, r)
            A, B = A[P], B[P]
            return A, B

        # Ideally, in above *_impl functions we should just call
        # np.random.seed() and they should produce the same sequence of random
        # numbers.  However, since Numba's PRNG uses NumPy's initialization
        # method for initializing PRNG, we cannot just set seed.  Instead, we
        # resort to this hack that generates a Python Random object with a fixed
        # seed and copies the state to Numba's internal NumPy PRNG state.  For
        # details please see https://github.com/numba/numba/issues/2782.
        r = self._follow_cpython(get_np_state_ptr())

        hpat_func1 = self.jit(locals={'A:return': 'distributed',
                                      'B:return': 'distributed'})(test_one_dim)

        # Test one-dimensional array indexing.
        for arr_len in [11, 111, 128, 120]:
            hpat_A, hpat_B = hpat_func1(arr_len)
            python_A, python_B = python_one_dim(arr_len, r)
            rank_bounds = self._rank_bounds(arr_len)
            np.testing.assert_allclose(hpat_A, python_A[slice(*rank_bounds)])
            np.testing.assert_allclose(hpat_B, python_B[slice(*rank_bounds)])

        # Test two-dimensional array indexing.  Like in one-dimensional case
        # above, in addition to NumPy version that is compiled by Numba, we
        # implement a Python version.
        def test_two_dim(arr_len):
            first_dim = arr_len // 2
            A = np.arange(arr_len).reshape(first_dim, 2)
            B = np.copy(A)
            P = np.random.permutation(first_dim)
            A, B = A[P], B[P]
            return A, B

        def python_two_dim(arr_len, r):
            first_dim = arr_len // 2
            A = np.arange(arr_len).reshape(first_dim, 2)
            B = np.copy(A)
            P = python_permutation(first_dim, r)
            A, B = A[P], B[P]
            return A, B

        hpat_func2 = self.jit(locals={'A:return': 'distributed',
                                      'B:return': 'distributed'})(test_two_dim)

        for arr_len in [18, 66, 128]:
            hpat_A, hpat_B = hpat_func2(arr_len)
            python_A, python_B = python_two_dim(arr_len, r)
            rank_bounds = self._rank_bounds(arr_len // 2)
            np.testing.assert_allclose(hpat_A, python_A[slice(*rank_bounds)])
            np.testing.assert_allclose(hpat_B, python_B[slice(*rank_bounds)])

        # Test that the indexed array is not modified if it is not being
        # assigned to.
        def test_rhs(arr_len):
            A = np.arange(arr_len)
            B = np.copy(A)
            P = np.random.permutation(arr_len)
            C = A[P]
            return A, B, C

        hpat_func3 = self.jit(locals={'A:return': 'distributed',
                                      'B:return': 'distributed',
                                      'C:return': 'distributed'})(test_rhs)

        for arr_len in [15, 23, 26]:
            A, B, _ = hpat_func3(arr_len)
            np.testing.assert_allclose(A, B)


if __name__ == "__main__":
    unittest.main()

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
import numba
import numpy as np
import pandas as pd
import random
import unittest
from numba import types

import sdc
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (get_rank,
                                  get_start_end,
                                  skip_numba_jit,
                                  skip_sdc_jit)


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


if __name__ == "__main__":
    unittest.main()

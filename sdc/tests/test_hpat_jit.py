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

import numba
import numpy as np
import pandas as pd
import platform
import unittest
from collections import defaultdict
from numba.typed import Dict

import sdc
from sdc import *
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import skip_numba_jit


class TestHpatJitIssues(TestCase):

    @unittest.skip("Dict is not supported as class member")
    def test_class_with_dict(self):
        @jitclass([('d', Dict)])
        class ClassWithDict:
            def __init__(self):
                self.d = Dict.empty(key_type=int32, value_type=int32)

        @numba.njit
        def test_impl():
            c = ClassWithDict()

            c.d[0] = 1

            return c.d[0]

        test_impl()

    @unittest.skip("Type infer from __init__ is not supported")
    def test_class_from_init(self):
        @jitclass()
        class ClassWithInt:
            def __init__(self):
                self.i = 0

        @numba.njit
        def test_impl():
            c = ClassWithInt()

            print(c.i)

        test_impl()

    @unittest.skip("list.sort with lambda is not supported")
    def test_list_sort_lambda(self):
        @numba.njit
        def sort_with_list_and_lambda():
            data = [5, 4, 3, 2, 1, 0]

            data.sort(key=lambda x: x)

            return data

        sort_with_list_and_lambda()

    @unittest.skip("list.sort with key is not supported")
    def test_list_sort_with_func(self):
        @numba.njit
        def key_func(x):
            return x

        @numba.njit
        def sort_with_list():
            data = [5, 4, 3, 2, 1, 0]

            data.sort(key=key_func)

            return data

        sort_with_list()

    @unittest.skip("sorted with lambda is not supported")
    def test_sorted_lambda(self):
        @numba.njit
        def sorted_with_list():
            data = [5, 4, 3, 2, 1, 0]

            sorted(data, key=lambda x: x)

            return data

        sorted_with_list()

    @unittest.skip("sorted with key is not supported")
    def test_sorted_with_func(self):
        @numba.njit
        def key_func(x):
            return x

        @numba.njit
        def sorted_with_list():
            data = [5, 4, 3, 2, 1, 0]

            sorted(data, key=key_func)

            return data

        sorted_with_list()

    @unittest.skip("iterate over tuple is not supported")
    def test_iterate_over_tuple(self):
        @numba.njit
        def func_iterate_over_tuple():
            t = ('1', 1, 1.)

            for i in t:
                print(i)

        func_iterate_over_tuple()

    @unittest.skip("try/except is not supported")
    def test_with_try_except(self):
        @numba.njit
        def func_with_try_except():
            try:
                return 0
            except BaseException:
                return 1

        func_with_try_except()

    @unittest.skip("raise is not supported")
    def test_with_raise(self):
        @numba.njit
        def func_with_raise(b):
            if b:
                return b
            else:
                raise "error"

        func_with_raise(True)

    @unittest.skip("defaultdict is not supported")
    def test_default_dict(self):
        @numba.njit
        def func_with_dict():
            d = defaultdict(int)

            return d['a']

        func_with_dict()


if __name__ == "__main__":
    unittest.main()

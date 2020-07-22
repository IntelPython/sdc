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
import unittest

from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import skip_numba_jit


DATA = [1.0, 2., 3., 4., 5.]
INDEX = [5, 4, 3, 2, 1]
NAME = "sname"


def series_apply_square_usecase(S):

    def square(x):
        return x ** 2

    return S.apply(square)


class TestSeries_apply(TestCase):

    def test_series_apply(self):
        test_impl = series_apply_square_usecase
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_apply_convert_type(self):
        def test_impl(S):
            def to_int(x):
                return int(x)
            return S.apply(to_int)
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_apply_index(self):
        test_impl = series_apply_square_usecase
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA, INDEX)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_apply_name(self):
        test_impl = series_apply_square_usecase
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA, name=NAME)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_apply_lambda(self):
        def test_impl(S):
            return S.apply(lambda a: 2 * a)
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_apply_np(self):
        def test_impl(S):
            return S.apply(np.log)
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_apply_register_jitable(self):
        @numba.extending.register_jitable
        def square(x):
            return x ** 2

        def test_impl(S):
            return S.apply(square)
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_numba_jit("'args' in apply is not supported")
    def test_series_apply_args(self):
        @numba.extending.register_jitable
        def subtract_custom_value(x, custom_value):
            return x - custom_value

        def test_impl(S):
            return S.apply(subtract_custom_value, args=(5,))
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_numba_jit("'kwds' in apply is not supported")
    def test_series_apply_kwds(self):
        @numba.extending.register_jitable
        def add_custom_values(x, **kwargs):
            for month in kwargs:
                x += kwargs[month]
            return x

        def test_impl(S):
            return S.apply(add_custom_values, june=30, july=20, august=25)
        hpat_func = self.jit(test_impl)

        S = pd.Series(DATA)
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))


if __name__ == "__main__":
    unittest.main()

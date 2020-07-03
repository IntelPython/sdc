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

import pandas as pd
import unittest

from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import skip_numba_jit, skip_sdc_jit


GLOBAL_VAL = 2


class TestSeries_map(TestCase):

    def test_series_map1(self):
        def test_impl(S):
            return S.map(lambda a: 2 * a)
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    def test_series_map_global1(self):
        def test_impl(S):
            return S.map(lambda a: a + GLOBAL_VAL)
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_numba_jit
    def test_series_map_tup1(self):
        def test_impl(S):
            return S.map(lambda a: (a, 2 * a))
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_numba_jit
    def test_series_map_tup_map1(self):
        def test_impl(S):
            A = S.map(lambda a: (a, 2 * a))
            return A.map(lambda a: a[1])
        hpat_func = self.jit(test_impl)

        S = pd.Series([1.0, 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))

    @skip_sdc_jit
    def test_series_map_dict(self):
        def test_impl(S):
            return S.map({2.: 42., 4.: 3.14})
        hpat_func = self.jit(test_impl)

        S = pd.Series([1., 2., 3., 4., 5.])
        pd.testing.assert_series_equal(hpat_func(S), test_impl(S))


if __name__ == "__main__":
    unittest.main()

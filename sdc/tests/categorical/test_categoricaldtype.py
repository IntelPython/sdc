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

from sdc.tests.test_base import TestCase

import pandas as pd
import numba as nb

from sdc.datatypes.categorical.types import CategoricalDtypeType


class CategoricalDtypeTest(TestCase):

    def _pd_dtype(self, ordered=True):
        return pd.CategoricalDtype(categories=['b', 'a'], ordered=ordered)

    def test_typeof(self):
        pd_dtype = self._pd_dtype()
        nb_dtype = nb.typeof(pd_dtype)

        assert(isinstance(nb_dtype, CategoricalDtypeType))
        assert(nb_dtype.categories == list(pd_dtype.categories))
        assert(nb_dtype.ordered == pd_dtype.ordered)

    def test_unboxing(self):
        @nb.njit
        def func(c):
            pass

        pd_dtype = self._pd_dtype()
        func(pd_dtype)

    def test_boxing(self):
        @nb.njit
        def func(c):
            return c

        pd_dtype = self._pd_dtype()
        boxed = func(pd_dtype)
        assert(boxed == pd_dtype)

    def test_lowering(self):
        pd_dtype = self._pd_dtype()

        @nb.njit
        def func():
            return pd_dtype

        boxed = func()
        assert(boxed == pd_dtype)

    def test_constructor(self):
        @nb.njit
        def func():
            return pd.CategoricalDtype(categories=('b', 'a'), ordered=True)

        boxed = func()
        assert(boxed == self._pd_dtype())

    def test_constructor_categories_list(self):
        @nb.njit
        def func():
            return pd.CategoricalDtype(categories=['b', 'a'], ordered=True)

        boxed = func()
        assert(boxed == self._pd_dtype())

    def test_constructor_categories_set(self):
        @nb.njit
        def func():
            return pd.CategoricalDtype(categories={'b', 'a'}, ordered=True)

        boxed = func()
        assert(boxed == self._pd_dtype())

    def test_constructor_no_order(self):
        @nb.njit
        def func():
            return pd.CategoricalDtype(categories=('b', 'a'))

        boxed = func()
        assert(boxed == self._pd_dtype(ordered=False))

    def test_constructor_no_categories(self):
        @nb.njit
        def func():
            return pd.CategoricalDtype()

        boxed = func()
        expected = pd.CategoricalDtype(ordered=None)
        assert(boxed == expected)
        assert(boxed.categories == expected.categories)
        assert(boxed.ordered == expected.ordered)

    def test_attribute_ordered(self):
        @nb.njit
        def func(c):
            return c.ordered

        pd_dtype = self._pd_dtype()
        ordered = func(pd_dtype)
        assert(ordered == pd_dtype.ordered)


if __name__ == "__main__":
    import unittest
    unittest.main()

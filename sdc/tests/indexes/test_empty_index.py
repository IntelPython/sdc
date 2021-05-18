# -*- coding: utf-8 -*-
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


class TestEmptyIndex(TestCase):
    """ Verifies basic support for empty DF and using special EmptyIndexType
        for respresnting it's index """

    def test_create_empty_df(self):
        def test_impl():
            df = pd.DataFrame({}, index=None)
            return len(df)
        sdc_func = self.jit(test_impl)

        result = sdc_func()
        result_ref = test_impl()
        self.assertEqual(result, result_ref)

    def test_unbox_empty_df(self):
        def test_impl(df):
            return len(df)
        sdc_func = self.jit(test_impl)

        df = pd.DataFrame({}, index=None)
        result = sdc_func(df)
        result_ref = test_impl(df)
        self.assertEqual(result, result_ref)

    def test_box_empty_df(self):
        def test_impl():
            df = pd.DataFrame({}, index=None)
            return df
        sdc_func = self.jit(test_impl)

        result = sdc_func()
        result_ref = test_impl()
        pd.testing.assert_frame_equal(result, result_ref)

    def test_empty_df_round_trip(self):
        def test_impl(df):
            return df
        sdc_func = self.jit(test_impl)

        df = pd.DataFrame({}, index=None)
        result = sdc_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_frame_equal(result, result_ref)

    def test_empty_df_unboxed_get_index_1(self):
        def test_impl(df):
            return df.index
        sdc_func = self.jit(test_impl)

        df = pd.DataFrame({}, index=None)
        result = sdc_func(df)
        result_ref = test_impl(df)
        pd.testing.assert_index_equal(result, result_ref)

    def test_empty_df_unboxed_get_index_2(self):

        def py_func(df):
            return df.index

        @self.jit
        def sdc_func(df):
            return df._index

        df = pd.DataFrame({}, index=None)
        result = sdc_func(df)
        result_ref = py_func(df)
        pd.testing.assert_index_equal(result, result_ref)

    def test_empty_df_created_get_index_1(self):
        def test_impl():
            df = pd.DataFrame({}, index=None)
            return df.index
        sdc_func = self.jit(test_impl)

        result = sdc_func()
        result_ref = test_impl()
        pd.testing.assert_index_equal(result, result_ref)

    def test_empty_df_created_get_index_2(self):

        def py_func():
            df = pd.DataFrame({}, index=None)
            return df.index

        @self.jit
        def sdc_func():
            df = pd.DataFrame({}, index=None)
            return df._index

        result = sdc_func()
        result_ref = py_func()
        pd.testing.assert_index_equal(result, result_ref)


if __name__ == "__main__":
    unittest.main()

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

import numpy as np
import pandas as pd
import unittest

from sdc.tests.indexes import TestRangeIndex, TestInt64Index
from sdc.tests.indexes.index_datagens import _generate_index_param_values


class TestIndexes(
        TestRangeIndex,
        TestInt64Index
        ):
    """ This suite combines tests from all concrete index-type suites and also adds
    tests for common use-cases that need to be checked for all index-types. """

    def assert_indexes_equal(self, index1, index2):
        # for SDC indexes that are represented with arrays (e.g. Uint64Index)
        supported_pandas_indexes = (pd.RangeIndex, pd.Int64Index, )
        if (not isinstance(index1, supported_pandas_indexes)
                or not isinstance(index2, supported_pandas_indexes)):
            index1 = np.asarray(index1)
            index2 = np.asarray(index2)
            np.testing.assert_array_equal(index1, index2)
        else:
            pd.testing.assert_index_equal(index1, index2)

    @unittest.skip("TODO: support boxing/unboxing and parent ref for Python ranges in Numba")
    def test_indexes_unbox_data_id_check(self):
        def test_impl(index):
            return index
        sdc_func = self.jit(test_impl)

        n = 11
        indexes_to_test = [
            pd.RangeIndex(n, name='abc'),  # only this one fails, other pass
            pd.Int64Index(np.arange(n), name='abc'),
        ]
        data_attr_names_map = {
            pd.RangeIndex: '_range',
            pd.Int64Index: '_data',
        }

        for index in indexes_to_test:
            with self.subTest(index_type=type(index)):
                result = sdc_func(index)
                result_ref = test_impl(index)

                data1, data2, data3 = map(
                    lambda x: getattr(x, data_attr_names_map[type(x)]),
                    [index, result, result_ref]
                )
                self.assertIs(data1, data3)
                self.assertIs(data2, data3)

    @unittest.skip("Needs writable native struct type members in Numba")
    def test_indexes_named_set_name(self):
        def test_impl(index):
            index.name = 'def'
            return index
        sdc_func = self.jit(test_impl)

        n = 11
        indexes_to_test = [
            pd.RangeIndex(n, name='abc'),
            pd.Int64Index(np.arange(n), name='abc'),
        ]

        for index in indexes_to_test:
            with self.subTest(index_type=type(index)):
                index1 = index.copy(deep=True)
                index2 = index.copy(deep=True)
                result = sdc_func(index1)
                result_ref = test_impl(index2)
                pd.testing.assert_index_equal(result, result_ref)

    @unittest.skip("Needs writable native struct type members and single common type for name")
    def test_indexes_unnamed_set_name(self):
        def test_impl(index):
            index.name = 'def'
            return index
        sdc_func = self.jit(test_impl)

        n = 11
        indexes_to_test = [
            pd.RangeIndex(n),
            pd.Int64Index(np.arange(n)),
        ]

        for index in indexes_to_test:
            with self.subTest(index_type=type(index)):
                index1 = index.copy(deep=True)
                index2 = index.copy(deep=True)
                result = sdc_func(index1)
                result_ref = test_impl(index2)
                pd.testing.assert_index_equal(result, result_ref)

    @unittest.skip("Need support unboxing pandas indexes with parent ref")
    def test_indexes_operator_is_unbox(self):
        def test_impl(index1, index2):
            return index1 is index2
        sdc_func = self.jit(test_impl)

        indexes_to_test = [
            pd.RangeIndex(1, 21, 3),
            pd.Int64Index([1, 2, 3, 5, 6, 3, 4]),
        ]

        for index in indexes_to_test:
            # positive testcase
            with self.subTest(subtest="same indexes"):
                index1 = index.copy(deep=True)
                index2 = index1
                result = sdc_func(index1, index2)
                result_ref = test_impl(index1, index2)
                self.assertEqual(result, result_ref)
                self.assertEqual(result, True)

            # negative testcase
            with self.subTest(subtest="not same indexes"):
                index1 = index.copy(deep=True)
                index2 = index.copy(deep=True)
                result = sdc_func(index1, index2)
                result_ref = test_impl(index1, index2)
                self.assertEqual(result, result_ref)
                self.assertEqual(result, False)

    def test_indexes_unbox_series_with_index(self):
        @self.jit
        def test_impl(S):
            # TO-DO: this actually includes calling 'index' attribute overload, should really be S._index,
            # but this requires separate type (e.g. DefaultIndexType) instead of types.none as default index
            return S.index

        n = 11
        for index in _generate_index_param_values(n):
            expected_res = pd.RangeIndex(n) if index is None else index
            with self.subTest(series_index=index):
                S = pd.Series(np.ones(n), index=index)
                result = test_impl(S)
                self.assert_indexes_equal(result, expected_res)

    def test_indexes_create_series_with_index(self):
        @self.jit
        def test_impl(data, index):
            S = pd.Series(data=data, index=index)
            return S.index

        n = 11
        series_data = np.ones(n)
        for index in _generate_index_param_values(n):
            expected_res = pd.RangeIndex(n) if index is None else index
            with self.subTest(series_index=index):
                result = test_impl(series_data, index)
                self.assert_indexes_equal(result, expected_res)

    def test_indexes_box_series_with_index(self):
        def test_impl(data, index):
            return pd.Series(data=data, index=index)
        sdc_func = self.jit(test_impl)

        n = 11
        series_data = np.ones(n)
        for index in _generate_index_param_values(n):
            with self.subTest(series_index=index):
                result = sdc_func(series_data, index)
                result_ref = test_impl(series_data, index)
                pd.testing.assert_series_equal(result, result_ref)

    def test_indexes_index_get_series_index(self):
        def test_impl(S):
            return S.index
        sdc_func = self.jit(test_impl)

        n = 11
        for index in _generate_index_param_values(n):
            with self.subTest(series_index=index):
                S = pd.Series(np.ones(n), index=index)
                result = sdc_func(S)
                result_ref = test_impl(S)
                self.assert_indexes_equal(result, result_ref)

    def test_indexes_index_unbox_df_with_index(self):
        @self.jit
        def test_impl(df):
            # TO-DO: this actually includes calling 'index' attribute overload, should really be df._index,
            # but this requires separate type (e.g. DefaultIndexType) instead of types.none as default index
            return df.index

        n = 11
        for index in _generate_index_param_values(n):
            expected_res = pd.RangeIndex(n) if index is None else index
            with self.subTest(df_index=index):
                df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)}, index=index)
                result = test_impl(df)
                self.assert_indexes_equal(result, expected_res)

    def test_indexes_index_create_df_with_index(self):
        @self.jit
        def test_impl(A, B, index):
            df = pd.DataFrame({'A': A, 'B': B}, index=index)
            return df.index

        n = 11
        A, B = np.ones(n), np.arange(n)
        for index in _generate_index_param_values(n):
            expected_res = pd.RangeIndex(n) if index is None else index
            with self.subTest(df_index=index):
                result = test_impl(A, B, index)
                self.assert_indexes_equal(result, expected_res)

    def test_indexes_index_box_df_with_index(self):
        def test_impl(A, B, index):
            return pd.DataFrame({'A': A, 'B': B}, index=index)
        sdc_func = self.jit(test_impl)

        n = 11
        A, B = np.ones(n), np.arange(n, dtype=np.intp)
        for index in _generate_index_param_values(n):
            with self.subTest(df_index=index):
                result = sdc_func(A, B, index)
                result_ref = test_impl(A, B, index)
                pd.testing.assert_frame_equal(result, result_ref)

    def test_indexes_index_get_df_index(self):
        def test_impl(df):
            return df.index
        sdc_func = self.jit(test_impl)

        n = 11
        for index in _generate_index_param_values(n):
            with self.subTest(df_index=index):
                df = pd.DataFrame({'A': np.ones(n)}, index=index)
                result = sdc_func(df)
                result_ref = test_impl(df)
                self.assert_indexes_equal(result, result_ref)


if __name__ == "__main__":
    unittest.main()

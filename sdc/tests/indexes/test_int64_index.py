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
from itertools import (combinations_with_replacement, product, )

from sdc.tests.indexes.index_datagens import (
    test_global_index_names,
    _generate_valid_int64_index_data,
    _generate_int64_indexes_fixed,
    )
from sdc.tests.test_base import TestCase


class TestInt64Index(TestCase):

    def test_int64_index_create_and_box(self):
        def test_impl(data, name):
            return pd.Int64Index(data, name=name)
        sdc_func = self.jit(test_impl)

        name = 'index'
        for data in _generate_valid_int64_index_data():
            with self.subTest(index_data=data):
                result = sdc_func(data, name)
                result_ref = test_impl(data, name)
                pd.testing.assert_index_equal(result, result_ref)

    def test_int64_index_unbox_and_box(self):
        def test_impl(index):
            return index
        sdc_func = self.jit(test_impl)

        n = 11
        for index in _generate_int64_indexes_fixed(n):
            with self.subTest(index=index):
                result = sdc_func(index)
                result_ref = test_impl(index)
                pd.testing.assert_index_equal(result, result_ref)

    def test_int64_index_create_param_copy_true(self):
        def test_impl(arr):
            return pd.Int64Index(arr, copy=True)
        sdc_func = self.jit(test_impl)

        index_data_to_test = [
            np.array([1, 2, 3, 5, 6, 3, 4], dtype=np.int64),
            list(np.array([1, 2, 3, 5, 6, 3, 4], dtype=np.int64)),
            pd.RangeIndex(11),
            pd.Int64Index([1, 2, 3, 5, 6, 3, 4]),
        ]

        for index_data in index_data_to_test:
            with self.subTest(index_data=index_data):
                result = sdc_func(index_data)
                result_ref = test_impl(index_data)
                pd.testing.assert_index_equal(result, result_ref)
                self.assertEqual(result._data is result_ref._data, False)

    def test_int64_index_create_param_copy_default(self):
        def test_impl(arr):
            return pd.Int64Index(arr)
        sdc_func = self.jit(test_impl)

        # only test data that has underlying array that can be referenced
        # and ensure it has int64 dtype as otherwise there will always be a copy
        index_data_to_test = [
            np.array([1, 2, 3, 5, 6, 3, 4], dtype=np.int64),
            pd.Int64Index([1, 2, 3, 5, 6, 3, 4]),
        ]

        for index_data in index_data_to_test:
            with self.subTest(index_data=index_data):
                result = sdc_func(index_data)
                result_ref = test_impl(index_data)
                pd.testing.assert_index_equal(result, result_ref)
                self.assertEqual(result._data is result_ref._data, True)

    def test_int64_index_create_param_dtype(self):
        def test_impl(n, dtype):
            return pd.Int64Index(np.arange(n), dtype=dtype)
        sdc_func = self.jit(test_impl)

        n = 11
        supported_dtypes = [None, np.int64, 'int64', np.int32, 'int32']
        for dtype in supported_dtypes:
            with self.subTest(dtype=dtype):
                result = sdc_func(n, dtype)
                result_ref = test_impl(n, dtype)
                pd.testing.assert_index_equal(result, result_ref)

    def test_int64_index_create_param_dtype_invalid(self):
        def test_impl(n, dtype):
            return pd.Int64Index(np.arange(n), dtype=dtype)
        sdc_func = self.jit(test_impl)

        n = 11
        invalid_dtypes = ['float', 'uint']
        for dtype in invalid_dtypes:
            with self.subTest(dtype=dtype):
                with self.assertRaises(Exception) as context:
                    test_impl(n, dtype)
                pandas_exception = context.exception

                with self.assertRaises(type(pandas_exception)) as context:
                    sdc_func(n, dtype)
                sdc_exception = context.exception
                self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_int64_index_attribute_dtype(self):
        def test_impl(index):
            return index.dtype
        sdc_func = self.jit(test_impl)

        n = 11
        index = pd.Int64Index(np.arange(n) * 2)
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    def test_int64_index_attribute_name(self):
        def test_impl(index):
            return index.name
        sdc_func = self.jit(test_impl)

        n = 11
        index_data = np.arange(n) * 2
        for name in test_global_index_names:
            with self.subTest(name=name):
                index = pd.Int64Index(index_data, name=name)
                result = sdc_func(index)
                result_ref = test_impl(index)
                self.assertEqual(result, result_ref)

    def test_int64_index_len(self):
        def test_impl(index):
            return len(index)
        sdc_func = self.jit(test_impl)

        n = 11
        index = pd.Int64Index(np.arange(n) * 2, name='index')
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    def test_int64_index_attribute_values(self):
        def test_impl(index):
            return index.values
        sdc_func = self.jit(test_impl)

        for data in _generate_valid_int64_index_data():
            index = pd.Int64Index(data)
            with self.subTest(index_data=data):
                result = sdc_func(index)
                result_ref = test_impl(index)
                np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_contains(self):
        def test_impl(index, value):
            return value in index
        sdc_func = self.jit(test_impl)

        index = pd.Int64Index([1, 11, 2])
        values_to_test = [-5, 15, 1, 11, 5, 6]
        for value in values_to_test:
            with self.subTest(value=value):
                result = sdc_func(index, value)
                result_ref = test_impl(index, value)
                np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_copy(self):
        def test_impl(index, new_name):
            return index.copy(name=new_name)
        sdc_func = self.jit(test_impl)

        for data in _generate_valid_int64_index_data():
            for name, new_name in product(test_global_index_names, repeat=2):
                index = pd.Int64Index(data, name=name)
                with self.subTest(index=index, new_name=new_name):
                    result = sdc_func(index, new_name)
                    result_ref = test_impl(index, new_name)
                    pd.testing.assert_index_equal(result, result_ref)

    def test_int64_index_copy_param_deep(self):
        def test_impl(index, deep):
            return index.copy(deep=deep)
        sdc_func = self.jit(test_impl)

        index = pd.Int64Index([1, 11, 2])
        for deep in [True, False]:
            with self.subTest(deep=deep):
                result = sdc_func(index, deep)
                result_ref = test_impl(index, deep)
                pd.testing.assert_index_equal(result, result_ref)
                # pandas uses ndarray views when copies index, so for python
                # case check that data arrays share the same memory
                self.assertEqual(
                    result._data is index._data,
                    result_ref._data.base is index._data
                )

    def test_int64_index_getitem_scalar(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        for data in _generate_valid_int64_index_data():
            index = pd.Int64Index(data)
            n = len(index)
            values_to_test = [-n, n // 2, n - 1]
            for idx in values_to_test:
                with self.subTest(index=index, idx=idx):
                    result = sdc_func(index, idx)
                    result_ref = test_impl(index, idx)
                    self.assertEqual(result, result_ref)

    def test_int64_index_getitem_scalar_idx_bounds(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        n = 11
        index = pd.Int64Index(np.arange(n) * 2, name='abc')
        values_to_test = [-(n + 1), n]
        for idx in values_to_test:
            with self.subTest(idx=idx):
                with self.assertRaises(Exception) as context:
                    test_impl(index, idx)
                pandas_exception = context.exception

                with self.assertRaises(type(pandas_exception)) as context:
                    sdc_func(index, idx)
                sdc_exception = context.exception
                self.assertIsInstance(sdc_exception, type(pandas_exception))
                self.assertIn("out of bounds", str(sdc_exception))

    def test_int64_index_getitem_slice(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        index_len = 11
        slices_params = combinations_with_replacement(
            [None, 0, -1, index_len // 2, index_len, index_len - 3, index_len + 3, -(index_len + 3)],
            3
        )

        for data in _generate_valid_int64_index_data():
            for slice_start, slice_stop, slice_step in slices_params:
                # slice step cannot be zero
                if slice_step == 0:
                    continue

                idx = slice(slice_start, slice_stop, slice_step)
                index = pd.Int64Index(data, name='abc')
                with self.subTest(index=index, idx=idx):
                    result = sdc_func(index, idx)
                    result_ref = test_impl(index, idx)
                    pd.testing.assert_index_equal(result, result_ref)

    def test_int64_index_iterator_1(self):
        def test_impl(index):
            res = []
            for i, label in enumerate(index):
                res.append((i, label))
            return res
        sdc_func = self.jit(test_impl)

        index = pd.Int64Index([5, 3, 2, 1, 7, 4])
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    def test_int64_index_iterator_2(self):
        def test_impl(index):
            res = []
            for label in index:
                if not label % 2:
                    res.append(label)
            return res
        sdc_func = self.jit(test_impl)

        index = pd.Int64Index([5, 3, 2, 1, 7, 4])
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    def test_int64_index_nparray(self):
        def test_impl(index):
            return np.array(index)
        sdc_func = self.jit(test_impl)

        n = 11
        index = pd.Int64Index(np.arange(n) * 2)
        result = sdc_func(index)
        result_ref = test_impl(index)
        np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_operator_eq_index(self):
        def test_impl(index1, index2):
            return index1 == index2
        sdc_func = self.jit(test_impl)

        n = 11
        for index1, index2 in product(_generate_int64_indexes_fixed(n), repeat=2):
            with self.subTest(index1=index1, index2=index2):
                result = np.asarray(sdc_func(index1, index2))   # FIXME_Numba#5157: remove np.asarray
                result_ref = test_impl(index1, index2)
                np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_operator_eq_scalar(self):
        def test_impl(A, B):
            return A == B
        sdc_func = self.jit(test_impl)

        n = 11
        A = pd.Int64Index(np.arange(n) * 2)
        scalars_to_test = [0, 22, 13, -5, 4.0]
        for B in scalars_to_test:
            for swap_operands in (False, True):
                if swap_operands:
                    A, B = B, A
                with self.subTest(left=A, right=B):
                    result = np.asarray(sdc_func(A, B))  # FIXME_Numba#5157: remove np.asarray
                    result_ref = test_impl(A, B)
                    np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_operator_eq_nparray(self):
        def test_impl(A, B):
            return A == B
        sdc_func = self.jit(test_impl)

        n = 11
        for A, B in product(
            _generate_int64_indexes_fixed(n),
            map(lambda x: np.array(x), _generate_int64_indexes_fixed(n))
        ):
            for swap_operands in (False, True):
                if swap_operands:
                    A, B = B, A
                with self.subTest(left=A, right=B):
                    result = np.asarray(sdc_func(A, B))  # FIXME_Numba#5157: remove np.asarray
                    result_ref = test_impl(A, B)
                    np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_operator_ne_index(self):
        def test_impl(index1, index2):
            return index1 != index2
        sdc_func = self.jit(test_impl)

        n = 11
        for index1, index2 in product(_generate_int64_indexes_fixed(n), repeat=2):
            with self.subTest(index1=index1, index2=index2):
                result = np.asarray(sdc_func(index1, index2))   # FIXME_Numba#5157: remove np.asarray
                result_ref = test_impl(index1, index2)
                np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_operator_is_nounbox(self):
        def test_impl_1(data):
            index1 = pd.Int64Index(data)
            index2 = index1
            return index1 is index2
        sdc_func_1 = self.jit(test_impl_1)

        def test_impl_2(data):
            index1 = pd.Int64Index(data)
            index2 = pd.Int64Index(data)
            return index1 is index2
        sdc_func_2 = self.jit(test_impl_2)

        # positive testcase
        index_data = [1, 2, 3, 5, 6, 3, 4]
        with self.subTest(subtest="same indexes"):
            result = sdc_func_1(index_data)
            result_ref = test_impl_1(index_data)
            self.assertEqual(result, result_ref)
            self.assertEqual(result, True)

        # negative testcase
        with self.subTest(subtest="not same indexes"):
            result = sdc_func_2(index_data)
            result_ref = test_impl_2(index_data)
            self.assertEqual(result, result_ref)
            self.assertEqual(result, False)

    def test_int64_index_getitem_by_mask(self):
        def test_impl(index, mask):
            return index[mask]
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        mask = np.random.choice([True, False], n)
        for index in _generate_int64_indexes_fixed(n):
            result = sdc_func(index, mask)
            result_ref = test_impl(index, mask)
            pd.testing.assert_index_equal(result, result_ref)

    def test_int64_index_support_reindexing(self):
        from sdc.datatypes.common_functions import sdc_reindex_series

        def pyfunc(data, index, name, by_index):
            S = pd.Series(data, index, name=name)
            return S.reindex(by_index)

        @self.jit
        def sdc_func(data, index, name, by_index):
            return sdc_reindex_series(data, index, name, by_index)

        n = 10
        np.random.seed(0)
        mask = np.random.choice([True, False], n)
        name = 'asdf'
        index1 = pd.Int64Index(np.arange(n))
        index2 = pd.Int64Index(np.arange(n))[::-1]
        result = sdc_func(mask, index1, name, index2)
        result_ref = pyfunc(mask, index1, name, index2)
        pd.testing.assert_series_equal(result, result_ref)

    def test_int64_index_support_join(self):
        from sdc.datatypes.common_functions import sdc_join_series_indexes

        def pyfunc(index1, index2):
            return index1.join(index2, how='outer', return_indexers=True)

        @self.jit
        def sdc_func(index1, index2):
            return sdc_join_series_indexes(index1, index2)

        index1 = pd.Int64Index(np.arange(-5, 5, 1), name='asv')
        index2 = pd.Int64Index(np.arange(0, 10, 2), name='df')
        result = sdc_func(index1, index2)
        result_ref = pyfunc(index1, index2)
        results_names = ['result index', 'left indexer', 'right indexer']
        for i, name in enumerate(results_names):
            result_elem = result[i]
            result_ref_elem = result_ref[i].values if not i else result_ref[i]
            np.testing.assert_array_equal(result_elem, result_ref_elem, f"Mismatch in {name}")

    def test_int64_index_support_take_from(self):
        from sdc.datatypes.common_functions import _sdc_take

        def pyfunc(index1, indexes):
            return index1.values.take(indexes)

        @self.jit
        def sdc_func(index1, indexes):
            return _sdc_take(index1, indexes)

        n, k = 1000, 200
        np.random.seed(0)
        index = pd.Int64Index(np.arange(n) * 2, name='asd')
        indexes = np.random.choice(np.arange(n), n)[:k]
        result = sdc_func(index, indexes)
        result_ref = pyfunc(index, indexes)
        np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_support_take_by(self):
        from sdc.datatypes.common_functions import _sdc_take

        def pyfunc(arr, index):
            return np.take(arr, index)

        @self.jit
        def sdc_func(arr, index):
            return _sdc_take(arr, index)

        n, k = 1000, 200
        np.random.seed(0)
        arr = np.arange(n) * 2
        index = pd.Int64Index(np.random.choice(np.arange(n), n)[:k])
        result = sdc_func(arr, index)
        result_ref = pyfunc(arr, index)
        np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_support_astype(self):
        from sdc.functions.numpy_like import astype

        def pyfunc(index):
            return index.values.astype(np.int64)

        @self.jit
        def sdc_func(index):
            return astype(index, np.int64)

        n = 100
        index = pd.Int64Index(np.arange(n) * 2, name='asd')
        np.testing.assert_array_equal(sdc_func(index), pyfunc(index))

    def test_int64_index_support_array_equal(self):
        from sdc.functions.numpy_like import array_equal

        def pyfunc(index1, index2):
            return np.array_equal(index1.values, index2.values)

        @self.jit
        def sdc_func(index1, index2):
            return array_equal(index1, index2)

        n = 11
        indexes_to_test = [
            pd.Int64Index(np.arange(n)),
            pd.Int64Index(np.arange(n), name='asd'),
            pd.Int64Index(np.arange(n) * 2, name='asd'),
            pd.Int64Index(np.arange(2 * n)),
        ]
        for index1, index2 in combinations_with_replacement(indexes_to_test, 2):
            with self.subTest(index1=index1, index2=index2):
                result = sdc_func(index1, index2)
                result_ref = pyfunc(index1, index2)
                self.assertEqual(result, result_ref)

    def test_int64_index_support_copy(self):
        from sdc.functions.numpy_like import copy

        @self.jit
        def sdc_func(index):
            return copy(index)

        for data in _generate_valid_int64_index_data():
            for name in test_global_index_names:
                index = pd.Int64Index(data, name=name)
                with self.subTest(index=index):
                    result = sdc_func(index)
                    pd.testing.assert_index_equal(result, index)

    def test_int64_index_support_append(self):
        from sdc.datatypes.common_functions import hpat_arrays_append

        def pyfunc(index1, index2):
            return index1.append(index2)

        @self.jit
        def sdc_func(index1, index2):
            return hpat_arrays_append(index1, index2)

        n = 11
        index1 = pd.Int64Index(np.arange(n), name='asv')
        index2 = pd.Int64Index(2 * np.arange(n), name='df')
        result = sdc_func(index1, index2)
        result_ref = pyfunc(index1, index2)
        np.testing.assert_array_equal(result, result_ref)

    def test_int64_index_ravel(self):
        def test_impl(index):
            return index.ravel()
        sdc_func = self.jit(test_impl)

        n = 11
        index = pd.Int64Index(np.arange(n) * 2)
        result = sdc_func(index)
        result_ref = test_impl(index)
        np.testing.assert_array_equal(result, result_ref)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2021, Intel Corporation All rights reserved.
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
from itertools import (combinations_with_replacement, product, combinations, )

from numba.core import types
from sdc.tests.indexes.index_datagens import (
    test_global_index_names,
    _generate_multi_indexes_fixed,
    _generate_multi_index_levels_unique,
    _generate_multi_index_levels,
    _generate_multi_indexes,
    _get_multi_index_base_index,
    get_sample_index,
    get_codes_from_levels,
    )
from sdc.tests.test_base import TestCase
from sdc.datatypes.indexes import *
from sdc.tests.test_utils import skip_numba_jit, assert_pandas_exception


class TestMultiIndex(TestCase):

    def test_multi_index_type_inferred(self):
        for index, name in product(_generate_multi_indexes(),
                                   test_global_index_names):
            with self.subTest(index=index):
                native_index_type = numba.typeof(index)
                self.assertIsInstance(native_index_type, MultiIndexType)

            index.name = name
            with self.subTest(index=index):
                native_index_type = numba.typeof(index)
                self.assertIsInstance(native_index_type, MultiIndexType)

    def test_multi_index_create_and_box(self):
        def test_impl(levels, codes):
            return pd.MultiIndex(levels, codes)
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        for data in _generate_multi_index_levels_unique():
            # creating pd.MultiIndex is only supported with levels and codes as tuples
            levels = tuple(data)
            codes = tuple(get_codes_from_levels(n, levels))
            with self.subTest(levels=levels, codes=codes):
                result = sdc_func(levels, codes)
                result_ref = test_impl(levels, codes)
                pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_create_invalid_inputs(self):
        def test_impl(levels, codes):
            return pd.MultiIndex(levels, codes)
        sdc_func = self.jit(test_impl)

        level_and_codes = [
            (['a', 'b', 'c'], [3, 0, 1, 2, 2]),  # code 3 is out of bounds
            (['a', 'b', 'c'], [1, 0, 1, -2, 2]),  # code -2 is out of bounds
            (['a', 'b', 'c', 'a', 'b'], [1, 0, 1, 2, 2])  # duplicate labels in level
        ]
        exc_strs = [
            "On one of the levels code max >= length of level.",
            "On one of the levels code value < -1",
            "Level values must be unique",
        ]

        for i, level_codes_pair in enumerate(level_and_codes):
            levels, codes = (level_codes_pair[0], ), (level_codes_pair[1], )
            test_msg = f"Inconsistent codes: levels={levels}, codes={codes}"
            sdc_exc_str = exc_strs[i]
            assert_pandas_exception(self, test_msg, sdc_exc_str, test_impl, sdc_func, (levels, codes))

    def test_multi_index_create_from_tuples(self):
        def test_impl():
            codes_max = 5
            levels = (
                ['a', 'b', 'c', 'd', 'e'],
                np.arange(codes_max)
            )
            codes = (
                np.arange(0, codes_max),
                np.arange(codes_max, 0, -1) - 1,
            )
            return pd.MultiIndex(levels, codes)
        sdc_func = self.jit(test_impl)

        result = sdc_func()
        result_ref = test_impl()
        pd.testing.assert_index_equal(result, result_ref)

    @skip_numba_jit("MultiIndexType ctor supports levels and codes as tuples only")
    def test_multi_index_create_from_lists(self):
        def test_impl():
            codes_max = 5
            levels = [
                ['a', 'b', 'c', 'd', 'e'],
                np.arange(codes_max),
            ]
            codes = [
                np.arange(0, codes_max),
                np.arange(codes_max, 0, -1) - 1,
            ]

            return pd.MultiIndex(levels, codes)
        sdc_func = self.jit(test_impl)

        result = sdc_func()
        result_ref = test_impl()
        pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_create_param_names(self):

        # using keyword arguments in typeref ctor, is not supported due to limitation of __call__ overload,
        # TO-DO: refactor this after @overload is supported for typerefs (see FIXME_Numba#XXXX):
        def test_impl(levels, codes, names):
            # return pd.MultiIndex(levels, codes, name=names)
            return pd.MultiIndex(levels, codes, None, None, None, False, names)
        sdc_func = self.jit(test_impl)

        n = 11
        max_codes = 5
        all_levels = [
            [5, 2, 1, 4, 3],
            np.arange(max_codes),
            pd.RangeIndex(max_codes),
            pd.RangeIndex(max_codes, name='abc'),
            pd.Int64Index([5, 2, 1, 4, 3]),
            pd.Int64Index([5, 2, 1, 4, 3], name='bce'),
        ]
        for data, names in product(
                combinations(all_levels, 2),
                combinations_with_replacement(test_global_index_names, 2)
            ):

            # all parameters are supported as tuples only in pd.MultiIndex ctor
            levels = tuple(data)
            codes = tuple(get_codes_from_levels(n, levels))
            _names = tuple(names)
            with self.subTest(levels=levels, codes=codes, names=_names):
                result = sdc_func(levels, codes, _names)
                result_ref = test_impl(levels, codes, _names)
                pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_unbox_and_box(self):
        def test_impl(index):
            return index
        sdc_func = self.jit(test_impl)

        np.random.seed(0)
        for index in _generate_multi_indexes():
            with self.subTest(index=index):
                result = sdc_func(index)
                result_ref = test_impl(index)
                pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_attribute_dtype(self):
        from numba.typed import List

        # index dtype cannot be returned (boxed), thus it only checks it can be used
        def test_impl(index):
            return List.empty_list(index.dtype)
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        result = sdc_func(index)
        expected = types.Tuple.from_types([types.unicode_type, types.intp])
        self.assertEqual(result._dtype, expected)

    def test_multi_index_attribute_name(self):
        def test_impl(index):
            return index.name
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        for name in test_global_index_names:
            index.name = name
            with self.subTest(name=name):
                result = sdc_func(index)
                result_ref = test_impl(index)
                self.assertEqual(result, result_ref)

    @skip_numba_jit("StringArrayType as index has no name. TO-DO: StringIndexType")
    def test_multi_index_attribute_names(self):
        def test_impl(index):
            return index.names
        sdc_func = self.jit(test_impl)

        np.random.seed(0)
        for index in _generate_multi_indexes():
            for names in combinations_with_replacement(
                    test_global_index_names,
                    index.nlevels):
                index.names = names
                with self.subTest(index=index):
                    result = sdc_func(index)
                    result_ref = test_impl(index)
                    self.assertEqual(result, result_ref)

    def test_multi_index_attribute_nlevels(self):
        def test_impl(index):
            return index.nlevels
        sdc_func = self.jit(test_impl)

        np.random.seed(0)
        for index in _generate_multi_indexes():
            with self.subTest(index=index):
                result = sdc_func(index)
                result_ref = test_impl(index)
                self.assertEqual(result, result_ref)

    def test_multi_index_len(self):
        def test_impl(index):
            return len(index)
        sdc_func = self.jit(test_impl)

        np.random.seed(0)
        for index in _generate_multi_indexes():
            with self.subTest(index=index):
                result = sdc_func(index)
                result_ref = test_impl(index)
                self.assertEqual(result, result_ref)

    def test_multi_index_attribute_values(self):
        def test_impl(index):
            return index.values
        sdc_func = self.jit(test_impl)

        np.random.seed(0)
        for index in _generate_multi_indexes():
            with self.subTest(index_data=index):
                result = sdc_func(index)
                result_ref = test_impl(index)
                # SDC MultiIndex.values return list but not numpy array
                self.assertEqual(result, list(result_ref))

    def test_multi_index_attribute_levels(self):
        def test_impl(index):
            return index.levels
        sdc_func = self.jit(test_impl)

        np.random.seed(0)
        for index in _generate_multi_indexes():
            with self.subTest(index_data=index):
                result = sdc_func(index)
                result_ref = test_impl(index)
                # SDC MultiIndex.levels return tuple of levels not list
                error_msg = f"Indexes'levels are different:\nresult={result},\nresult_ref{result_ref}"
                self.assertEqual(len(result), len(result_ref), error_msg)
                self.assertTrue(map(
                    lambda x, y: pd.testing.assert_index_equal(x, y),
                    zip(result, result_ref)),
                    error_msg
                )

    def test_multi_index_attribute_codes(self):
        def test_impl(index):
            return index.codes
        sdc_func = self.jit(test_impl)

        np.random.seed(0)
        for index in _generate_multi_indexes():
            with self.subTest(index_data=index):
                result = sdc_func(index)
                result_ref = test_impl(index)
                # SDC MultiIndex.levels return tuple of levels not list
                error_msg = f"Indexes'levels are different:\nresult={result},\nresult_ref{result_ref}"
                self.assertEqual(len(result), len(result_ref), error_msg)
                self.assertTrue(map(
                    lambda x, y: np.testing.assert_array_equal(x, y),
                    zip(result, result_ref)),
                    error_msg
                )

    def test_multi_index_contains(self):
        def test_impl(index, value):
            return value in index
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        values_to_test = [('a', 1), ('a', 4), ('e', 1), ('x', 5)]
        for value in values_to_test:
            with self.subTest(value=value):
                result = sdc_func(index, value)
                result_ref = test_impl(index, value)
                np.testing.assert_array_equal(result, result_ref)

    def test_multi_index_getitem_scalar(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        idxs_to_test = [0, n // 2, n - 1, -1]
        for idx in idxs_to_test:
            with self.subTest(idx=idx):
                result = sdc_func(index, idx)
                result_ref = test_impl(index, idx)
                self.assertEqual(result, result_ref)

    def test_multi_index_getitem_scalar_idx_bounds(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        idxs_to_test = [-(n + 1), n]
        for idx in idxs_to_test:
            with self.subTest(idx=idx):
                with self.assertRaises(Exception) as context:
                    test_impl(index, idx)
                pandas_exception = context.exception

                with self.assertRaises(type(pandas_exception)) as context:
                    sdc_func(index, idx)
                sdc_exception = context.exception
                self.assertIsInstance(sdc_exception, type(pandas_exception))
                self.assertIn("out of bounds", str(sdc_exception))

    def test_multi_index_getitem_slice(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        n = 17
        index = get_sample_index(n, MultiIndexType)
        slices_params = combinations_with_replacement(
            [None, 0, -1, n // 2, n, n - 3, n + 3, -(n + 3)],
            2
        )

        for slice_start, slice_stop in slices_params:
            for slice_step in [1, -1, 2]:
                idx = slice(slice_start, slice_stop, slice_step)
                with self.subTest(idx=idx):
                    result = sdc_func(index, idx)
                    result_ref = test_impl(index, idx)
                    pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_iterator_1(self):
        def test_impl(index):
            res = []
            for i, label in enumerate(index):
                res.append((i, label))
            return res
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    def test_multi_index_iterator_2(self):
        def test_impl(index):
            res = []
            for label in index:
                str_part, _ = label
                if str_part == 'a':
                    res.append(label)
            return res
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    @skip_numba_jit("Requires np.array of complex dtypes (tuples) support in Numba")
    def test_multi_index_nparray(self):
        def test_impl(index):
            return np.array(index)
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        result = sdc_func(index)
        result_ref = test_impl(index)
        np.testing.assert_array_equal(result, result_ref)

    def test_multi_index_operator_eq_index(self):
        def test_impl(index1, index2):
            return index1 == index2
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        indexes_to_test = list(_generate_multi_indexes_fixed(n))
        for index1, index2 in combinations_with_replacement(indexes_to_test, 2):
            with self.subTest(index1=index1, index2=index2):
                result = np.asarray(sdc_func(index1, index2))   # FIXME_Numba#5157: remove np.asarray
                result_ref = test_impl(index1, index2)
                np.testing.assert_array_equal(result, result_ref)

    def test_multi_index_operator_eq_scalar(self):
        def test_impl(A, B):
            return A == B
        sdc_func = self.jit(test_impl)

        n = 11
        A = get_sample_index(n, MultiIndexType)
        scalars_to_test = [('a', 1), ('a', 4), ('e', 1), ('x', 5)]
        for B in scalars_to_test:
            for swap_operands in (False, True):
                if swap_operands:
                    A, B = B, A
                with self.subTest(left=A, right=B):
                    result = np.asarray(sdc_func(A, B))  # FIXME_Numba#5157: remove np.asarray
                    result_ref = test_impl(A, B)
                    np.testing.assert_array_equal(result, result_ref)

    @skip_numba_jit("Requires np.array of complex dtypes (tuples) support in Numba")
    def test_multi_index_operator_eq_nparray(self):
        def test_impl(A, B):
            return A == B
        sdc_func = self.jit(test_impl)

        n = 11
        for A, B in product(
            _generate_multi_indexes_fixed(n),
            map(lambda x: np.array(x), _generate_multi_indexes_fixed(n))
        ):
            for swap_operands in (False, True):
                if swap_operands:
                    A, B = B, A
                with self.subTest(left=A, right=B):
                    result = np.asarray(sdc_func(A, B))  # FIXME_Numba#5157: remove np.asarray
                    result_ref = test_impl(A, B)
                    np.testing.assert_array_equal(result, result_ref)

    def test_multi_index_operator_ne_index(self):
        def test_impl(index1, index2):
            return index1 != index2
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        indexes_to_test = list(_generate_multi_indexes_fixed(n))
        for index1, index2 in combinations_with_replacement(indexes_to_test, 2):
            with self.subTest(index1=index1, index2=index2):
                result = np.asarray(sdc_func(index1, index2))   # FIXME_Numba#5157: remove np.asarray
                result_ref = test_impl(index1, index2)
                np.testing.assert_array_equal(result, result_ref)

    def test_multi_index_operator_is_nounbox(self):
        def test_impl_1():
            index1 = pd.MultiIndex(
                levels=(['a', 'b', 'c'], [1, 2, 3]),
                codes=([0, 1, 0, 1, 2], [0, 0, 1, 1, 2])
            )
            index2 = index1
            return index1 is index2
        sdc_func_1 = self.jit(test_impl_1)

        def test_impl_2():
            index1 = pd.MultiIndex(
                levels=(['a', 'b', 'c'], [1, 2, 3]),
                codes=([0, 1, 0, 1, 2], [0, 0, 1, 1, 2])
            )
            index2 = pd.MultiIndex(
                levels=(['a', 'b', 'c'], [1, 2, 3]),
                codes=([0, 1, 0, 1, 2], [0, 0, 1, 1, 2])
            )
            return index1 is index2
        sdc_func_2 = self.jit(test_impl_2)

        # positive testcase
        with self.subTest(subtest="same indexes"):
            result = sdc_func_1()
            result_ref = test_impl_1()
            self.assertEqual(result, result_ref)
            self.assertEqual(result, True)

        # negative testcase
        with self.subTest(subtest="not same indexes"):
            result = sdc_func_2()
            result_ref = test_impl_2()
            self.assertEqual(result, result_ref)
            self.assertEqual(result, False)

    def test_multi_index_getitem_by_mask(self):
        def test_impl(index, mask):
            return index[mask]
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        mask = np.random.choice([True, False], n)
        for index in _generate_multi_indexes_fixed(n):
            result = sdc_func(index, mask)
            result_ref = test_impl(index, mask)
            pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_getitem_by_array(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        n, k = 11, 7
        np.random.seed(0)
        idx = np.random.choice(np.arange(n), k)
        for index in _generate_multi_indexes_fixed(n):
            result = sdc_func(index, idx)
            result_ref = test_impl(index, idx)
            pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_reindex_equal_indexes(self):

        def test_func(index1, index2):
            return index1.reindex(index2)
        sdc_func = self.jit(test_func)

        n = 10
        index1 = get_sample_index(n, MultiIndexType)
        index2 = index1.copy(deep=True)

        result = sdc_func(index1, index2)
        result_ref = test_func(index1, index2)
        pd.testing.assert_index_equal(result[0], result_ref[0])
        np.testing.assert_array_equal(result[1], result_ref[1])

    def test_multi_index_reindex(self):

        def test_impl(index1, index2):
            return index1.reindex(index2)
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        base_index = _get_multi_index_base_index(n)
        index1 = base_index[:n] 
        size_range = np.arange(len(index1))
        reindex_by = list(map(
            lambda x: base_index.take(x),
            [
                size_range,  # same index as index1
                np.random.choice(size_range, n),  # random values from index1 with duplicates
                np.random.choice(size_range, n, replace=False),   # random unique values from index1
                np.random.choice(np.arange(len(base_index)), n),  # random values from larger set
                size_range[:n // 2],  # shorter index
                np.random.choice(size_range, 2*n), # longer index
            ]
        ))

        for index2 in reindex_by:
            with self.subTest(index2=index2):
                result = sdc_func(index1, index2)
                result_ref = test_impl(index1, index2)
                pd.testing.assert_index_equal(result[0], result_ref[0])
                np.testing.assert_array_equal(result[1], result_ref[1])

    def test_multi_index_equals(self):
        def test_impl(index1, index2):
            return index1.equals(index2)
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        indexes_to_test = list(_generate_multi_indexes_fixed(n))
        for index1, index2 in combinations_with_replacement(indexes_to_test, 2):
            with self.subTest(index1=index1, index2=index2):
                result = sdc_func(index1, index2)
                result_ref = test_impl(index1, index2)
                self.assertEqual(result, result_ref)

    def test_multi_index_ravel(self):
        def test_impl(index):
            return index.ravel()
        sdc_func = self.jit(test_impl)

        n = 11
        index = get_sample_index(n, MultiIndexType)
        result = sdc_func(index)
        result_ref = test_impl(index)
        # SDC MultiIndex.values return list but not numpy array
        np.testing.assert_array_equal(result, list(result_ref))

    def test_multi_index_take(self):
        def test_impl(index, value):
            return index.take(value)
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        index_pos = np.arange(n)
        values_to_test = [
            np.random.choice(index_pos, 2*n),
            list(np.random.choice(index_pos, n, replace=False)),
            pd.RangeIndex(n // 2),
            pd.Int64Index(index_pos[n // 2:])
        ]
        for index, value in product(_generate_multi_indexes_fixed(n), values_to_test):
            with self.subTest(index=index, value=value):
                result = sdc_func(index, value)
                result_ref = test_impl(index, value)
                pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_append(self):
        def test_impl(index, other):
            return index.append(other)
        sdc_func = self.jit(test_impl)

        index = pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
        other = pd.MultiIndex.from_tuples(
            [('a', 3), ('c', 1), ('c', 3), ('b', 2), ('b', 3)])
        result = sdc_func(index, other)
        result_ref = test_impl(index, other)
        pd.testing.assert_index_equal(result, result_ref)

    @skip_numba_jit("MultiIndexType.join is not implemented yet")
    def test_multi_index_join(self):
        def test_impl(index, other):
            return index.join(other, 'outer', return_indexers=True)
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        indexes_to_test = list(_generate_multi_indexes_fixed(n))
        for index, other in combinations_with_replacement(indexes_to_test, 2):
            with self.subTest(index=index, other=other):
                result = sdc_func(index, other)
                result_ref = test_impl(index, other)
                # check_names=False, since pandas behavior is not type-stable
                pd.testing.assert_index_equal(result[0], result_ref[0], check_names=False)
                np.testing.assert_array_equal(result[1], result_ref[1])
                np.testing.assert_array_equal(result[2], result_ref[2])

    def test_multi_index_from_product(self):
        def test_impl(levels):
            return pd.MultiIndex.from_product(levels)
        sdc_func = self.jit(test_impl)

        np.random.seed(0)
        for data in _generate_multi_index_levels():
            # creating pd.MultiIndex is only supported with levels and codes as tuples
            levels = tuple(data)
            with self.subTest(levels=levels):
                result = sdc_func(levels)
                result_ref = test_impl(levels)
                pd.testing.assert_index_equal(result, result_ref)

    def test_multi_index_from_tuples(self):
        def test_impl(data):
            return pd.MultiIndex.from_tuples(data)
        sdc_func = self.jit(test_impl)

        n = 100
        np.random.seed(0)
        for index in _generate_multi_indexes_fixed(n):
            data = list(index.values)
            with self.subTest(data=data):
                result = sdc_func(data)
                result_ref = test_impl(data)
                pd.testing.assert_index_equal(result, result_ref)


if __name__ == "__main__":
    unittest.main()

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

from numba.core.errors import TypingError
from sdc.tests.indexes.index_datagens import (
    test_global_index_names,
    _generate_valid_range_params,
    _generate_range_indexes_fixed,
    _generate_index_param_values,
    )
from sdc.tests.test_base import TestCase
from sdc.utilities.sdc_typing_utils import kwsparams2list
from sdc.tests.test_series import _make_func_from_text


class TestRangeIndex(TestCase):

    def test_range_index_create_and_box(self):
        def test_impl(start, stop, step, name):
            return pd.RangeIndex(start, stop, step, name=name)
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            start, stop, step = params
            for name in test_global_index_names:
                with self.subTest(start=start, stop=stop, step=step, name=name):
                    result = sdc_func(start, stop, step, name)
                    result_ref = test_impl(start, stop, step, name)
                    pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_unbox_and_box(self):
        def test_impl(index):
            return index
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            start, stop, step = params
            for name in test_global_index_names:
                index = pd.RangeIndex(start, stop, step, name=name)
                with self.subTest(index=index):
                    result = sdc_func(index)
                    result_ref = test_impl(index)
                    pd.testing.assert_index_equal(result, result_ref)

    @unittest.skip("TODO: add support for integers as floats in ctor")
    def test_range_index_create_from_floats(self):
        def test_impl(*args):
            return pd.RangeIndex(*args)
        sdc_func = self.jit(test_impl)

        start, stop, step = 3.0, 15.0, 2.0
        result = sdc_func(start, stop, step)
        result_ref = test_impl(start, stop, step)
        pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_create_invalid_1(self):
        def test_impl(start, stop, step):
            return pd.RangeIndex(start, stop, step)
        sdc_func = self.jit(test_impl)

        # zero step is not allowed by pandas
        start, stop, step = 3, 5, 0
        with self.assertRaises(Exception) as context:
            test_impl(start, stop, step)
        pandas_exception = context.exception

        with self.assertRaises(type(pandas_exception)) as context:
            sdc_func(start, stop, step)
        sdc_exception = context.exception
        self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_range_index_create_invalid_2(self):
        def test_impl():
            return pd.RangeIndex(name='index')
        sdc_func = self.jit(test_impl)

        # all start, stop and step cannot be None at the same time
        with self.assertRaises(Exception) as context:
            test_impl()
        pandas_exception = context.exception

        with self.assertRaises(type(pandas_exception)) as context:
            sdc_func()
        sdc_exception = context.exception
        self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_range_index_create_defaults(self):
        func_lines = [
            'def test_impl():',
            '  return pd.RangeIndex({})'
        ]
        test_impl_text = '\n'.join(func_lines)

        # use non default values for all parameters except one (tested)
        non_default_params = {'start': 2, 'stop': 7, 'step': 2, 'name': "'index'"}
        for arg in non_default_params.keys():
            with self.subTest(omitted=arg):
                kwargs = {key: val for key, val in non_default_params.items() if key != arg}
                func_text = test_impl_text.format(', '.join(kwsparams2list(kwargs)))
                test_impl = _make_func_from_text(func_text, global_vars={'pd': pd})
                sdc_func = self.jit(test_impl)
                result = sdc_func()
                result_ref = test_impl()
                pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_create_param_copy(self):
        def test_impl(stop, value):
            return pd.RangeIndex(stop, copy=value)
        sdc_func = self.jit(test_impl)

        with self.assertRaises(TypingError) as raises:
            sdc_func(11, False)
        self.assertIn("SDCLimitation: pd.RangeIndex(). Unsupported parameter",
                      str(raises.exception))

    def test_range_index_create_param_name_literal_str(self):
        def test_impl(stop):
            return pd.RangeIndex(stop, name='index')
        sdc_func = self.jit(test_impl)

        n = 11
        result = sdc_func(n)
        result_ref = test_impl(n)
        pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_create_param_dtype(self):
        def test_impl(stop, dtype):
            return pd.RangeIndex(stop, dtype=dtype)
        sdc_func = self.jit(test_impl)

        n = 11
        supported_dtypes = [None, np.int64, 'int64', np.int32, 'int32']
        for dtype in supported_dtypes:
            with self.subTest(dtype=dtype):
                result = sdc_func(n, dtype)
                result_ref = test_impl(n, dtype)
                pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_create_param_dtype_invalid(self):
        def test_impl(stop, dtype):
            return pd.RangeIndex(stop, dtype=dtype)
        sdc_func = self.jit(test_impl)

        n = 11
        invalid_dtypes = ['float']
        for dtype in invalid_dtypes:
            with self.subTest(dtype=dtype):
                with self.assertRaises(Exception) as context:
                    test_impl(n, dtype)
                pandas_exception = context.exception

                with self.assertRaises(type(pandas_exception)) as context:
                    sdc_func(n, dtype)
                sdc_exception = context.exception
                self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_range_index_attribute_start(self):
        def test_impl(*args):
            index = pd.RangeIndex(*args)
            return index.start
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            start, stop, step = params
            with self.subTest(start=start, stop=stop, step=step):
                result = sdc_func(*params)
                result_ref = test_impl(*params)
                self.assertEqual(result, result_ref)

    def test_range_index_attribute_stop(self):
        def test_impl(*args):
            index = pd.RangeIndex(*args)
            return index.stop
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            start, stop, step = params
            with self.subTest(start=start, stop=stop, step=step):
                result = sdc_func(*params)
                result_ref = test_impl(*params)
                self.assertEqual(result, result_ref)

    def test_range_index_attribute_step(self):
        def test_impl(*args):
            index = pd.RangeIndex(*args)
            return index.step
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            start, stop, step = params
            with self.subTest(start=start, stop=stop, step=step):
                result = sdc_func(*params)
                result_ref = test_impl(*params)
                self.assertEqual(result, result_ref)

    def test_range_index_attribute_dtype(self):
        def test_impl(index):
            return index.dtype
        sdc_func = self.jit(test_impl)

        index = pd.RangeIndex(11)
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    def test_range_index_attribute_name(self):
        def test_impl(index):
            return index.name
        sdc_func = self.jit(test_impl)

        n = 11
        for name in test_global_index_names:
            with self.subTest(name=name):
                index = pd.RangeIndex(n, name=name)
                result = sdc_func(index)
                result_ref = test_impl(index)
                self.assertEqual(result, result_ref)

    def test_range_index_len(self):
        def test_impl(*args):
            index = pd.RangeIndex(*args)
            return len(index)
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            start, stop, step = params
            with self.subTest(start=start, stop=stop, step=step):
                result = sdc_func(*params)
                result_ref = test_impl(*params)
                self.assertEqual(result, result_ref)

    def test_range_index_attribute_values(self):
        def test_impl(index):
            return index.values
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            index = pd.RangeIndex(*params)
            with self.subTest(index=index):
                result = sdc_func(index)
                result_ref = test_impl(index)
                np.testing.assert_array_equal(result, result_ref)

    def test_range_index_contains(self):
        def test_impl(index, value):
            return value in index
        sdc_func = self.jit(test_impl)

        index = pd.RangeIndex(1, 11, 2)
        values_to_test = [-5, 15, 1, 11, 5, 6]
        for value in values_to_test:
            with self.subTest(value=value):
                result = sdc_func(index, value)
                result_ref = test_impl(index, value)
                np.testing.assert_array_equal(result, result_ref)

    def test_range_index_copy(self):
        def test_impl(index, new_name):
            return index.copy(name=new_name)
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            start, stop, step = params
            for name, new_name in product(test_global_index_names, repeat=2):
                index = pd.RangeIndex(start, stop, step, name=name)
                with self.subTest(index=index, new_name=new_name):
                    result = sdc_func(index, new_name)
                    result_ref = test_impl(index, new_name)
                    pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_getitem_scalar(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        for params in _generate_valid_range_params():
            index = pd.RangeIndex(*params)
            n = len(index)
            if not n:  # test only non-empty ranges
                continue
            values_to_test = [-n, n // 2, n - 1]
            for idx in values_to_test:
                with self.subTest(index=index, idx=idx):
                    result = sdc_func(index, idx)
                    result_ref = test_impl(index, idx)
                    self.assertEqual(result, result_ref)

    def test_range_index_getitem_scalar_idx_bounds(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        n = 11
        index = pd.RangeIndex(n, name='abc')
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

    def test_range_index_getitem_slice(self):
        def test_impl(index, idx):
            return index[idx]
        sdc_func = self.jit(test_impl)

        index_len = 17
        start_values, step_values = [0, 5, -5], [1, 2, 7]
        slices_params = combinations_with_replacement(
            [None, 0, -1, index_len // 2, index_len, index_len - 3, index_len + 3, -(index_len + 3)],
            2
        )

        for start, step, slice_step in product(start_values, step_values, step_values):
            stop = start + index_len
            for slice_start, slice_stop in slices_params:
                idx = slice(slice_start, slice_stop, slice_step)
                index = pd.RangeIndex(start, stop, step, name='abc')
                with self.subTest(index=index, idx=idx):
                    result = sdc_func(index, idx)
                    result_ref = test_impl(index, idx)
                    pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_iterator_1(self):
        def test_impl(index):
            res = []
            for i, label in enumerate(index):
                res.append((i, label))
            return res
        sdc_func = self.jit(test_impl)

        index = pd.RangeIndex(1, 21, 3)
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    def test_range_index_iterator_2(self):
        def test_impl(index):
            res = []
            for label in index:
                if not label % 2:
                    res.append(label)
            return res
        sdc_func = self.jit(test_impl)

        index = pd.RangeIndex(1, 21, 3)
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertEqual(result, result_ref)

    def test_range_index_nparray(self):
        def test_impl(index):
            return np.array(index)
        sdc_func = self.jit(test_impl)

        index = pd.RangeIndex(1, 21, 3)
        result = sdc_func(index)
        result_ref = test_impl(index)
        np.testing.assert_array_equal(result, result_ref)

    def test_range_index_operator_eq_index_1(self):
        """ Verifies operator.eq implementation for pandas RangeIndex in a case of equal range sizes """
        def test_impl(index1, index2):
            return index1 == index2
        sdc_func = self.jit(test_impl)

        n = 11
        for index1, index2 in product(_generate_range_indexes_fixed(n), repeat=2):
            with self.subTest(index1=index1, index2=index2):
                result = np.asarray(sdc_func(index1, index2))   # FIXME_Numba#5157: remove np.asarray
                result_ref = test_impl(index1, index2)
                np.testing.assert_array_equal(result, result_ref)

    def test_range_index_operator_eq_index_2(self):
        """ Verifies operator.eq implementation for pandas RangeIndex in a case of non equal range sizes """
        def test_impl(index1, index2):
            return index1 == index2
        sdc_func = self.jit(test_impl)

        index1 = pd.RangeIndex(1, 22, 5)
        index2 = pd.RangeIndex(1, 22, 10)
        with self.assertRaises(Exception) as context:
            test_impl(index1, index2)
        pandas_exception = context.exception

        with self.assertRaises(type(pandas_exception)) as context:
            sdc_func(index1, index2)
        sdc_exception = context.exception
        self.assertIn(str(sdc_exception), str(pandas_exception))

    def test_range_index_operator_eq_scalar(self):
        """ Verifies operator.eq implementation for pandas RangeIndex and a scalar value """
        def test_impl(A, B):
            return A == B
        sdc_func = self.jit(test_impl)

        n = 11
        A = pd.RangeIndex(n)
        scalars_to_test = [
            A.start,
            float(A.start),
            A.start + 1,
            (A.start + A.stop) / 2,
            A.stop,
        ]
        for B in scalars_to_test:
            for swap_operands in (False, True):
                if swap_operands:
                    A, B = B, A
                with self.subTest(left=A, right=B):
                    result = np.asarray(sdc_func(A, B))  # FIXME_Numba#5157: remove np.asarray
                    result_ref = test_impl(A, B)
                    np.testing.assert_array_equal(result, result_ref)

    def test_range_index_operator_eq_nparray(self):
        """ Verifies operator.eq implementation for pandas RangeIndex and a numpy array """
        def test_impl(A, B):
            return A == B
        sdc_func = self.jit(test_impl)

        n = 11
        for A, B in product(
            _generate_range_indexes_fixed(n),
            map(lambda x: np.array(x), _generate_range_indexes_fixed(n))
        ):
            for swap_operands in (False, True):
                if swap_operands:
                    A, B = B, A
                with self.subTest(left=A, right=B):
                    result = np.asarray(sdc_func(A, B))  # FIXME_Numba#5157: remove np.asarray
                    result_ref = test_impl(A, B)
                    np.testing.assert_array_equal(result, result_ref)

    def test_range_index_operator_ne_index(self):
        """ Verifies operator.ne implementation for pandas RangeIndex in a case of non equal range sizes """
        def test_impl(index1, index2):
            return index1 != index2
        sdc_func = self.jit(test_impl)

        n = 11
        for index1, index2 in product(_generate_range_indexes_fixed(n), repeat=2):
            with self.subTest(index1=index1, index2=index2):
                result = np.asarray(sdc_func(index1, index2))   # FIXME_Numba#5157: remove np.asarray
                result_ref = test_impl(index1, index2)
                np.testing.assert_array_equal(result, result_ref)

    def test_range_index_operator_is_nounbox(self):
        def test_impl_1(*args):
            index1 = pd.RangeIndex(*args)
            index2 = index1
            return index1 is index2
        sdc_func_1 = self.jit(test_impl_1)

        def test_impl_2(*args):
            index1 = pd.RangeIndex(*args)
            index2 = pd.RangeIndex(*args)
            return index1 is index2
        sdc_func_2 = self.jit(test_impl_2)

        # positive testcase
        params = 1, 21, 3
        with self.subTest(subtest="same indexes"):
            result = sdc_func_1(*params)
            result_ref = test_impl_1(*params)
            self.assertEqual(result, result_ref)
            self.assertEqual(result, True)

        # negative testcase
        with self.subTest(subtest="not same indexes"):
            result = sdc_func_2(*params)
            result_ref = test_impl_2(*params)
            self.assertEqual(result, result_ref)
            self.assertEqual(result, False)

    def test_range_index_getitem_by_mask(self):
        def test_impl(index, mask):
            return index[mask]
        sdc_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        mask = np.random.choice([True, False], n)
        for index in _generate_range_indexes_fixed(n):
            result = sdc_func(index, mask)
            result_ref = test_impl(index, mask)
            pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_support_reindexing(self):
        from sdc.datatypes.common_functions import sdc_reindex_series

        def pyfunc(data, index, name, by_index):
            S = pd.Series(data, index, name=name)
            return S.reindex(by_index)

        @self.jit
        def sdc_func(data, index, name, by_index):
            return sdc_reindex_series(data, index, name, by_index)

        n = 100
        np.random.seed(0)
        mask = np.random.choice([True, False], n)
        name = 'asdf'
        index1 = pd.RangeIndex(n)
        index2 = index1[::-1]
        result = sdc_func(mask, index1, name, index2)
        result_ref = pyfunc(mask, index1, name, index2)
        pd.testing.assert_series_equal(result, result_ref)

    def test_range_index_support_join(self):
        from sdc.datatypes.common_functions import sdc_join_series_indexes

        def pyfunc(index1, index2):
            return index1.join(index2, how='outer', return_indexers=True)

        @self.jit
        def sdc_func(index1, index2):
            return sdc_join_series_indexes(index1, index2)

        index1 = pd.RangeIndex(1, 21, 3, name='asv')
        index2 = pd.RangeIndex(19, -1, -3, name='df')
        result = sdc_func(index1, index2)
        result_ref = pyfunc(index1, index2)
        results_names = ['result index', 'left indexer', 'right indexer']
        for i, name in enumerate(results_names):
            result_elem = result[i]
            result_ref_elem = result_ref[i].values if not i else result_ref[i]
            np.testing.assert_array_equal(result_elem, result_ref_elem, f"Mismatch in {name}")

    def test_range_index_support_take(self):
        from sdc.datatypes.common_functions import _sdc_take

        def pyfunc(index1, indexes):
            return index1.values.take(indexes)

        @self.jit
        def sdc_func(index1, indexes):
            return _sdc_take(index1, indexes)

        n, k = 1000, 200
        np.random.seed(0)
        index = pd.RangeIndex(stop=3 * n, step=3, name='asd')
        indexes = np.random.choice(np.arange(n), n)[:k]
        result = sdc_func(index, indexes)
        result_ref = pyfunc(index, indexes)
        np.testing.assert_array_equal(result, result_ref)

    def test_range_index_support_astype(self):
        from sdc.functions.numpy_like import astype

        def pyfunc(index):
            return index.values.astype(np.int64)

        @self.jit
        def sdc_func(index):
            return astype(index, np.int64)

        index = pd.RangeIndex(stop=11, name='asd')
        np.testing.assert_array_equal(sdc_func(index), pyfunc(index))

    def test_range_index_support_array_equal(self):
        from sdc.functions.numpy_like import array_equal

        def pyfunc(index1, index2):
            return np.array_equal(index1.values, index2.values)

        @self.jit
        def sdc_func(index1, index2):
            return array_equal(index1, index2)

        for params1, params2 in product(_generate_valid_range_params(), repeat=2):
            for name1, name2 in product(test_global_index_names, repeat=2):
                index1 = pd.RangeIndex(*params1, name=name1)
                index2 = pd.RangeIndex(*params2, name=name2)
                with self.subTest(index1=index1, index2=index2):
                    result = sdc_func(index1, index2)
                    result_ref = pyfunc(index1, index2)
                    self.assertEqual(result, result_ref)

    def test_range_index_support_copy(self):
        from sdc.functions.numpy_like import copy

        @self.jit
        def sdc_func(index):
            return copy(index)

        for params in _generate_valid_range_params():
            for name in test_global_index_names:
                index = pd.RangeIndex(*params, name=name)
                with self.subTest(index=index):
                    result = sdc_func(index)
                    pd.testing.assert_index_equal(result, index)

    def test_range_index_support_append(self):
        from sdc.datatypes.common_functions import hpat_arrays_append

        def pyfunc(index1, index2):
            return index1.append(index2)

        @self.jit
        def sdc_func(index1, index2):
            return hpat_arrays_append(index1, index2)

        n = 11
        index1 = pd.RangeIndex(1, 21, 3, name='asv')
        index2 = pd.RangeIndex(19, -1, -3, name='df')
        result = sdc_func(index1, index2)
        result_ref = pyfunc(index1, index2)
        np.testing.assert_array_equal(result, result_ref)

    def test_range_index_ravel(self):
        def test_impl(index):
            return index.ravel()
        sdc_func = self.jit(test_impl)

        n = 11
        index = pd.RangeIndex(n)
        result = sdc_func(index)
        result_ref = test_impl(index)
        np.testing.assert_array_equal(result, result_ref)


if __name__ == "__main__":
    unittest.main()

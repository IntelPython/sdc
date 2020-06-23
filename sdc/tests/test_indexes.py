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

from itertools import (combinations_with_replacement, product, filterfalse)

from sdc.tests.test_base import TestCase
from sdc.utilities.sdc_typing_utils import kwsparams2list
from sdc.tests.test_series import _make_func_from_text
from numba.core.errors import TypingError


test_global_index_names = [None, 'abc', 'index']
test_global_range_member_values = [1, 2, 10, -5, 0, None]


def _generate_valid_range_params():

    def valid_params_predicate(range_params):
        # if step is zero or all start/stop/step are None range is invalid
        return (range_params[-1] == 0
                or all(map(lambda x: x is None, range_params)))

    return filterfalse(
        valid_params_predicate,
        combinations_with_replacement(test_global_range_member_values, 3)
    )


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

    @unittest.skip("TODO: support boxing/unboxing and parent ref for Python ranges in Numba")
    def test_range_index_unbox_data_id_check(self):
        def test_impl(index):
            return index
        sdc_func = self.jit(test_impl)

        index = pd.RangeIndex(11, name='abc')
        result = sdc_func(index)
        result_ref = test_impl(index)
        self.assertIs(index._range, result_ref._range)
        self.assertIs(result._range, result_ref._range)

    @unittest.skip("TODO: add support for integers as floats in ctor")
    def test_range_index_create_from_floats(self):
        def test_impl(*args):
            return pd.RangeIndex(*args)
        sdc_func = self.jit(test_impl)

        start, stop, step = 3.0, 15.0, 2.0
        result = sdc_func(start, stop, step)
        result_ref = test_impl(start, stop, step)
        pd.testing.assert_index_equal(result, result_ref)

    def test_range_index_create_invalid1(self):
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

    def test_range_index_create_invalid2(self):
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
        supported_dtypes = [None, np.int64, 'int64']
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
        invalid_dtypes = ['float', np.int32, 'int32']
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


if __name__ == "__main__":
    unittest.main()

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
import platform
import unittest
from itertools import combinations, combinations_with_replacement, product

from numba.core.config import IS_32BITS
from numba.core.errors import TypingError

from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (skip_numba_jit,
                                  skip_sdc_jit,
                                  _make_func_from_text,
                                  gen_frand_array)


def _make_func_use_binop1(operator):
    func_text = "def test_impl(A, B):\n"
    func_text += "   return A {} B\n".format(operator)
    return _make_func_from_text(func_text)


def _make_func_use_binop2(operator):
    func_text = "def test_impl(A, B):\n"
    func_text += "   A {} B\n".format(operator)
    func_text += "   return A\n"
    return _make_func_from_text(func_text)


def _make_func_use_method_arg1(method):
    func_text = "def test_impl(A, B):\n"
    func_text += "   return A.{}(B)\n".format(method)
    return _make_func_from_text(func_text)


class TestSeries_ops(TestCase):

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support comparing Series of different lengths')
    def test_series_operators_int(self):
        """Verifies using all various Series arithmetic binary operators on two integer Series with default indexes"""
        n = 11
        np.random.seed(0)
        data_to_test = [np.arange(-5, -5 + n, dtype=np.int32),
                        np.ones(n + 3, dtype=np.int32),
                        np.random.randint(-5, 5, n + 7)]

        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')
        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = self.jit(test_impl)
            for data_left, data_right in combinations_with_replacement(data_to_test, 2):
                # integers to negative powers are not allowed
                if (operator == '**' and np.any(data_right < 0)):
                    data_right = np.abs(data_right)

                with self.subTest(left=data_left, right=data_right, operator=operator):
                    S1 = pd.Series(data_left)
                    S2 = pd.Series(data_right)
                    # check_dtype=False because SDC implementation always returns float64 Series
                    pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2), check_dtype=False)

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support division/modulo/etc by zero')
    def test_series_operators_int_scalar(self):
        """Verifies using all various Series arithmetic binary operators
           on an integer Series with default index and a scalar value"""
        n = 11
        np.random.seed(0)
        data_to_test = [np.arange(-5, -5 + n, dtype=np.int32),
                        np.ones(n + 3, dtype=np.int32),
                        np.random.randint(-5, 5, n + 7)]
        scalar_values = [1, -1, 0, 3, 7, -5]

        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')
        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = self.jit(test_impl)
            for data, scalar, swap_operands in product(data_to_test, scalar_values, (False, True)):

                S = pd.Series(data)
                left, right = (S, scalar) if swap_operands else (scalar, S)

                # integers to negative powers are not allowed
                if (operator == '**' and np.any(right < 0)):
                    right = abs(right)

                with self.subTest(left=left, right=right, operator=operator):
                    # check_dtype=False because SDC implementation always returns float64 Series
                    pd.testing.assert_series_equal(hpat_func(left, right), test_impl(left, right), check_dtype=False)

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support comparing Series of different lengths')
    def test_series_operators_float(self):
        """Verifies using all various Series arithmetic binary operators on two float Series with default indexes"""
        n = 11
        np.random.seed(0)
        data_to_test = [np.arange(-5, -5 + n, dtype=np.float32),
                        np.ones(n + 3, dtype=np.float32),
                        np.random.ranf(n + 7)]

        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')
        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = self.jit(test_impl)
            for data_left, data_right in combinations_with_replacement(data_to_test, 2):
                with self.subTest(left=data_left, right=data_right, operator=operator):
                    S1 = pd.Series(data_left)
                    S2 = pd.Series(data_right)
                    # check_dtype=False because SDC implementation always returns float64 Series
                    pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2), check_dtype=False)

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support division/modulo/etc by zero')
    def test_series_operators_float_scalar(self):
        """Verifies using all various Series arithmetic binary operators
           on a float Series with default index and a scalar value"""
        n = 11
        np.random.seed(0)
        data_to_test = [np.arange(-5, -5 + n, dtype=np.float32),
                        np.ones(n + 3, dtype=np.float32),
                        np.random.ranf(n + 7)]
        scalar_values = [1., -1., 0., -0., 7., -5.]

        arithmetic_binops = ('+', '-', '*', '/', '//', '%', '**')
        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = self.jit(test_impl)
            for data, scalar, swap_operands in product(data_to_test, scalar_values, (False, True)):
                S = pd.Series(data)
                left, right = (S, scalar) if swap_operands else (scalar, S)
                with self.subTest(left=left, right=right, operator=operator):
                    pd.testing.assert_series_equal(hpat_func(S, scalar), test_impl(S, scalar), check_dtype=False)

    @skip_numba_jit('Not implemented in new-pipeline yet')
    def test_series_operators_inplace(self):
        arithmetic_binops = ('+=', '-=', '*=', '/=', '//=', '%=', '**=')

        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop2(operator)
            hpat_func = self.jit(test_impl)

            # TODO: extend to test arithmetic operations between numeric Series of different dtypes
            n = 11
            A1 = pd.Series(np.arange(1, n, dtype=np.float64), name='A')
            A2 = A1.copy(deep=True)
            B = pd.Series(np.ones(n - 1), name='B')
            hpat_func(A1, B)
            test_impl(A2, B)
            pd.testing.assert_series_equal(A1, A2)

    @skip_numba_jit('Not implemented in new-pipeline yet')
    def test_series_operators_inplace_scalar(self):
        arithmetic_binops = ('+=', '-=', '*=', '/=', '//=', '%=', '**=')

        for operator in arithmetic_binops:
            test_impl = _make_func_use_binop2(operator)
            hpat_func = self.jit(test_impl)

            # TODO: extend to test arithmetic operations between numeric Series of different dtypes
            n = 11
            S1 = pd.Series(np.arange(1, n, dtype=np.float64), name='A')
            S2 = S1.copy(deep=True)
            hpat_func(S1, 1)
            test_impl(S2, 1)
            pd.testing.assert_series_equal(S1, S2)

    @skip_numba_jit('operator.neg for SeriesType is not implemented in yet')
    def test_series_operator_neg(self):
        def test_impl(A):
            return -A
        hpat_func = self.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        pd.testing.assert_series_equal(hpat_func(A), test_impl(A))

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support Series indexes')
    def test_series_operators_comp_numeric(self):
        """Verifies using all various Series comparison binary operators on two integer Series with various indexes"""
        n = 11
        data_left = [1, 2, -1, 3, 4, 2, -3, 5, 6, 6, 0]
        data_right = [3, 2, -2, 1, 4, 1, -5, 6, 6, 3, -1]
        dtype_to_index = {'None': None,
                          'int': np.arange(n, dtype='int'),
                          'float': np.arange(n, dtype='float'),
                          'string': ['aa', 'aa', '', '', 'b', 'b', 'cccc', None, 'dd', 'ddd', None]}

        comparison_binops = ('<', '>', '<=', '>=', '!=', '==')
        for operator in comparison_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = self.jit(test_impl)
            for dtype, index_data in dtype_to_index.items():
                with self.subTest(operator=operator, index_dtype=dtype, index=index_data):
                    A = pd.Series(data_left, index=index_data)
                    B = pd.Series(data_right, index=index_data)
                    pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support comparing to inf')
    def test_series_operators_comp_numeric_scalar(self):
        """Verifies using all various Series comparison binary operators on an integer Series and scalar values"""
        S = pd.Series([1, 2, -1, 3, 4, 2, -3, 5, 6, 6, 0])

        scalar_values = [2, 2.0, -3, np.inf, -np.inf, np.PZERO, np.NZERO]
        comparison_binops = ('<', '>', '<=', '>=', '!=', '==')
        for operator in comparison_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = self.jit(test_impl)
            for scalar in scalar_values:
                with self.subTest(left=S, right=scalar, operator=operator):
                    pd.testing.assert_series_equal(hpat_func(S, scalar), test_impl(S, scalar))

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support comparing to inf')
    def test_series_operators_comp_str_scalar(self):
        """Verifies using all various Series comparison binary operators on an string Series and scalar values"""
        S = pd.Series(['aa', 'aa', '', '', 'b', 'b', 'cccc', None, 'dd', 'ddd', None])

        scalar_values = ['a', 'aa', 'ab', 'ba', '']
        comparison_binops = ('<', '>', '<=', '>=', '!=', '==')
        for operator in comparison_binops:
            test_impl = _make_func_use_binop1(operator)
            hpat_func = self.jit(test_impl)
            for scalar in scalar_values:
                with self.subTest(left=S, right=scalar, operator=operator):
                    pd.testing.assert_series_equal(hpat_func(S, scalar), test_impl(S, scalar))

    @skip_numba_jit
    def test_series_operators_inplace_array(self):
        def test_impl(A, B):
            A += B
            return A
        hpat_func = self.jit(test_impl)

        n = 11
        A = np.arange(n)**2.0  # TODO: use 2 for test int casting
        B = pd.Series(np.ones(n))
        np.testing.assert_array_equal(hpat_func(A.copy(), B), test_impl(A, B))

    @skip_numba_jit('Functionally test passes, but in old-style it checked fusion of parfors.\n'
                    'TODO: implement the same checks in new-pipeline')
    def test_series_fusion1(self):
        def test_impl(A, B):
            return A + B + 1
        hpat_func = self.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False)
        # self.assertEqual(count_parfor_REPs(), 1)

    @skip_numba_jit('Functionally test passes, but in old-style it checked fusion of parfors.\n'
                    'TODO: implement the same checks in new-pipeline')
    def test_series_fusion2(self):
        def test_impl(A, B):
            S = B + 2
            if A.iat[0] == 0:
                S = A + 1
            return S + B
        hpat_func = self.jit(test_impl)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n)**2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False)
        # self.assertEqual(count_parfor_REPs(), 3)

    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_add_numeric_scalar(self):
        """Verifies Series.operator.add implementation for numeric series and scalar second operand"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        dtype_to_index = {'None': None,
                          'int': np.arange(n, dtype='int'),
                          'float': np.arange(n, dtype='float'),
                          'string': ['aa', 'aa', 'b', 'b', 'cccc', 'dd', 'ddd']}

        int_scalar = 24
        for dtype, index_data in dtype_to_index.items():
            with self.subTest(index_dtype=dtype, index=index_data):
                if platform.system() == 'Windows' and not IS_32BITS:
                    A = pd.Series(np.arange(n, dtype=np.int64), index=index_data)
                else:
                    A = pd.Series(np.arange(n), index=index_data)
                result = hpat_func(A, int_scalar)
                result_ref = test_impl(A, int_scalar)
                pd.testing.assert_series_equal(result, result_ref, check_dtype=False, check_names=False)

        float_scalar = 24.0
        for dtype, index_data in dtype_to_index.items():
            with self.subTest(index_dtype=dtype, index=index_data):
                if platform.system() == 'Windows' and not IS_32BITS:
                    A = pd.Series(np.arange(n, dtype=np.int64), index=index_data)
                else:
                    A = pd.Series(np.arange(n), index=index_data)
                ref_result = test_impl(A, float_scalar)
                result = hpat_func(A, float_scalar)
                pd.testing.assert_series_equal(result, ref_result, check_dtype=False, check_names=False)

    def test_series_operator_add_numeric_same_index_default(self):
        """Verifies implementation of Series.operator.add between two numeric Series
        with default indexes and same size"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        dtypes_to_test = (np.int32, np.int64, np.float32, np.float64)
        for dtype_left, dtype_right in combinations(dtypes_to_test, 2):
            with self.subTest(left_series_dtype=dtype_left, right_series_dtype=dtype_right):
                A = pd.Series(np.arange(n), dtype=dtype_left)
                B = pd.Series(np.arange(n)**2, dtype=dtype_right)
                pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False)

    @skip_numba_jit
    @skip_sdc_jit("TODO: find out why pandas aligning series indexes produces Int64Index when common dtype is float\n"
                  "AssertionError: Series.index are different\n"
                  "Series.index classes are not equivalent\n"
                  "[left]:  Float64Index([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype='float64')\n"
                  "[right]: Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='int64')\n")
    def test_series_operator_add_numeric_same_index_numeric(self):
        """Verifies implementation of Series.operator.add between two numeric Series
           with the same numeric indexes of different dtypes"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        dtypes_to_test = (np.int32, np.int64, np.float32, np.float64)
        for dtype_left, dtype_right in combinations(dtypes_to_test, 2):
            with self.subTest(left_series_dtype=dtype_left, right_series_dtype=dtype_right):
                A = pd.Series(np.arange(n), index=np.arange(n, dtype=dtype_left))
                B = pd.Series(np.arange(n)**2, index=np.arange(n, dtype=dtype_right))
                pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False)

    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_add_numeric_same_index_numeric_fixme(self):
        """ Same as test_series_operator_add_same_index_numeric but with w/a for the problem.
        Can be deleted when the latter is fixed """
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        index_dtypes_to_test = (np.int32, np.int64, np.float32, np.float64)
        for dtype_left, dtype_right in combinations(index_dtypes_to_test, 2):
            # FIXME: skip the sub-test if one of the dtypes is float and the other is integer
            if not (np.issubdtype(dtype_left, np.integer) and np.issubdtype(dtype_right, np.integer)
                    or np.issubdtype(dtype_left, np.float) and np.issubdtype(dtype_right, np.float)):
                continue

            with self.subTest(left_series_dtype=dtype_left, right_series_dtype=dtype_right):
                A = pd.Series(np.arange(n), index=np.arange(n, dtype=dtype_left))
                B = pd.Series(np.arange(n)**2, index=np.arange(n, dtype=dtype_right))
                pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False)

    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_add_numeric_same_index_str(self):
        """Verifies implementation of Series.operator.add between two numeric Series with the same string indexes"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        A = pd.Series(np.arange(n), index=['a', 'c', 'e', 'c', 'b', 'a', 'o'])
        B = pd.Series(np.arange(n)**2, index=['a', 'c', 'e', 'c', 'b', 'a', 'o'])
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_add_numeric_align_index_int(self):
        """Verifies implementation of Series.operator.add between two numeric Series with non-equal integer indexes"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 11
        index_A = [0, 1, 1, 2, 3, 3, 3, 4, 6, 8, 9]
        index_B = [0, 1, 1, 3, 4, 4, 5, 5, 6, 6, 9]
        np.random.shuffle(index_A)
        np.random.shuffle(index_B)
        A = pd.Series(np.arange(n), index=index_A)
        B = pd.Series(np.arange(n)**2, index=index_B)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_add_numeric_align_index_str(self):
        """Verifies implementation of Series.operator.add between two numeric Series with non-equal string indexes"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 11
        index_A = ['', '', 'aa', 'aa', 'ae', 'ae', 'b', 'ccc', 'cccc', 'oo', 's']
        index_B = ['', '', 'aa', 'aa', 'cc', 'cccc', 'e', 'f', 'h', 'oo', 's']
        np.random.shuffle(index_A)
        np.random.shuffle(index_B)
        A = pd.Series(np.arange(n), index=index_A)
        B = pd.Series(np.arange(n)**2, index=index_B)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    @skip_numba_jit('TODO: fix Series.sort_values to handle both None and '' in string series')
    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_add_numeric_align_index_str_fixme(self):
        """Same as test_series_operator_add_align_index_str but with None values in string indexes"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 11
        index_A = ['', '', 'aa', 'aa', 'ae', 'b', 'ccc', 'cccc', 'oo', None, None]
        index_B = ['', '', 'aa', 'aa', 'cccc', 'f', 'h', 'oo', 's', None, None]
        np.random.shuffle(index_A)
        np.random.shuffle(index_B)
        A = pd.Series(np.arange(n), index=index_A)
        B = pd.Series(np.arange(n)**2, index=index_B)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_add_numeric_align_index_other_dtype(self):
        """Verifies implementation of Series.operator.add between two numeric Series
        with non-equal integer indexes of different dtypes"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        A = pd.Series(np.arange(3*n), index=np.arange(-n, 2*n, 1, dtype=np.int64))
        B = pd.Series(np.arange(3*n)**2, index=np.arange(0, 3*n, 1, dtype=np.float64))
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    @skip_sdc_jit('Arithmetic operations on Series with different sizes are not supported in old-style')
    def test_series_operator_add_numeric_diff_series_sizes(self):
        """Verifies implementation of Series.operator.add between two numeric Series with different sizes"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        size_A, size_B = 7, 25
        A = pd.Series(np.arange(size_A))
        B = pd.Series(np.arange(size_B)**2)
        result = hpat_func(A, B)
        result_ref = test_impl(A, B)
        pd.testing.assert_series_equal(result, result_ref, check_dtype=False, check_names=False)

    @skip_sdc_jit('Arithmetic operations on Series requiring alignment of indexes are not supported in old-style')
    def test_series_operator_add_align_index_int_capacity(self):
        """Verifies implementation of Series.operator.add and alignment of numeric indexes of large size"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 20000
        np.random.seed(0)
        index1 = np.random.randint(-30, 30, n)
        index2 = np.random.randint(-30, 30, n)
        A = pd.Series(np.random.ranf(n), index=index1)
        B = pd.Series(np.random.ranf(n), index=index2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    @skip_sdc_jit('Arithmetic operations on Series requiring alignment of indexes are not supported in old-style')
    def test_series_operator_add_align_index_str_capacity(self):
        """Verifies implementation of Series.operator.add and alignment of string indexes of large size"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 2000
        np.random.seed(0)
        valid_ids = ['', 'aaa', 'a', 'b', 'ccc', 'ef', 'ff', 'fff', 'fa', 'dddd']
        index1 = [valid_ids[i] for i in np.random.randint(0, len(valid_ids), n)]
        index2 = [valid_ids[i] for i in np.random.randint(0, len(valid_ids), n)]
        A = pd.Series(np.random.ranf(n), index=index1)
        B = pd.Series(np.random.ranf(n), index=index2)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    @skip_sdc_jit
    def test_series_operator_add_str_same_index_default(self):
        """Verifies implementation of Series.operator.add between two string Series
        with default indexes and same size"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        A = pd.Series(['a', '', 'ae', 'b', 'cccc', 'oo', None])
        B = pd.Series(['b', 'aa', '', 'b', 'o', None, 'oo'])
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_add_str_align_index_int(self):
        """Verifies implementation of Series.operator.add between two string Series with non-equal integer indexes"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        np.random.seed(0)
        index_A = [0, 1, 1, 2, 3, 3, 3, 4, 6, 8, 9]
        index_B = [0, 1, 1, 3, 4, 4, 5, 5, 6, 6, 9]
        np.random.shuffle(index_A)
        np.random.shuffle(index_B)
        data = ['', '', 'aa', 'aa', None, 'ae', 'b', 'ccc', 'cccc', None, 'oo']
        A = pd.Series(data, index=index_A)
        B = pd.Series(data, index=index_B)
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B), check_dtype=False, check_names=False)

    def test_series_operator_add_result_name1(self):
        """Verifies name of the Series resulting from appying Series.operator.add to different arguments"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        series_names = ['A', '', None, 'B']
        for left_name, right_name in combinations(series_names, 2):
            S1 = pd.Series(np.arange(n), name=left_name)
            S2 = pd.Series(np.arange(n, 0, -1), name=right_name)
            with self.subTest(left_series_name=left_name, right_series_name=right_name):
                # check_dtype=False because SDC implementation always returns float64 Series
                pd.testing.assert_series_equal(hpat_func(S1, S2), test_impl(S1, S2), check_dtype=False)

        # also verify case when second operator is scalar
        scalar = 3.0
        with self.subTest(scalar=scalar):
            S1 = pd.Series(np.arange(n), name='A')
            pd.testing.assert_series_equal(hpat_func(S1, scalar), test_impl(S1, scalar), check_dtype=False)

    @unittest.expectedFailure
    def test_series_operator_add_result_name2(self):
        """Verifies implementation of Series.operator.add differs from Pandas
           in returning unnamed Series when both operands are named Series with the same name"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        S1 = pd.Series(np.arange(n), name='A')
        S2 = pd.Series(np.arange(n, 0, -1), name='A')
        result = hpat_func(S1, S2)
        result_ref = test_impl(S1, S2)
        # check_dtype=False because SDC implementation always returns float64 Series
        pd.testing.assert_series_equal(result, result_ref, check_dtype=False)

    @unittest.expectedFailure
    def test_series_operator_add_series_dtype_promotion(self):
        """Verifies implementation of Series.operator.add differs from Pandas
           in dtype of resulting Series that is fixed to float64"""
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        dtypes_to_test = (np.int32, np.int64, np.float32, np.float64)
        for dtype_left, dtype_right in combinations(dtypes_to_test, 2):
            with self.subTest(left_series_dtype=dtype_left, right_series_dtype=dtype_right):
                A = pd.Series(np.array(np.arange(n), dtype=dtype_left))
                B = pd.Series(np.array(np.arange(n)**2, dtype=dtype_right))
                pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))

    @skip_sdc_jit('Arithmetic operations on string series not implemented in old-pipeline')
    def test_series_operator_add_str_scalar(self):
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        series_data = ['a', '', 'ae', 'b', 'cccc', 'oo', None]
        S = pd.Series(series_data)
        values_to_test = [' ', 'wq', '', '23']
        for scalar in values_to_test:
            with self.subTest(left=series_data, right=scalar):
                result_ref = test_impl(S, scalar)
                result = hpat_func(S, scalar)
                pd.testing.assert_series_equal(result, result_ref)

            with self.subTest(left=scalar, right=series_data):
                result_ref = test_impl(scalar, S)
                result = hpat_func(scalar, S)
                pd.testing.assert_series_equal(result, result_ref)

    @skip_sdc_jit('Arithmetic operations on string series not implemented in old-pipeline')
    def test_series_operator_add_str_unsupported(self):
        def test_impl(A, B):
            return A + B
        hpat_func = self.jit(test_impl)

        n = 7
        series_data = ['a', '', 'ae', 'b', 'cccc', 'oo', None]
        S = pd.Series(series_data)
        other_operands = [
            1,
            3.0,
            pd.Series(np.arange(n)),
            pd.Series([True, False, False, True, False, True, True]),
        ]

        for operand in other_operands:
            with self.subTest(right=operand):
                with self.assertRaises(TypingError) as raises:
                    hpat_func(S, operand)
                expected_msg = 'Operator add(). Not supported for not-comparable operands.'
                self.assertIn(expected_msg, str(raises.exception))

    @skip_sdc_jit('Arithmetic operations on string series not implemented in old-pipeline')
    def test_series_operator_mul_str_scalar(self):
        def test_impl(A, B):
            return A * B
        hpat_func = self.jit(test_impl)

        series_data = ['a', '', 'ae', 'b', ' ', 'cccc', 'oo', None]
        S = pd.Series(series_data)
        values_to_test = [-1, 0, 2, 5]
        for scalar in values_to_test:
            with self.subTest(left=series_data, right=scalar):
                result_ref = test_impl(S, scalar)
                result = hpat_func(S, scalar)
                pd.testing.assert_series_equal(result, result_ref)

            with self.subTest(left=scalar, right=series_data):
                result_ref = test_impl(scalar, S)
                result = hpat_func(scalar, S)
                pd.testing.assert_series_equal(result, result_ref)

    @skip_sdc_jit
    def test_series_operator_mul_str_same_index_default(self):
        """Verifies implementation of Series.operator.add between two string Series
        with default indexes and same size"""
        def test_impl(A, B):
            return A * B
        hpat_func = self.jit(test_impl)

        A = pd.Series(['a', '', 'ae', 'b', 'cccc', 'oo', None])
        B = pd.Series([-1, 2, 0, 5, 3, -5, 4])
        pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))

    @skip_sdc_jit('Arithmetic operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_mul_str_align_index_int1(self):
        """ Verifies implementation of Series.operator.add between two string Series
            with integer indexes containg same unique values (so alignment doesn't produce NaNs) """
        def test_impl(A, B):
            return A * B
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        shuffled_data = np.arange(n, dtype=np.int)
        np.random.shuffle(shuffled_data)
        index_A = shuffled_data
        np.random.shuffle(shuffled_data)
        index_B = shuffled_data
        str_series_values = ['', '', 'aa', 'aa', None, 'ae', 'b', 'ccc', 'cccc', None, 'oo']
        int_series_values = np.random.randint(-5, 5, n)

        A = pd.Series(str_series_values, index=index_A)
        B = pd.Series(int_series_values, index=index_B)
        for swap_operands in (False, True):
            if swap_operands:
                A, B = B, A
            with self.subTest(left=A, right=B):
                result = hpat_func(A, B)
                result_ref = test_impl(A, B)
                pd.testing.assert_series_equal(result, result_ref)

    @unittest.expectedFailure   # pandas can't calculate this due to adding NaNs to int series during alignment
    def test_series_operator_mul_str_align_index_int2(self):
        """ Verifies implementation of Series.operator.add between two string Series
            with integer indexes that cannot be aligned without NaNs """
        def test_impl(A, B):
            return A * B
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        index_A = [0, 1, 1, 2, 3, 3, 3, 4, 6, 8, 9]
        index_B = [0, 1, 1, 3, 4, 4, 5, 5, 6, 6, 9]
        np.random.shuffle(index_A)
        np.random.shuffle(index_B)
        str_series_values = ['', '', 'aa', 'aa', None, 'ae', 'b', 'ccc', 'cccc', None, 'oo']
        int_series_values = np.random.randint(-5, 5, n)

        A = pd.Series(str_series_values, index=index_A)
        B = pd.Series(int_series_values, index=index_B)
        for swap_operands in (False, True):
            if swap_operands:
                A, B = B, A
            with self.subTest(left=A, right=B):
                result = hpat_func(A, B)
                result_ref = test_impl(A, B)
                pd.testing.assert_series_equal(result, result_ref)

    @skip_sdc_jit('Arithmetic operations on string series not implemented in old-pipeline')
    def test_series_operator_mul_str_unsupported(self):
        def test_impl(A, B):
            return A * B
        hpat_func = self.jit(test_impl)

        series_data = ['a', '', 'ae', 'b', 'cccc', 'oo', None]
        S = pd.Series(series_data)
        other_operands = [
            'abc',
            3.0,
            pd.Series(series_data),
            pd.Series([True, False, False, True, False, True, True]),
        ]

        for operand in other_operands:
            with self.subTest(right=operand):
                with self.assertRaises(TypingError) as raises:
                    hpat_func(S, operand)
                expected_msg = 'Operator mul(). Not supported between operands of types:'
                self.assertIn(expected_msg, str(raises.exception))

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support Series indexes')
    def test_series_operator_lt_index_mismatch1(self):
        """Verifies correct exception is raised when comparing Series with non equal integer indexes"""
        def test_impl(A, B):
            return A < B
        hpat_func = self.jit(test_impl)

        n = 11
        np.random.seed(0)
        index1 = np.arange(n)
        index2 = np.copy(index1)
        np.random.shuffle(index2)
        A = pd.Series([1, 2, -1, 3, 4, 2, -3, 5, 6, 6, 0], index=index1)
        B = pd.Series([3, 2, -2, 1, 4, 1, -5, 6, 6, 3, -1], index=index2)

        with self.assertRaises(Exception) as context:
            test_impl(A, B)
        exception_ref = context.exception

        self.assertRaises(type(exception_ref), hpat_func, A, B)

    @skip_sdc_jit('Old-style implementation of operators doesn\'t support comparing Series of different lengths')
    def test_series_operator_lt_index_mismatch2(self):
        """Verifies correct exception is raised when comparing Series of different size with default indexes"""
        def test_impl(A, B):
            return A < B
        hpat_func = self.jit(test_impl)

        A = pd.Series([1, 2, -1, 3, 4, 2])
        B = pd.Series([3, 2, -2, 1, 4, 1, -5, 6, 6, 3, -1])

        with self.assertRaises(Exception) as context:
            test_impl(A, B)
        exception_ref = context.exception

        self.assertRaises(type(exception_ref), hpat_func, A, B)

    @skip_numba_jit('Numba propagates different exception:\n'
                    'numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)\n'
                    'Internal error at <numba.core.typeinfer.IntrinsicCallConstraint ...\n'
                    '\'Signature\' object is not iterable')
    @skip_sdc_jit('Typing checks not implemented for Series operators in old-style')
    def test_series_operator_lt_index_mismatch3(self):
        """Verifies correct exception is raised when comparing two Series with non-comparable indexes"""
        def test_impl(A, B):
            return A < B
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1, 2, -1, 3, 4, 2])
        S2 = pd.Series(['a', 'b', '', None, '2', 'ccc'])

        with self.assertRaises(TypingError) as raises:
            hpat_func(S1, S2)
        msg = 'Operator lt(). Not supported for series with not-comparable indexes.'
        self.assertIn(msg, str(raises.exception))

    @skip_sdc_jit('Comparison operations on Series with non-default indexes are not supported in old-style')
    @skip_numba_jit("TODO: find out why pandas aligning series indexes produces Int64Index when common dtype is float\n"
                    "AssertionError: Series.index are different\n"
                    "Series.index classes are not equivalent\n"
                    "[left]:  Float64Index([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype='float64')\n"
                    "[right]: Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='int64')\n")
    def test_series_operator_lt_index_dtype_promotion(self):
        """Verifies implementation of Series.operator.lt between two numeric Series
           with the same numeric indexes of different dtypes"""
        def test_impl(A, B):
            return A < B
        hpat_func = self.jit(test_impl)

        n = 7
        index_dtypes_to_test = (np.int32, np.int64, np.float32, np.float64)
        for dtype_left, dtype_right in combinations(index_dtypes_to_test, 2):
            with self.subTest(left_series_dtype=dtype_left, right_series_dtype=dtype_right):
                A = pd.Series(np.arange(n), index=np.arange(n, dtype=dtype_left))
                B = pd.Series(np.arange(n)**2, index=np.arange(n, dtype=dtype_right))
                pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))

    @skip_sdc_jit('Comparison operations on Series with non-default indexes are not supported in old-style')
    def test_series_operator_lt_index_dtype_promotion_fixme(self):
        """ Same as test_series_operator_lt_index_dtype_promotion but with w/a for the problem.
        Can be deleted when the latter is fixed """
        def test_impl(A, B):
            return A < B
        hpat_func = self.jit(test_impl)

        n = 7
        index_dtypes_to_test = (np.int32, np.int64, np.float32, np.float64)
        for dtype_left, dtype_right in combinations(index_dtypes_to_test, 2):
            # FIXME: skip the sub-test if one of the dtypes is float and the other is integer
            if not (np.issubdtype(dtype_left, np.integer) and np.issubdtype(dtype_right, np.integer)
                    or np.issubdtype(dtype_left, np.float) and np.issubdtype(dtype_right, np.float)):
                continue

            with self.subTest(left_series_dtype=dtype_left, right_series_dtype=dtype_right):
                A = pd.Series(np.arange(n), index=np.arange(n, dtype=dtype_left))
                B = pd.Series(np.arange(n)**2, index=np.arange(n, dtype=dtype_right))
                pd.testing.assert_series_equal(hpat_func(A, B), test_impl(A, B))

    @skip_sdc_jit('Typing checks not implemented for Series operators in old-style')
    def test_series_operator_lt_unsupported_dtypes(self):
        """Verifies correct exception is raised when comparing two Series with non-comparable dtypes"""
        def test_impl(A, B):
            return A < B
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1, 2, -1, 3, 4, 2])
        S2 = pd.Series(['a', 'b', '', None, '2', 'ccc'])

        with self.assertRaises(TypingError) as raises:
            hpat_func(S1, S2)
        msg = 'Operator lt(). Not supported for not-comparable operands.'
        self.assertIn(msg, str(raises.exception))

    @skip_sdc_jit
    def test_series_operator_lt_str(self):
        """Verifies implementation of Series.operator.lt between two string Series with default indexes"""
        def test_impl(A, B):
            return A < B
        hpat_func = self.jit(test_impl)

        A = pd.Series(['a', '', 'ae', 'b', 'cccc', 'oo', None])
        B = pd.Series(['b', 'aa', '', 'b', 'o', None, 'oo'])
        result = hpat_func(A, B)
        result_ref = test_impl(A, B)
        pd.testing.assert_series_equal(result, result_ref)

    def test_series_binops_numeric(self):
        arithmetic_methods = ('add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow')

        n = 11
        cases_series = [
            pd.Series(np.arange(1, n), name='A'),
            pd.Series(np.ones(n - 1), name='B'),
            pd.Series(np.arange(1, n) / 2, name='C'),
        ]

        for method in arithmetic_methods:
            test_impl = _make_func_use_method_arg1(method)
            hpat_func = self.jit(test_impl)
            for S1, S2 in combinations(cases_series, 2):
                with self.subTest(S1=S1, S2=S2, method=method):
                    # check_dtype=False because SDC arithmetic methods return only float Series
                    pd.testing.assert_series_equal(
                        hpat_func(S1, S2),
                        test_impl(S1, S2),
                        check_dtype=False)

    def test_series_binops_scalar(self):
        arithmetic_methods = ('add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow')

        n = 11
        cases_series = [
            pd.Series(np.arange(1, n)),
            pd.Series(np.ones(n - 1)),
            pd.Series(np.arange(1, n) / 2),
        ]
        cases_scalars = [0, 5, 0.5]

        for method in arithmetic_methods:
            test_impl = _make_func_use_method_arg1(method)
            hpat_func = self.jit(test_impl)

            for S1, scalar in product(cases_series, cases_scalars):
                with self.subTest(S1=S1, scalar=scalar, method=method):
                    # check_dtype=False because SDC arithmetic methods return only float Series
                    pd.testing.assert_series_equal(
                        hpat_func(S1, scalar),
                        test_impl(S1, scalar),
                        check_dtype=False)

    def test_series_binops_comp_numeric(self):
        comparison_methods = ('lt', 'gt', 'le', 'ge', 'ne', 'eq')

        n = 11
        np.random.seed(0)
        data_to_test = [np.arange(-5, -5 + n, dtype=np.float64),
                        gen_frand_array(n),
                        np.ones(n, dtype=np.int32),
                        np.random.randint(-5, 5, n)]

        for method in comparison_methods:
            test_impl = _make_func_use_method_arg1(method)
            hpat_func = self.jit(test_impl)

            for data1, data2 in product(data_to_test, repeat=2):
                A = pd.Series(data1)
                B = pd.Series(data2)
                with self.subTest(A=A, B=B):
                    pd.testing.assert_series_equal(
                        hpat_func(A, B),
                        test_impl(A, B),
                        check_names=False)

    def test_series_binops_comp_numeric_scalar(self):
        comparison_methods = ('lt', 'gt', 'le', 'ge', 'eq', 'ne')

        n = 11
        np.random.seed(0)
        data_to_test = [np.arange(-5, -5 + n, dtype=np.float64),
                        gen_frand_array(n),
                        np.ones(n, dtype=np.int32),
                        np.random.randint(-5, 5, n)]
        scalar_values = [1, -1, 0, 3, 7, -5, 4.2]

        for method in comparison_methods:
            test_impl = _make_func_use_method_arg1(method)
            hpat_func = self.jit(test_impl)

            for data, scalar in product(data_to_test, scalar_values):
                S = pd.Series(data)
                with self.subTest(S=S, scalar=scalar, method=method):
                    pd.testing.assert_series_equal(
                        hpat_func(S, scalar),
                        test_impl(S, scalar),
                        check_names=False)

    def test_series_binop_add_numeric(self):
        """Verifies implementation of Series.add method and fill_value param support on two float Series"""

        def test_impl(S1, S2, value):
            return S1.add(S2, fill_value=value)

        sdc_func = self.jit(test_impl)

        n = 100
        np.random.seed(0)
        cases_data = [
            np.arange(n, dtype=np.float64),
            gen_frand_array(n, nancount=25),
        ]
        cases_index = [
            None,
            np.arange(n),
            np.random.choice(np.arange(n), n, replace=False),
        ]
        cases_value = [
            None,
            np.nan,
            4,
            5.5
        ]

        for value, (arr1, arr2), (index1, index2) in product(
                cases_value,
                combinations_with_replacement(cases_data, 2),
                combinations_with_replacement(cases_index, 2)):
            S1 = pd.Series(arr1, index1)
            S2 = pd.Series(arr2, index2)
            with self.subTest(value=value, S1=S1, S2=S2):
                result = sdc_func(S1, S2, value)
                result_ref = test_impl(S1, S2, value)
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_binop_add_scalar_numeric(self):
        """Verifies implementation of Series.add method and fill_value param support on float Series and a scalar"""

        def test_impl(S1, S2, value):
            return S1.add(S2, fill_value=value)

        sdc_func = self.jit(test_impl)

        S1 = pd.Series([1, np.nan, 3, np.nan, 5, 6, 7, np.nan, 9])
        cases_value = [
            None,
            np.nan,
            4,
            5.5
        ]
        cases_scalar = [
            -2,
            5.5,
            np.nan
        ]

        for fill_value, scalar in product(cases_value, cases_scalar):
            with self.subTest(fill_value=fill_value, scalar=scalar):
                result = sdc_func(S1, scalar, fill_value)
                result_ref = test_impl(S1, scalar, fill_value)
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_binop_add_numeric_diff_sizes(self):
        """Verifies implementation of Series.add method and fill_value param support
        on two float Series with default indexes and different sizes"""

        def test_impl(a, b, value):
            return a.add(b, fill_value=value)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1, np.nan, 3, np.nan, 5, 6, 7, np.nan, 9])
        S2 = pd.Series([1, np.nan, 3, 4, np.nan, 6])
        values_to_test = [
            None,
            np.nan,
            2,
            2.1
        ]

        for value in values_to_test:
            with self.subTest(fill_value=value):
                result = hpat_func(S1, S2, value)
                result_ref = test_impl(S1, S2, value)
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_binop_lt_numeric(self):
        """Verifies implementation of Series.lt method and fill_value param support on two float Series"""

        def test_impl(S1, S2, value):
            return S1.lt(S2, fill_value=value)

        sdc_func = self.jit(test_impl)

        n = 100
        np.random.seed(0)
        cases_data = [
            np.arange(n, dtype=np.float64),
            gen_frand_array(n, nancount=25),
        ]
        cases_index = [
            None,
            np.arange(n),
            pd.RangeIndex(n)
        ]
        cases_value = [
            None,
            np.nan,
            4,
            5.5
        ]

        for value, (arr1, arr2), (index1, index2) in product(
                cases_value,
                combinations_with_replacement(cases_data, 2),
                combinations_with_replacement(cases_index, 2)):
            S1 = pd.Series(arr1, index1)
            S2 = pd.Series(arr2, index2)
            with self.subTest(value=value, S1=S1, S2=S2):
                result = sdc_func(S1, S2, value)
                result_ref = test_impl(S1, S2, value)
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_lt_scalar_numeric(self):
        """Verifies implementation of Series.lt method and fill_value param support on float Series and a scalar"""

        def test_impl(S1, S2, value):
            return S1.lt(S2, fill_value=value)

        sdc_func = self.jit(test_impl)

        S1 = pd.Series([1, np.nan, 3, np.nan, 5, 6, 7, np.nan, 9])
        cases_value = [
            None,
            np.nan,
            4,
            5.5
        ]
        cases_scalar = [
            -2,
            5.5,
            np.nan
        ]

        for fill_value, scalar in product(cases_value, cases_scalar):
            with self.subTest(S1=S1, fill_value=fill_value, scalar=scalar):
                result = sdc_func(S1, scalar, fill_value)
                result_ref = test_impl(S1, scalar, fill_value)
                pd.testing.assert_series_equal(result, result_ref)

    @unittest.expectedFailure  # Numba issue with 1/0 is different (inf) than in Numpy (nan)
    def test_series_binop_floordiv_numeric(self):
        def test_impl(a, b, value):
            return a.floordiv(b, fill_value=value)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series([1., -5., 2., 2., np.nan, 2., 1.])
        S2 = pd.Series([0., -2., 3., 2., 0., 2., 2.])

        fill_values = [
            None,
            np.nan,
            2,
            2.1
        ]

        for fill_value in fill_values:
            with self.subTest(fill_value=fill_value):
                result = hpat_func(S1, S2, fill_value)
                result_ref = test_impl(S1, S2, fill_value)
                pd.testing.assert_series_equal(result, result_ref)

    def test_series_binop_add_same_non_unique_index(self):
        """Verifies addition of two Series with equal indexes with duplicate values that don't require alignment"""

        def test_impl(a, b, value):
            return a.add(b, fill_value=value)
        hpat_func = self.jit(test_impl)

        n = 1000
        np.random.seed(0)
        series_values = [-5, 5, 1/3, 2, -25.5, 1, 0, np.nan, np.inf]
        index = np.random.choice(np.arange(n // 2), n)
        S1 = pd.Series(np.random.choice(series_values, n), index)
        S2 = pd.Series(np.random.choice(series_values, n), index)

        fill_values = [
            None,
            np.nan,
            2,
            2.1
        ]

        for fill_value in fill_values:
            with self.subTest(fill_value=fill_value):
                result = hpat_func(S1, S2, fill_value)
                result_ref = test_impl(S1, S2, fill_value)
                pd.testing.assert_series_equal(result, result_ref)

    @skip_numba_jit("Arithmetic methods for string series not implemented yet")
    def test_series_add_str(self):
        def test_impl(a, b, value):
            return a.add(b, fill_value=value)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['a', 'bb', 'cc', None, 'd', 'ed'])
        S2 = pd.Series(['aa', 'b', 'cc', 'a', None, 'de'])
        fill_value = 'asd'

        result = hpat_func(S1, S2, fill_value)
        result_ref = test_impl(S1, S2, fill_value)
        pd.testing.assert_series_equal(result, result_ref)

    def test_series_lt_str(self):
        """Verifies implementation of Series.lt method and fill_value param support on two string Series"""

        def test_impl(a, b, value):
            return a.lt(b, fill_value=value)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['a', 'bb', 'cc', None, 'd', 'ed'])
        S2 = pd.Series(['aa', 'b', 'cc', 'a', None, 'de'])
        fill_value = 'asd'

        result = hpat_func(S1, S2, fill_value)
        result_ref = test_impl(S1, S2, fill_value)
        pd.testing.assert_series_equal(result, result_ref)

    def test_series_lt_str_scalar(self):
        """Verifies implementation of Series.lt method and fill_value param support on a string Series and a scalar"""

        def test_impl(a, b, value):
            return a.lt(b, fill_value=value)
        hpat_func = self.jit(test_impl)

        S1 = pd.Series(['a', 'bb', 'cc', None, 'd', 'ed', None])
        fill_value = 'x1'
        cases_scalar = ['x', 'xy']

        for scalar in cases_scalar:
            with self.subTest(scalar=scalar):
                result = hpat_func(S1, scalar, fill_value)
                result_ref = test_impl(S1, scalar, fill_value)
                pd.testing.assert_series_equal(result, result_ref)


if __name__ == "__main__":
    unittest.main()

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


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import platform
import pyarrow.parquet as pq
import sdc
import string
import unittest
from itertools import combinations, combinations_with_replacement, islice, permutations, product
from numba import types
from numba.config import IS_32BITS
from numba.errors import TypingError
from numba.special import literally

from sdc.tests.test_series_apply import TestSeries_apply
from sdc.tests.test_series_map import TestSeries_map
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_OneDs,
                                  count_array_REPs,
                                  count_parfor_REPs,
                                  get_start_end,
                                  sdc_limitation,
                                  skip_inline,
                                  skip_numba_jit,
                                  skip_parallel,
                                  skip_sdc_jit,
                                  create_series_from_values,
                                  take_k_elements)
from sdc.tests.gen_test_data import ParquetGenerator

from sdc.tests.test_utils import test_global_input_data_unicode_kind1
from sdc.datatypes.common_functions import SDCLimitation


_cov_corr_series = [(pd.Series(x), pd.Series(y)) for x, y in [
    (
        [np.nan, -2., 3., 9.1],
        [np.nan, -2., 3., 5.0],
    ),
    # TODO(quasilyte): more intricate data for complex-typed series.
    # Some arguments make assert_almost_equal fail.
    # Functions that yield mismaching results:
    # _column_corr_impl and _column_cov_impl.
    (
        [complex(-2., 1.0), complex(3.0, 1.0)],
        [complex(-3., 1.0), complex(2.0, 1.0)],
    ),
    (
        [complex(-2.0, 1.0), complex(3.0, 1.0)],
        [1.0, -2.0],
    ),
    (
        [1.0, -4.5],
        [complex(-4.5, 1.0), complex(3.0, 1.0)],
    ),
]]

min_float64 = np.finfo('float64').min
max_float64 = np.finfo('float64').max

test_global_input_data_float64 = [
    [11., 35.2, -24., 0., np.NZERO, np.NINF, np.PZERO, min_float64],
    [1., np.nan, -1., 0., min_float64, max_float64, max_float64, min_float64],
    [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
]

min_int64 = np.iinfo('int64').min
max_int64 = np.iinfo('int64').max
max_uint64 = np.iinfo('uint64').max

test_global_input_data_signed_integer64 = [
    [1, -1, 0],
    [min_int64, max_int64, max_int64, min_int64],
]

test_global_input_data_integer64 = test_global_input_data_signed_integer64 + [[max_uint64, max_uint64]]

test_global_input_data_numeric = test_global_input_data_integer64 + test_global_input_data_float64

test_global_input_data_unicode_kind4 = [
    'ascii',
    '12345',
    '1234567890',
    '¡Y tú quién te crees?',
    '🐍⚡',
    '大处着眼，小处着手。',
]

def gen_srand_array(size, nchars=8):
    """Generate array of strings of specified size based on [a-zA-Z] + [0-9]"""
    accepted_chars = list(string.ascii_letters + string.digits)
    rands_chars = np.array(accepted_chars, dtype=(np.str_, 1))

    np.random.seed(100)
    return np.random.choice(rands_chars, size=nchars * size).view((np.str_, nchars))


def gen_frand_array(size, min=-100, max=100, nancount=0):
    """Generate array of float of specified size based on [-100-100]"""
    np.random.seed(100)
    res = (max - min) * np.random.sample(size) + min
    if nancount:
        res[np.random.choice(np.arange(size), nancount)] = np.nan
    return res


def gen_strlist(size, nchars=8, accepted_chars=None):
    """Generate list of strings of specified size based on accepted_chars"""
    if not accepted_chars:
        accepted_chars = string.ascii_letters + string.digits
    generated_chars = islice(permutations(accepted_chars, nchars), size)

    return [''.join(chars) for chars in generated_chars]


def series_values_from_argsort_result(series, argsorted):
    """
        Rearranges series values according to pandas argsort result.
        Used in tests to verify correct work of Series.argsort implementation for unstable sortings.
    """
    argsort_indices = argsorted.values
    result = np.empty_like(series.values)
    # pandas argsort returns -1 in positions of NaN elements
    nan_values_mask = argsort_indices == -1
    if np.any(nan_values_mask):
        result[nan_values_mask] = np.nan

    # pandas argsort returns indexes in series values after all nans were dropped from it
    # hence drop the NaN values, rearrange the rest with argsort result and assign them back to their positions
    series_notna_values = series.dropna().values
    result[~nan_values_mask] = series_notna_values.take(argsort_indices[~nan_values_mask])

    return result


#   Restores a series and checks the correct arrangement of indices,
#   taking into account the same elements for unstable sortings
#   Example: pd.Series([15, 3, 7, 3, 1],[2, 4, 6, 8, 10])
#   Result can be pd.Series([1, 3, 3, 7, 15],[10, 4, 8, 6, 2]) or pd.Series([1, 3, 3, 7, 15],[10, 8, 4, 6, 2])
#   if indices correct - return 0; wrong - return 1
def restore_series_sort_values(series, my_result_index, ascending):
    value_dict = {}
    nan_list = []
    data = np.copy(series.data)
    index = np.copy(series.index)
    for value in range(len(data)):
        # if np.isnan(data[value]):
        if series.isna()[index[value]]:
            nan_list.append(index[value])
        if data[value] in value_dict:
            value_dict[data[value]].append(index[value])
        else:
            value_dict[data[value]] = [index[value]]
    na = series.isna().sum()
    sort = np.argsort(data)
    result = np.copy(my_result_index)
    if not ascending:
        sort[:len(result)-na] = sort[:len(result)-na][::-1]
    for i in range(len(result)-na):
        check = 0
        for j in value_dict[data[sort[i]]]:
            if j == result[i]:
                check = 1
        if check == 0:
            return 1
    for i in range(len(result)-na, len(result)):
        check = 0
        for j in nan_list:
            if result[i] == j:
                check = 1
        if check == 0:
            return 1
    return 0


def _make_func_from_text(func_text, func_name='test_impl'):
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars[func_name]
    return test_impl


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


def ljust_usecase(series, width):
    return series.str.ljust(width)


def ljust_with_fillchar_usecase(series, width, fillchar):
    return series.str.ljust(width, fillchar)


def rjust_usecase(series, width):
    return series.str.rjust(width)


def rjust_with_fillchar_usecase(series, width, fillchar):
    return series.str.rjust(width, fillchar)


def istitle_usecase(series):
    return series.str.istitle()


def isspace_usecase(series):
    return series.str.isspace()


def isalpha_usecase(series):
    return series.str.isalpha()


def islower_usecase(series):
    return series.str.islower()


def isalnum_usecase(series):
    return series.str.isalnum()


def isnumeric_usecase(series):
    return series.str.isnumeric()


def isdigit_usecase(series):
    return series.str.isdigit()


def isdecimal_usecase(series):
    return series.str.isdecimal()


def isupper_usecase(series):
    return series.str.isupper()


def lower_usecase(series):
    return series.str.lower()


def upper_usecase(series):
    return series.str.upper()


def strip_usecase(series, to_strip=None):
    return series.str.strip(to_strip)


def lstrip_usecase(series, to_strip=None):
    return series.str.lstrip(to_strip)


def rstrip_usecase(series, to_strip=None):
    return series.str.rstrip(to_strip)


def contains_usecase(series, pat, case=True, flags=0, na=None, regex=True):
    return series.str.contains(pat, case, flags, na, regex)


class TestSeries(
    TestSeries_apply,
    TestSeries_map,
    TestCase
):

    # SDC operator methods returns only float Series
    def test_series_add(self):
        def test_impl(S1, S2, value):
            return S1.add(S2, fill_value=value)

        sdc_func = self.jit(test_impl)

        data = [0, 1, 2, 3, 4]
        index = [3, 4, 3, 9, 2]
        value = None

        S1 = pd.Series(data, index)
        S2 = pd.Series(index, data)
        print(sdc_func(S1, S2, value))


if __name__ == "__main__":
    unittest.main()

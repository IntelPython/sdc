# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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

from sdc.tests.tests_perf.test_perf_base import *
from sdc.tests.tests_perf.test_perf_utils import *
from sdc.tests.test_utils import *
import pandas as pd


def usecase_series_min(input_data):
    start_time = time.time()
    res = input_data.min()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_max(input_data):
    start_time = time.time()
    res = input_data.max()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_abs(input_data):
    start_time = time.time()
    res = input_data.abs()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_value_counts(input_data):
    start_time = time.time()
    res = input_data.value_counts()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_nsmallest(input_data):
    start_time = time.time()
    res = input_data.nsmallest()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_nlargest(input_data):
    start_time = time.time()
    res = input_data.nlargest()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_var(input_data):
    start_time = time.time()
    res = input_data.var()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_shift(input_data):
    start_time = time.time()
    res = input_data.shift()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_copy(input_data):
    start_time = time.time()
    res = input_data.copy()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_sum(input_data):
    start_time = time.time()
    res = input_data.sum()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_idxmax(input_data):
    start_time = time.time()
    res = input_data.idxmax()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_idxmin(input_data):
    start_time = time.time()
    res = input_data.idxmin()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_prod(input_data):
    start_time = time.time()
    res = input_data.prod()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_quantile(input_data):
    start_time = time.time()
    res = input_data.quantile()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_mean(input_data):
    start_time = time.time()
    res = input_data.mean()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_unique(input_data):
    start_time = time.time()
    res = input_data.unique()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_cumsum(input_data):
    start_time = time.time()
    res = input_data.cumsum()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_nunique(input_data):
    start_time = time.time()
    res = input_data.nunique()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_count(input_data):
    start_time = time.time()
    res = input_data.count()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_median(input_data):
    start_time = time.time()
    res = input_data.median()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_argsort(input_data):
    start_time = time.time()
    res = input_data.argsort()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_sort_values(input_data):
    start_time = time.time()
    res = input_data.sort_values()
    finish_time = time.time()

    return finish_time - start_time, res


def usecase_series_dropna(input_data):
    start_time = time.time()
    res = input_data.dropna()
    finish_time = time.time()

    return finish_time - start_time, res


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series.TestSeriesMethods
class TestSeriesMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.total_data_length = {
            'series_float_min': [10 ** 9],
            'series_float_max': [10 ** 9],
            'series_float_abs': [3 * 10 ** 8],
            'series_float_value_counts': [3 * 10 ** 5],
            'series_float_nsmallest': [10 ** 9],
            'series_float_nlargest': [10 ** 9],
            'series_float_var': [5 * 10 ** 8],
            'series_float_shift': [5 * 10 ** 8],
            'series_float_sum': [10 ** 9],
            'series_float_idxmax': [10 ** 9],
            'series_float_idxmin': [10 ** 9],
            'series_float_prod': [5 * 10 ** 8],
            'series_float_quantile': [10 ** 8],
            'series_float_mean': [10 ** 8],
            'series_float_unique': [10 ** 5],
            'series_float_cumsum': [2 * 10 ** 8],
            'series_float_nunique': [10 ** 5],
            'series_float_count': [2 * 10 ** 9],
            'series_float_median': [10 ** 8],
            'series_float_argsort': [10 ** 5],
            'series_float_sort_values': [10 ** 5],
            'series_float_dropna': [2 * 10 ** 8]
        }

    def _test_series(self, pyfunc, name, input_data=None):
        input_data = input_data or test_global_input_data_float64
        full_input_data_length = sum(len(i) for i in input_data)
        hpat_func = sdc.jit(pyfunc)
        for data_length in self.total_data_length[name]:
            data = perf_data_gen_fixed_len(input_data, full_input_data_length, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data, iter_number=self.iter_number)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = get_times(hpat_func, test_data, iter_number=self.iter_number)
            self.test_results.add(name, 'JIT', test_data.size, exec_times, boxing_results=boxing_times,
                                  compile_results=compile_results)

            exec_times, _ = get_times(pyfunc, test_data, iter_number=self.iter_number)
            self.test_results.add(name, 'Reference', test_data.size, test_results=exec_times)

    def test_series_float_min(self):
        self._test_series(usecase_series_min, 'series_float_min')

    def test_series_float_max(self):
        self._test_series(usecase_series_max, 'series_float_max')

    def test_series_float_abs(self):
        self._test_series(usecase_series_abs, 'series_float_abs')

    def test_series_float_value_counts(self):
        self._test_series(usecase_series_value_counts, 'series_float_value_counts')

    def test_series_float_nsmallest(self):
        self._test_series(usecase_series_nsmallest, 'series_float_nsmallest')

    def test_series_float_nlargest(self):
        self._test_series(usecase_series_nlargest, 'series_float_nlargest')

    def test_series_float_var(self):
        self._test_series(usecase_series_var, 'series_float_var')

    def test_series_float_shift(self):
        self._test_series(usecase_series_shift, 'series_float_shift')

    def test_series_float_sum(self):
        self._test_series(usecase_series_sum, 'series_float_sum')

    def test_series_float_idxmax(self):
        self._test_series(usecase_series_idxmax, 'series_float_idxmax')

    def test_series_float_idxmin(self):
        self._test_series(usecase_series_idxmin, 'series_float_idxmin')

    def test_series_float_prod(self):
        self._test_series(usecase_series_prod, 'series_float_prod')

    def test_series_float_quantile(self):
        self._test_series(usecase_series_quantile, 'series_float_quantile')

    def test_series_float_mean(self):
        self._test_series(usecase_series_quantile, 'series_float_mean')

    def test_series_float_unique(self):
        self._test_series(usecase_series_unique, 'series_float_unique')

    def test_series_float_cumsum(self):
        self._test_series(usecase_series_cumsum, 'series_float_cumsum')

    def test_series_float_nunique(self):
        self._test_series(usecase_series_nunique, 'series_float_nunique')

    def test_series_float_count(self):
        self._test_series(usecase_series_count, 'series_float_count')

    def test_series_float_median(self):
        self._test_series(usecase_series_median, 'series_float_median')

    def test_series_float_argsort(self):
        self._test_series(usecase_series_argsort, 'series_float_argsort')

    def test_series_float_sort_values(self):
        self._test_series(usecase_series_sort_values, 'series_float_sort_values')

    def test_series_float_dropna(self):
        self._test_series(usecase_series_dropna, 'series_float_dropna')

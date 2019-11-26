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

import time
import random

import pandas
import sdc

from .test_perf_base import TestBase
from sdc.tests.test_utils import test_global_input_data_float64
from .test_perf_utils import calc_compilation, get_times, perf_data_gen_fixed_len

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


class TestSeries(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.total_data_length = {
            'series_min': [10 ** 9],
            'series_max': [10 ** 9],
            'series_abs': [3 * 10 ** 8],
            'series_value_counts': [3 * 10 ** 5],
            'series_nsmallest': [10 ** 9],
            'series_nlargest': [10 ** 9],
            'series_var': [5 * 10 ** 8],
            'series_shift': [5 * 10 ** 8],
            'series_copy': [10 ** 8],
            'series_sum': [10 ** 9],
            'series_idxmax': [10 ** 9],
            'series_idxmin': [10 ** 9],
            'series_prod': [5 * 10 ** 8],
            'series_quantile': [10 ** 8],
            'series_mean': [10 ** 8],
            'series_unique': [10 ** 5],
            'series_cumsum': [2 * 10 ** 8],
            'series_nunique': [10 ** 5],
            'series_count': [2 * 10 ** 9],
            'series_median': [10 ** 8],
            'series_argsort': [10 ** 5],
            'series_sort_values': [10 ** 5],
            'series_dropna': [2 * 10 ** 8]
        }

    def _test_jitted(self, pyfunc, record, *args, **kwargs):
        # compilation time
        record["compile_results"] = calc_compilation(pyfunc, *args, **kwargs)

        sdc_func = sdc.jit(pyfunc)

        # Warming up
        sdc_func(*args, **kwargs)

        # execution and boxing time
        record["test_results"], record["boxing_results"] = \
            get_times(sdc_func, *args, **kwargs)

    def _test_python(self, pyfunc, record, *args, **kwargs):
        record["test_results"], _ = \
            get_times(pyfunc, *args, **kwargs)

    def _test_case(self, pyfunc, name):
        input_data = test_global_input_data_float64
        full_input_data_length = sum(len(i) for i in input_data)
        for data_length in self.total_data_length[name]:
            base = {
                "test_name": name,
                "data_size": data_length,
            }
            data = perf_data_gen_fixed_len(input_data, full_input_data_length,
                                           data_length)
            test_data = pandas.Series(data)

            record = base.copy()
            record["test_type"] = 'SDC'
            self._test_jitted(pyfunc, record, test_data)
            self.test_results.add(**record)

            record = base.copy()
            record["test_type"] = 'Python'
            self._test_python(pyfunc, record, test_data)
            self.test_results.add(**record)

    def test_series_float_min(self):
        self._test_case(usecase_series_min, 'series_min')

    def test_series_float_max(self):
        self._test_case(usecase_series_max, 'series_max')

    def test_series_float_abs(self):
        self._test_case(usecase_series_abs, 'series_abs')

    def test_series_float_value_counts(self):
        self._test_case(usecase_series_value_counts, 'series_value_counts')

    def test_series_float_nsmallest(self):
        self._test_case(usecase_series_nsmallest, 'series_nsmallest')

    def test_series_float_nlargest(self):
        self._test_case(usecase_series_nlargest, 'series_nlargest')

    def test_series_float_var(self):
        self._test_case(usecase_series_var, 'series_var')

    def test_series_float_shift(self):
        self._test_case(usecase_series_shift, 'series_shift')

    def test_series_float_copy(self):
        self._test_case(usecase_series_shift, 'series_copy')

    def test_series_float_sum(self):
        self._test_case(usecase_series_sum, 'series_sum')

    def test_series_float_idxmax(self):
        self._test_case(usecase_series_idxmax, 'series_idxmax')

    def test_series_float_idxmin(self):
        self._test_case(usecase_series_idxmin, 'series_idxmin')

    def test_series_float_prod(self):
        self._test_case(usecase_series_prod, 'series_prod')

    def test_series_float_quantile(self):
        self._test_case(usecase_series_quantile, 'series_quantile')

    def test_series_float_mean(self):
        self._test_case(usecase_series_quantile, 'series_mean')

    def test_series_float_unique(self):
        self._test_case(usecase_series_unique, 'series_unique')

    def test_series_float_cumsum(self):
        self._test_case(usecase_series_cumsum, 'series_cumsum')

    def test_series_float_nunique(self):
        self._test_case(usecase_series_nunique, 'series_nunique')

    def test_series_float_count(self):
        self._test_case(usecase_series_count, 'series_count')

    def test_series_float_median(self):
        self._test_case(usecase_series_median, 'series_median')

    def test_series_float_argsort(self):
        self._test_case(usecase_series_argsort, 'series_argsort')

    def test_series_float_sort_values(self):
        self._test_case(usecase_series_sort_values, 'series_sort_values')

    def test_series_float_dropna(self):
        self._test_case(usecase_series_dropna, 'series_dropna')

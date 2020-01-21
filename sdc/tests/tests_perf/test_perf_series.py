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

import pandas as pd
import numpy as np

import time
import random

import pandas
import sdc

from .test_perf_base import TestBase
from sdc.tests.test_utils import test_global_input_data_float64
from .test_perf_utils import calc_compilation, get_times, perf_data_gen_fixed_len
from .generator import test_perf_generator, test_perf_generator_two_par, usecase_gen


def usecase_series_astype_int(input_data):
    # astype to int8
    start_time = time.time()
    input_data.astype(np.int8)
    finish_time = time.time()
    res_time = finish_time - start_time

    return res_time, input_data


def usecase_series_chain_add_and_sum(A, B):
    start_time = time.time()
    res = (A + B).sum()
    finish_time = time.time()
    res_time = finish_time - start_time

    return res_time, res


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series.TestSeriesMethods.test_series_{method_name}
class TestSeriesMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

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

    def _test_case(self, pyfunc, name, total_data_length, input_data=test_global_input_data_float64):
        test_name = 'Series.{}'.format(name)
        full_input_data_length = sum(len(i) for i in input_data)
        for data_length in total_data_length:
            base = {
                "test_name": test_name,
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

    def _test_binary_operations(self, pyfunc, name, total_data_length, input_data=None):
        test_name = 'Series.{}'.format(name)
        np.random.seed(0)
        hpat_func = sdc.jit(pyfunc)
        for data_length in total_data_length:

            # TODO: replace with generic function to generate random sequence of floats
            data1 = np.random.ranf(data_length)
            data2 = np.random.ranf(data_length)
            A = pd.Series(data1)
            B = pd.Series(data2)

            compile_results = calc_compilation(pyfunc, A, B, iter_number=self.iter_number)

            # Warming up
            hpat_func(A, B)

            exec_times, boxing_times = get_times(hpat_func, A, B, iter_number=self.iter_number)
            self.test_results.add(test_name, 'SDC', A.size, exec_times, boxing_times,
                                  compile_results=compile_results, num_threads=self.num_threads)
            exec_times, _ = get_times(pyfunc, A, B, iter_number=self.iter_number)
            self.test_results.add(test_name, 'Python', A.size, exec_times, num_threads=self.num_threads)

    def test_series_float_astype_int(self):
        self._test_case(usecase_gen('astype(np.int8)'), 'series_astype_int', [10 ** 5],
                        input_data=[test_global_input_data_float64[0]])

    def test_series_chain_add_and_sum(self):
        self._test_series_binary_operations(usecase_series_chain_add_and_sum,
                                            'series_chain_add_and_sum',
                                            [20 * 10 ** 7, 25 * 10 ** 7, 30 * 10 ** 7])


# (method_name, parameters, total_data_length, call_expression)
cases = [
    ('abs', '', [3 * 10 ** 8]),
    ('apply', 'lambda x: x', [10 ** 7]),
    ('argsort', '', [10 ** 5]),
    ('at', '', [10 ** 7], 'at[3]'),
    ('copy', '', [10 ** 8]),
    ('count', '', [2 * 10 ** 9]),
    ('cumsum', '', [2 * 10 ** 8]),
    ('describe', '', [10 ** 7]),
    ('dropna', '', [2 * 10 ** 8]),
    ('fillna', '-1', [2 * 10 ** 7]),
    ('head', '', [10 ** 8]),
    ('idxmax', '', [10 ** 9]),
    ('idxmin', '', [10 ** 9]),
    ('isna', '', [2 * 10 ** 7]),
    ('max', '', [10 ** 9]),
    ('mean', '', [10 ** 8]),
    ('median', '', [10 ** 8]),
    ('min', '', [10 ** 9]),
    ('min', 'skipna=False', [10 ** 7]),
    ('nlargest', '', [4 * 10 ** 7]),
    ('nsmallest', '', [10 ** 9]),
    ('nunique', '', [10 ** 5]),
    ('prod', '', [5 * 10 ** 8]),
    ('pct_change', 'periods=1, limit=None, freq=None', [10 ** 7]),
    ('quantile', '', [10 ** 8]),
    ('shift', '', [5 * 10 ** 8]),
    ('sort_values', '', [10 ** 5]),
    ('std', '', [10 ** 7]),
    ('sum', '', [10 ** 9]),
    ('value_counts', '', [3 * 10 ** 5]),
    ('var', '', [5 * 10 ** 8]),
    ('unique', '', [10 ** 5]),
]

cases_two_par = [
    ('add', '', [10 ** 7]),
    ('append', '', [10 ** 7]),
    ('corr', '', [10 ** 7]),
    ('cov', '', [10 ** 8]),
    ('div', '', [10 ** 7]),
    ('eq', '', [10 ** 7]),
    ('floordiv', '', [10 ** 7]),
    ('ge', '', [10 ** 7]),
    ('pow', '', [10 ** 7]),
]


test_perf_generator(cases, TestSeriesMethods, 'series')
test_perf_generator_two_par(cases_two_par, TestSeriesMethods, 'series')

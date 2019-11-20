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

import itertools
import os
import time
import unittest
from contextlib import contextmanager

import pandas as pd
import numpy as np

from hpat.tests.test_utils import *
from hpat.tests.tests_perf.test_perf_utils import *

min_float64 = np.finfo('float64').min
max_float64 = np.finfo('float64').max

test_global_input_data_float64 = [
    [1., np.nan, -1., 0., min_float64, max_float64, max_float64, min_float64],
    [np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.NINF, np.NZERO]
]

def usecase_series_abs(input_data):
    start_time = time.time()
    input_data.abs()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_cov(input_data):
    start_time = time.time()
    input_data.cov()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_value_counts(input_data):
    start_time = time.time()
    input_data.value_counts()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_min(input_data):
    start_time = time.time()
    input_data.min()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_max(input_data):
    start_time = time.time()
    input_data.max()
    finish_time = time.time()

    return finish_time - start_time


@contextmanager
def do_jit(f):
    """Context manager to jit function"""
    cfunc = hpat.jit(f)
    try:
        yield cfunc
    finally:
        del cfunc


def calc_time(func, *args, **kwargs):
    """Calculate execution time of specified function."""
    start_time = time.time()
    func(*args, **kwargs)
    finish_time = time.time()

    return finish_time - start_time


def calc_compile_time(func, *args, **kwargs):
    """Calculate compile time as difference between first 2 runs."""
    return calc_time(func, *args, **kwargs) - calc_time(func, *args, **kwargs)


def calc_compilation(pyfunc, data, iter_number=5):
    """Calculate compile time several times."""
    compile_times = []
    for _ in range(iter_number):
        with do_jit(pyfunc) as cfunc:
            compile_time = calc_compile_time(cfunc, data)
            compile_times.append(compile_time)

    return compile_times


def get_times(f, test_data, iter_number=5):
    """Get time of boxing+unboxing and internal execution"""
    exec_times = []
    boxing_times = []
    for _ in range(iter_number):
        ext_start = time.time()
        int_result = f(test_data)
        ext_finish = time.time()

        exec_times.append(int_result)
        boxing_times.append(max(ext_finish - ext_start - int_result, 0))

    return exec_times, boxing_times

# python -m hpat.runtests hpat.tests.tests_perf.test_perf_series.TestSeriesFloatMethods
class TestSeriesFloatMethods(unittest.TestCase):
    iter_number = 5

    @classmethod
    def setUpClass(cls):
        print("SetUp")
        cls.test_results = TestResults()
        if is_true(os.environ.get('LOAD_PREV_RESULTS')):
            cls.test_results.load()

        cls.total_data_length = [10**4 + 513, 10**5 + 2025]
        cls.num_threads = int(os.environ.get('NUMBA_NUM_THREADS', config.NUMBA_NUM_THREADS))
        cls.threading_layer = os.environ.get('NUMBA_THREADING_LAYER', config.THREADING_LAYER)

    @classmethod
    def tearDownClass(cls):
        print("tearnDown")
        cls.test_results.print()
        cls.test_results.dump()

    def _test_series_float(self, pyfunc, name, input_data=None):
        print("test")
        input_data = input_data or test_global_input_data_float64
        hpat_func = hpat.jit(pyfunc)
        for data_length in itertools.product(self.total_data_length):
            data = self.generate(input_data, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data, iter_number=self.iter_number)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = get_times(hpat_func, test_data, iter_number=self.iter_number)
            self.test_results.add(test_name=name, test_type='JIT', data_size=test_data.size, test_results=exec_times,
                                  boxing_results=boxing_times, compile_results=compile_results, num_threads=self.num_threads)
            exec_times, _ = get_times(pyfunc, test_data, iter_number=self.iter_number)
            self.test_results.add(test_name=name, test_type='Reference', data_size=test_data.size,
                                  test_results=exec_times, num_threads=self.num_threads)

    def generate(self, input, maxlen):
        result = []
        i = 0
        N = len(input)
        while len(result) < maxlen[0]:
            n = (i - i // 2) % N
            result.extend(input[n])
            i += 1
        return result[:maxlen[0]]

    def test_series_float_abs(self):
        print("abs")
        self._test_series_float(usecase_series_abs, 'series_float_abs')

    # def test_series_float_cov(self):
    #     print("cov")
    #     self._test_series_float(usecase_series_cov, 'series_float_cov')
    #
    # def test_series_float_value_counts(self):
    #     print("value-counts")
    #     self._test_series_float(usecase_series_value_counts, 'series_float_value_counts')
    #
    # def test_series_float_min(self):
    #     print("min")
    #     self._test_series_float(usecase_series_min, 'series_float_min')
    #
    # def test_series_float_max(self):
    #     print("max")
    #     self._test_series_float(usecase_series_max, 'series_float_max')

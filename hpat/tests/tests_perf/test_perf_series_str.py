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

from hpat.tests.test_utils import *
from hpat.tests.tests_perf.test_perf_utils import *


def usecase_series_len(input_data):
    start_time = time.time()
    input_data.str.len()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_capitalize(input_data):
    start_time = time.time()
    input_data.str.capitalize()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_lower(input_data):
    start_time = time.time()
    input_data.str.lower()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_swapcase(input_data):
    start_time = time.time()
    input_data.str.swapcase()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_title(input_data):
    start_time = time.time()
    input_data.str.title()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_upper(input_data):
    start_time = time.time()
    input_data.str.upper()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_lstrip(input_data):
    start_time = time.time()
    input_data.str.lstrip()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_rstrip(input_data):
    start_time = time.time()
    input_data.str.rstrip()
    finish_time = time.time()

    return finish_time - start_time


def usecase_series_strip(input_data):
    start_time = time.time()
    input_data.str.strip()
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


class TestSeriesStringMethods(unittest.TestCase):
    iter_number = 5

    @classmethod
    def setUpClass(cls):
        cls.test_results = TestResults()
        if is_true(os.environ.get('LOAD_PREV_RESULTS')):
            cls.test_results.load()

        cls.total_data_length = [10**4 + 513, 10**5 + 2025]
        cls.width = [16, 64, 512, 1024]
        cls.num_threads = int(os.environ.get('NUMBA_NUM_THREADS', config.NUMBA_NUM_THREADS))
        cls.threading_layer = os.environ.get('NUMBA_THREADING_LAYER', config.THREADING_LAYER)

    @classmethod
    def tearDownClass(cls):
        cls.test_results.print()
        cls.test_results.dump()

    def _test_series_str(self, pyfunc, name, input_data=None):
        input_data = input_data or test_global_input_data_unicode_kind4
        hpat_func = hpat.jit(pyfunc)
        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(input_data, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data, iter_number=self.iter_number)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = get_times(hpat_func, test_data, iter_number=self.iter_number)
            self.test_results.add(name, 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_threads)
            exec_times, _ = get_times(pyfunc, test_data, iter_number=self.iter_number)
            self.test_results.add(name, 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_threads)

    def test_series_str_len(self):
        self._test_series_str(usecase_series_len, 'series_str_len')

    def test_series_str_capitalize(self):
        self._test_series_str(usecase_series_capitalize, 'series_str_capitalize')

    def test_series_str_lower(self):
        self._test_series_str(usecase_series_lower, 'series_str_lower')

    def test_series_str_swapcase(self):
        self._test_series_str(usecase_series_swapcase, 'series_str_swapcase')

    def test_series_str_title(self):
        self._test_series_str(usecase_series_title, 'series_str_title')

    def test_series_str_upper(self):
        self._test_series_str(usecase_series_upper, 'series_str_upper')

    def test_series_str_lstrip(self):
        input_data = ['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]
        self._test_series_str(usecase_series_lstrip, 'series_str_lstrip', input_data=input_data)

    def test_series_str_rstrip(self):
        input_data = ['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]
        self._test_series_str(usecase_series_rstrip, 'series_str_rstrip', input_data=input_data)

    def test_series_str_strip(self):
        input_data = ['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]
        self._test_series_str(usecase_series_strip, 'series_str_strip', input_data=input_data)


if __name__ == "__main__":
    unittest.main()

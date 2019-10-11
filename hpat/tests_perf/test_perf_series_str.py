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

import hpat
from hpat.tests_perf.test_perf_utils import *


STRIP_CASES = [
    'ascii',
    'tú quiénc te crees?',
    '大处 着眼，c小处着手c。大大c大处'
]


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


def calc_time(f, data):
    """Calculate execution time of specified function."""
    start_time = time.time()
    f(data)
    finish_time = time.time()

    return finish_time - start_time


def calc_compile_time(f, data):
    """Calculate compile time as difference between first 2 runs."""
    return calc_time(f, data) - calc_time(f, data)


def calc_compilation(pyfunc, data):
    """Calculate compile time several times."""
    compile_iteration_number = 5

    compile_times = []
    for _ in range(compile_iteration_number):
        with do_jit(pyfunc) as cfunc:
            compile_times.append(calc_compile_time(cfunc, data))

    return compile_times


def calc_results(f, test_data):
    """Calculate time of boxing+unboxing and internal execution"""
    test_iteration_number = 5

    exec_times = []
    boxing_times = []
    for _ in range(test_iteration_number):
        ext_start = time.time()

        int_result = f(test_data)

        ext_finish = time.time()
        exec_times.append(int_result)
        boxing_times.append(max(ext_finish - ext_start - int_result, 0))

    return exec_times, boxing_times


class TestSeriesStringMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_results = TestResults()
        cls.test_results.load()

        cls.total_data_length = [10**4 + 513, 10**5 + 2025]
        cls.width = [16, 64, 512, 1024]
        cls.num_thread = int(os.environ.get('NUMBA_NUM_THREADS', config.NUMBA_NUM_THREADS))

    @classmethod
    def tearDownClass(cls):
        cls.test_results.print()
        cls.test_results.dump()

    def test_series_str_len(self):
        pyfunc = usecase_series_len
        hpat_func = hpat.jit(pyfunc)

        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(STRIP_CASES, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_len', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_len', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)


    def test_series_str_capitalize(self):
        pyfunc = usecase_series_capitalize
        hpat_func = hpat.jit(pyfunc)

        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(STRIP_CASES, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_capitalize', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_capitalize', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)

    def test_series_str_lower(self):
        pyfunc = usecase_series_lower
        hpat_func = hpat.jit(pyfunc)

        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(STRIP_CASES, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_lower', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_lower', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)

    def test_series_str_swapcase(self):
        pyfunc = usecase_series_swapcase
        hpat_func = hpat.jit(pyfunc)

        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(STRIP_CASES, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_swapcase', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_swapcase', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)

    def test_series_str_title(self):
        pyfunc = usecase_series_title
        hpat_func = hpat.jit(pyfunc)

        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(STRIP_CASES, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_title', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_title', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)

    def test_series_str_upper(self):
        pyfunc = usecase_series_upper
        hpat_func = hpat.jit(pyfunc)

        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(STRIP_CASES, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_upper', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_upper', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)

    def test_series_str_lstrip(self):
        pyfunc = usecase_series_lstrip
        hpat_func = hpat.jit(pyfunc)

        strip_cases = ['\t{}  '.format(case) for case in STRIP_CASES]
        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(strip_cases, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_lstrip', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_lstrip', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)

    def test_series_str_rstrip(self):
        pyfunc = usecase_series_rstrip
        hpat_func = hpat.jit(pyfunc)

        strip_cases = ['\t{}  '.format(case) for case in STRIP_CASES]
        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(strip_cases, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_rstrip', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_rstrip', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)

    def test_series_str_strip(self):
        pyfunc = usecase_series_strip
        hpat_func = hpat.jit(pyfunc)

        strip_cases = ['\t{}  '.format(case) for case in STRIP_CASES]
        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(strip_cases, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = calc_results(hpat_func, test_data)
            self.test_results.add('series_str_strip', 'JIT', test_data.size, data_width, exec_times,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_thread)
            exec_times, _ = calc_results(pyfunc, test_data)
            self.test_results.add('series_str_strip', 'Reference', test_data.size, data_width, exec_times,
                                  num_threads=self.num_thread)


if __name__ == "__main__":
    unittest.main()

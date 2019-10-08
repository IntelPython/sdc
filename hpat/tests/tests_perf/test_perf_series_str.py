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

import unittest
import time

import pandas as pd

import hpat

from hpat.tests.tests_perf.test_perf_utils import *


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


def calc_compilation_time(f, test_data):
    first_start = time.time()
    f(test_data)
    first_finish = time.time()
    first_result = first_finish - first_start

    second_start = time.time()
    f(test_data)
    second_finish = time.time()
    second_result = second_finish - second_start

    return first_result - second_result


def calc_results(f, test_data, test_name, test_type, test_data_width, compilation_time=0):
    test_iteration_number = 5

    exec_times = []
    boxing_times = []
    for local_iter in range(test_iteration_number):
        ext_start = time.time()

        int_result = f(test_data)

        ext_finish = time.time()
        exec_times.append(int_result)
        boxing_times.append(ext_finish - ext_start - int_result)

    add_results(test_name, test_type, test_data_width, exec_times, boxing_times, compilation_time=compilation_time)


class TestSeriesStringMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(self):
        self.total_data_size_bytes = 1.0E+07

    @classmethod
    def tearDownClass(self):
        print_results()

    def test_series_str_len(self):
        pyfunc = usecase_series_len
        hpat_func = hpat.jit(pyfunc)

        test_data = pd.Series(perf_data_gen(STRIP_CASES, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(STRIP_CASES, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_len', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_len', 'Reference', data_width)

    def test_series_str_capitalize(self):
        pyfunc = usecase_series_capitalize
        hpat_func = hpat.jit(pyfunc)

        test_data = pd.Series(perf_data_gen(STRIP_CASES, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(STRIP_CASES, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_capitalize', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_capitalize', 'Reference', data_width)

    def test_series_str_lower(self):
        pyfunc = usecase_series_lower
        hpat_func = hpat.jit(pyfunc)

        test_data = pd.Series(perf_data_gen(STRIP_CASES, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(STRIP_CASES, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_lower', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_lower', 'Reference', data_width)

    def test_series_str_swapcase(self):
        pyfunc = usecase_series_swapcase
        hpat_func = hpat.jit(pyfunc)

        test_data = pd.Series(perf_data_gen(STRIP_CASES, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(STRIP_CASES, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_swapcase', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_swapcase', 'Reference', data_width)

    def test_series_str_title(self):
        pyfunc = usecase_series_title
        hpat_func = hpat.jit(pyfunc)

        test_data = pd.Series(perf_data_gen(STRIP_CASES, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(STRIP_CASES, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_title', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_title', 'Reference', data_width)

    def test_series_str_upper(self):
        pyfunc = usecase_series_upper
        hpat_func = hpat.jit(pyfunc)

        test_data = pd.Series(perf_data_gen(STRIP_CASES, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(STRIP_CASES, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_upper', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_upper', 'Reference', data_width)

    def test_series_str_lstrip(self):
        pyfunc = usecase_series_lstrip
        hpat_func = hpat.jit(pyfunc)

        strip_cases = ['\t{}  '.format(case) for case in STRIP_CASES]
        test_data = pd.Series(perf_data_gen(strip_cases, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(strip_cases, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_lstrip', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_lstrip', 'Reference', data_width)

    def test_series_str_rstrip(self):
        pyfunc = usecase_series_rstrip
        hpat_func = hpat.jit(pyfunc)

        strip_cases = ['\t{}  '.format(case) for case in STRIP_CASES]
        test_data = pd.Series(perf_data_gen(strip_cases, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(strip_cases, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_rstrip', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_rstrip', 'Reference', data_width)

    def test_series_str_strip(self):
        pyfunc = usecase_series_strip
        hpat_func = hpat.jit(pyfunc)

        strip_cases = ['\t{}  '.format(case) for case in STRIP_CASES]
        test_data = pd.Series(perf_data_gen(strip_cases, 256, self.total_data_size_bytes))
        compilation_time = calc_compilation_time(hpat_func, test_data)

        for data_width in [16, 64, 512, 1024]:
            test_data = pd.Series(perf_data_gen(strip_cases, data_width, self.total_data_size_bytes))
            calc_results(hpat_func, test_data, 'series_str_strip', 'JIT', data_width, compilation_time)
            calc_results(pyfunc, test_data, 'series_str_strip', 'Reference', data_width)


if __name__ == "__main__":
    unittest.main()

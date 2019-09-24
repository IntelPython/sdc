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
import numba
import time
import sys
import pandas


def perf_data_gen(tmpl, max_item_len, max_bytes_size):

    result = []
    obj_size = sys.getsizeof(tmpl, -1)

    while (obj_size < max_bytes_size) and (obj_size >= 0):
        for item in tmpl:
            local_item = item
            local_item_len = len(local_item)

            while (local_item_len < max_item_len) and (local_item_len >= 0):
                local_item += item
                local_item_len = len(local_item)

            result.append(local_item)

        obj_size = sys.getsizeof(result, -1)

    return result


test_results_data = pandas.DataFrame(columns=['name', 'type', 'width', 'Time(s)'])


def add_results(test_name, test_type, test_data_width, test_results):
    local_results = pandas.DataFrame({'name': test_name,
                                      'type': test_type,
                                      'width': test_data_width,
                                      'Time(s)': test_results})
    global test_results_data
#     test_results_data = pandas.concat([test_results_data, local_results])
    test_results_data = test_results_data.append(local_results)


def print_results():
    global test_results_data

    # Following code is terrible. needs to be redeveloped
    # print(test_results_data)
    median_col = test_results_data.groupby(['name', 'type', 'width'])['Time(s)'].median()
    min_col = test_results_data.groupby(['name', 'type', 'width'])['Time(s)'].min()
    max_col = test_results_data.groupby(['name', 'type', 'width'])['Time(s)'].max()

    test_results_data = test_results_data.set_index(['name', 'type', 'width'])
    test_results_data['median'] = median_col
    test_results_data['min'] = min_col
    test_results_data['max'] = max_col
    test_results_data = test_results_data.reset_index()

    print(test_results_data.groupby(['name', 'type', 'width', 'median', 'min', 'max']).first())


STRIP_CASES = [
    'ascii',
    'tú quiénc te crees?',
    '大处 着眼，c小处着手c。大大c大处'
]


def usecase_split(input_data):
    test_iteration_number = 5
    iter_time = []

    for local_iter in range(test_iteration_number):
        start_time = time.time()

        for string in input_data:
            string.split('c')

        finish_time = time.time()
        local_result = finish_time - start_time

        iter_time.append(local_result)

    return iter_time


def usecase_join(input_data):
    test_iteration_number = 5
    iter_time = []

    for local_iter in range(test_iteration_number):
        start_time = time.time()

        for string in input_data:
            'X'.join([string, string, string, string])

        finish_time = time.time()
        local_result = finish_time - start_time

        iter_time.append(local_result)

    return iter_time


def usecase_center(input_data):
    result_str_grow_factor = 1.4
    test_iteration_number = 5
    iter_time = []

    for local_iter in range(test_iteration_number):
        start_time = time.time()

        for string in input_data:
            new_string_len = int(len(string) * result_str_grow_factor)
            '''The width must be an Integer'''

            string.center(new_string_len, '+')

        finish_time = time.time()
        local_result = finish_time - start_time

        iter_time.append(local_result)

    return iter_time


class TestStringMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(self):
        self.total_data_size_bytes = 1.0E+07

    @classmethod
    def tearDownClass(self):
        print_results()

    def test_unicode_split(self):
        pyfunc = usecase_split
        hpat_func = numba.njit(pyfunc)

        for data_width in [16, 64, 512, 1024]:
            test_data = perf_data_gen(STRIP_CASES, data_width, self.total_data_size_bytes)
            add_results('unicode_split', 'JIT', data_width, hpat_func(test_data))
            add_results('unicode_split', 'Reference', data_width, pyfunc(test_data))

    def test_unicode_join(self):
        pyfunc = usecase_join
        hpat_func = numba.njit(pyfunc)

        for data_width in [16, 64, 512, 1024]:
            test_data = perf_data_gen(STRIP_CASES, data_width, self.total_data_size_bytes)
            add_results('unicode_join', 'JIT', data_width, hpat_func(test_data))
            add_results('unicode_join', 'Reference', data_width, pyfunc(test_data))

    def test_unicode_center(self):
        pyfunc = usecase_center
        hpat_func = numba.njit(pyfunc, parallel=True)

        for data_width in [16, 64, 512, 1024]:
            test_data = perf_data_gen(STRIP_CASES, 32, self.total_data_size_bytes)
            add_results('unicode_center', 'JIT', data_width, hpat_func(test_data))
            add_results('unicode_center', 'Reference', data_width, pyfunc(test_data))


if __name__ == "__main__":
    unittest.main()

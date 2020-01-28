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

import itertools
import os
import time
import unittest
from contextlib import contextmanager

import pandas as pd

from sdc.tests.test_utils import *
from sdc.tests.tests_perf.test_perf_base import TestBase
from sdc.tests.tests_perf.test_perf_utils import *
from .generator import generate_test_cases
from .generator import TestCase as TC


test_global_input_data_unicode_kind1 = [
    'ascii',
    '12345',
    '1234567890',
]


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series_str.TestSeriesStringMethods.test_series_str_{method_name}
class TestSeriesStringMethods(TestBase):
    iter_number = 5
    results_class = TestResultsStr

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.width = [16, 64, 512, 1024]

    def _test_case(self, pyfunc, name, total_data_length, input_data=None, *args, **kwargs):
        test_name = 'series_str_{}'.format(name)
        input_data = input_data or test_global_input_data_unicode_kind4
        hpat_func = sdc.jit(pyfunc)
        for data_length, data_width in itertools.product(total_data_length, self.width):
            data = perf_data_gen_fixed_len(input_data, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data, iter_number=self.iter_number)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = get_times(hpat_func, test_data, iter_number=self.iter_number)

            self.test_results.add(test_name, 'SDC', test_data.size, exec_times, data_width,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_threads)
            exec_times, _ = get_times(pyfunc, test_data, iter_number=self.iter_number)
            self.test_results.add(test_name, 'Python', test_data.size, exec_times, data_width,
                                  num_threads=self.num_threads)


cases = [
    TC(name='center', params='1', size=[10 ** 4, 10 ** 5],  input_data=test_global_input_data_unicode_kind1),
    TC(name='endswith', params='"e"', size=[10 ** 4, 10 ** 5]),
    TC(name='find', params='"e"', size=[10 ** 4, 10 ** 5]),
    TC(name='len', size=[10 ** 4, 10 ** 5]),
    TC(name='ljust', params='1', size=[10 ** 4, 10 ** 5], input_data=test_global_input_data_unicode_kind1),
    TC(name='lower', size=[10 ** 4, 10 ** 5]),
    TC(name='lstrip', size=[10 ** 4, 10 ** 5],
       input_data=['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]),
    TC(name='rjust', params='1', size=[10 ** 4, 10 ** 5], input_data=test_global_input_data_unicode_kind1),
    TC(name='rstrip', size=[10 ** 4, 10 ** 5],
       input_data=['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]),
    TC(name='startswith', params='"e"', size=[10 ** 4, 10 ** 5]),
    TC(name='strip', size=[10 ** 4, 10 ** 5],
       input_data=['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]),
    TC(name='upper', size=[10 ** 4, 10 ** 5]),
    TC(name='zfill', params='1', size=[10 ** 4, 10 ** 5], input_data=test_global_input_data_unicode_kind1),
]

generate_test_cases(cases, TestSeriesStringMethods, 'series', 'str')

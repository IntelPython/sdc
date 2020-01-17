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
from sdc.io.csv_ext import to_varname


def usecase_gen(call_expression):
    func_name = 'usecase_func'

    func_text = f"""\
def {func_name}(input_data):
  start_time = time.time()
  res = input_data.str.{call_expression}
  finish_time = time.time()
  return finish_time - start_time, res
"""
    print(func_text)
    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    _gen_impl = loc_vars[func_name]

    return _gen_impl


class TestSeriesStringMethods(TestBase):
    iter_number = 5
    results_class = TestResultsStr

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.total_data_length = [10**4, 10**5]
        cls.width = [16, 64, 512, 1024]
        cls.num_threads = int(os.environ.get('NUMBA_NUM_THREADS', config.NUMBA_NUM_THREADS))
        cls.threading_layer = os.environ.get('NUMBA_THREADING_LAYER', config.THREADING_LAYER)

    def _test_series_str(self, pyfunc, name, input_data=None):
        input_data = input_data or test_global_input_data_unicode_kind4
        hpat_func = sdc.jit(pyfunc)
        for data_length, data_width in itertools.product(self.total_data_length, self.width):
            data = perf_data_gen_fixed_len(input_data, data_width, data_length)
            test_data = pd.Series(data)

            compile_results = calc_compilation(pyfunc, test_data, iter_number=self.iter_number)
            # Warming up
            hpat_func(test_data)

            exec_times, boxing_times = get_times(hpat_func, test_data, iter_number=self.iter_number)
            self.test_results.add(name, 'JIT', test_data.size, exec_times, data_width,
                                  boxing_times, compile_results=compile_results, num_threads=self.num_threads)
            exec_times, _ = get_times(pyfunc, test_data, iter_number=self.iter_number)
            self.test_results.add(name, 'Reference', test_data.size, exec_times, data_width,
                                  num_threads=self.num_threads)


def test_gen(name, params, input_data):
    func_name = 'func'

    func_text = f"""\
def {func_name}(self):
  self._test_series_str(usecase_gen('{name}({params})'), 'series_str_{name}', input_data={input_data})
"""

    global_vars = {'usecase_gen': usecase_gen}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _gen_impl = loc_vars[func_name]

    return _gen_impl


cases = [
    ('capitalize', ''),
    ('center', 'width=1'),
    ('endswith', '"e"'),
    ('find', '"e"'),
    ('len', ''),
    ('lower', ''),
    ('lstrip', '', ['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]),
    ('rstrip', '', ['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]),
    ('startswith', '"e"'),
    ('strip', '', ['\t{}  '.format(case) for case in test_global_input_data_unicode_kind4]),
    ('swapcase', ''),
    ('title', ''),
    ('upper', ''),
]


for params in cases:
    if len(params) == 3:
        func, param, input_data = params
    else:
        func, param = params
        input_data = None
    name = func
    if param:
        name += "_" + to_varname(param).replace('__', '_')
    func_name = 'test_series_str_{}'.format(name)
    print(func_name)
    setattr(TestSeriesStringMethods, func_name, test_gen(func, param, input_data))

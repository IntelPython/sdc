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

import pandas as pd
import numpy as np

import time
import random

import pandas
import sdc

from .test_perf_base import TestBase
from sdc.tests.test_utils import test_global_input_data_float64
from .test_perf_utils import calc_compilation, get_times, perf_data_gen_fixed_len


def usecase_gen(call_expression):
    func_name = 'usecase_func'

    func_text = f"""\
def {func_name}(input_data):
  start_time = time.time()
  res = input_data.{call_expression}
  finish_time = time.time()
  return finish_time - start_time, res
"""

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    _gen_impl = loc_vars[func_name]

    return _gen_impl


def usecase_gen_two_par(name, par):
    func_name = 'usecase_func'

    func_text = f"""\
def {func_name}(A, B):
  start_time = time.time()
  res = A.{name}(B, {par})
  finish_time = time.time()
  return finish_time - start_time, res
"""

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    _gen_impl = loc_vars[func_name]

    return _gen_impl


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


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series.TestSeriesMethods
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
        full_input_data_length = sum(len(i) for i in input_data)
        for data_length in total_data_length:
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

    def _test_series_binary_operations(self, pyfunc, name, total_data_length, input_data=None):
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
            self.test_results.add(name, 'JIT', A.size, exec_times, boxing_times,
                                  compile_results=compile_results, num_threads=self.num_threads)
            exec_times, _ = get_times(pyfunc, A, B, iter_number=self.iter_number)
            self.test_results.add(name, 'Reference', A.size, exec_times, num_threads=self.num_threads)

    def test_series_float_astype_int(self):
        self._test_case(usecase_gen('astype(np.int8)'), 'series_astype_int', [10 ** 5],
                        input_data=[test_global_input_data_float64[0]])

    def test_series_chain_add_and_sum(self):
        self._test_series_binary_operations(usecase_series_chain_add_and_sum,
                                            [20 * 10 ** 7, 25 * 10 ** 7, 30 * 10 ** 7],
                                            'series_chain_add_and_sum')


def test_gen(name, params, data_length):
    func_name = 'func'

    func_text = f"""\
def {func_name}(self):
  self._test_case(usecase_gen('{name}({params})'), 'series_{name}', {data_length})
"""

    global_vars = {'usecase_gen': usecase_gen}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _gen_impl = loc_vars[func_name]

    return _gen_impl


def test_gen_two_par(name, params, data_length):
    func_name = 'func'

    func_text = f"""\
def {func_name}(self):
  self._test_series_binary_operations(usecase_gen_two_par('{name}', '{params}'), 'series_{name}', {data_length})
"""

    global_vars = {'usecase_gen_two_par': usecase_gen_two_par}

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _gen_impl = loc_vars[func_name]

    return _gen_impl

cases = {
    'abs': ('abs', '', [3 * 10 ** 8]),
    'argsort': ('argsort', '', [10 ** 5]),
    'copy': ('copy', '', [10 ** 8]),
    'count': ('count', '', [2 * 10 ** 9]),
    'cumsum': ('cumsum', '', [2 * 10 ** 8]),
    'describe': ('describe', '', [10 ** 7]),
    'dropna': ('dropna', '', [2 * 10 ** 8]),
    'fillna': ('fillna', '-1', [2 * 10 ** 7]),
    'idxmax': ('idxmax', '', [10 ** 9]),
    'idxmin': ('idxmin', '', [10 ** 9]),
    'isna': ('isna', '', [2 * 10 ** 7]),
    'max': ('max', '', [10 ** 9]),
    'mean': ('mean', '', [10 ** 8]),
    'median': ('median', '', [10 ** 8]),
    'min': ('min', '', [10 ** 9]),
    'min_skipna_True': ('min', 'skipna=True', [10 ** 7]),
    'nlargest': ('nlargest', '', [4 * 10 ** 7]),
    'nsmallest': ('nsmallest', '', [10 ** 9]),
    'nunique': ('nunique', '', [10 ** 5]),
    'prod': ('prod', '', [5 * 10 ** 8]),
    'quantile': ('quantile', '', [10 ** 8]),
    'shift': ('shift', '', [5 * 10 ** 8]),
    'sort_values': ('sort_values', '', [10 ** 5]),
    'std': ('std', '', [10 ** 7]),
    'sum': ('sum', '', [10 ** 9]),
    'value_counts': ('value_counts', '', [3 * 10 ** 5]),
    'var': ('var', '', [5 * 10 ** 8]),
    'unique': ('unique', '', [10 ** 5]),
}

cases_two_par = {'add': ('add', '', [10 ** 7]),
                 'append': ('append', '', [10 ** 7]),
                 'corr': ('corr', '', [10 ** 7]),
                 'cov': ('cov', '', [10 ** 8]),
                 'pow': ('pow', '', [10 ** 7])}

for name, params in cases.items():
    func = params[0]
    param = params[1]
    length = params[2]
    func_name = 'test_series_float_{}'.format(name)
    setattr(TestSeriesMethods, func_name, test_gen(func, param, length))

for name, params in cases_two_par.items():
    func = params[0]
    param = params[1]
    length = params[2]
    func_name = 'test_series_float_{}'.format(name)
    setattr(TestSeriesMethods, func_name, test_gen_two_par(func, param, length))

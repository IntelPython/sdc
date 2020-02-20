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

import time

import pandas
import numpy as np

from sdc.tests.test_utils import test_global_input_data_float64
from sdc.tests.tests_perf.test_perf_base import TestBase
from sdc.tests.tests_perf.test_perf_utils import perf_data_gen_fixed_len
from .generator import generate_test_cases
from .generator import TestCase as TC


rolling_usecase_tmpl = """
def series_rolling_{method_name}_usecase(data, {extra_usecase_params}):
    start_time = time.time()
    for i in range({ncalls}):
        res = data.rolling({rolling_params}).{method_name}({method_params})
    end_time = time.time()
    return end_time - start_time, res
"""


def get_rolling_params(window=100, min_periods=None):
    """Generate supported rolling parameters"""
    rolling_params = [f'{window}']
    if min_periods:
        rolling_params.append(f'min_periods={min_periods}')

    return ', '.join(rolling_params)


def gen_series_rolling_usecase(method_name, rolling_params=None,
                               extra_usecase_params='',
                               method_params='', ncalls=1):
    """Generate series rolling method use case"""
    if not rolling_params:
        rolling_params = get_rolling_params()

    func_text = rolling_usecase_tmpl.format(**{
        'method_name': method_name,
        'extra_usecase_params': extra_usecase_params,
        'rolling_params': rolling_params,
        'method_params': method_params,
        'ncalls': ncalls
    })

    global_vars = {'np': np, 'time': time}
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _series_rolling_usecase = loc_vars[f'series_rolling_{method_name}_usecase']

    return _series_rolling_usecase


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series_rolling.TestSeriesRollingMethods
class TestSeriesRollingMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.map_ncalls_dlength = {
            'mean': (100, [2 * 10 ** 5]),
            'sum': (100, [8 * 10 ** 5]),
        }

    def _test_case(self, pyfunc, name, total_data_length, data_num=1,
                   input_data=test_global_input_data_float64):
        test_name = 'Series.rolling.{}'.format(name)

        if input_data is None:
            input_data = test_global_input_data_float64

        full_input_data_length = sum(len(i) for i in input_data)
        for data_length in total_data_length:
            base = {
                'test_name': test_name,
                'data_size': data_length,
            }
            data = perf_data_gen_fixed_len(input_data, full_input_data_length, data_length)
            test_data = pandas.Series(data)

            args = [test_data]
            for i in range(data_num - 1):
                np.random.seed(i)
                extra_data = np.random.ranf(data_length)
                args.append(pandas.Series(extra_data))

            self._test_jit(pyfunc, base, *args)
            self._test_py(pyfunc, base, *args)

    def _test_series_rolling_method(self, name, rolling_params=None,
                                    extra_usecase_params='', method_params=''):
        ncalls, total_data_length = self.map_ncalls_dlength[name]
        usecase = gen_series_rolling_usecase(name, rolling_params=rolling_params,
                                             extra_usecase_params=extra_usecase_params,
                                             method_params=method_params, ncalls=ncalls)
        data_num = 1
        if extra_usecase_params:
            data_num += len(extra_usecase_params.split(', '))
        self._test_case(usecase, name, total_data_length, data_num=data_num)

    def test_series_rolling_mean(self):
        self._test_series_rolling_method('mean')

    def test_series_rolling_sum(self):
        self._test_series_rolling_method('sum')


cases = [
    TC(name='apply', size=[10 ** 7], params='func=lambda x: np.nan if len(x) == 0 else x.mean()'),
    TC(name='corr', size=[10 ** 7]),
    TC(name='count', size=[10 ** 7]),
    TC(name='cov', size=[10 ** 7]),
    TC(name='kurt', size=[10 ** 7]),
    TC(name='max', size=[10 ** 7]),
    TC(name='median', size=[10 ** 7]),
    TC(name='min', size=[10 ** 7]),
    TC(name='quantile', size=[10 ** 7], params='0.2'),
    TC(name='skew', size=[10 ** 7]),
    TC(name='std', size=[10 ** 7]),
    TC(name='var', size=[10 ** 7]),
]


generate_test_cases(cases, TestSeriesRollingMethods, 'series', 'rolling({})'.format(get_rolling_params()))

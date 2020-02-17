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
import string
import time

import numba
import pandas
import numpy as np

from sdc.tests.test_utils import test_global_input_data_float64
from sdc.tests.tests_perf.test_perf_base import TestBase
from sdc.tests.tests_perf.test_perf_utils import (calc_compilation, get_times,
                                                  perf_data_gen_fixed_len)
from .generator import generate_test_cases
from .generator import TestCase as TC


def get_rolling_params(window=100, min_periods=None):
    """Generate supported rolling parameters"""
    rolling_params = [f'{window}']
    if min_periods:
        rolling_params.append(f'min_periods={min_periods}')

    return ', '.join(rolling_params)


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series_rolling.TestSeriesRollingMethods
class TestSeriesRollingMethods(TestBase):
    # more than 19 columns raise SystemError: CPUDispatcher() returned a result with an error set
    max_columns_num = 19

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

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


cases = [
    TC(name='apply', size=[10 ** 7], params='func=lambda x: np.nan if len(x) == 0 else x.mean()'),
    TC(name='corr', size=[10 ** 7]),
    TC(name='count', size=[10 ** 7]),
    TC(name='cov', size=[10 ** 7]),
    TC(name='kurt', size=[10 ** 7]),
    TC(name='max', size=[10 ** 7]),
    TC(name='mean', size=[10 ** 7]),
    TC(name='median', size=[10 ** 7]),
    TC(name='min', size=[10 ** 7]),
    TC(name='quantile', size=[10 ** 7], params='0.2'),
    TC(name='skew', size=[10 ** 7]),
    TC(name='std', size=[10 ** 7]),
    TC(name='sum', size=[4 * 10 ** 7]),
    TC(name='var', size=[10 ** 7]),
]


generate_test_cases(cases, TestSeriesRollingMethods, 'series', 'rolling({})'.format(get_rolling_params()))

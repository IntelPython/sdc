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

from sdc.tests.test_utils import test_global_input_data_float64
from sdc.tests.tests_perf.test_perf_base import TestBase
from sdc.tests.tests_perf.test_perf_utils import (calc_compilation, get_times,
                                                  perf_data_gen_fixed_len)
from .generator import test_perf_generator


def get_rolling_params(window=100, min_periods=None):
    """Generate supported rolling parameters"""
    rolling_params = [f'{window}']
    if min_periods:
        rolling_params.append(f'min_periods={min_periods}')

    return ', '.join(rolling_params)


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_df_rolling.TestDFRollingMethods
class TestDFRollingMethods(TestBase):
    # more than 19 columns raise SystemError: CPUDispatcher() returned a result with an error set
    max_columns_num = 19

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _test_jitted(self, pyfunc, record, *args, **kwargs):
        # compilation time
        record['compile_results'] = calc_compilation(pyfunc, *args, **kwargs)

        cfunc = numba.njit(pyfunc)

        # Warming up
        cfunc(*args, **kwargs)

        # execution and boxing time
        record['test_results'], record['boxing_results'] = get_times(cfunc, *args, **kwargs)

    def _test_python(self, pyfunc, record, *args, **kwargs):
        record['test_results'], _ = get_times(pyfunc, *args, **kwargs)

    def _test_case(self, pyfunc, name, total_data_length, test_name=None,
                   input_data=test_global_input_data_float64, columns_num=10):
        if columns_num > self.max_columns_num:
            columns_num = self.max_columns_num

        test_name = 'DataFrame.rolling.{}'.format(name)

        full_input_data_length = sum(len(i) for i in input_data)
        for data_length in total_data_length:
            base = {
                'test_name': test_name,
                'data_size': data_length,
            }
            data = perf_data_gen_fixed_len(input_data, full_input_data_length, data_length)
            test_data = pandas.DataFrame({col: data for col in string.ascii_uppercase[:columns_num]})

            record = base.copy()
            record['test_type'] = 'SDC'
            self._test_jitted(pyfunc, record, test_data)
            self.test_results.add(**record)

            record = base.copy()
            record['test_type'] = 'Python'
            self._test_python(pyfunc, record, test_data)
            self.test_results.add(**record)


cases = [
    ('min', '', [2 * 10 ** 5]),
]

test_perf_generator(cases, TestDFRollingMethods, 'df', 'rolling({}).'.format(get_rolling_params()))

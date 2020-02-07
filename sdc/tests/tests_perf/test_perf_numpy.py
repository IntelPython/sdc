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

import pandas
import numpy as np

import time
import random

import sdc

from .test_perf_base import TestBase
from sdc.tests.test_utils import test_global_input_data_float64
from .test_perf_utils import calc_compilation, get_times, perf_data_gen_fixed_len
from .generator import generate_test_cases
from .generator import TestCase as TC
from .generator import CallExpression as CE
from sdc.functions import numpy_like


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_numpy.TestFunctions.test_function_{name}
class TestFunctions(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _test_jitted(self, pyfunc, record, *args, **kwargs):
        # compilation time
        record["compile_results"] = calc_compilation(pyfunc, *args, **kwargs)

        sdc_func = sdc.jit(pyfunc)

        # execution and boxing time
        record["test_results"], record["boxing_results"] = \
            get_times(sdc_func, *args, **kwargs)

    def _test_python(self, pyfunc, record, *args, **kwargs):
        record["test_results"], _ = \
            get_times(pyfunc, *args, **kwargs)

    def _test_case(self, cases, name, total_data_length, data_num=1, input_data=test_global_input_data_float64):
        test_name = '{}'.format(name)

        if input_data is None:
            input_data = test_global_input_data_float64

        full_input_data_length = sum(len(i) for i in input_data)
        for data_length in total_data_length:
            base = {
                "test_name": test_name,
                "data_size": data_length,
            }
            data = perf_data_gen_fixed_len(input_data, full_input_data_length,
                                           data_length)
            test_data = np.array(data)

            args = [test_data]
            for i in range(data_num - 1):
                np.random.seed(i)
                extra_data = np.random.ranf(data_length)
                args.append(np.array(extra_data))

            for case in cases:
                record = base.copy()
                record["test_type"] = case['type_']
                if case['jitted']:
                    self._test_jitted(case['func'], record, *args)
                else:
                    self._test_python(case['func'], record, *args)
                self.test_results.add(**record)

cases = [
    TC(name='astype', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='data.astype(np.int64)', jitted=False),
        CE(type_='Numba', code='data.astype(np.int64)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.astype(data, np.int64)', jitted=True),
    ], usecase_params='data'),
    TC(name='copy', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.copy(data)', jitted=False),
        CE(type_='Numba', code='np.copy(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.copy(data)', jitted=True),
    ], usecase_params='data'),
]

generate_test_cases(cases, TestFunctions, 'function')

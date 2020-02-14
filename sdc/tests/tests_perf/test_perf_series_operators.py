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
import random

import sdc

from .test_perf_base import TestBase
from sdc.tests.test_utils import test_global_input_data_float64
from .generator import generate_test_cases
from .generator import TestCase as TC
from .data_generator import gen_series
from .test_perf_series import test_integer64


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series_operators.TestSeriesOperatorMethods.test_series_{name}
class TestSeriesOperatorMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _test_case(self, pyfunc, name, total_data_length, typ, data_num=1, input_data=test_global_input_data_float64):
        test_name = 'Series.{}'.format(name)

        for data_length in total_data_length:
            base = {
                "test_name": test_name,
                "data_size": data_length,
            }

            args = gen_series(data_num, data_length, input_data, typ)

            self._test_jit(pyfunc, base, *args)
            self._test_py(pyfunc, base, *args)


cases = [
    TC(name='operator.add', size=[10 ** 7], call_expr='A + B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.eq', size=[10 ** 7], call_expr='A == B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.floordiv', size=[10 ** 7], call_expr='A // B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.ge', size=[10 ** 7], call_expr='A >= B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.gt', size=[10 ** 7], call_expr='A > B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.le', size=[10 ** 7], call_expr='A <= B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.lt', size=[10 ** 7], call_expr='A < B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.mod', size=[10 ** 7], call_expr='A % B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.mul', size=[10 ** 7], call_expr='A * B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.ne', size=[10 ** 7], call_expr='A != B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.pow', size=[10 ** 7], call_expr='A ** B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.sub', size=[10 ** 7], call_expr='A - B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='operator.truediv', size=[10 ** 7], call_expr='A / B', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
]

generate_test_cases(cases, TestSeriesOperatorMethods, 'series')

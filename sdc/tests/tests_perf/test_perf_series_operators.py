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


"""
python -m sdc.runtests
sdc.tests.tests_perf.test_perf_series_operators.TestSeriesOperatorMethods.test_series_operator_{name}
"""


class TestSeriesOperatorMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def gen_data(self, data_num, data_length, input_data):
        data = []
        full_input_data_length = sum(len(i) for i in input_data)
        data.append(perf_data_gen_fixed_len(input_data, full_input_data_length,
                                            data_length))
        for i in range(data_num - 1):
            np.random.seed(i)
            data.append(np.random.ranf(data_length))

        return data

    def gen_args(self, data_num, data_length, input_data):
        datas = self.gen_data(data_num, data_length, input_data)
        args = []
        for data in datas:
            test_data = pandas.Series(data)
            args.append(test_data)

        return args

    def gen_base(self, test_name, data_length):
        base = {
            "test_name": test_name,
            "data_size": data_length,
        }

        return base

    def _test_case(self, pyfunc, name, total_data_length, data_num=1, input_data=test_global_input_data_float64):
        test_name = 'Series.{}'.format(name)

        if input_data is None:
            input_data = test_global_input_data_float64

        for data_length in total_data_length:
            base = self.gen_base(test_name, data_length)

            args = self.gen_args(data_num, data_length, input_data)

            self.test_jit(pyfunc, base, *args)
            self.test_py(pyfunc, base, *args)


cases = [
    TC(name='operator_add', size=[10 ** 7], call_expr='A + B', usecase_params='A, B', data_num=2),
    TC(name='operator_eq', size=[10 ** 7], call_expr='A == B', usecase_params='A, B', data_num=2),
    TC(name='operator_floordiv', size=[10 ** 7], call_expr='A // B', usecase_params='A, B', data_num=2),
    TC(name='operator_ge', size=[10 ** 7], call_expr='A >= B', usecase_params='A, B', data_num=2),
    TC(name='operator_gt', size=[10 ** 7], call_expr='A > B', usecase_params='A, B', data_num=2),
    TC(name='operator_le', size=[10 ** 7], call_expr='A <= B', usecase_params='A, B', data_num=2),
    TC(name='operator_lt', size=[10 ** 7], call_expr='A < B', usecase_params='A, B', data_num=2),
    TC(name='operator_mod', size=[10 ** 7], call_expr='A % B', usecase_params='A, B', data_num=2),
    TC(name='operator_mul', size=[10 ** 7], call_expr='A * B', usecase_params='A, B', data_num=2),
    TC(name='operator_ne', size=[10 ** 7], call_expr='A != B', usecase_params='A, B', data_num=2),
    TC(name='operator_pow', size=[10 ** 7], call_expr='A ** B', usecase_params='A, B', data_num=2),
    TC(name='operator_sub', size=[10 ** 7], call_expr='A - B', usecase_params='A, B', data_num=2),
    TC(name='operator_truediv', size=[10 ** 7], call_expr='A / B', usecase_params='A, B', data_num=2),
]

generate_test_cases(cases, TestSeriesOperatorMethods, 'series')

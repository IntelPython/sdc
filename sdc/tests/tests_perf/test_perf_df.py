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

import numba
import sdc

from sdc.tests.test_utils import *
from sdc.tests.tests_perf.test_perf_base import TestBase
from .generator import generate_test_cases
from .generator import TestCase as TC
from .data_generator import gen_df
from .test_perf_series import test_integer64


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_df.TestDataFrameMethods.test_df_{method_name}
class TestDataFrameMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _test_case(self, pyfunc, name, total_data_length, input_data, typ, data_num=1):
        test_name = 'DataFrame.{}'.format(name)

        for data_length in total_data_length:
            base = {
                "test_name": test_name,
                "data_size": data_length,
            }

            args = gen_df(data_num, data_length, input_data, typ)

            self._test_jit(pyfunc, base, *args)
            self._test_py(pyfunc, base, *args)


cases = [
    TC(name='append', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='count', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='drop', size=[10 ** 8], params=['columns="f0"'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='max', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='mean', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='median', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='min', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='pct_change', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='prod', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='std', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='sum', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='var', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
]

generate_test_cases(cases, TestDataFrameMethods, 'df')

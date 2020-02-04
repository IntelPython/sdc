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

import numpy as np

import pandas
import numba
import sdc

from sdc.tests.tests_perf.test_perf_base import TestBase
from sdc.tests.tests_perf.test_perf_utils import calc_compilation, get_times, perf_data_gen_fixed_len
from .generator import generate_test_cases
from .generator import TestCase as TC


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_df.TestDataFrameMethods.test_df_{method_name}
class TestDataFrameMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _test_case(self, pyfunc, name, total_data_length, input_data, typ, data_num=1):
        test_name = 'DataFrame.{}'.format(name)

        full_input_data_length = sum(len(i) for i in input_data)
        for data_length in total_data_length:
            base = {
                "test_name": test_name,
                "data_size": data_length,
            }
            data = perf_data_gen_fixed_len(input_data, full_input_data_length,
                                           data_length)
            test_data = pandas.DataFrame({f"f{i}": data for i in range(3)})

            args = [test_data]
            for i in range(data_num - 1):
                np.random.seed(i)
                if typ == 'float':
                    extra_data = np.random.ranf(data_length)
                elif typ == 'int':
                    extra_data = np.random.randint(10 ** 4, size=data_length)
                args.append(pandas.DataFrame({f"f{i}": extra_data for i in range(3)}))

            record = base.copy()
            record["test_type"] = 'SDC'
            self._test_jitted(pyfunc, record, *args)
            self.test_results.add(**record)

            record = base.copy()
            record["test_type"] = 'Python'
            self._test_python(pyfunc, record, *args)
            self.test_results.add(**record)


cases = [
    TC(name='append', size=[10 ** 7], params=['other'], data_num=2),
    TC(name='count', size=[10 ** 7]),
    TC(name='drop', size=[10 ** 8], params=['columns="f0"']),
    TC(name='max', size=[10 ** 7]),
    TC(name='mean', size=[10 ** 7]),
    TC(name='median', size=[10 ** 7]),
    TC(name='min', size=[10 ** 7]),
    TC(name='pct_change', size=[10 ** 7]),
    TC(name='prod', size=[10 ** 7]),
    TC(name='std', size=[10 ** 7]),
    TC(name='sum', size=[10 ** 7]),
    TC(name='var', size=[10 ** 7]),
]

generate_test_cases(cases, TestDataFrameMethods, 'df')

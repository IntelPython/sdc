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
from .data_generator import gen_numpy_arr


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_numpy.TestFunctions.test_function_{name}
class TestFunctions(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _test_case(self, cases, name, total_data_length, data_num=1, input_data=test_global_input_data_float64):
        test_name = '{}'.format(name)

        if input_data is None:
            input_data = test_global_input_data_float64

        for data_length in total_data_length:
            base = {
                "test_name": test_name,
                "data_size": data_length,
            }

            args = gen_numpy_arr(data_num, data_length, input_data)

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
    TC(name='nanargmin', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.nanargmin(data)', jitted=False),
        CE(type_='SDC', code='sdc.functions.numpy_like.nanargmin(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='nanargmax', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.nanargmax(data)', jitted=False),
        CE(type_='SDC', code='sdc.functions.numpy_like.nanargmax(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='argmax', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.argmax(data)', jitted=False),
        CE(type_='Numba', code='np.argmax(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.argmax(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='argmin', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.argmin(data)', jitted=False),
        CE(type_='Numba', code='np.argmin(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.argmin(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='copy', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.copy(data)', jitted=False),
        CE(type_='Numba', code='np.copy(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.copy(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='isnan', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.isnan(data)', jitted=False),
        CE(type_='Numba', code='np.isnan(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.isnan(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='nanmean', size=[10 ** 8], call_expr=[
        CE(type_='Python', code='np.nanmean(data)', jitted=False),
        CE(type_='Numba', code='np.nanmean(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.nanmean(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='nansum', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.nansum(data)', jitted=False),
        CE(type_='SDC', code='sdc.functions.numpy_like.nansum(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='nanprod', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.nanprod(data)', jitted=False),
        CE(type_='Numba', code='np.nanprod(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.nanprod(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='nanvar', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.nanvar(data)', jitted=False),
        CE(type_='Numba', code='np.nanvar(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.nanvar(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='sum', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.sum(data)', jitted=False),
        CE(type_='Numba', code='np.sum(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.sum(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='nanmin', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.nanmin(data)', jitted=False),
        CE(type_='Numba', code='np.nanmin(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.nanmin(data)', jitted=True),
    ], usecase_params='data'),
    TC(name='nanmax', size=[10 ** 7], call_expr=[
        CE(type_='Python', code='np.nanmax(data)', jitted=False),
        CE(type_='Numba', code='np.nanmax(data)', jitted=True),
        CE(type_='SDC', code='sdc.functions.numpy_like.nanmax(data)', jitted=True),
    ], usecase_params='data'),
]

generate_test_cases(cases, TestFunctions, 'function')

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


test_integer64 = [[1, -1, 0],
                  [-9223372, 9223372, 9223372036, -9223372036],
                  [1844674, 9551615]]


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series.TestSeriesMethods.test_series_{method_name}
class TestSeriesMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _test_case(self, pyfunc, name, total_data_length, input_data, typ, data_num=1):
        test_name = 'Series.{}'.format(name)

        full_input_data_length = sum(len(i) for i in input_data)
        for data_length in total_data_length:
            base = {
                "test_name": test_name,
                "data_size": data_length,
            }
            data = perf_data_gen_fixed_len(input_data, full_input_data_length,
                                           data_length)
            test_data = pandas.Series(data)

            args = [test_data]
            for i in range(data_num - 1):
                np.random.seed(i)
                if typ == 'float':
                    extra_data = np.random.ranf(data_length)
                elif typ == 'int':
                    extra_data = np.random.randint(10 ** 4, size=data_length)
                args.append(pandas.Series(extra_data))

            self.test_jit(pyfunc, base, *args)
            self.test_py(pyfunc, base, *args)


cases = [
    TC(name='abs', size=[10 ** 8],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='add', size=[10 ** 7], params=['other'],
       input_data=[test_global_input_data_float64, test_integer64],  data_num=2),
    TC(name='append', size=[10 ** 7], params=['other'],
       input_data=[test_global_input_data_float64, test_integer64], data_num=2),
    TC(name='apply', size=[10 ** 7], params=['lambda x: x'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='argsort', size=[10 ** 4],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='astype', size=[10 ** 5], call_expr='data.astype(np.int8)', usecase_params='data',
       input_data=[[test_global_input_data_float64[0]], [test_integer64[0]]]),
    TC(name='at', size=[10 ** 7], call_expr='data.at[3]', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='chain_add_and_sum', size=[20 * 10 ** 6, 25 * 10 ** 6, 30 * 10 ** 6],
       call_expr='(A + B).sum()', usecase_params='A, B', data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='copy', size=[10 ** 8],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='corr', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='count', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='cov', size=[10 ** 8], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='cumsum', size=[10 ** 8], params=['', 'skipna=False'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='describe', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='div', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='dropna', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='eq', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='fillna', size=[10 ** 7], params=['-1'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='floordiv', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='ge', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='gt', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='head', size=[10 ** 8],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='iat', size=[10 ** 7], call_expr='data.iat[100000]', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='idxmax', size=[10 ** 8], params=['', 'skipna=False'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='idxmin', size=[10 ** 8], params=['', 'skipna=False'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='iloc', size=[10 ** 7], call_expr='data.iloc[100000]', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='index', size=[10 ** 7], call_expr='data.index', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='isin', size=[10 ** 7], call_expr='data.isin([0])', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='isna', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='isnull', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='le', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='loc', size=[10 ** 7], call_expr='data.loc[0]', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='lt', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='map', size=[10 ** 7], params=['lambda x: x * 2', '{2.: 42., 4.: 3.14}'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='max', size=[10 ** 8], params=['', 'skipna=False', 'skipna=True'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='mean', size=[10 ** 8], params=['', 'skipna=False', 'skipna=True'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='median', size=[10 ** 8], params=['', 'skipna=False', 'skipna=True'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='min', size=[10 ** 8], params=['', 'skipna=False', 'skipna=True'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='mod', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='mul', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='ndim', size=[10 ** 7], call_expr='data.ndim', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='ne', size=[10 ** 8], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='nlargest', size=[10 ** 6],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='notna', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='nsmallest', size=[10 ** 6],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='nunique', size=[10 ** 7],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='prod', size=[10 ** 8], params=['', 'skipna=False', 'skipna=True'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='pct_change', size=[10 ** 7], params=['periods=1, limit=None, freq=None'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='pow', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='quantile', size=[10 ** 8],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='rename', size=[10 ** 7], call_expr='data.rename("new_series")', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='shape', size=[10 ** 7], call_expr='data.shape', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='shift', size=[10 ** 8],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='size', size=[10 ** 7], call_expr='data.size', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='sort_values', size=[10 ** 5],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='std', size=[10 ** 7], params=['', 'skipna=False', 'skipna=True'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='sub', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='sum', size=[10 ** 8], params=['', 'skipna=False', 'skipna=True'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='take', size=[10 ** 7], call_expr='data.take([0])', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='truediv', size=[10 ** 7], params=['other'], data_num=2,
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='values', size=[10 ** 7], call_expr='data.values', usecase_params='data',
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='value_counts', size=[10 ** 6],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='var', size=[10 ** 8], params=['', 'skipna=False', 'skipna=True'],
       input_data=[test_global_input_data_float64, test_integer64]),
    TC(name='unique', size=[10 ** 5],
       input_data=[test_global_input_data_float64, test_integer64]),
]

generate_test_cases(cases, TestSeriesMethods, 'series')

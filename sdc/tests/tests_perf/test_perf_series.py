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

import pandas as pd
import numpy as np

import time
import random

import pandas
import sdc

from .test_perf_base import TestBase
from sdc.tests.test_utils import test_global_input_data_float64
from .test_perf_utils import calc_compilation, get_times, perf_data_gen_fixed_len
from .generator import generate_test_cases
from .generator import TestCase as TC


def usecase_series_astype_int(input_data):
    # astype to int8
    start_time = time.time()
    input_data.astype(np.int8)
    finish_time = time.time()
    res_time = finish_time - start_time

    return res_time, input_data


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_series.TestSeriesMethods.test_series_{method_name}
class TestSeriesMethods(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _test_jitted(self, pyfunc, record, *args, **kwargs):
        # compilation time
        record["compile_results"] = calc_compilation(pyfunc, *args, **kwargs)

        sdc_func = sdc.jit(pyfunc)

        # Warming up
        sdc_func(*args, **kwargs)

        # execution and boxing time
        record["test_results"], record["boxing_results"] = \
            get_times(sdc_func, *args, **kwargs)

    def _test_python(self, pyfunc, record, *args, **kwargs):
        record["test_results"], _ = \
            get_times(pyfunc, *args, **kwargs)

    def _test_case(self, pyfunc, name, total_data_length, data_num=1, input_data=test_global_input_data_float64):
        test_name = 'Series.{}'.format(name)

        if input_data == []:
            input_data = test_global_input_data_float64

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
                extra_data = np.random.ranf(data_length)
                args.append(pandas.Series(extra_data))

            record = base.copy()
            record["test_type"] = 'SDC'
            self._test_jitted(pyfunc, record, *args)
            self.test_results.add(**record)

            record = base.copy()
            record["test_type"] = 'Python'
            self._test_python(pyfunc, record, *args)
            self.test_results.add(**record)


cases = [
    TC(name='abs', size=[10 ** 8]),
    TC(name='add', size=[10 ** 7], data_num=2),
    TC(name='append', size=[10 ** 7], data_num=2),
    TC(name='apply', params='lambda x: x', size=[10 ** 7]),
    TC(name='argsort', size=[10 ** 4]),
    TC(name='astype', size=[10 ** 5], call_expression='input_data.astype(np.int8)', input_data=[test_global_input_data_float64[0]]),
    TC(name='at', size=[10 ** 7], call_expression='input_data.at[3]'),
    TC(name='chain_add_and_sum', size=[20 * 10 ** 6, 25 * 10 ** 6, 30 * 10 ** 6], call_expression='(A + B).sum()', data_num=2),
    TC(name='copy', size=[10 ** 8]),
    TC(name='corr', size=[10 ** 7], data_num=2),
    TC(name='count', size=[10 ** 7]),
    TC(name='cov', size=[10 ** 8], data_num=2),
    TC(name='cumsum', size=[10 ** 8]),
    TC(name='describe', size=[10 ** 7]),
    TC(name='div', size=[10 ** 7], data_num=2),
    TC(name='dropna', size=[10 ** 7]),
    TC(name='eq', size=[10 ** 7], data_num=2),
    TC(name='fillna', params='-1', size=[10 ** 7]),
    TC(name='floordiv', size=[10 ** 7], data_num=2),
    TC(name='ge', size=[10 ** 7], data_num=2),
    TC(name='gt', size=[10 ** 7], data_num=2),
    TC(name='head', size=[10 ** 8]),
    TC(name='iat', size=[10 ** 7], call_expression='input_data.iat[100000]'),
    TC(name='idxmax', size=[10 ** 8]),
    TC(name='idxmin', size=[10 ** 8]),
    TC(name='iloc', size=[10 ** 7], call_expression='input_data.iloc[100000]'),
    TC(name='index', size=[10 ** 7], call_expression='input_data.index'),
    TC(name='isin', size=[10 ** 7], call_expression='input_data.isin([0])'),
    TC(name='isna', size=[10 ** 7]),
    TC(name='isnull', size=[10 ** 7]),
    TC(name='le', size=[10 ** 7], data_num=2),
    TC(name='loc', size=[10 ** 7], call_expression='input_data.loc[0]'),
    TC(name='lt', size=[10 ** 7], data_num=2),
    TC(name='max', size=[10 ** 8]),
    TC(name='max', params='skipna=False', size=[10 ** 8]),
    TC(name='mean', size=[10 ** 8]),
    TC(name='median', size=[10 ** 8]),
    TC(name='min', size=[10 ** 8]),
    TC(name='min', params='skipna=False', size=[10 ** 7]),
    TC(name='mod', size=[10 ** 7], data_num=2),
    TC(name='mul', size=[10 ** 7], data_num=2),
    TC(name='ndim', size=[10 ** 7], call_expression='input_data.ndim'),
    TC(name='ne', size=[10 ** 8], data_num=2),
    TC(name='nlargest', size=[10 ** 6]),
    TC(name='notna', size=[10 ** 7]),
    TC(name='nsmallest', size=[10 ** 6]),
    TC(name='nunique', size=[10 ** 7]),
    TC(name='prod', size=[10 ** 8]),
    TC(name='pct_change', params='periods=1, limit=None, freq=None', size=[10 ** 7]),
    TC(name='pow', size=[10 ** 7], data_num=2),
    TC(name='quantile', size=[10 ** 8]),
    TC(name='rename', size=[10 ** 7], call_expression='input_data.rename("new_series")'),
    TC(name='shape', size=[10 ** 7], call_expression='input_data.shape'),
    TC(name='shift', size=[10 ** 8]),
    TC(name='size', size=[10 ** 7], call_expression='input_data.size'),
    TC(name='sort_values', size=[10 ** 5]),
    TC(name='std', size=[10 ** 7]),
    TC(name='sub', size=[10 ** 7], data_num=2),
    TC(name='sum', size=[10 ** 8]),
    TC(name='take', size=[10 ** 7], call_expression='input_data.take([0])'),
    TC(name='truediv', size=[10 ** 7], data_num=2),
    TC(name='values', size=[10 ** 7], call_expression='input_data.values'),
    TC(name='value_counts', size=[10 ** 6]),
    TC(name='var', size=[10 ** 8]),
    TC(name='unique', size=[10 ** 5]),
]


generate_test_cases(cases, TestSeriesMethods, 'series')

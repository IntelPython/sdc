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
import numpy
import pandas

from sdc.tests.tests_perf.test_perf_base import TestBase
from sdc.tests.test_series import gen_strlist, gen_frand_array
from sdc.utilities.sdc_typing_utils import kwsparams2list


min_groupby_int = -50
max_groupby_int = 50
n_groups_default = 200


groupby_usecase_tmpl = """
def df_groupby_{method_name}_usecase(data):
    start_time = time.time()
    res = data.groupby({groupby_params}).{method_name}({method_params})
    end_time = time.time()
    return end_time - start_time, res
"""


def get_groupby_params(**kwargs):
    """Generate supported groupby parameters"""

    # only supported groupby parameters are here
    df_params_defaults = {
        'by': "'A'",
        'sort': 'True'
    }
    groupby_params = {k: kwargs.get(k, df_params_defaults[k]) for k in df_params_defaults}
    return ', '.join(kwsparams2list(groupby_params))


def gen_df_groupby_usecase(method_name, groupby_params=None, method_params=''):
    """Generate df groupby method use case"""

    groupby_params = {} if groupby_params is None else groupby_params
    groupby_params = get_groupby_params(**groupby_params)

    func_text = groupby_usecase_tmpl.format(**{
        'method_name': method_name,
        'groupby_params': groupby_params,
        'method_params': method_params
    })

    global_vars = {'np': numpy, 'time': time}
    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _df_groupby_usecase = loc_vars[f'df_groupby_{method_name}_usecase']

    return _df_groupby_usecase


# python -m sdc.runtests sdc.tests.tests_perf.test_perf_df_groupby.TestDFGroupByMethods
class TestDFGroupByMethods(TestBase):
    # more than 19 columns raise SystemError: CPUDispatcher() returned a result with an error set
    max_columns_num = 19

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.total_data_length = {
            'count': [2 * 10 ** 5],
            'max': [2 * 10 ** 5],
            'mean': [2 * 10 ** 5],
            'median': [2 * 10 ** 5],
            'min': [2 * 10 ** 5],
            'prod': [2 * 10 ** 5],
            'std': [2 * 10 ** 5],
            'sum': [2 * 10 ** 5],
            'var': [2 * 10 ** 5],
        }

    def _gen_df(self, data, columns_num=10):
        """Generate DataFrame based on input data"""

        if len(data) < columns_num:
            data *= columns_num // len(data)
            data.extend(data[:(columns_num % len(data))])
        columns_seq = zip(string.ascii_uppercase[:columns_num], data)
        return pandas.DataFrame(dict(columns_seq))

    def _test_case(self, pyfunc, name, usecase_name=None,
                   input_data=None,
                   columns_num=10):
        """
        Test DataFrame.groupby method
        :param pyfunc: Python function to test which calls tested method inside
        :param name: name of the tested method, e.g. min
        :param input_data: initial data used for generating test data
        :param columns_num: number of columns in generated DataFrame """
        if columns_num > self.max_columns_num:
            columns_num = self.max_columns_num

        usecase_name = f'{name}' if usecase_name is None else f'{usecase_name}'
        for data_length in self.total_data_length[name]:
            base = {
                'test_name': f'DataFrame.groupby.{usecase_name}',
                'data_size': data_length,
            }

            numpy.random.seed(0)
            data = []
            if input_data is None:
                data.append(numpy.random.randint(min_groupby_int, max_groupby_int, data_length))
                for _ in range(1, columns_num):
                    data.append(numpy.random.ranf(data_length))
            else:
                for i in range(columns_num):
                    if i < len(input_data):
                        col_data = numpy.random.choice(input_data[i], data_length)
                    else:
                        col_data = numpy.random.ranf(data_length)
                    data.append(col_data)

            test_data = self._gen_df(data, columns_num=columns_num)
            args = [test_data]

            record = base.copy()
            record['test_type'] = 'SDC'
            self._test_jitted(pyfunc, record, *args)
            self.test_results.add(**record)

            record = base.copy()
            record['test_type'] = 'Python'
            self._test_python(pyfunc, record, *args)
            self.test_results.add(**record)

    def _test_df_groupby_method(self, name, usecase_name=None, groupby_params=None, method_params='', input_data=None):
        usecase = gen_df_groupby_usecase(name, groupby_params=groupby_params, method_params=method_params)
        self._test_case(usecase, name, usecase_name=usecase_name, input_data=input_data)

    def test_df_groupby_count_sort_false(self):
        self._test_df_groupby_method('count', groupby_params={'sort': 'False'})

    def test_df_groupby_max_sort_false(self):
        self._test_df_groupby_method('max', groupby_params={'sort': 'False'})

    def test_df_groupby_mean_sort_false(self):
        self._test_df_groupby_method('mean', groupby_params={'sort': 'False'})

    def test_df_groupby_median_sort_false(self):
        self._test_df_groupby_method('median', groupby_params={'sort': 'False'})

    def test_df_groupby_min_sort_false(self):
        self._test_df_groupby_method('min', groupby_params={'sort': 'False'})

    def test_df_groupby_prod_sort_false(self):
        self._test_df_groupby_method('prod', groupby_params={'sort': 'False'})

    def test_df_groupby_std_sort_false(self):
        self._test_df_groupby_method('std', groupby_params={'sort': 'False'})

    def test_df_groupby_sum_sort_false(self):
        self._test_df_groupby_method('sum', groupby_params={'sort': 'False'})

    def test_df_groupby_var_sort_false(self):
        self._test_df_groupby_method('var', groupby_params={'sort': 'False'})

    def test_df_groupby_mean_sort_true(self):
        self._test_df_groupby_method('mean', usecase_name='mean_sort_true')

    def test_df_groupby_mean_by_float_sort_false(self):
        self._test_df_groupby_method('mean',
                                     usecase_name='by_float_mean',
                                     input_data=[gen_frand_array(n_groups_default)],
                                     groupby_params={'sort': 'False'})

    def test_df_groupby_mean_by_str_sort_false(self):
        self._test_df_groupby_method('mean',
                                     usecase_name='by_str_mean',
                                     input_data=[gen_strlist(n_groups_default, 3, 'abcdef')],
                                     groupby_params={'sort': 'False'})

# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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

import unittest
import os
import time
import numba

from sdc.tests.test_utils import *
from sdc.tests.tests_perf.test_perf_base import TestBase
from sdc.tests.tests_perf.test_perf_utils import *


def usecase_split(input_data):
    test_iteration_number = 5
    iter_time = []

    for local_iter in range(test_iteration_number):
        start_time = time.time()

        for string in input_data:
            string.split('c')

        finish_time = time.time()
        local_result = finish_time - start_time

        iter_time.append(local_result)

    return iter_time


def usecase_join(input_data):
    test_iteration_number = 5
    iter_time = []

    for local_iter in range(test_iteration_number):
        start_time = time.time()

        for string in input_data:
            'X'.join([string, string, string, string])

        finish_time = time.time()
        local_result = finish_time - start_time

        iter_time.append(local_result)

    return iter_time


def usecase_center(input_data):
    result_str_grow_factor = 1.4
    test_iteration_number = 5
    iter_time = []

    for local_iter in range(test_iteration_number):
        start_time = time.time()

        for string in input_data:
            # The width must be an Integer
            new_string_len = int(len(string) * result_str_grow_factor)

            string.center(new_string_len, '+')

        finish_time = time.time()
        local_result = finish_time - start_time

        iter_time.append(local_result)

    return iter_time


class TestStringMethods(TestBase):
    results_class = TestResultsStr

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.total_data_size_bytes = [1.0E+04]
        cls.width = [16, 64, 512, 1024]

    def _test_unicode(self, pyfunc, name):
        hpat_func = numba.njit(pyfunc)
        for data_size in self.total_data_size_bytes:
            for data_width in self.width:
                test_data = perf_data_gen(test_global_input_data_unicode_kind4, data_width, data_size)
                self.test_results.add(name, 'JIT', len(test_data), hpat_func(test_data), data_width)
                self.test_results.add(name, 'Reference', len(test_data), pyfunc(test_data), data_width)

    def test_unicode_split(self):
        self._test_unicode(usecase_split, 'unicode_split')

    def test_unicode_join(self):
        self._test_unicode(usecase_join, 'unicode_join')

    def test_unicode_center(self):
        self._test_unicode(usecase_center, 'unicode_center')


if __name__ == "__main__":
    unittest.main()

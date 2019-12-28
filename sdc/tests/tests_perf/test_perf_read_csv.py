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

import time

import pandas
import pyarrow.csv
import sdc

from sdc.tests.tests_perf.test_perf_base import TestBase
from sdc.tests.tests_perf.test_perf_utils import calc_compilation, get_times

from sdc.tests.tests_perf.gen_csv import generate_csv


def make_func(file_name):
    """Create function for testing.
    It is necessary because file_name should be constant for jitted function.
    """
    def _function():
        start = time.time()
        df = pandas.read_csv(file_name)
        return time.time() - start, df
    return _function


def make_func_pyarrow(file_name):
    """Create function implemented via PyArrow."""
    def _function():
        start = time.time()
        df = sdc.io.csv_ext.pandas_read_csv(file_name)
        return time.time() - start, df
    return _function


class TestPandasReadCSV(TestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rows = 10**6
        cls.columns = 10
        cls.generated_file = generate_csv(cls.rows, cls.columns)

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

    def _test_case(self, pyfunc, name):
        base = {
            "test_name": name,
            "data_size": f"[{self.rows},{self.columns}]",
        }

        record = base.copy()
        record["test_type"] = 'SDC'
        self._test_jitted(pyfunc, record)
        self.test_results.add(**record)

        record = base.copy()
        record["test_type"] = 'Python'
        self._test_python(pyfunc, record)
        self.test_results.add(**record)

    def test_read_csv(self):
        self._test_case(make_func(self.generated_file), 'read_csv')

    def test_read_csv_pyarrow(self):
        pyfunc = make_func_pyarrow(self.generated_file)
        name = 'read_csv'

        base = {
            "test_name": name,
            "data_size": f"[{self.rows},{self.columns}]",
        }

        record = base.copy()
        record["test_type"] = 'PyArrow'
        self._test_python(pyfunc, record)
        self.test_results.add(**record)


if __name__ == "__main__":
    print("Gererate data files...")
    generate_csv(rows=10**6, columns=10)
    print("Data files generated!")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

import gc
import logging
import sys
import sdc
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import pandas
from numba import config

"""
Utility functions collection to support performance testing of
functions implemented in the project

Data generators:
    perf_data_gen() generates list of items with fixed length

Data handling:
    add_results() add an experiment timing results to globla storage
    print_results() print all timing results from global storage

"""


def setup_logging():
    """Setup logger"""
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    return logger



def is_true(input_string):
    if isinstance(input_string, str):
        input_string = input_string.lower()
    return input_string in ['yes', 'y', 'true', 't', '1', True]


def get_size(obj):
    """Sum size of object and its members."""
    size = 0
    processed_ids = set()
    objects = [obj]
    while objects:
        need_refer = []
        for obj in objects:
            if id(obj) in processed_ids:
                continue
            processed_ids.add(id(obj))
            need_refer.append(obj)
            size += sys.getsizeof(obj)
        objects = gc.get_referents(*need_refer)
    return size


def multiply_oneds_data(tmpl, max_len):
    """Multiply specified 1D like data."""
    result = copy(tmpl)
    while len(result) < max_len:
        result += tmpl

    # Trim result to max_len
    return result[:max_len]


def multiply_data(tmpl, max_item_len):
    """Multiply specified 2D like data."""
    result = []
    for item in tmpl:
        result += multiply_oneds_data(item, max_item_len)

    return result


def perf_data_gen(tmpl, max_item_len, max_bytes_size):
    """
    Data generator produces 2D like data.
                  tmpl: list of input template string
          max_item_len: length (in elements) of resulted string in an element of the result array
        max_bytes_size: maximum size in bytes of the return data

                return: list of data
    """
    result = []
    while get_size(result) < max_bytes_size:
        result.extend(multiply_data(tmpl, max_item_len))

    # Trim result to max_bytes_size
    while result and get_size(result) > max_bytes_size:
        del result[-1]

    return result


def perf_data_gen_fixed_len(tmpl, max_item_len, max_obj_len):
    """
    Data generator produces 2D like data.
                  tmpl: list of input template string
          max_item_len: length (in elements) of resulted string in an element of the result array
           max_obj_len: maximum length of the return data

                return: list of data
    """
    result = []
    while len(result) < max_obj_len:
        result.extend(multiply_data(tmpl, max_item_len))

    # Trim result to max_obj_len
    return result[:max_obj_len]


@contextmanager
def do_jit(f):
    """Context manager to jit function"""
    cfunc = sdc.jit(f)
    try:
        yield cfunc
    finally:
        del cfunc


def calc_time(func, *args, **kwargs):
    """Calculate execution time of specified function."""
    start_time = time.time()
    func(*args, **kwargs)
    finish_time = time.time()

    return finish_time - start_time


def calc_compile_time(func, *args, **kwargs):
    """Calculate compile time as difference between first 2 runs."""
    return calc_time(func, *args, **kwargs) - calc_time(func, *args, **kwargs)


def calc_compilation(pyfunc, *args, iter_number=5):
    """Calculate compile time several times."""
    compile_times = []
    for _ in range(iter_number):
        with do_jit(pyfunc) as cfunc:
            compile_time = calc_compile_time(cfunc, *args)
            compile_times.append(compile_time)

    return compile_times


def get_times(f, *args, iter_number=5):
    """Get time of boxing+unboxing and internal execution"""
    exec_times = []
    boxing_times = []
    for _ in range(iter_number):
        ext_start = time.time()
        int_result, _ = f(*args)
        ext_finish = time.time()

        exec_times.append(int_result)
        boxing_times.append(max(ext_finish - ext_start - int_result, 0))

    return exec_times, boxing_times


class ResultsDriver:
    """Base class. Load and dump results."""

    def __init__(self, file_name, raw_file_name=None):
        self.file_name = file_name
        self.raw_file_name = raw_file_name or f'raw_{file_name}'


class ExcelResultsDriver(ResultsDriver):
    # openpyxl need to be installed

    def dump_grouped_data(self, grouped_data, logger=None):
        try:
            with pandas.ExcelWriter(self.file_name) as writer:
                grouped_data.to_excel(writer)
        except ModuleNotFoundError as e:
            if logger:
                msg = 'Could not dump the results to "%s": %s'
                logger.warning(msg, self.file_name, e)

    def dump_test_results_data(self, test_results_data, logger=None):
        try:
            with pandas.ExcelWriter(self.raw_file_name) as writer:
                test_results_data.to_excel(writer, index=False)
        except ModuleNotFoundError as e:
            if logger:
                msg = 'Could not dump raw results to "%s": %s'
                logger.warning(msg, self.raw_file_name, e)

    def load(self, logger=None):
        raw_perf_results_xlsx = Path(self.raw_file_name)
        if not raw_perf_results_xlsx.exists():
            return
        with raw_perf_results_xlsx.open('rb') as fd:
            # xlrd need to be installed
            try:
                return pandas.read_excel(fd)
            except ModuleNotFoundError as e:
                if logger:
                    msg = 'Could not load previous results from %s: %s'
                    logger.warning(msg, self.raw_file_name, e)


class CSVResultsDriver(ResultsDriver):

    def dump_grouped_data(self, grouped_data, logger=None):
        grouped_data.to_csv(self.file_name)

    def dump_test_results_data(self, test_results_data, logger=None):
        test_results_data.to_csv(self.raw_file_name)

    def load(self, logger=None):
        raw_perf_results_csv = Path(self.raw_file_name)
        if not raw_perf_results_csv.exists():
            return
        with raw_perf_results_csv.open('rb') as fd:
            try:
                return pandas.read_csv(fd)
            except ModuleNotFoundError as e:
                if logger:
                    msg = 'Could not load previous results from %s: %s'
                    logger.warning(msg, self.raw_file_name, e)


class TestResults:
    index = ['name', 'nthreads', 'type', 'size']
    test_results_data = pandas.DataFrame(index=index)
    logger = setup_logging()

    def __init__(self, drivers=None):
        self.drivers = drivers or []

    @property
    def grouped_data(self):
        """
        Group global storage results
        Example:
                                                    median       min       max  compilation(median)  boxing(median)
            name           type      size  width
            series_str_len JIT       33174 16     0.005283  0.005190  0.005888             0.163459        0.001801
                                     6201  64     0.001473  0.001458  0.001886             0.156071        0.000528
                                     1374  512    0.001087  0.001066  0.001268             0.154500        0.000972
                                     729   1024   0.000998  0.000993  0.001235             0.155549        0.001002
                           Reference 33174 16     0.007499  0.007000  0.010999                  NaN        0.000000
                                     6201  64     0.001998  0.001498  0.002002                  NaN        0.000000
                                     1374  512    0.000541  0.000500  0.000960                  NaN        0.000000
                                     729   1024   0.000500  0.000000  0.000502                  NaN        0.000000
        """
        if self.test_results_data.empty:
            return None

        median_col = self.test_results_data.groupby(self.index)['Time(s)'].median()
        min_col = self.test_results_data.groupby(self.index)['Time(s)'].min()
        max_col = self.test_results_data.groupby(self.index)['Time(s)'].max()
        compilation_col = self.test_results_data.groupby(self.index)['Compile(s)'].median(skipna=False)
        boxing_col = self.test_results_data.groupby(self.index)['Boxing(s)'].median(skipna=False)

        test_results_data = self.test_results_data.set_index(self.index)
        test_results_data['median'] = median_col
        test_results_data['min'] = min_col
        test_results_data['max'] = max_col
        test_results_data['compile'] = compilation_col
        test_results_data['boxing'] = boxing_col
        test_results_data = test_results_data.reset_index()

        columns = ['median', 'min', 'max', 'compile', 'boxing']
        return test_results_data.groupby(self.index)[columns].first().sort_values(self.index)

    def add(self, test_name, test_type, data_size, test_results,
            boxing_results=None, compile_results=None, num_threads=config.NUMBA_NUM_THREADS):
        """
        Add performance testing timing results into global storage
                  test_name: Name of test (1st column in grouped result)
                  test_type: Type of test (3rd column in grouped result)
                  data_size: Size of input data (4s column in grouped result)
               test_results: List of timing results of the experiment
             boxing_results: List of timing results of the overhead (boxing/unboxing)
           compilation_time: Timing result of compilation
                num_threads: Value from NUMBA_NUM_THREADS (2nd column in grouped result)
        """
        data = {
            'name': test_name,
            'nthreads': num_threads,
            'type': test_type,
            'size': data_size,
            'Time(s)': test_results,
            'Compile(s)': compile_results,
            'Boxing(s)': boxing_results
        }
        local_results = pandas.DataFrame(data)
        self.test_results_data = self.test_results_data.append(local_results, sort=False)

    def print(self):
        """
        Print performance testing results from global data storage
        """
        print(self.grouped_data.to_string())

    def dump(self):
        """
        Dump performance testing results from global data storage to excel
        """
        for d in self.drivers:
            d.dump_grouped_data(self.grouped_data, self.logger)
            d.dump_test_results_data(self.test_results_data, self.logger)

    def load(self):
        """
        Load existing performance testing results from excel to global data storage
        """
        for d in self.drivers:
            test_results_data = d.load(self.logger)
            if test_results_data is not None:
                self.test_results_data = test_results_data
                break


class TestResultsStr(TestResults):
    index = ['name', 'nthreads', 'type', 'size', 'width']

    def add(self, test_name, test_type, data_size, test_results, data_width=None,
            boxing_results=None, compile_results=None, num_threads=config.NUMBA_NUM_THREADS):
        """
        Add performance testing timing results into global storage
                  test_name: Name of test (1st column in grouped result)
                  test_type: Type of test (3rd column in grouped result)
                  data_size: Size of input data (4s column in grouped result)
               test_results: List of timing results of the experiment
                 data_width: Scalability attribute for str input data (5s column in grouped result)
             boxing_results: List of timing results of the overhead (boxing/unboxing)
           compilation_time: Timing result of compilation
                num_threads: Value from NUMBA_NUM_THREADS (2nd column in grouped result)
        """
        data = {
            'name': test_name,
            'nthreads': num_threads,
            'type': test_type,
            'size': data_size,
            'width': data_width,
            'Time(s)': test_results,
            'Compile(s)': compile_results,
            'Boxing(s)': boxing_results
        }
        local_results = pandas.DataFrame(data, index=self.index)
        self.test_results_data = self.test_results_data.append(local_results, sort=False)


if __name__ == "__main__":
    data = perf_data_gen(['Test example'], 64, 1.0E+03)
    print("Result data:", data)

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
import sys

import pandas

"""
Utility functions collection to support performance testing of
functions implemented in the project

Data generators:
    perf_data_gen() generates list of items with fixed length

Data handling:
    add_results() add an experiment timing results to globla storage
    print_results() print all timing results from global storage

"""


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


def multiply_data(tmpl, max_item_len):
    """Multiply specified 2D like data."""
    result = []
    for item in tmpl:
        local_item = item
        local_item_len = len(local_item)

        while (local_item_len < max_item_len) and (local_item_len >= 0):
            local_item += item
            local_item_len = len(local_item)

        # Trim local_item to max_item_len
        local_item = local_item[:max_item_len]
        result.append(local_item)

    return result


def perf_data_gen(tmpl, max_item_len, max_bytes_size):
    """
    Data generator produces 2D like data.
                  tmpl: list of input template string
          max_item_len: length (in elements) of resulted string in an element of the result array
        max_bytes_size: maximum size in bytes of the return data

                return: list of strings
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

                return: list of strings
    """
    result = []
    while len(result) < max_obj_len:
        result.extend(multiply_data(tmpl, max_item_len))

    # Trim result to max_obj_len
    return result[:max_obj_len]


test_results_data = pandas.DataFrame()


def add_results(test_name, test_type, data_size, data_width, test_results, boxing_results=None, compile_results=None):
    """
    Add performance testing timing results into global storage
              test_name: Name of test (1st column in grouped result)
              test_type: Type of test (2nd column in grouped result)
        test_data_width: Scalability attribute for input data (3rd column in grouped result)
           test_results: List of timing results of the experiment
         boxing_results: List of timing results of the overhead (boxing/unboxing)
       compilation_time: Timing result of compilation
    """

    local_results = pandas.DataFrame({'name': test_name,
                                      'type': test_type,
                                      'size': data_size,
                                      'width': data_width,
                                      'Time(s)': test_results,
                                      'Compilation(s)': compile_results,
                                      'Boxing(s)': boxing_results})
    global test_results_data
#     test_results_data = pandas.concat([test_results_data, local_results])
    test_results_data = test_results_data.append(local_results)


def print_results():
    """
    Print performance testing results from global data storage
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
    global test_results_data

    if test_results_data.empty:
        return None

    # Following code is terrible. needs to be redeveloped
    # print(test_results_data)
    index = ['name', 'type', 'size', 'width']
    median_col = test_results_data.groupby(index)['Time(s)'].median()
    min_col = test_results_data.groupby(index)['Time(s)'].min()
    max_col = test_results_data.groupby(index)['Time(s)'].max()

    test_results_data = test_results_data.set_index(index)
    test_results_data['median'] = median_col
    test_results_data['min'] = min_col
    test_results_data['max'] = max_col
    test_results_data['compilation(median)'] = test_results_data.groupby(index)['Compilation(s)'].median(skipna=False)
    test_results_data['boxing(median)'] = test_results_data.groupby(index)['Boxing(s)'].median(skipna=False)
    test_results_data = test_results_data.reset_index()

    columns = ['median', 'min', 'max', 'compilation(median)', 'boxing(median)']
    grouped_data = test_results_data.groupby(index)[columns].first().sort_values(index)
    print(grouped_data.to_string())

    with pandas.ExcelWriter('perf_results.xlsx') as writer:
        grouped_data.to_excel(writer)


if __name__ == "__main__":
    data = perf_data_gen(['Test example'], 64, 1.0E+03)
    print("Result data:", data)

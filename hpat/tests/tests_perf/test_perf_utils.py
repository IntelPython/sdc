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


def perf_data_gen(tmpl, max_item_len, max_bytes_size):
    """
    Data generator produces 2D like data.
                  tmpl: list of input template string
          max_item_len: length (in elements) of resulted string in an element of the result array
        max_bytes_size: maximum size in bytes of the return data
                return: list of strings
    """

    result = []
    obj_size = sys.getsizeof(tmpl, -1)

    while (obj_size < max_bytes_size) and (obj_size >= 0):
        for item in tmpl:
            local_item = item
            local_item_len = len(local_item)

            while (local_item_len < max_item_len) and (local_item_len >= 0):
                local_item += item
                local_item_len = len(local_item)

            result.append(local_item)

        obj_size = sys.getsizeof(result, -1)

    return result


test_results_data = pandas.DataFrame(columns=['name', 'type', 'width', 'Time(s)'])


def add_results(test_name, test_type, test_data_width, test_results, boxing_results, compilation_time=0):
    """
    Add performance testing timing results into global storage
              test_name: Name of test (1st column in grouped result)
              test_type: Type of test (2nd column in grouped result)
        test_data_width: Scalability attribute for input data (3rd column in grouped result)
           test_results: List of timing results of the experiment
         boxing_results: List of timing results of the overhead (boxing/unboxing)
    """

    local_results = pandas.DataFrame({'name': test_name,
                                      'type': test_type,
                                      'compilation(s)': compilation_time,
                                      'width': test_data_width,
                                      'Time(s)': test_results,
                                      'Boxing(s)': boxing_results})
    global test_results_data
#     test_results_data = pandas.concat([test_results_data, local_results])
    test_results_data = test_results_data.append(local_results)


def print_results():
    """
    Print performance testing results from global data storage
    Example:
        name                  type compilation(s) width median   min      max      boxing_median boxing_min boxing_max
        series_str_capitalize JIT  0.953211       16    0.026692 0.026562 0.029201  0.004213     -0.010934  0.004599
                                                  64    0.033153 0.033100 0.033440 -0.001908     -0.002187  0.013767
                                                  512   0.076871 0.075122 0.079951  0.013765      0.001243  0.018627
                                                  1024  0.121077 0.120640 0.121638  0.019550      0.004309  0.034610
        series_str_len        JIT  0.171878       16    0.001839 0.001832 0.001922 -0.001838     -0.001848  0.013701
                                                  64    0.002869 0.002868 0.003138 -0.002868     -0.002872  0.012714
                                                  512   0.009281 0.009254 0.009306  0.006368      0.006341  0.021974
                                                  1024  0.016713 0.016613 0.017197  0.014309     -0.000975  0.045882
    """

    global test_results_data

    # Following code is terrible. needs to be redeveloped
    # print(test_results_data)
    index = ['name', 'type', 'width']
    median_col = test_results_data.groupby(index)['Time(s)'].median()
    min_col = test_results_data.groupby(index)['Time(s)'].min()
    max_col = test_results_data.groupby(index)['Time(s)'].max()

    boxing = {
        'median': test_results_data.groupby(index)['Boxing(s)'].median(),
        'min': test_results_data.groupby(index)['Boxing(s)'].min(),
        'max': test_results_data.groupby(index)['Boxing(s)'].max()
    }

    test_results_data = test_results_data.set_index(index)
    test_results_data['median'] = median_col
    test_results_data['min'] = min_col
    test_results_data['max'] = max_col
    test_results_data['boxing_median'] = boxing['median']
    test_results_data['boxing_min'] = boxing['min']
    test_results_data['boxing_max'] = boxing['max']
    test_results_data = test_results_data.reset_index()

    columns = ['name', 'type', 'compilation(s)', 'width', 'median', 'min', 'max',
               'boxing_median', 'boxing_min', 'boxing_max']
    print(test_results_data.groupby(columns).first().to_string())


if __name__ == "__main__":
    data = perf_data_gen(['Test example'], 64, 1.0E+03)
    print("Result data:", data)

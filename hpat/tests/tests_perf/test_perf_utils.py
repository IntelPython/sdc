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


def add_results(test_name, test_type, test_data_width, test_results):
    """
    Add performance testing timing results into global storage
              test_name: Name of test (1st column in grouped result)
              test_type: Type of test (2nd column in grouped result)
        test_data_width: Scalability attribute for input data (3rd column in grouped result)
           test_results: List of timing results of the experiment
    """

    local_results = pandas.DataFrame({'name': test_name,
                                      'type': test_type,
                                      'width': test_data_width,
                                      'Time(s)': test_results})
    global test_results_data
#     test_results_data = pandas.concat([test_results_data, local_results])
    test_results_data = test_results_data.append(local_results)


def print_results():
    """
    Print performance testing results from global data storage

    Example:
        name           type      width median   min      max
        unicode_center JIT       16    0.717773 0.672643 0.759156
                                 64    0.681843 0.664885 0.732811
                                 512   0.697704 0.670506 0.733021
                                 1024  0.672802 0.665532 0.712082
                       Reference 16    0.441212 0.418706 0.478509
                                 64    0.431903 0.423517 0.473604
                                 512   0.429890 0.421120 0.478813
                                 1024  0.429866 0.422612 0.468271
        unicode_join   JIT       16    0.677574 0.658613 0.704243
                                 64    0.638922 0.631664 0.681562
                                 512   0.831152 0.797906 0.850718
                                 1024  1.076941 1.041046 1.123236
                       Reference 16    0.273045 0.254438 0.280705
                                 64    0.249600 0.240926 0.272831
                                 512   0.394346 0.386359 0.431976
                                 1024  0.553249 0.542781 0.607437
    """

    global test_results_data

    # Following code is terrible. needs to be redeveloped
    # print(test_results_data)
    median_col = test_results_data.groupby(['name', 'type', 'width'])['Time(s)'].median()
    min_col = test_results_data.groupby(['name', 'type', 'width'])['Time(s)'].min()
    max_col = test_results_data.groupby(['name', 'type', 'width'])['Time(s)'].max()

    test_results_data = test_results_data.set_index(['name', 'type', 'width'])
    test_results_data['median'] = median_col
    test_results_data['min'] = min_col
    test_results_data['max'] = max_col
    test_results_data = test_results_data.reset_index()

    print(test_results_data.groupby(['name', 'type', 'width', 'median', 'min', 'max']).first())


if __name__ == "__main__":
    data = perf_data_gen(['Test example'], 64, 1.0E+03)
    print("Result data:", data)

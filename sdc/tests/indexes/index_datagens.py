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

import numpy as np
import pandas as pd
from itertools import (product, combinations_with_replacement, filterfalse, chain)

from sdc.tests.test_utils import gen_strlist
from sdc.datatypes.indexes import *

test_global_index_names = [None, 'abc', 'index']
test_global_range_member_values = [1, 2, 10, -5, 0, None]


def _generate_all_range_params():

    def valid_params_predicate(range_params):
        # if step is zero or all start/stop/step are None range is invalid
        return (range_params[-1] == 0
                or all(map(lambda x: x is None, range_params)))

    return filterfalse(
        valid_params_predicate,
        combinations_with_replacement(test_global_range_member_values, 3)
    )


def _generate_positional_range_params():

    # for PositionalIndexType represented ranges only
    starts, stops, steps = [0, ], [1, 2, 10, ], [1, ]
    return product(starts, stops, steps)


def _generate_custom_range_params():

    # for non PositionalIndexType represented range objects
    def valid_positional_index_predicate(range_params):
        index = pd.RangeIndex(*range_params)
        return index.start == 0 and index.stop > 0 and index.step == 1

    return filterfalse(
        valid_positional_index_predicate,
        _generate_all_range_params()
    )


def _generate_positional_indexes_fixed(size, start=1, step=3):
    yield pd.RangeIndex(size)
    yield pd.RangeIndex(size, name='abc')


def _generate_custom_range_indexes_fixed(size, start=1, step=3):
    yield pd.RangeIndex(stop=step * size, step=step)
    yield pd.RangeIndex(stop=2*step*size, step=2*step)
    yield pd.RangeIndex(start=start, stop=start + size*step - step//2, step=step)
    yield pd.RangeIndex(start=start + step, stop=start + (size + 1)*step, step=step)


def _generate_range_indexes_fixed(size, start=1, step=3):
    return chain(
            _generate_positional_indexes_fixed(size, start, step),
            _generate_custom_range_indexes_fixed(size, start, step),
        )


def _generate_index_param_values(n):
    return chain(
        [None],
        _generate_range_indexes_fixed(n),
        _generate_int64_indexes_fixed(n),
        [np.arange(n) / 2],
        [np.arange(n, dtype=np.uint64)],
        [gen_strlist(n)],
    )


def _generate_valid_int64_index_data():
    n = 100
    yield np.arange(n)
    yield np.arange(n) % 2
    yield np.ones(n, dtype=np.int16)
    yield list(np.arange(n))
    yield pd.RangeIndex(n)
    yield pd.Int64Index(np.arange(n))
    yield np.arange(n) * 2
    yield np.arange(2 * n)


def _generate_int64_indexes_fixed(size):
    yield pd.Int64Index(np.arange(size))
    yield pd.Int64Index(np.arange(size), name='abc')
    yield pd.Int64Index([i if i % 2 else 0 for i in range(size)])
    yield pd.Int64Index([i // 2 for i in range(size)])
    yield pd.Int64Index(np.ones(size))


def get_sample_index(size, sdc_index_type):
    if sdc_index_type is PositionalIndexType:
        return pd.RangeIndex(size)
    if sdc_index_type is RangeIndexType:
        return pd.RangeIndex(-1, size - 1, 1)
    if sdc_index_type is Int64IndexType:
        return pd.Int64Index(np.arange(size))

    assert False, f"Refusing to create index of non-specific index type: {sdc_index_type}"

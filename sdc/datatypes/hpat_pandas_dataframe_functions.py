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

'''
| :class:`pandas.DataFrame` functions and operators implementations in Intel SDC
| Also, it contains Numba internal operators which are required for DataFrame type handling
'''

import operator
import pandas
import numpy

import sdc

from numba import types
from numba.extending import (overload, overload_method, overload_attribute)
from sdc.hiframes.pd_dataframe_ext import DataFrameType
from numba.errors import TypingError
from sdc.datatypes.hpat_pandas_series_functions import TypeChecker

def sdc_pandas_dataframe_reduce_columns(df, name, params):
    saved_columns = df.columns
    n_cols = len(saved_columns)
    data_args = tuple('data{}'.format(i) for i in range(n_cols))
    all_params = ['df']
    for key, value in params:
        all_params.append('{}={}'.format(key, value))
    func_definition = 'def _reduce_impl({}):'.format(', '.join(all_params))
    func_lines = [func_definition]
    for i, d in enumerate(data_args):
        line = '  {} = hpat.hiframes.api.init_series(hpat.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}))'
        func_lines.append(line.format(d + '_S', i))
        func_lines.append('  {} = {}.{}()'.format(d + '_O', d + '_S', name))
    func_lines.append('  data = np.array(({},))'.format(
        ", ".join(d + '_O' for d in data_args)))
    func_lines.append('  index = hpat.str_arr_ext.StringArray(({},))'.format(
        ', '.join('"{}"'.format(c) for c in saved_columns)))
    func_lines.append('  return hpat.hiframes.api.init_series(data, index)')
    loc_vars = {}
    func_text = '\n'.join(func_lines)

    exec(func_text, {'hpat': sdc, 'np': numpy}, loc_vars)
    _reduce_impl = loc_vars['_reduce_impl']

    return _reduce_impl

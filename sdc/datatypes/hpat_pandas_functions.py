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
| :module:`pandas` functions and operators implementations in Intel SDC
'''

import pandas
import pyarrow.csv

import numba
import sdc

from numba.extending import overload
from sdc.hiframes.api import fix_df_array

import pandas as pd
from numba import types

from sdc.io.csv_ext import (
    _gen_csv_reader_py_pyarrow_py_func,
    _gen_csv_reader_py_pyarrow_func_text_dataframe,
)
from sdc.hiframes.hiframes_untyped import HiFramesPassImpl


def infer_usecols(col_names):
    # usecols_var = self._get_arg('read_csv', rhs.args, kws, 6, 'usecols', '')
    usecols = list(range(len(col_names)))
    # if usecols_var != '':
    #     err_msg = "pd.read_csv() usecols should be constant list of ints"
    #     usecols = self._get_str_or_list(usecols_var, err_msg=err_msg, typ=int)
    return usecols


@overload(pandas.read_csv)
def sdc_pandas_read_csv(fname, skiprows=0):
    signature =        "fname, skiprows=0"  # nopep8

    assert isinstance(fname, numba.types.Literal)
    assert isinstance(skiprows, numba.types.Literal) or skiprows == 0

    sep = ','

    fname_const = fname.literal_value
    if isinstance(skiprows, numba.types.Literal):
        skiprows = skiprows.literal_value
    col_names = 0
    skiprows, col_names, dtype_map = \
        HiFramesPassImpl.infer_column_names_and_types_from_constant_filename(
            fname_const, skiprows, col_names)

    usecols = infer_usecols(col_names)

    date_cols = []
    columns, out_types = HiFramesPassImpl._get_csv_col_info_core(dtype_map, date_cols, col_names)

    # generate function text with signature and returning DataFrame
    func_text, func_name = _gen_csv_reader_py_pyarrow_func_text_dataframe(
        columns, out_types, usecols, sep, skiprows, signature)
    # print(func_text)

    # compile with Python
    csv_reader_py = _gen_csv_reader_py_pyarrow_py_func(func_text, func_name)

    return csv_reader_py

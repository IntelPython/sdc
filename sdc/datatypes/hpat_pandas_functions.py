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
def sdc_pandas_read_csv(
    filepath_or_buffer,
    sep=',',
    delimiter=None,
    # Column and Index Locations and Names
    header="infer",
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    prefix=None,
    mangle_dupe_cols=True,
    # General Parsing Configuration
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skipinitialspace=False,
    skiprows=None,
    skipfooter=0,
    nrows=None,
    # NA and Missing Data Handling
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    verbose=False,
    skip_blank_lines=True,
    # Datetime Handling
    parse_dates=False,
    infer_datetime_format=False,
    keep_date_col=False,
    date_parser=None,
    dayfirst=False,
    cache_dates=True,
    # Iteration
    iterator=False,
    chunksize=None,
    # Quoting, Compression, and File Format
    compression="infer",
    thousands=None,
    decimal=b".",
    lineterminator=None,
    quotechar='"',
    # quoting=csv.QUOTE_MINIMAL,  # not supported
    doublequote=True,
    escapechar=None,
    comment=None,
    encoding=None,
    dialect=None,
    # Error Handling
    error_bad_lines=True,
    warn_bad_lines=True,
    # Internal
    delim_whitespace=False,
    # low_memory=_c_parser_defaults["low_memory"],  # not supported
    memory_map=False,
    float_precision=None,
):
    signature = """
        filepath_or_buffer,
        sep=',',
        delimiter=None,
        # Column and Index Locations and Names
        header="infer",
        names=None,
        index_col=None,
        usecols=None,
        squeeze=False,
        prefix=None,
        mangle_dupe_cols=True,
        # General Parsing Configuration
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skipinitialspace=False,
        skiprows=None,
        skipfooter=0,
        nrows=None,
        # NA and Missing Data Handling
        na_values=None,
        keep_default_na=True,
        na_filter=True,
        verbose=False,
        skip_blank_lines=True,
        # Datetime Handling
        parse_dates=False,
        infer_datetime_format=False,
        keep_date_col=False,
        date_parser=None,
        dayfirst=False,
        cache_dates=True,
        # Iteration
        iterator=False,
        chunksize=None,
        # Quoting, Compression, and File Format
        compression="infer",
        thousands=None,
        decimal=b".",
        lineterminator=None,
        quotechar='"',
        # quoting=csv.QUOTE_MINIMAL,  # not supported
        doublequote=True,
        escapechar=None,
        comment=None,
        encoding=None,
        dialect=None,
        # Error Handling
        error_bad_lines=True,
        warn_bad_lines=True,
        # Internal
        delim_whitespace=False,
        # low_memory=_c_parser_defaults["low_memory"],  # not supported
        memory_map=False,
        float_precision=None,
    """

    assert isinstance(filepath_or_buffer, numba.types.Literal)
    assert isinstance(sep, numba.types.Literal) or sep == ','
    assert isinstance(delimiter, numba.types.Literal) or delimiter is None
    assert isinstance(skiprows, numba.types.Literal) or skiprows is None

    if isinstance(sep, numba.types.Literal):
        sep = sep.literal_value

    if isinstance(delimiter, numba.types.Literal):
        delimiter = delimiter.literal_value

    # Alias sep -> delimiter.
    if delimiter is None:
        delimiter = sep

    fname_const = filepath_or_buffer.literal_value
    if skiprows is None:
        skiprows = 0
    if isinstance(skiprows, numba.types.Literal):
        skiprows = skiprows.literal_value
    col_names = 0
    skiprows, col_names, dtype_map = \
        HiFramesPassImpl.infer_column_names_and_types_from_constant_filename(
            fname_const, skiprows, col_names, sep=delimiter)

    usecols = infer_usecols(col_names)

    date_cols = []
    columns, out_types = HiFramesPassImpl._get_csv_col_info_core(dtype_map, date_cols, col_names)

    # generate function text with signature and returning DataFrame
    func_text, func_name = _gen_csv_reader_py_pyarrow_func_text_dataframe(
        columns, out_types, usecols, delimiter, skiprows, signature)
    # print(func_text)

    # compile with Python
    csv_reader_py = _gen_csv_reader_py_pyarrow_py_func(func_text, func_name)

    return csv_reader_py

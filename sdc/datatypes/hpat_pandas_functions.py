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
import numpy as np

from numba import types

from sdc.io.csv_ext import (
    _gen_csv_reader_py_pyarrow_py_func,
    _gen_csv_reader_py_pyarrow_func_text_dataframe,
)
from sdc.str_arr_ext import string_array_type


def get_dtypes(df):
    dtypes = []
    for d in df.dtypes.values:
        try:
            numba_type = numba.typeof(d).dtype
            array_type = types.Array(numba_type, 1, 'C')
        except:
            array_type = string_array_type
        dtypes.append(array_type)
    return dtypes


def infer_column_names_and_types_from_constant_filename(fname_const, skiprows, names, delimiter):
    rows_to_read = 100  # TODO: tune this
    df = pd.read_csv(fname_const, delimiter=delimiter, names=names, skiprows=skiprows, nrows=rows_to_read)
    # TODO: string_array, categorical, etc.
    dtypes = get_dtypes(df)
    col_names = df.columns.to_list()
    # overwrite column names like Pandas if explicitly provided
    if names:
        col_names[-len(names):] = names
    else:
        # a row is used for names if not provided
        skiprows += 1
    dtype_map = dict(zip(col_names, dtypes))
    return skiprows, col_names, dtype_map


def _get_csv_col_info_core(dtype_map, date_cols, col_names):
    if isinstance(dtype_map, types.Type):
        typ = dtype_map
        return col_names, [typ] * len(col_names)

    columns = []
    out_types = []
    for i, (col_name, typ) in enumerate(dtype_map.items()):
        columns.append(col_name)
        # get array dtype
        if i in date_cols:
            typ = types.Array(types.NPDatetime('ns'), 1, 'C')
        out_types.append(typ)
    return columns, out_types


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

    # read_csv can infer result DataFrame type from file or from params

    # for inferring from file this parameters should be literal or omitted
    infer_from_file = all([
        isinstance(filepath_or_buffer, types.Literal),
        isinstance(sep, (types.Literal, types.Omitted)) or sep == ',',
        isinstance(delimiter, (types.Literal, types.Omitted)) or delimiter is None,
        isinstance(skiprows, (types.Literal, types.Omitted)) or skiprows is None,
    ])

    # for inference from params dtype and (names or usecols) shoud present
    # names, dtype and usecols should be literal tuples after rewrite pass (see. RewriteReadCsv)
    infer_from_params = all([
        isinstance(dtype, types.Tuple),
        any([
            isinstance(names, types.Tuple) and isinstance(usecols, types.Tuple),
            isinstance(names, types.Tuple) and isinstance(usecols, (types.Omitted, type(None))),
            isinstance(names, (types.Omitted, type(None))) and isinstance(usecols, types.Tuple),
        ]),
    ])

    # cannot create function if parameters provide not enough info
    assert infer_from_file or infer_from_params

    if isinstance(filepath_or_buffer, types.Literal):
        filepath_or_buffer = filepath_or_buffer.literal_value

    if isinstance(sep, types.Literal):
        sep = sep.literal_value

    if isinstance(delimiter, types.Literal):
        delimiter = delimiter.literal_value

    # Alias sep -> delimiter.
    if delimiter is None:
        delimiter = sep

    if isinstance(header, types.Literal):
        header = header.literal_value

    if isinstance(names, types.Tuple):
        assert all(isinstance(name, types.Literal) for name in names)
        names = [name.literal_value for name in names]

    if isinstance(usecols, types.Tuple):
        assert all(isinstance(col, types.Literal) for col in usecols)
        usecols = [col.literal_value for col in usecols]

    if isinstance(dtype, types.Tuple):
        assert all(isinstance(key, types.Literal) for key in dtype[::2])
        keys = (k.literal_value for k in dtype[::2])

        values = dtype[1::2]
        values = [v.typing_key if isinstance(v, types.Function) else v for v in values]
        values = [types.Array(numba.from_dtype(np.dtype(v.literal_value)), 1, 'C') if isinstance(v, types.Literal) else v for v in values]
        values = [types.Array(types.int_, 1, 'C') if v == int else v for v in values]
        values = [types.Array(types.float64, 1, 'C') if v == float else v for v in values]
        values = [string_array_type if v == str else v for v in values]

        dtype = dict(zip(keys, values))

    if isinstance(skiprows, types.Literal):
        skiprows = skiprows.literal_value

    if skiprows is None:
        skiprows = 0

    # inferencing from params has priority over inferencing from file
    # in case of both are available

    if infer_from_params:
        if header == 'infer':
            if names is None:
                header = 0
                if usecols:
                    col_names = None
                else:
                    # require infer from file -> file should be constant
                    assert infer_from_file
            else:
                header = None
                col_names = names
        elif header is None:
            if names is None:
                # list of integers
                # infer from file number of columns -> file should be const
                assert infer_from_file
            else:
                col_names = names
        else:  # Integer
            # names does not metter
            # infer from file -> file shoudl be const
            assert infer_from_file
        # [int] not supported

        # all names should be in dtype
        return_columns = usecols if usecols else col_names
        assert all(n in dtype for n in return_columns)
        col_typs = [dtype[n] for n in return_columns]

    elif infer_from_file:
        # read file and get data types
        skiprows, col_names, dtype_map = infer_column_names_and_types_from_constant_filename(
            filepath_or_buffer, skiprows, names, delimiter)

        # usecols should be constant list of ints
        usecols = list(range(len(col_names)))

        # convert type data
        date_cols = []
        columns, out_types = _get_csv_col_info_core(dtype_map, date_cols, col_names)
        col_names, col_typs = columns, out_types

    else:
        return None

    dtype_present = not isinstance(dtype, (types.Omitted, type(None)))

    # generate function text with signature and returning DataFrame
    func_text, func_name = _gen_csv_reader_py_pyarrow_func_text_dataframe(
        col_names, col_typs, dtype_present, usecols, signature)

    # compile with Python
    csv_reader_py = _gen_csv_reader_py_pyarrow_py_func(func_text, func_name)

    return csv_reader_py

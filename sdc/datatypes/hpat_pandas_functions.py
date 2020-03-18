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

from numba import types, numpy_support

from sdc.io.csv_ext import (
    _gen_csv_reader_py_pyarrow_py_func,
    _gen_csv_reader_py_pyarrow_func_text_dataframe,
)
from sdc.str_arr_ext import string_array_type


def get_numba_array_types_for_csv(df):
    """Extracts Numba array types from the given DataFrame."""
    result = []
    for numpy_type in df.dtypes.values:
        try:
            numba_type = numpy_support.from_dtype(numpy_type)
        except NotImplementedError:
            numba_type = None

        if numba_type:
            array_type = types.Array(numba_type, 1, 'C')
        else:
            # default type for CSV is string
            array_type = string_array_type

        result.append(array_type)
    return result


def infer_column_names_and_types_from_constant_filename(fname_const, skiprows, names, usecols, delimiter):
    rows_to_read = 100  # TODO: tune this
    df = pd.read_csv(fname_const, delimiter=delimiter, names=names, usecols=usecols, skiprows=skiprows, nrows=rows_to_read)
    # TODO: string_array, categorical, etc.
    col_names = df.columns.to_list()
    # overwrite column names like Pandas if explicitly provided
    if names:
        col_names[-len(names):] = names
    col_typs = get_numba_array_types_for_csv(df)
    return col_names, col_typs


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
        isinstance(names, (types.Tuple, types.Omitted, type(None))),
        isinstance(usecols, (types.Tuple, types.Omitted, type(None))),
        isinstance(skiprows, (types.Literal, types.Omitted)) or skiprows is None,
    ])

    # for inference from params dtype and (names or usecols) shoud present
    # names, dtype and usecols should be literal tuples after rewrite pass (see. RewriteReadCsv)
    # header not supported
    infer_from_params = all([
        isinstance(dtype, types.Tuple),
        any([
            isinstance(names, types.Tuple) and isinstance(usecols, types.Tuple),
            isinstance(names, types.Tuple) and isinstance(usecols, (types.Omitted, type(None))),
            isinstance(names, (types.Omitted, type(None))) and isinstance(usecols, types.Tuple),
        ]),
        isinstance(header, types.Omitted) or header == 'infer',
    ])

    # cannot create function if parameters provide not enough info
    assert infer_from_file or infer_from_params

    if infer_from_file:
        # parameters should be constants and are important only for inference from file

        if isinstance(filepath_or_buffer, types.Literal):
            filepath_or_buffer = filepath_or_buffer.literal_value

        if isinstance(sep, types.Literal):
            sep = sep.literal_value

        if isinstance(delimiter, types.Literal):
            delimiter = delimiter.literal_value

        # Alias sep -> delimiter.
        if delimiter is None:
            delimiter = sep

        if isinstance(skiprows, types.Literal):
            skiprows = skiprows.literal_value

    # names and usecols influence on both inferencing from file and from params
    if isinstance(names, types.Tuple):
        assert all(isinstance(name, types.Literal) for name in names)
        names = [name.literal_value for name in names]

    if isinstance(usecols, types.Tuple):
        assert all(isinstance(col, types.Literal) for col in usecols)
        usecols = [col.literal_value for col in usecols]

    if infer_from_params:
        # dtype should be constants and is important only for inference from params
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

    # in case of both are available
    # inferencing from params has priority over inferencing from file
    if infer_from_params:
        col_names = names
        # all names should be in dtype
        return_columns = usecols if usecols else names
        col_typs = [dtype[n] for n in return_columns]

    elif infer_from_file:
        col_names, col_typs = infer_column_names_and_types_from_constant_filename(
            filepath_or_buffer, skiprows, names, usecols, delimiter)

    else:
        return None

    dtype_present = not isinstance(dtype, (types.Omitted, type(None)))

    # generate function text with signature and returning DataFrame
    func_text, func_name = _gen_csv_reader_py_pyarrow_func_text_dataframe(
        col_names, col_typs, dtype_present, usecols, signature)

    # compile with Python
    csv_reader_py = _gen_csv_reader_py_pyarrow_py_func(func_text, func_name)

    return csv_reader_py

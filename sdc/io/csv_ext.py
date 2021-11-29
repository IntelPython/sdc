# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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

import contextlib
import functools

import llvmlite.binding as ll
from llvmlite import ir as lir
from collections import defaultdict
import numba
from numba.core import typeinfer, ir, ir_utils, types
from numba.extending import overload, intrinsic, register_model, models, box
from numba.core.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes)
from numba.core import analysis
from numba.parfors import array_analysis

import sdc
from sdc import distributed, distributed_analysis
from sdc import objmode
from sdc.types import Categorical

import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.csv as csv

from sdc.str_arr_type import string_array_type
from sdc.datatypes.indexes.positional_index_type import PositionalIndexType
from sdc.extensions.sdc_arrow_table_ext import ArrowTableType
from sdc.extensions.sdc_string_view_type import StdStringViewType
from sdc.hiframes.pd_dataframe_type import DataFrameType
from sdc.hiframes.pd_dataframe_ext import get_structure_maps

from numba.core.errors import TypingError
from numba.np import numpy_support


# TODO: move to hpat.common
def to_varname(string):
    """Converts string to correct Python variable name.
    Replaces unavailable symbols with _ and insert _ if string starts with digit.
    """
    import re
    return re.sub(r'\W|^(?=\d)', '_', string)


@contextlib.contextmanager
def pyarrow_cpu_count(cpu_count=pa.cpu_count()):
    old_cpu_count = pa.cpu_count()
    pa.set_cpu_count(cpu_count)
    try:
        yield
    finally:
        pa.set_cpu_count(old_cpu_count)


def pyarrow_cpu_count_equal_numba_num_treads(func):
    """Decorator. Set pyarrow cpu_count the same as NUMBA_NUM_THREADS."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with pyarrow_cpu_count(numba.config.NUMBA_NUM_THREADS):
            return func(*args, **kwargs)

    return wrapper


@pyarrow_cpu_count_equal_numba_num_treads
def do_read_csv(filepath_or_buffer, sep, delimiter, names, usecols, dtype, skiprows, parse_dates):

    if delimiter is None:
        delimiter = sep

    pa_options = get_pyarrow_read_csv_options(
                    delimiter, names, usecols, dtype, skiprows, parse_dates)

    table = csv.read_csv(
        filepath_or_buffer,
        read_options=pa_options[0],
        parse_options=pa_options[1],
        convert_options=pa_options[2],
    )

    return table


def csv_reader_infer_nb_arrow_type(
    filepath_or_buffer, delimiter=',', names=None, usecols=None, dtype=None, skiprows=None, parse_dates=False
):

    read_opts, parse_opts, convert_opts = get_pyarrow_read_csv_options(
                                                delimiter, names, usecols, dtype, skiprows, parse_dates)
    csv_reader = csv.open_csv(filepath_or_buffer,
                              read_options=read_opts,
                              parse_options=parse_opts,
                              convert_options=convert_opts)

    table_schema = csv_reader.schema

    nb_arrow_column_types = []
    for i, pa_data_type in enumerate(table_schema.types):
        nb_type = numpy_support.from_dtype(pa_data_type.to_pandas_dtype())

        if isinstance(nb_type, types.PyObject):
            if pa_data_type == pa.string():
                nb_type = StdStringViewType()
            else:
                raise TypingError("Cannot infer numba type for: ", pa_data_type, f"of column={table_schema.names[i]}")

        nb_arrow_column_types.append(nb_type)

    table_column_names = table_schema.names if not names else (names if usecols is None else usecols)

    arrow_table_type = ArrowTableType(nb_arrow_column_types, table_column_names)
    return arrow_table_type


def csv_reader_infer_nb_pandas_type(
    filepath_or_buffer, delimiter=',', names=None, usecols=None, dtype=None, skiprows=None, parse_dates=False
):

    # infer column types from the first block (similarly as Arrow does this)
    # TO-DO: tune the block size or allow user configure it via env var
    rows_to_read = 1000
    df = pd.read_csv(filepath_or_buffer, delimiter=delimiter, names=names,
                     usecols=usecols, dtype=dtype, skiprows=skiprows, nrows=rows_to_read,
                     parse_dates=parse_dates)

    try:
        df_type = numba.typeof(df)
    except ValueError:
        nb_col_types = []
        for col_name in df.columns:
            try:
                series_type = numba.typeof(df[col_name])
                col_type = series_type.data
            except ValueError:
                col_type = string_array_type
            nb_col_types.append(col_type)

        nb_col_types = tuple(nb_col_types)
        nb_col_names = tuple(df.columns)
        column_loc, _, _ = get_structure_maps(nb_col_types, nb_col_names)
        df_type = DataFrameType(nb_col_types, PositionalIndexType(), nb_col_names, column_loc=column_loc)

    return df_type


def csv_reader_get_pyarrow_read_options(names, skiprows):

    # if in the pd.read_csv call names=None the column names are inferred from the first row
    # but instead of using them here as column_names argument, we rely on pyarrow
    # autogenerated names, since this simplifies mapping of usecols pandas argument to
    # include_columns in ConvertOptions when usecols are column indices (not names)
    autogenerate_column_names = bool(names)

    read_options = pa.csv.ReadOptions(
        skip_rows=skiprows,
        # column_names=column_names,
        autogenerate_column_names=autogenerate_column_names,
    )

    return read_options


def csv_reader_get_pyarrow_parse_options(delimiter):

    parse_options = csv.ParseOptions(
        delimiter=delimiter,
    )
    return parse_options


def csv_reader_get_pyarrow_convert_options(names, usecols, dtype, parse_dates):

    include_columns = None  # default value (include all CSV columns)

    # if names is not given then column names will be defined from from the first row of CSV file
    # otherwise pyarrow autogenerated column names will be used (see ReadOptions), so
    # map pandas usecols to pyarrow include_columns accordingly
    if usecols:
        if type(usecols[0]) == str:
            if names:
                include_columns = [f'f{names.index(col)}' for col in usecols]
            else:
                include_columns = usecols  # no autogenerated names
        elif type(usecols[0]) == int:
            include_columns = [f'f{i}' for i in usecols]
        else:
            assert False, f"Failed building pyarrow ConvertOptions due to usecols param value: {usecols}"

    if dtype:
        # dtype pandas read_csv argument maps to pyarrow column_types dict, but column names
        # must match those that are read from CSV (if names is None) or pyarrows generated names (otherwise)
        if isinstance(dtype, dict):
            if names:
                names_list = list(names)
                column_types = {}
                for k, v in dtype.items():
                    # TO-DO: check this is aligned with include_columns
                    column_name = "f{}".format(names_list.index(k))
                    if isinstance(v, pd.CategoricalDtype):
                        column_type = pa.string()
                    else:
                        column_type = pa.from_numpy_dtype(v)
                    column_types[column_name] = column_type

            else:
                column_types = {k: pa.from_numpy_dtype(v) for k, v in dtype.items()}

        else:  # single dtype for all columns
            pa_dtype = pa.from_numpy_dtype(dtype)
            if names:
                column_types = {f"f{names_list.index(k)}": pa_dtype for k in names}
            elif usecols:
                column_types = dict.fromkeys(usecols, pa_dtype)
            else:
                column_types = pa_dtype
    else:
        column_types = None

    # TO-DO: support all possible parse_dates values (now only list of column positions is supported)
    try:
        for column in parse_dates:
            name = f"f{column}"
            # starting from pyarrow=3.0.0 strings are parsed to DateType (converted back to 'object'
            # when using to_pandas), but not TimestampType (that is used to represent np.datetime64)
            # see: pyarrow.from_numpy_dtype(np.datetime64('NaT', 's'))
            # so make pyarrow infer needed type manually
            column_types[name] = pa.timestamp('s')
    except (KeyError, TypeError):
        pass

    convert_options = csv.ConvertOptions(
        column_types=column_types,
        strings_can_be_null=True,
        include_columns=include_columns,
    )
    return convert_options


def get_pyarrow_read_csv_options(delimiter, names, usecols, dtype, skiprows, parse_dates):
    """ This function attempts to map pandas read_csv parameters to pyarrow read_csv options to be used """

    read_opts = csv_reader_get_pyarrow_read_options(names, skiprows)
    parse_opts = csv_reader_get_pyarrow_parse_options(delimiter)
    convert_opts = csv_reader_get_pyarrow_convert_options(names, usecols, dtype, parse_dates)

    return (read_opts, parse_opts, convert_opts)

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

'''
| :module:`pandas` functions and operators implementations in Intel SDC
'''

import pandas as pd
import numpy as np

import numba
from numba import types, objmode, literally
from numba.np import numpy_support
from numba.core.errors import TypingError
from numba.extending import overload

from sdc.io.csv_ext import (
    do_read_csv,
    csv_reader_infer_nb_arrow_type,
    csv_reader_infer_nb_pandas_type,
)
from sdc.str_arr_ext import string_array_type
from sdc.types import CategoricalDtypeType, Categorical
from sdc.datatypes.categorical.pdimpl import _reconstruct_CategoricalDtype
from sdc.utilities.utils import sdc_overload
from sdc.utilities.sdc_typing_utils import has_python_value
from sdc.extensions.sdc_arrow_table_type import PyarrowTableType
from sdc.extensions.sdc_arrow_table_ext import (
    arrow_reader_create_tableobj,
    apply_converters,
    combine_df_columns,
    create_df_from_columns,
    decref_pyarrow_table,
)


def get_df_col_type_from_dtype(dtype):
    if isinstance(dtype, types.Function):
        if dtype.typing_key == int:
            return types.Array(types.int_, 1, 'C')
        elif dtype.typing_key == float:
            return types.Array(types.float64, 1, 'C')
        elif dtype.typing_key == str:
            return string_array_type
        else:
            assert False, f"map_dtype_to_col_type: failing to infer column type for dtype={dtype}"

    if isinstance(dtype, types.StringLiteral):
        if dtype.literal_value == 'str':
            return string_array_type
        else:
            return types.Array(numba.from_dtype(np.dtype(dtype.literal_value)), 1, 'C')

    if isinstance(dtype, types.NumberClass):
        return types.Array(dtype.dtype, 1, 'C')

    if isinstance(dtype, CategoricalDtypeType):
        return Categorical(dtype)


def _get_py_col_dtype(ctype):
    """ Re-creates column dtype as python type to be used in read_csv call """
    dtype = ctype.dtype
    if ctype == string_array_type:
        return 'str'
    if isinstance(ctype, Categorical):
        return _reconstruct_CategoricalDtype(ctype.pd_dtype)
    return numpy_support.as_dtype(dtype)


def get_nbtype_literal_values(nbtype):
    assert all(isinstance(x, types.Literal) for x in nbtype), \
           f"Attempt to unliteral values of {nbtype} failed"

    return [x.literal_value for x in nbtype]


@sdc_overload(pd.read_csv)
def sdc_pandas_read_csv_ovld(
    filepath_or_buffer, sep=',', delimiter=None, header="infer", names=None, index_col=None,
    usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None,
    converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None,
    skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False,
    skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False,
    date_parser=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression="infer",
    thousands=None, decimal=b".", lineterminator=None, quotechar='"', doublequote=True, escapechar=None,
    comment=None, encoding=None, dialect=None, error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False,
    # low_memory=_c_parser_defaults["low_memory"],  # not supported
    memory_map=False, float_precision=None,
):

    # this overload is for inferencing from constant filename only, inferencing from params is TBD
    def sdc_pandas_read_csv_impl(
        filepath_or_buffer, sep=',', delimiter=None, header="infer", names=None, index_col=None,
        usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None,
        converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None,
        skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False,
        skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False,
        date_parser=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression="infer",
        thousands=None, decimal=b".", lineterminator=None, quotechar='"', doublequote=True, escapechar=None,
        comment=None, encoding=None, dialect=None, error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False,
        # low_memory=_c_parser_defaults["low_memory"],  # not supported
        memory_map=False, float_precision=None,
    ):
        # this forwards to the overload that accepts supported arguments only
        return sdc_internal_read_csv(filepath_or_buffer,
                                     sep=sep,
                                     delimiter=delimiter,
                                     names=names,
                                     usecols=usecols,
                                     dtype=dtype,
                                     converters=converters,
                                     skiprows=skiprows,
                                     parse_dates=parse_dates)

    return sdc_pandas_read_csv_impl


def read_csv_via_pyarrow():
    pass


def sdc_internal_read_csv(filepath_or_buffer, sep, delimiter, names, usecols, dtype,
                          converters, skiprows, parse_dates):
    pass


@overload(sdc_internal_read_csv, prefer_literal=True)
def sdc_internal_read_csv_ovld(filepath_or_buffer, sep, delimiter, names, usecols, dtype,
                               converters, skiprows, parse_dates):

    # print("Typing sdc_internal_read_csv, args:\n\t",
    #       filepath_or_buffer, sep, delimiter, names, usecols, dtype, converters, skiprows, parse_dates)

    if not isinstance(filepath_or_buffer, types.Literal):
        def sdc_internal_read_csv_impl(filepath_or_buffer, sep, delimiter, names, usecols, dtype,
                                       converters, skiprows, parse_dates):
            return literally(filepath_or_buffer)

        return sdc_internal_read_csv_impl

    accepted_args_types = {
        'filepath_or_buffer': (types.StringLiteral, ),
        'sep': (types.StringLiteral, types.Omitted),
        'delimiter': (types.StringLiteral, types.NoneType, types.Omitted),
        'names': (types.BaseAnonymousTuple, types.NoneType, types.Omitted),
        'usecols': (types.BaseAnonymousTuple, types.NoneType, types.Omitted),
        'skiprows': (types.IntegerLiteral, types.NoneType, types.Omitted),
        'parse_dates': (types.BaseAnonymousTuple, types.BooleanLiteral, types.Omitted),
        'converters': (types.LiteralStrKeyDict, types.NoneType, types.Omitted)
    }

    args_names = accepted_args_types.keys()
    args_py_defaults = dict.fromkeys(args_names, None)
    args_py_defaults.pop('filepath_or_buffer')  # no default value
    args_py_defaults.update({'sep': ',', 'parse_dates': False})

    param_types = locals()
    param_types = {k: v for k, v in param_types.items() if k in args_names}

    def _param_checker(x, accepted_types, defaults):
        is_default = has_python_value(x, defaults[x]) if x in defaults else False
        return isinstance(param_types[x], accepted_types[x]) or is_default

    check_const_args = {x: _param_checker(x, accepted_args_types, args_py_defaults) for x in args_names}
    assert all(check_const_args.values()), \
           f"""SDC read_csv can work with const args affecting column type inference only.
               \tGiven param_types: {param_types}
               \tCheck results: {check_const_args}"""

    # parameters should be constants when inferring DF type from a csv file
    py_filepath_or_buffer = filepath_or_buffer.literal_value

    if (isinstance(names, (types.Omitted, types.NoneType)) or names is None):
        py_names = None
    else:
        py_names = get_nbtype_literal_values(names)

    if (isinstance(usecols, (types.Omitted, types.NoneType)) or usecols is None):
        py_usecols = None
    else:
        py_usecols = get_nbtype_literal_values(usecols)

    py_sep = sep.literal_value if isinstance(sep, types.Literal) else sep == ','
    py_delimiter = delimiter.literal_value if isinstance(delimiter, types.Literal) else None

    if py_delimiter is None:
        py_delimiter = py_sep
    py_skiprows = skiprows.literal_value if isinstance(skiprows, types.Literal) else None

    if isinstance(parse_dates, types.Literal):
        py_parse_dates = parse_dates.literal_value
    elif isinstance(parse_dates, types.BaseAnonymousTuple):
        py_parse_dates = get_nbtype_literal_values(parse_dates)
    else:
        assert False, "sdc_internal_read_csv: parse_dates parameter must be literal"

    if isinstance(dtype, types.Tuple):
        # dtype is a tuple of format ('A', A_dtype, 'B', B_dtype, ...) after RewriteReadCsv
        keys = [k.literal_value for k in dtype[::2]]
        values = list(map(get_df_col_type_from_dtype, dtype[1::2]))
        py_dtype = dict(zip(keys, map(_get_py_col_dtype, values)))
    elif isinstance(dtype, types.LiteralStrKeyDict):
        keys = dtype.fields
        values = list(map(get_df_col_type_from_dtype, dtype.types))
        py_dtype = dict(zip(keys, map(_get_py_col_dtype, values)))
    elif isinstance(dtype, types.NoneType) or dtype is None:
        py_dtype = None
    elif isinstance(dtype, types.NumberClass):
        py_dtype = dtype.key.key
    else:
        assert False, f"Not supported dtype parameter received, with numba type: {dtype}"

    # infer the resulting DF type as a numba type
    pandas_df_type = csv_reader_infer_nb_pandas_type(
        py_filepath_or_buffer,
        delimiter=py_delimiter,
        names=py_names,
        usecols=py_usecols,
        dtype=py_dtype,
        skiprows=py_skiprows,
        parse_dates=py_parse_dates
    )

    col_names = pandas_df_type.columns
    col_types = pandas_df_type.data

    py_col_dtypes = {cname: _get_py_col_dtype(ctype) for cname, ctype in zip(col_names, col_types)}
    cat_columns_list = [name for name in col_names if isinstance(py_col_dtypes[name], pd.CategoricalDtype)]

    # need to re-order usecols as they appear in pandas_df_type in different order as in pyarrow table
    if py_usecols is not None:
        def _check_usecol_type(py_val, py_type):
            return all([isinstance(x, py_type) for x in py_val])

        if _check_usecol_type(py_usecols, str):
            py_usecols = tuple([c for c in col_names if c in set(py_usecols)])
        elif _check_usecol_type(py_usecols, int):
            py_usecols = tuple(sorted(py_usecols))
        else:
            assert False, f"Unsupported usecols param value: {py_usecols}"

    use_user_converters = not (isinstance(converters, types.NoneType) or converters is None)
    if not use_user_converters:

        # dtype parameter is deliberately captured into objmode as global value to avoid
        # IR grow due to passing large tuples as function arguments
        def sdc_internal_read_csv_impl(filepath_or_buffer, sep, delimiter, names, usecols, dtype,
                                       converters, skiprows, parse_dates):
            with objmode(df=pandas_df_type):
                pa_table = do_read_csv(
                    filepath_or_buffer,
                    sep=sep,
                    delimiter=delimiter,
                    names=names,
                    usecols=py_usecols,
                    dtype=py_col_dtypes,
                    skiprows=skiprows,
                    parse_dates=parse_dates
                )

                df = pa_table.rename_columns(col_names).to_pandas(categories=cat_columns_list)

                # fix when PyArrow will support predicted categories
                for cat_column_name in cat_columns_list:
                    df[cat_column_name].cat.set_categories(py_col_dtypes[cat_column_name].categories, inplace=True)

            return df

        return sdc_internal_read_csv_impl

    else:

        converterted_cols = set(converters.fields)
        py_col_dtypes.update(dict.fromkeys(converterted_cols, 'str'))
        arrow_table_type = csv_reader_infer_nb_arrow_type(py_filepath_or_buffer,
                                                          delimiter=py_delimiter,
                                                          names=py_names,
                                                          usecols=py_usecols,
                                                          dtype=py_col_dtypes,
                                                          skiprows=py_skiprows,
                                                          parse_dates=py_parse_dates)

        n_cols = len(col_names)
        pa_table_type = PyarrowTableType()
        non_converted_cols_ids = [i for i in range(n_cols) if col_names[i] not in converterted_cols]

        all_columns_converted = set(col_names) == converterted_cols
        if all_columns_converted:
            objmode_ret_column = types.none
        else:
            objmode_ret_tup_types = []
            for i, name in enumerate(col_names):
                if name in converterted_cols:
                    objmode_ret_tup_types.append(types.none)
                else:
                    objmode_ret_tup_types.append(pandas_df_type.get_series_type(i))
            objmode_ret_column = types.Tuple.from_types(objmode_ret_tup_types)

        def sdc_internal_read_csv_impl(filepath_or_buffer, sep, delimiter, names, usecols, dtype,
                                       converters, skiprows, parse_dates):
            with objmode(pa_table=pa_table_type, maybe_unboxed_columns=objmode_ret_column):
                pa_table = do_read_csv(
                    filepath_or_buffer,
                    sep=sep,
                    delimiter=delimiter,
                    names=names,
                    usecols=py_usecols,
                    dtype=py_col_dtypes,
                    skiprows=skiprows,
                    parse_dates=parse_dates
                )

                pa_table = pa_table.rename_columns(col_names)

                if all_columns_converted:
                    maybe_unboxed_columns = None
                else:
                    # if converters are not applied to some columns, convert them to pandas series
                    # using pyarrow API calls and return from objmode in a tuple of all DF columns
                    # where they reside at corresponding positions and other elements are None-s
                    ret_cols = [None] * n_cols
                    for i in non_converted_cols_ids:
                        col_as_series = pa_table.column(col_names[i]).to_pandas(categories=cat_columns_list)
                        # fix when PyArrow will support predicted categories
                        if isinstance(col_as_series, pd.CategoricalDtype):
                            col_as_series.cat.set_categories(py_col_dtypes[col_names[i]], inplace=True)
                        ret_cols[i] = col_as_series

                    maybe_unboxed_columns = tuple(ret_cols)

            arrow_table = arrow_reader_create_tableobj(arrow_table_type, pa_table)
            decref_pyarrow_table(pa_table)
            converted_columns = apply_converters(arrow_table, converters)
            all_df_columns = combine_df_columns(maybe_unboxed_columns, converted_columns)

            res_df = create_df_from_columns(col_names, all_df_columns)
            return res_df

        return sdc_internal_read_csv_impl


sdc_pandas_read_csv_ovld.__doc__ = r"""
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.read_csv

    Limitations
    -----------
    - Parameters \
        ``header``, \
        ``index_col``, \
        ``squeeze``, \
        ``prefix``, \
        ``mangle_dupe_cols``, \
        ``engine``, \
        ``converters``, \
        ``true_values``, \
        ``false_values``, \
        ``skipinitialspace``, \
        ``skipfooter``, \
        ``nrows``, \
        ``na_values``, \
        ``keep_default_na``, \
        ``na_filter``, \
        ``verbose``, \
        ``skip_blank_lines``, \
        ``parse_dates``, \
        ``infer_datetime_format``, \
        ``keep_date_col``, \
        ``date_parser``, \
        ``dayfirst``, \
        ``cache_dates``, \
        ``iterator``, \
        ``chunksize``, \
        ``compression``, \
        ``thousands``, \
        ``decimal``, \
        ``lineterminator``, \
        ``quotechar``, \
        ``quoting``, \
        ``doublequote``, \
        ``escapechar``, \
        ``comment``, \
        ``encoding``, \
        ``dialect``, \
        ``error_bad_lines``, \
        ``warn_bad_lines``, \
        ``delim_whitespace``, \
        ``low_memory``, \
        ``memory_map`` and \
        ``float_precision`` \
        are currently unsupported by Intel Scalable Dataframe Compiler.
    - Resulting DataFrame type could be inferred from constant file name of from parameters. \
        ``filepath_or_buffer`` could be constant for inferencing from file. \
        ``filepath_or_buffer`` could be variable for inferencing from parameters if ``dtype`` is constant. \
        If both ``filepath_or_buffer`` and ``dtype`` are constants then default is inferencing from parameters.
    - For inferring from parameters ``names`` or ``usecols`` should be provided additionally to ``dtype``.
    - For inferring from file ``sep``, ``delimiter`` and ``skiprows`` should be constants or omitted.
    - ``names`` and ``usecols`` should be constants or omitted for both types of inferrencing.
    - ``usecols`` with list of ints is unsupported by Intel Scalable Dataframe Compiler.

    Examples
    --------
    Inference from file. File name is constant. \
    Resulting DataFrame depends on CSV file content at the moment of compilation.

    >>> pd.read_csv('data.csv')  # doctest: +SKIP

    Inference from file. File name, ``names``, ``usecols``, ``delimiter`` and ``skiprow`` are constants. \
    Resulting DataFrame contains one column ``A`` \
    with type of column depending on CSV file content at the moment of compilation.

    >>> pd.read_csv('data.csv', names=['A','B'], usecols=['A'], delimiter=';', skiprows=2)  # doctest: +SKIP

    Inference from parameters. File name, ``delimiter`` and ``skiprow`` are variables. \
    ``names``, ``usecols`` and ``dtype`` are constants. \
    Resulting DataFrame contains column ``A`` with type ``np.float64``.

    >>> pd.read_csv(file_name, names=['A','B'], usecols=['A'], dtype={'A': np.float64}, \
                    delimiter=some_char, skiprows=some_int)  # doctest: +SKIP
"""

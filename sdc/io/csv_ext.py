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


import contextlib
import functools

import llvmlite.binding as ll
from llvmlite import ir as lir
from .. import hio
from collections import defaultdict
import numba
from numba.core import typeinfer, ir, ir_utils, types
from numba.core.typing.templates import signature
from numba.extending import overload, intrinsic, register_model, models, box
from numba.core.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes)
from numba.core import analysis
from numba.parfors import array_analysis
import sdc
from sdc import distributed, distributed_analysis
from sdc.utilities.utils import (debug_prints, alloc_arr_tup, empty_like_type,
                                 _numba_to_c_type_map)
from sdc.distributed_analysis import Distribution
from sdc.str_ext import string_type
from sdc.str_arr_ext import (string_array_type, to_string_list,
                              cp_str_list_to_array, str_list_to_array,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, num_total_chars,
                              getitem_str_offset, copy_str_arr_slice)
from sdc.timsort import copyElement_tup, getitem_arr_tup
from sdc import objmode
import pandas as pd
import numpy as np

from sdc.types import CategoricalDtypeType, Categorical

import pyarrow
import pyarrow.csv


class CsvReader(ir.Stmt):
    def __init__(self, file_name, df_out, sep, df_colnames, out_vars, out_types, usecols, loc, skiprows=0):
        self.file_name = file_name
        self.df_out = df_out
        self.sep = sep
        self.df_colnames = df_colnames
        self.out_vars = out_vars
        self.out_types = out_types
        self.usecols = usecols
        self.loc = loc
        self.skiprows = skiprows

    def __repr__(self):  # pragma: no cover
        # TODO
        return "{} = ReadCsv()".format(self.df_out)


def csv_array_analysis(csv_node, equiv_set, typemap, array_analysis):
    post = []
    # empty csv nodes should be deleted in remove dead
    assert len(csv_node.out_vars) > 0, "empty csv in array analysis"

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for col_var in csv_node.out_vars:
        typ = typemap[col_var.name]
        # TODO: string_series_type also?
        if typ == string_array_type:
            continue
        (shape, c_post) = array_analysis._gen_shape_call(
            equiv_set, col_var, typ.ndim, None)
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, {})

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


array_analysis.array_analysis_extensions[CsvReader] = csv_array_analysis


def csv_distributed_analysis(csv_node, array_dists):
    for v in csv_node.out_vars:
        if v.name not in array_dists:
            array_dists[v.name] = Distribution.OneD

    return


distributed_analysis.distributed_analysis_extensions[CsvReader] = csv_distributed_analysis


def csv_typeinfer(csv_node, typeinferer):
    for col_var, typ in zip(csv_node.out_vars, csv_node.out_types):
        typeinferer.lock_type(col_var.name, typ, loc=csv_node.loc)
    return


typeinfer.typeinfer_extensions[CsvReader] = csv_typeinfer


def visit_vars_csv(csv_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting csv vars for:", csv_node)
        print("cbdata: ", sorted(cbdata.items()))

    # update output_vars
    new_out_vars = []
    for col_var in csv_node.out_vars:
        new_var = visit_vars_inner(col_var, callback, cbdata)
        new_out_vars.append(new_var)

    csv_node.out_vars = new_out_vars
    csv_node.file_name = visit_vars_inner(csv_node.file_name, callback, cbdata)
    return


# add call to visit csv variable
ir_utils.visit_vars_extensions[CsvReader] = visit_vars_csv


def remove_dead_csv(csv_node, lives, arg_aliases, alias_map, func_ir, typemap):
    # TODO
    new_df_colnames = []
    new_out_vars = []
    new_out_types = []
    new_usecols = []

    for i, col_var in enumerate(csv_node.out_vars):
        if col_var.name in lives:
            new_df_colnames.append(csv_node.df_colnames[i])
            new_out_vars.append(csv_node.out_vars[i])
            new_out_types.append(csv_node.out_types[i])
            new_usecols.append(csv_node.usecols[i])

    csv_node.df_colnames = new_df_colnames
    csv_node.out_vars = new_out_vars
    csv_node.out_types = new_out_types
    csv_node.usecols = new_usecols

    if len(csv_node.out_vars) == 0:
        return None

    return csv_node


ir_utils.remove_dead_extensions[CsvReader] = remove_dead_csv


def csv_usedefs(csv_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # output columns are defined
    def_set.update({v.name for v in csv_node.out_vars})
    use_set.add(csv_node.file_name.name)

    return analysis._use_defs_result(usemap=use_set, defmap=def_set)


analysis.ir_extension_usedefs[CsvReader] = csv_usedefs


def get_copies_csv(csv_node, typemap):
    # csv doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in csv_node.out_vars)
    return set(), kill_set


ir_utils.copy_propagate_extensions[CsvReader] = get_copies_csv


def apply_copies_csv(csv_node, var_dict, name_var_table, typemap, calltypes, save_copies):
    """apply copy propagate in csv node"""

    # update output_vars
    new_out_vars = []
    for col_var in csv_node.out_vars:
        new_var = replace_vars_inner(col_var, var_dict)
        new_out_vars.append(new_var)

    csv_node.out_vars = new_out_vars
    csv_node.file_name = replace_vars_inner(csv_node.file_name, var_dict)
    return


ir_utils.apply_copy_propagate_extensions[CsvReader] = apply_copies_csv


def build_csv_definitions(csv_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in csv_node.out_vars:
        definitions[col_var.name].append(csv_node)

    return definitions


ir_utils.build_defs_extensions[CsvReader] = build_csv_definitions


def csv_distributed_run(csv_node, array_dists, typemap, calltypes, typingctx, targetctx, dist_pass):
    parallel = False

    n_cols = len(csv_node.out_vars)
    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    # get column variables
    arg_names = ", ".join("arr" + str(i) for i in range(n_cols))
    func_text = "def csv_impl(fname):\n"
    func_text += "    ({},) = _csv_reader_py(fname)\n".format(arg_names)
    # print(func_text)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    csv_impl = loc_vars['csv_impl']

    csv_reader_py = _gen_csv_reader_py(
        csv_node.df_colnames, csv_node.out_types, csv_node.usecols,
        csv_node.sep, typingctx, targetctx, parallel, csv_node.skiprows)

    f_block = compile_to_numba_ir(csv_impl,
                                  {'_csv_reader_py': csv_reader_py},
                                  typingctx, (string_type,),
                                  typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [csv_node.file_name])
    nodes = f_block.body[:-3]
    for i in range(len(csv_node.out_vars)):
        nodes[-len(csv_node.out_vars) + i].target = csv_node.out_vars[i]

    # get global array sizes by calling allreduce on chunk lens
    # TODO: get global size from C
    for arr in csv_node.out_vars:
        def f(A):
            return sdc.distributed_api.dist_reduce(len(A), np.int32(_op))
        f_block = compile_to_numba_ir(
            f, {'sdc': sdc, 'np': np,
                '_op': sdc.distributed_api.Reduce_Type.Sum.value},
            typingctx, (typemap[arr.name],), typemap, calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr])
        nodes += f_block.body[:-2]
        size_var = nodes[-1].target
        dist_pass._array_sizes[arr.name] = [size_var]
        out, start_var, end_var = dist_pass._gen_1D_div(
            size_var, arr.scope, csv_node.loc, "$alloc", "get_node_portion",
            sdc.distributed_api.get_node_portion)
        dist_pass._array_starts[arr.name] = [start_var]
        dist_pass._array_counts[arr.name] = [end_var]
        nodes += out

    return nodes


distributed.distributed_run_extensions[CsvReader] = csv_distributed_run


class StreamReaderType(types.Opaque):
    def __init__(self):
        super(StreamReaderType, self).__init__(name='StreamReaderType')


stream_reader_type = StreamReaderType()
register_model(StreamReaderType)(models.OpaqueModel)


@box(StreamReaderType)
def box_stream_reader(typ, val, c):
    return val


def _get_dtype_str(t):
    dtype = t.dtype

    if isinstance(t, Categorical):
        # return categorical representation
        # for some reason pandas and pyarrow read_csv() return CategoricalDtype with
        # ordered=False in case when dtype is with ordered=None
        return str(t).replace('ordered=None', 'ordered=False')

    if dtype == types.NPDatetime('ns'):
        dtype = 'NPDatetime("ns")'
    if t == string_array_type:
        # HACK: add string_array_type to numba.types
        # FIXME: fix after Numba #3372 is resolved
        types.string_array_type = string_array_type
        return 'string_array_type'
    return '{}[::1]'.format(dtype)


def _get_pd_dtype_str(t):
    dtype = t.dtype
    if isinstance(t, Categorical):
        return 'pd.{}'.format(t.pd_dtype)
    if dtype == types.NPDatetime('ns'):
        dtype = 'str'
    if t == string_array_type:
        return 'str'
    return 'np.{}'.format(dtype)


# XXX: temporary fix pending Numba's #3378
# keep the compiled functions around to make sure GC doesn't delete them and
# the reference to the dynamic function inside them
# (numba/lowering.py:self.context.add_dynamic_addr ...)
compiled_funcs = []


# TODO: move to hpat.common
def to_varname(string):
    """Converts string to correct Python variable name.
    Replaces unavailable symbols with _ and insert _ if string starts with digit.
    """
    import re
    return re.sub(r'\W|^(?=\d)','_', string)


@contextlib.contextmanager
def pyarrow_cpu_count(cpu_count=pyarrow.cpu_count()):
    old_cpu_count = pyarrow.cpu_count()
    pyarrow.set_cpu_count(cpu_count)
    try:
        yield
    finally:
        pyarrow.set_cpu_count(old_cpu_count)


def pyarrow_cpu_count_equal_numba_num_treads(func):
    """Decorator. Set pyarrow cpu_count the same as NUMBA_NUM_THREADS."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with pyarrow_cpu_count(numba.config.NUMBA_NUM_THREADS):
            return func(*args, **kwargs)
    return wrapper


@pyarrow_cpu_count_equal_numba_num_treads
def pandas_read_csv(
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
    """Implements pandas.read_csv via pyarrow.csv.read_csv.
    This function has the same interface as pandas.read_csv.
    """

    if delimiter is None:
        delimiter = sep

    autogenerate_column_names = bool(names)

    include_columns = None

    if usecols:
        if type(usecols[0]) == str:
            if names:
                include_columns = [f'f{names.index(col)}' for col in usecols]
            else:
                include_columns = usecols
        elif type(usecols[0]) == int:
            include_columns = [f'f{i}' for i in usecols]
        else:
            # usecols should be all str or int
            assert False

    # try:
    #     keys = [k for k, v in dtype.items() if isinstance(v, pd.CategoricalDtype)]
    #     if keys:
    #         for k in keys:
    #             del dtype[k]
    #         names_list = list(names)
    #         categories = [f"f{names_list.index(k)}" for k in keys]
    # except: pass

    categories = []

    if dtype:
        if names:
            names_list = list(names)
            if isinstance(dtype, dict):
                column_types = {}
                for k, v in dtype.items():
                    column_name = "f{}".format(names_list.index(k))
                    if isinstance(v, pd.CategoricalDtype):
                        categories.append(column_name)
                        column_type = pyarrow.string()
                    else:
                        column_type = pyarrow.from_numpy_dtype(v)
                    column_types[column_name] = column_type
            else:
                pa_dtype = pyarrow.from_numpy_dtype(dtype)
                column_types = {f"f{names_list.index(k)}": pa_dtype for k in names}
        elif usecols:
            if isinstance(dtype, dict):
                column_types = {k: pyarrow.from_numpy_dtype(v) for k, v in dtype.items()}
            else:
                column_types = {k: pyarrow.from_numpy_dtype(dtype) for k in usecols}
        else:
            if isinstance(dtype, dict):
                column_types = {k: pyarrow.from_numpy_dtype(v) for k, v in dtype.items()}
            else:
                column_types = pyarrow.from_numpy_dtype(dtype)
    else:
        column_types = None

    try:
        for column in parse_dates:
            name = f"f{column}"
            # TODO: Try to help pyarrow infer date type - set DateType.
            # dtype[name] = pyarrow.from_numpy_dtype(np.datetime64) # string
            del column_types[name]
    except: pass

    parse_options = pyarrow.csv.ParseOptions(
        delimiter=delimiter,
    )

    read_options = pyarrow.csv.ReadOptions(
        skip_rows=skiprows,
        # column_names=column_names,
        autogenerate_column_names=autogenerate_column_names,
    )

    convert_options = pyarrow.csv.ConvertOptions(
        column_types=column_types,
        strings_can_be_null=True,
        include_columns=include_columns,
    )

    table = pyarrow.csv.read_csv(
        filepath_or_buffer,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    dataframe = table.to_pandas(
        # categories=categories or None,
    )

    if names:
        if usecols and len(names) != len(usecols):
            if isinstance(usecols[0], int):
                dataframe.columns = [names[col] for col in usecols]
            elif isinstance(usecols[0], str):
                dataframe.columns = [name for name in names if name in usecols]
        else:
            dataframe.columns = names

    # fix when PyArrow will support predicted categories
    if isinstance(dtype, dict):
        for column_name, column_type in dtype.items():
            if isinstance(column_type, pd.CategoricalDtype):
                dataframe[column_name] = dataframe[column_name].astype(column_type)

    return dataframe


def _gen_csv_reader_py_pyarrow(col_names, col_typs, usecols, sep, typingctx, targetctx, parallel, skiprows):
    func_text, func_name = _gen_csv_reader_py_pyarrow_func_text(col_names, col_typs, usecols, sep, skiprows)
    csv_reader_py = _gen_csv_reader_py_pyarrow_py_func(func_text, func_name)
    return _gen_csv_reader_py_pyarrow_jit_func(csv_reader_py)


def _gen_csv_reader_py_pyarrow_func_text_core(col_names, col_typs, dtype_present, usecols, signature=None):
    # TODO: support non-numpy types like strings
    date_inds = ", ".join(str(i) for i, t in enumerate(col_typs) if t.dtype == types.NPDatetime('ns'))
    return_columns = usecols if usecols and isinstance(usecols[0], str) else col_names
    nb_objmode_vars = ", ".join([
        '{}="{}"'.format(to_varname(cname), _get_dtype_str(t))
        for cname, t in zip(return_columns, col_typs)
    ])
    pd_dtype_strs = ", ".join([
        "'{}': {}".format(cname, _get_pd_dtype_str(t))
        for cname, t in zip(return_columns, col_typs)
    ])

    if signature is None:
        signature = "filepath_or_buffer"
    func_text = "def csv_reader_py({}):\n".format(signature)
    func_text += "  with objmode({}):\n".format(nb_objmode_vars)
    func_text += "    df = pandas_read_csv(filepath_or_buffer,\n"

    # pyarrow reads unnamed header as " ", pandas reads it as "Unnamed: N"
    # during inference from file names should be raplaced with "Unnamed: N"
    # passing names to pyarrow means that one row is header and should be skipped
    if col_names and any(map(lambda x: x.startswith('Unnamed: '), col_names)):
        func_text += "        names={},\n".format(col_names)
        func_text += "        skiprows=(skiprows and skiprows + 1) or 1,\n"
    else:
        func_text += "        names=names,\n"
        func_text += "        skiprows=skiprows,\n"

    func_text += "        parse_dates=[{}],\n".format(date_inds)

    # Python objects (e.g. str, np.float) could not be jitted and passed to objmode
    # so they are hardcoded to function
    # func_text += "        dtype={{{}}},\n".format(pd_dtype_strs) if dtype_present else \
    #              "        dtype=dtype,\n"
    # dtype is hardcoded because datetime should be read as string
    func_text += "        dtype={{{}}},\n".format(pd_dtype_strs)

    func_text += "        usecols=usecols,\n"
    func_text += "        sep=sep,\n"
    func_text += "        delimiter=delimiter,\n"
    func_text += "    )\n"
    for cname in return_columns:
        func_text += "    {} = df['{}'].values\n".format(to_varname(cname), cname)
        # func_text += "    print({})\n".format(cname)
    return func_text, 'csv_reader_py'


def _gen_csv_reader_py_pyarrow_func_text(col_names, col_typs, usecols):
    func_text, func_name = _gen_csv_reader_py_pyarrow_func_text_core(col_names, col_typs, usecols)

    func_text += "  return ({},)\n".format(", ".join(to_varname(c) for c in col_names))

    return func_text, func_name


def _gen_csv_reader_py_pyarrow_func_text_dataframe(col_names, col_typs, dtype_present, usecols, signature):
    func_text, func_name = _gen_csv_reader_py_pyarrow_func_text_core(
        col_names, col_typs, dtype_present, usecols, signature)
    return_columns = usecols if usecols and isinstance(usecols[0], str) else col_names

    func_text += "  return sdc.hiframes.pd_dataframe_ext.init_dataframe({}, None, {})\n".format(
        ", ".join(to_varname(c) for c in return_columns),
        ", ".join("'{}'".format(c) for c in return_columns)
    )

    return func_text, func_name


def _gen_csv_reader_py_pyarrow_py_func(func_text, func_name):
    locals = {}
    exec(func_text, globals(), locals)
    func = locals[func_name]
    return func


def _gen_csv_reader_py_pyarrow_jit_func(csv_reader_py):
    # TODO: no_cpython_wrapper=True crashes for some reason
    jit_func = numba.njit(csv_reader_py)
    compiled_funcs.append(jit_func)
    return jit_func


_gen_csv_reader_py = _gen_csv_reader_py_pyarrow

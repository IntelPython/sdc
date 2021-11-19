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

import numpy as np
import pandas as pd
import pyarrow as pa

import numba
from numba import njit, objmode, literally
from numba.core import types, cgutils
from numba.core.errors import TypingError
from numba.core.imputils import impl_ret_borrowed
from numba.core.pythonapi import unbox, NativeValue
from numba.extending import overload, intrinsic, typeof_impl
import llvmlite.binding as ll
from llvmlite import ir as lir

import sdc
from sdc import harrow_reader
from sdc.utilities.utils import sdc_overload
from sdc.utilities.prange_utils import parallel_chunks
from sdc.extensions.sdc_arrow_table_type import ArrowTableType, PyarrowTableType
from sdc.extensions.sdc_string_view_type import StdStringViewType
from sdc.extensions.sdc_string_view_ext import string_view_create
from sdc.extensions.sdc_hashmap_ext import load_native_func


load_native_func("create_arrow_table", harrow_reader)
load_native_func("get_table_len", harrow_reader)
load_native_func("get_table_cell", harrow_reader)


@typeof_impl.register(pa.lib.Table)
def _typeof_pyarrow_table(val, c):
    return PyarrowTableType()


@unbox(PyarrowTableType)
def unbox_pyarrow_table(typ, val, c):
    # incref pyobject, as Numba releases it when returning from objmode
    c.pyapi.incref(val)
    return NativeValue(val)


@intrinsic
def decref_pyarrow_table(typingctx, pa_table):
    ret_type = types.void

    def codegen(context, builder, sig, args):
        pa_table_val = args[0]
        context.get_python_api(builder).decref(pa_table_val)

    return ret_type(pa_table), codegen


@intrinsic
def arrow_reader_table_len(typingctx, a_table):

    ret_type = types.int64

    def codegen(context, builder, sig, args):
        arrow_table = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
        fnty = lir.FunctionType(lir.IntType(64),
                                [lir.IntType(8).as_pointer(), ])
        func_name = f"get_table_len"
        fn_arrow_table_len = cgutils.get_or_insert_function(
            builder.module, fnty, name=func_name)

        res = builder.call(fn_arrow_table_len,
                           [arrow_table.table_ptr, ])

        return res

    return ret_type(a_table), codegen


@overload(len)
def len_arrow_table_ovld(a_table):
    if not isinstance(a_table, ArrowTableType):
        return None

    def len_arrow_table_impl(a_table):
        return arrow_reader_table_len(a_table)
    return len_arrow_table_impl


@intrinsic
def arrow_reader_create_tableobj(typingctx, table_type, pa_table):

    ret_type = table_type.instance_type

    def codegen(context, builder, sig, args):
        table_pyobject = args[1]

        nrt_table = context.nrt.get_nrt_api(builder)
        arrow_table = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer(),               # ptr to pyarrow table
                                 arrow_table.meminfo.type.as_pointer(),     # meminfo to fill
                                 lir.IntType(8).as_pointer(), ])            # NRT API func table
        func_name = f"create_arrow_table"
        fn = cgutils.get_or_insert_function(builder.module, fnty, name=func_name)

        builder.call(fn, [table_pyobject,
                          arrow_table._get_ptr_by_name('meminfo'),
                          nrt_table])

        arrow_table.table_ptr = context.nrt.meminfo_data(builder, arrow_table.meminfo)
        return arrow_table._getvalue()

    return ret_type(table_type, pa_table), codegen


@intrinsic
def alloc_res_storage_for_number(typingctx, nbtype):

    number_dtype = nbtype.dtype
    ret_type = types.voidptr

    def codegen(context, builder, sig, args):
        lir_res_type = context.get_value_type(number_dtype)
        res_ptr = cgutils.alloca_once(builder, lir_res_type)
        return builder.bitcast(res_ptr, cgutils.voidptr_t)

    return ret_type(nbtype), codegen


def alloc_res_storage(nbtype):
    pass


@overload(alloc_res_storage)
def alloc_res_storage_ovld(nbtype):

    if isinstance(nbtype, types.NumberClass):
        def alloc_res_storage_impl(nbtype):
            return alloc_res_storage_for_number(nbtype)
        return alloc_res_storage_impl

    elif isinstance(nbtype.instance_type, StdStringViewType):
        def alloc_res_storage_impl(nbtype):
            return string_view_create()
        return alloc_res_storage_impl

    else:
        assert False, f"No implementation of alloc_res_storage for type: {nbtype}"


@intrinsic
def read_from_storage(typingctx, nbtype, ret):

    ret_type = nbtype.dtype if isinstance(nbtype, types.NumberClass) else nbtype.instance_type

    def codegen(context, builder, sig, args):
        ret_val = args[1]

        if isinstance(nbtype, types.NumberClass):
            lir_ret_type = context.get_value_type(ret_type).as_pointer()
            caster_ptr = builder.bitcast(ret_val,
                                         lir_ret_type)
            return builder.load(caster_ptr)
        else:
            # currently this branch is used for StringViewType instance (thus incref is needed)
            return impl_ret_borrowed(context, builder, ret_type, ret_val)

    return ret_type(nbtype, ret), codegen


def df_alloc_column_of_dtype(dtype, size):
    pass


@overload(df_alloc_column_of_dtype)
def df_alloc_column_of_dtype_ovld(dtype, size):

    if isinstance(dtype, types.NumberClass):
        res_dtype = dtype.dtype

        def df_alloc_column_of_dtype_impl(dtype, size):
            return np.empty(size, dtype=res_dtype)

        return df_alloc_column_of_dtype_impl

    elif isinstance(dtype.instance_type, types.UnicodeType):
        def df_alloc_column_of_dtype_impl(dtype, size):
            return ['' for _ in range(size)]

        return df_alloc_column_of_dtype_impl

    else:
        assert False, f"Function df_alloc_column_of_dtype not supported for dtype: {dtype}"


def arrow_reader_get_table_cell(table, col_idx, row_idx, ret):
    pass


@overload(arrow_reader_get_table_cell)
def arrow_reader_get_table_cell_ovld(table, col_idx, row_idx, ret):
    if not isinstance(table, ArrowTableType):
        return None

    # FIXME_Numba#7568: using literally erases the compilation-failure traces in error messages
    # for debugging use else block only or use logging module with DEBUG
    if not isinstance(col_idx, types.IntegerLiteral):

        def arrow_reader_get_table_cell_impl(table, col_idx, row_idx, ret):
            return literally(col_idx)
        return arrow_reader_get_table_cell_impl
    else:

        def arrow_reader_get_table_cell_impl(table, col_idx, row_idx, ret):
            return arrow_reader_get_table_cell_internal(table, col_idx, row_idx, ret)
        return arrow_reader_get_table_cell_impl


@intrinsic
def arrow_reader_get_table_cell_internal(typingctx, table, col_idx, row_idx, ret):

    assert isinstance(col_idx, types.IntegerLiteral), \
           f"arrow_reader_get_table_cell_internal: col_idx must be literal integer, given: {col_idx}"

    col_idx_literal_val = col_idx.literal_value
    col_type = table.dtypes[col_idx_literal_val]
    ret_type = types.int8

    def codegen(context, builder, sig, args):
        col_idx_val = context.get_constant(types.int64, col_idx_literal_val)
        row_idx_val, ret_val = args[2:]

        arrow_table = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
        fnty = lir.FunctionType(lir.IntType(8),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(64),
                                 lir.IntType(64),
                                 lir.IntType(8).as_pointer()])
        func_name = f"get_table_cell"
        fn_arrow_get_table_cell = cgutils.get_or_insert_function(builder.module, fnty, name=func_name)

        # pointers to table elements (e.g. for string data) returned by this function are raw pointers
        # not extending the lifetime of whole table (thus no incref), it's only safe to use them
        # in scenarios like during read_csv conversion when table is guaranteed to be alive
        # if context.enable_nrt:
        #     context.nrt.incref(builder, sig.args[0], args[0])

        # get the native value pointer, where the result will be stored, StdStringView objects are numba wrappers
        # for real std::string_view* (data_ptr), so they need special handling
        if isinstance(col_type, StdStringViewType):
            str_view_struct = cgutils.create_struct_proxy(col_type)(context, builder, value=ret_val)
            native_val_ptr = str_view_struct.data_ptr
        else:
            native_val_ptr = ret_val

        ret_code = builder.call(fn_arrow_get_table_cell,
                                [arrow_table.table_ptr,
                                 col_idx_val,
                                 row_idx_val,
                                 native_val_ptr])

        return ret_code

    return ret_type(table, col_idx, row_idx, ret), codegen


def apply_converter_to_column(table, col_idx, func):
    pass


@overload(apply_converter_to_column, inline='always')
def apply_converter_to_column_ovld(table, col_idx, func):

    if not isinstance(col_idx, types.IntegerLiteral):
        return None

    if isinstance(func, (types.Dispatcher, types.Function)):
        col_idx_literal_val = col_idx.literal_value
        col_dtype = table.dtypes[col_idx_literal_val]
        res_dtype = func.get_call_type(func.dispatcher.typingctx, (col_dtype, ), ()).return_type

        def apply_converter_to_column_impl(table, col_idx, func):
            table_size = len(table)
            prange_chunks = parallel_chunks(table_size)
            n_chunks = len(prange_chunks)

            res = df_alloc_column_of_dtype(res_dtype, table_size)
            for j in numba.prange(n_chunks):
                res_storage = alloc_res_storage(col_dtype)
                for i in range(prange_chunks[j].start, prange_chunks[j].stop):
                    arrow_reader_get_table_cell(table, col_idx_literal_val, i, res_storage)
                    cell_val = read_from_storage(col_dtype, res_storage)
                    res[i] = func(cell_val)
                # no release_res_storage as meminfo managed objects are freed automatically

            return res

        return apply_converter_to_column_impl

    # if func (i.e. converter) was not provided this column should not be converted, so return None
    elif isinstance(func, (types.NoneType, types.Omitted) or func is None):
        def apply_converter_to_column_impl(table, col_idx, func):
            return None
        return apply_converter_to_column_impl

    else:
        assert False, f"No implementation of apply_converter_to_column with func of type: {func}"


def apply_converters(table, converters):
    pass


@overload(apply_converters, prefer_literal=True, jit_options={'parallel': True})
def apply_converters_ovld(table, converters):
    if not isinstance(table, ArrowTableType):
        return None
    if not isinstance(converters, (types.LiteralStrKeyDict, types.NoneType)):
        return None

    n_cols = len(table.dtypes)
    keys_tuple = table.names
    converted_fields = set() if not isinstance(converters, types.LiteralStrKeyDict) else set(converters.fields)
    has_converter = tuple(True if col_name in converted_fields else False for col_name in keys_tuple)

    func_impl_name = 'apply_converters_impl'
    indent = '  '
    res_col_names = ', '.join([f'res_col_{i}' for i in range(n_cols)])

    def get_block(i):
        converter_val = "None" if not has_converter[i] else f'converters[\'{keys_tuple[i]}\']'
        return '\n'.join([
            f'{indent}res_col_{i} = apply_converter_to_column(table, {i}, {converter_val})',
        ])
    all_blocks = '\n'.join([get_block(i) for i in range(n_cols)])
    func_lines = [
        f'def {func_impl_name}(table, converters):',
        all_blocks,
        f'{indent}return ({res_col_names})'
    ]
    func_text = '\n'.join(func_lines)
    use_globals, use_locals = {'keys_tuple': keys_tuple,
                               'has_converter': has_converter,
                               'apply_converter_to_column': apply_converter_to_column}, {}
    exec(func_text, use_globals, use_locals)
    return use_locals[func_impl_name]


def create_df_from_columns(names, columns):
    pass


@overload(create_df_from_columns, prefer_literal=True)
def create_df_from_columns_ovld(names, columns):

    n_cols = len(names)
    func_impl_name = 'create_df_from_columns_impl'
    indent = '  '
    all_columns = '\n'.join([f'{indent}{indent}names[{i}]: columns[{i}],' for i in range(n_cols)])
    func_lines = [
        f'def {func_impl_name}(names, columns):',
        f'{indent}res = pd.DataFrame({{',
        all_columns,
        f'{indent}}})',
        f'{indent}return res'
    ]
    func_text = '\n'.join(func_lines)
    use_globals, use_locals = {'pd': pd, }, {}
    exec(func_text, use_globals, use_locals)
    return use_locals[func_impl_name]


@intrinsic
def combine_df_columns(typingctx, cols1, cols2):

    init_cols1, init_cols2 = cols1, cols2
    if (isinstance(cols1, types.NoneType) and isinstance(cols2, types.NoneType)
        or not (isinstance(cols1, (types.NoneType, types.BaseAnonymousTuple))
                and isinstance(cols2, (types.NoneType, types.BaseAnonymousTuple)))):
        return

    if not (isinstance(cols1, types.NoneType) or isinstance(cols2, types.NoneType)):
        assert len(cols1) == len(cols2), \
               f"Combined tuples must have the same length: given len1={len(cols1)}, len2={len(cols2)}"

    if isinstance(cols1, types.NoneType):
        cols1 = types.Tuple.from_types([types.none for _ in range(len(cols2))])
    if isinstance(cols2, types.NoneType):
        cols2 = types.Tuple.from_types([types.none for _ in range(len(cols1))])

    n_cols = len(cols1)
    cols1_none_mask = np.asarray(list(map(lambda x: isinstance(x, types.NoneType), cols1)))
    cols2_none_mask = np.asarray(list(map(lambda x: isinstance(x, types.NoneType), cols2)))
    assert np.all(cols1_none_mask ^ cols2_none_mask), \
           "Ambigous arguments during elementwise combine (only one should be not None), " \
           f"given cols1={cols1}, cols2={cols2}"

    take_first_element = [not isinstance(x, types.NoneType) for x in cols1]
    ret_type = types.Tuple.from_types([cols1[i] if take_first_element[i] else cols2[i] for i in range(n_cols)])

    def codegen(context, builder, sig, args):
        cols1_val, cols2_val = args
        if isinstance(init_cols1, types.NoneType):
            return impl_ret_borrowed(context, builder, init_cols2, cols2_val)
        elif isinstance(init_cols2, types.NoneType):
            return impl_ret_borrowed(context, builder, init_cols1, cols1_val)
        else:
            res_values = []
            for i in range(n_cols):
                selected_tuple = cols1_val if take_first_element[i] else cols2_val
                val = builder.extract_value(selected_tuple, i)
                if context.enable_nrt:
                    context.nrt.incref(builder, ret_type[i], val)
                res_values.append(val)

            res = context.make_tuple(builder, ret_type, res_values)
            return res

    return ret_type(init_cols1, init_cols2), codegen

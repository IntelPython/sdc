# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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
import numba
import sdc
from numba import types
from numba.core import cgutils
from numba.np.arrayobj import make_array
from numba.extending import overload, intrinsic, overload_method
from sdc.str_ext import string_type

from numba.core.ir_utils import (compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard)

_get_file_size = types.ExternalFunction("get_file_size", types.int64(types.voidptr))
_file_read = types.ExternalFunction("file_read", types.void(types.voidptr, types.voidptr, types.intp))
_file_read_parallel = types.ExternalFunction(
    "file_read_parallel", types.void(
        types.voidptr, types.voidptr, types.intp, types.intp))

file_write = types.ExternalFunction("file_write", types.void(types.voidptr, types.voidptr, types.intp))

_file_write_parallel = types.ExternalFunction(
    "file_write_parallel",
    types.void(
        types.voidptr,
        types.voidptr,
        types.intp,
        types.intp,
        types.intp))


def _handle_np_fromfile(assign, lhs, rhs):
    """translate np.fromfile() to native
    """
    # TODO: dtype in kws
    if len(rhs.args) != 2:  # pragma: no cover
        raise ValueError(
            "np.fromfile(): file name and dtype expected")

    # FIXME: import here since hio has hdf5 which might not be available
    from .. import hio
    from .. import transport_seq as transport

    import llvmlite.binding as ll
    ll.add_symbol('get_file_size', transport.get_file_size)
    ll.add_symbol('file_read', hio.file_read)
    ll.add_symbol('file_read_parallel', transport.file_read_parallel)
    _fname = rhs.args[0]
    _dtype = rhs.args[1]

    def fromfile_impl(fname, dtype):
        size = get_file_size(fname)
        dtype_size = get_dtype_size(dtype)
        A = np.empty(size // dtype_size, dtype=dtype)
        file_read(fname, A, size)
        read_arr = A

    f_block = compile_to_numba_ir(fromfile_impl,
                                  {'np': np,
                                   'get_file_size': get_file_size,
                                   'file_read': file_read,
                                   'get_dtype_size': get_dtype_size}).blocks.popitem()[1]
    replace_arg_nodes(f_block, [_fname, _dtype])
    nodes = f_block.body[:-3]  # remove none return
    nodes[-1].target = lhs
    return nodes


@intrinsic
def get_dtype_size(typingctx, dtype=None):
    assert isinstance(dtype, types.DTypeSpec)

    def codegen(context, builder, sig, args):
        num_bytes = context.get_abi_sizeof(context.get_data_type(dtype.dtype))
        return context.get_constant(types.intp, num_bytes)
    return types.intp(dtype), codegen


@overload_method(types.Array, 'tofile')
def tofile_overload(arr, fname):
    # FIXME: import here since hio has hdf5 which might not be available
    from .. import hio
    from .. import transport_seq as transport

    import llvmlite.binding as ll
    ll.add_symbol('file_write', hio.file_write)
    ll.add_symbol('file_write_parallel', transport.file_write_parallel)
    # TODO: fix Numba to convert literal
    if fname == string_type or isinstance(fname, types.StringLiteral):
        def tofile_impl(arr, fname):
            A = np.ascontiguousarray(arr)
            dtype_size = get_dtype_size(A.dtype)
            file_write(fname._data, A.ctypes, dtype_size * A.size)

        return tofile_impl

# from llvmlite import ir as lir
# @intrinsic
# def print_array_ptr(typingctx, arr_ty):
#     assert isinstance(arr_ty, types.Array)
#     def codegen(context, builder, sig, args):
#         out = make_array(sig.args[0])(context, builder, args[0])
#         cgutils.printf(builder, "%p ", out.data)
#         cgutils.printf(builder, "%lf ", builder.bitcast(out.data, lir.IntType(64).as_pointer()))
#         return context.get_dummy_value()
#     return types.void(arr_ty), codegen


def file_write_parallel(fname, arr, start, count):
    pass

# TODO: fix A.ctype inlined case
@overload(file_write_parallel)
def file_write_parallel_overload(fname, arr, start, count):
    if fname == string_type:  # avoid str literal
        def _impl(fname, arr, start, count):
            A = np.ascontiguousarray(arr)
            dtype_size = get_dtype_size(A.dtype)
            elem_size = dtype_size * sdc.distributed_lower.get_tuple_prod(A.shape[1:])
            # sdc.cprint(start, count, elem_size)
            _file_write_parallel(fname._data, A.ctypes, start, count, elem_size)
        return _impl


def file_read_parallel(fname, arr, start, count):
    return


@overload(file_read_parallel)
def file_read_parallel_overload(fname, arr, start, count):
    if fname == string_type:
        def _impl(fname, arr, start, count):
            dtype_size = get_dtype_size(arr.dtype)
            _file_read_parallel(fname._data, arr.ctypes, start * dtype_size, count * dtype_size)
        return _impl


def file_read(fname, arr, size):
    return


@overload(file_read)
def file_read_overload(fname, arr, size):
    if fname == string_type:
        return lambda fname, arr, size: _file_read(fname._data, arr.ctypes, size)


def get_file_size(fname):
    return 0


@overload(get_file_size)
def get_file_size_overload(fname):
    if fname == string_type:
        return lambda fname: _get_file_size(fname._data)

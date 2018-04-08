import numpy as np
from numba import types
from numba.extending import overload, intrinsic, overload_method
from hpat.str_ext import string_type

from numba.ir_utils import (compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard)

get_file_size = types.ExternalFunction("get_file_size", types.int64(string_type))
file_read = types.ExternalFunction("file_read",
                types.void(string_type, types.voidptr, types.intp))
file_read_parallel = types.ExternalFunction("file_read_parallel",
                types.void(string_type, types.voidptr, types.intp, types.intp))

file_write = types.ExternalFunction("file_write",
                types.void(string_type, types.voidptr, types.intp))

# @overload(np.fromfile)
# def fromfile_overload(fname_t, dtype_t):
#     if fname_t != string_type:
#         raise("np.fromfile() invalid filename type")
#     if dtype_t is not None and not isinstance(dtype_t, types.DTypeSpec):
#         raise("np.fromfile() invalid dtype")
#
#     # FIXME: import here since hio has hdf5 which might not be available
#     import hio
#     import llvmlite.binding as ll
#     ll.add_symbol('get_file_size', hio.get_file_size)
#     ll.add_symbol('file_read', hio.file_read)
#
#     def fromfile_impl(fname, dtype):
#         size = get_file_size(fname)
#         dtype_size = get_dtype_size(dtype)
#         A = np.empty(size//dtype_size, dtype=dtype)
#         file_read(fname, A.ctypes, size)
#         return A
#
#     return fromfile_impl

def _handle_np_fromfile(assign, lhs, rhs):
    """translate np.fromfile() to native
    """
    # TODO: dtype in kws
    if len(rhs.args) != 2:  # pragma: no cover
        raise ValueError(
            "np.fromfile(): file name and dtype expected")

    # FIXME: import here since hio has hdf5 which might not be available
    import hio
    import llvmlite.binding as ll
    ll.add_symbol('get_file_size', hio.get_file_size)
    ll.add_symbol('file_read', hio.file_read)
    ll.add_symbol('file_read_parallel', hio.file_read_parallel)
    _fname = rhs.args[0]
    _dtype = rhs.args[1]

    def fromfile_impl(fname, dtype):
        size = get_file_size(fname)
        dtype_size = get_dtype_size(dtype)
        A = np.empty(size//dtype_size, dtype=dtype)
        file_read(fname, A.ctypes, size)
        read_arr = A

    f_block = compile_to_numba_ir(
        fromfile_impl, {'np': np, 'get_file_size': get_file_size,
        'file_read': file_read, 'get_dtype_size': get_dtype_size}).blocks.popitem()[1]
    replace_arg_nodes(f_block, [_fname, _dtype])
    nodes = f_block.body[:-3]  # remove none return
    nodes[-1].target = lhs
    return nodes


@intrinsic
def get_dtype_size(typingctx, dtype):
    assert isinstance(dtype, types.DTypeSpec)
    def codegen(context, builder, sig, args):
        num_bytes = context.get_abi_sizeof(context.get_data_type(dtype.dtype))
        return context.get_constant(types.intp, num_bytes)
    return types.intp(dtype), codegen

@overload_method(types.Array, 'tofile')
def tofile_overload(arr_ty, fname_ty):
    # FIXME: import here since hio has hdf5 which might not be available
    import hio
    import llvmlite.binding as ll
    ll.add_symbol('file_write', hio.file_write)
    if fname_ty == string_type:
        def tofile_impl(arr, fname):
            A = np.ascontiguousarray(arr)
            dtype_size = get_dtype_size(A.dtype)
            file_write(fname, A.ctypes, dtype_size * A.size)

        return tofile_impl

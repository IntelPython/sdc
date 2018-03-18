import numpy as np
from numba import types
from numba.extending import overload, intrinsic
from hpat.str_ext import string_type

get_file_size = types.ExternalFunction("get_file_size", types.int64(string_type))

@overload(np.fromfile)
def fromfile_overload(fname_t, dtype_t):
    if fname_t != string_type:
        raise("np.fromfile() invalid filename type")
    if dtype_t is not None and not isinstance(dtype_t, types.DTypeSpec):
        raise("np.fromfile() invalid dtype")

    # FIXME: import here since hio has hdf5 which might not be available
    import hio
    import llvmlite.binding as ll
    ll.add_symbol('get_file_size', hio.get_file_size)
    ll.add_symbol('file_read', hio.file_read)

    file_read = types.ExternalFunction("file_read",
                    types.void(string_type, types.CPointer(dtype_t.dtype), types.intp))

    def fromfile_impl(fname, dtype):
        size = get_file_size(fname)
        dtype_size = get_dtype_size(dtype)
        A = np.empty(size//dtype_size, dtype=dtype)
        file_read(fname, A.ctypes, size)
        return A

    return fromfile_impl


@intrinsic
def get_dtype_size(typingctx, dtype):
    assert isinstance(dtype, types.DTypeSpec)
    def codegen(context, builder, sig, args):
        num_bytes = context.get_abi_sizeof(context.get_data_type(dtype.dtype))
        return context.get_constant(types.intp, num_bytes)
    return types.intp(dtype), codegen

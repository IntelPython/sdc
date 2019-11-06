# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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


import operator
from numba import types, cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
import hpat.io
from hpat.io import pio_api
from hpat.utils import _numba_to_c_type_map
from hpat.io.pio_api import (h5file_type, h5dataset_or_group_type, h5dataset_type, h5group_type)
from hpat.str_ext import string_type, gen_get_unicode_chars, gen_std_str_to_unicode

from llvmlite import ir as lir
import llvmlite.binding as ll
import hpat.io
if hpat.config._has_h5py:
    import h5py
    from hpat.io import _hdf5
    ll.add_symbol('hpat_h5_open', _hdf5.hpat_h5_open)
    ll.add_symbol('hpat_h5_open_dset_or_group_obj', _hdf5.hpat_h5_open_dset_or_group_obj)
    ll.add_symbol('hpat_h5_size', _hdf5.hpat_h5_size)
    ll.add_symbol('hpat_h5_read', _hdf5.hpat_h5_read)
    ll.add_symbol('hpat_h5_get_type_enum', _hdf5.hpat_h5_get_type_enum)
    ll.add_symbol('hpat_h5_create_dset', _hdf5.hpat_h5_create_dset)
    ll.add_symbol('hpat_h5_create_group', _hdf5.hpat_h5_create_group)
    ll.add_symbol('hpat_h5_write', _hdf5.hpat_h5_write)
    ll.add_symbol('hpat_h5_close', _hdf5.hpat_h5_close)
    ll.add_symbol('h5g_get_num_objs', _hdf5.h5g_get_num_objs)
    ll.add_symbol('h5g_get_objname_by_idx', _hdf5.h5g_get_objname_by_idx)
    ll.add_symbol('h5g_close', _hdf5.hpat_h5g_close)

h5file_lir_type = lir.IntType(64)

if hpat.config._has_h5py:
    # hid_t is 32bit in 1.8 but 64bit in 1.10
    if h5py.version.hdf5_version_tuple[1] == 8:
        h5file_lir_type = lir.IntType(32)
    else:
        assert h5py.version.hdf5_version_tuple[1] == 10

h5g_close = types.ExternalFunction("h5g_close", types.none(h5group_type))


@lower_builtin(operator.getitem, h5file_type, string_type)
@lower_builtin(operator.getitem, h5dataset_or_group_type, string_type)
def h5_open_dset_lower(context, builder, sig, args):
    fg_id, dset_name = args
    dset_name = gen_get_unicode_chars(context, builder, dset_name)

    fnty = lir.FunctionType(h5file_lir_type, [h5file_lir_type, lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_open_dset_or_group_obj")
    return builder.call(fn, [fg_id, dset_name])


if hpat.config._has_h5py:
    @lower_builtin(h5py.File, string_type, string_type)
    @lower_builtin(h5py.File, string_type, string_type, types.int64)
    def h5_open(context, builder, sig, args):
        fname = args[0]
        mode = args[1]
        fname = gen_get_unicode_chars(context, builder, fname)
        mode = gen_get_unicode_chars(context, builder, mode)

        is_parallel = context.get_constant(types.int64, 0) if len(args) < 3 else args[2]
        fnty = lir.FunctionType(h5file_lir_type, [lir.IntType(
            8).as_pointer(), lir.IntType(8).as_pointer(), lir.IntType(64)])
        fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_open")
        return builder.call(fn, [fname, mode, is_parallel])


@lower_builtin(pio_api.h5size, h5dataset_or_group_type, types.int32)
def h5_size(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(
        64), [h5file_lir_type, lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_size")
    return builder.call(fn, [args[0], args[1]])


@lower_builtin(pio_api.h5read, h5dataset_or_group_type, types.int32,
               types.UniTuple, types.UniTuple, types.int64,
               types.npytypes.Array)
def h5_read(context, builder, sig, args):
    # extra last arg type for type enum
    arg_typs = [h5file_lir_type, lir.IntType(32),
                lir.IntType(64).as_pointer(), lir.IntType(64).as_pointer(),
                lir.IntType(64), lir.IntType(8).as_pointer(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_read")
    out = make_array(sig.args[5])(context, builder, args[5])
    # store size vars array struct to pointer
    count_ptr = cgutils.alloca_once(builder, args[2].type)
    builder.store(args[2], count_ptr)
    size_ptr = cgutils.alloca_once(builder, args[3].type)
    builder.store(args[3], size_ptr)
    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[5].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum))
    call_args = [args[0], args[1],
                 builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
                 builder.bitcast(size_ptr, lir.IntType(
                     64).as_pointer()), args[4],
                 builder.bitcast(out.data, lir.IntType(8).as_pointer()),
                 builder.load(typ_arg)]

    return builder.call(fn, call_args)


@lower_builtin(pio_api.h5close, h5file_type)
@lower_builtin("h5file.close", h5file_type)
def h5_close(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [h5file_lir_type])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_close")
    builder.call(fn, args)
    return context.get_dummy_value()


@lower_builtin("h5group.create_dataset", h5group_type, string_type, types.UniTuple, string_type)
@lower_builtin("h5file.create_dataset", h5file_type, string_type, types.UniTuple, string_type)
@lower_builtin(pio_api.h5create_dset, h5file_type, string_type, types.UniTuple, string_type)
def h5_create_dset(context, builder, sig, args):
    fg_id, dset_name, counts, dtype_str = args

    dset_name = gen_get_unicode_chars(context, builder, dset_name)
    dtype_str = gen_get_unicode_chars(context, builder, dtype_str)

    # extra last arg type for type enum
    arg_typs = [h5file_lir_type, lir.IntType(8).as_pointer(), lir.IntType(32),
                lir.IntType(64).as_pointer(),
                lir.IntType(32)]
    fnty = lir.FunctionType(h5file_lir_type, arg_typs)

    fn = builder.module.get_or_insert_function(
        fnty, name="hpat_h5_create_dset")

    ndims = sig.args[2].count
    ndims_arg = lir.Constant(lir.IntType(32), ndims)

    # store size vars array struct to pointer
    count_ptr = cgutils.alloca_once(builder, counts.type)
    builder.store(counts, count_ptr)

    t_fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    t_fn = builder.module.get_or_insert_function(
        t_fnty, name="hpat_h5_get_type_enum")
    typ_arg = builder.call(t_fn, [dtype_str])

    call_args = [fg_id, dset_name, ndims_arg,
                 builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
                 typ_arg]

    return builder.call(fn, call_args)


@lower_builtin("h5group.create_group", h5group_type, string_type)
@lower_builtin("h5file.create_group", h5file_type, string_type)
@lower_builtin(pio_api.h5create_group, h5file_type, string_type)
def h5_create_group(context, builder, sig, args):
    fg_id, gname = args
    gname = gen_get_unicode_chars(context, builder, gname)

    fnty = lir.FunctionType(h5file_lir_type,
                            [h5file_lir_type, lir.IntType(8).as_pointer()])

    fn = builder.module.get_or_insert_function(
        fnty, name="hpat_h5_create_group")
    return builder.call(fn, [fg_id, gname])

# _h5_str_typ_table = {
#     'i1':0,
#     'u1':1,
#     'i4':2,
#     'i8':3,
#     'f4':4,
#     'f8':5
#     }


@lower_builtin(pio_api.h5write, h5dataset_type, types.int32,
               types.UniTuple, types.UniTuple, types.int64, types.Array)
def h5_write(context, builder, sig, args):
    # extra last arg type for type enum
    arg_typs = [h5file_lir_type, lir.IntType(32),
                lir.IntType(64).as_pointer(), lir.IntType(64).as_pointer(),
                lir.IntType(64), lir.IntType(8).as_pointer(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_write")
    out = make_array(sig.args[5])(context, builder, args[5])
    # store size vars array struct to pointer
    count_ptr = cgutils.alloca_once(builder, args[2].type)
    builder.store(args[2], count_ptr)
    size_ptr = cgutils.alloca_once(builder, args[3].type)
    builder.store(args[3], size_ptr)
    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[5].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum))
    call_args = [args[0], args[1],
                 builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
                 builder.bitcast(size_ptr, lir.IntType(
                     64).as_pointer()), args[4],
                 builder.bitcast(out.data, lir.IntType(8).as_pointer()),
                 builder.load(typ_arg)]

    return builder.call(fn, call_args)


@lower_builtin("h5file.keys", h5file_type)
@lower_builtin("h5group.keys", h5dataset_or_group_type)
def lower_dict_get(context, builder, sig, args):
    def h5f_keys_imp(file_id):
        obj_name_list = []
        nobjs = pio_api.h5g_get_num_objs(file_id)
        for i in range(nobjs):
            obj_name = pio_api.h5g_get_objname_by_idx(file_id, i)
            obj_name_list.append(obj_name)
        return obj_name_list

    res = context.compile_internal(builder, h5f_keys_imp, sig, args)
    return res


@lower_builtin(pio_api.h5g_get_num_objs, h5file_type)
@lower_builtin(pio_api.h5g_get_num_objs, h5dataset_or_group_type)
def h5g_get_num_objs_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64),
                            [h5file_lir_type])
    fn = builder.module.get_or_insert_function(fnty, name="h5g_get_num_objs")
    return builder.call(fn, args)


@lower_builtin(pio_api.h5g_get_objname_by_idx, h5file_type, types.int64)
@lower_builtin(pio_api.h5g_get_objname_by_idx, h5dataset_or_group_type, types.int64)
def h5g_get_objname_by_idx_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [h5file_lir_type, lir.IntType(64)])
    fn = builder.module.get_or_insert_function(
        fnty, name="h5g_get_objname_by_idx")
    res = builder.call(fn, args)
    res = gen_std_str_to_unicode(context, builder, res, True)
    return res

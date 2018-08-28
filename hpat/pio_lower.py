from numba import types, cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
from hpat import pio_api
from hpat.utils import _numba_to_c_type_map
from hpat.pio_api import h5file_type, h5dataset_or_group_type
from hpat.str_ext import StringType
import h5py
from llvmlite import ir as lir
import hio
import llvmlite.binding as ll
ll.add_symbol('hpat_h5_open', hio.hpat_h5_open)
ll.add_symbol('hpat_h5_open_dset_or_group_obj', hio.hpat_h5_open_dset_or_group_obj)
ll.add_symbol('hpat_h5_size', hio.hpat_h5_size)
ll.add_symbol('hpat_h5_read', hio.hpat_h5_read)
ll.add_symbol('hpat_h5_get_type_enum', hio.hpat_h5_get_type_enum)
ll.add_symbol('hpat_h5_create_dset', hio.hpat_h5_create_dset)
ll.add_symbol('hpat_h5_create_group', hio.hpat_h5_create_group)
ll.add_symbol('hpat_h5_write', hio.hpat_h5_write)
ll.add_symbol('hpat_h5_close', hio.hpat_h5_close)
ll.add_symbol('h5g_get_num_objs', hio.h5g_get_num_objs)
ll.add_symbol('h5g_get_objname_by_idx', hio.h5g_get_objname_by_idx)

h5file_lir_type = lir.IntType(64)

# hid_t is 32bit in 1.8 but 64bit in 1.10
if h5py.version.hdf5_version_tuple[1] == 8:
    h5file_lir_type = lir.IntType(32)
else:
    assert h5py.version.hdf5_version_tuple[1] == 10


@lower_builtin("getitem", h5file_type, StringType)
def h5_open_dset_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="get_c_str")
    val1 = builder.call(fn, [args[1]])

    fnty = lir.FunctionType(h5file_lir_type, [h5file_lir_type, lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_open_dset_or_group_obj")
    return builder.call(fn, [args[0], val1])


@lower_builtin(h5py.File, StringType, StringType)
@lower_builtin(h5py.File, StringType, StringType, types.int64)
def h5_open(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="get_c_str")
    val1 = builder.call(fn, [args[0]])
    val2 = builder.call(fn, [args[1]])
    is_parallel = context.get_constant(types.int64, 0) if len(args) < 3 else args[2]
    fnty = lir.FunctionType(h5file_lir_type, [lir.IntType(
        8).as_pointer(), lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_open")
    return builder.call(fn, [val1, val2, is_parallel])


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


@lower_builtin(pio_api.h5create_dset, h5file_type, StringType,
               types.containers.UniTuple, StringType)
def h5_create_dset(context, builder, sig, args):
    # insert the dset_name string arg
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="get_c_str")
    val2 = builder.call(fn, [args[1]])

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
    count_ptr = cgutils.alloca_once(builder, args[2].type)
    builder.store(args[2], count_ptr)

    t_fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    t_fn = builder.module.get_or_insert_function(
        t_fnty, name="hpat_h5_get_type_enum")
    typ_arg = builder.call(t_fn, [args[3]])

    call_args = [args[0], val2, ndims_arg,
                 builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
                 typ_arg]

    return builder.call(fn, call_args)


@lower_builtin(pio_api.h5create_group, h5file_type, StringType)
def h5_create_group(context, builder, sig, args):
    # insert the group_name string arg
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="get_c_str")
    val2 = builder.call(fn, [args[1]])

    fnty = lir.FunctionType(h5file_lir_type,
                            [h5file_lir_type, lir.IntType(8).as_pointer()])

    fn = builder.module.get_or_insert_function(
        fnty, name="hpat_h5_create_group")
    return builder.call(fn, [args[0], val2])

# _h5_str_typ_table = {
#     'i1':0,
#     'u1':1,
#     'i4':2,
#     'i8':3,
#     'f4':4,
#     'f8':5
#     }


@lower_builtin(pio_api.h5write, h5file_type, h5file_type, types.int32,
               types.containers.UniTuple, types.containers.UniTuple, types.int64,
               types.npytypes.Array)
def h5_write(context, builder, sig, args):
    # extra last arg type for type enum
    arg_typs = [h5file_lir_type, h5file_lir_type, lir.IntType(32),
                lir.IntType(64).as_pointer(), lir.IntType(64).as_pointer(),
                lir.IntType(64), lir.IntType(8).as_pointer(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_write")
    out = make_array(sig.args[6])(context, builder, args[6])
    # store size vars array struct to pointer
    count_ptr = cgutils.alloca_once(builder, args[3].type)
    builder.store(args[3], count_ptr)
    size_ptr = cgutils.alloca_once(builder, args[4].type)
    builder.store(args[4], size_ptr)
    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[6].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum))
    call_args = [args[0], args[1], args[2],
                 builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
                 builder.bitcast(size_ptr, lir.IntType(
                     64).as_pointer()), args[5],
                 builder.bitcast(out.data, lir.IntType(8).as_pointer()),
                 builder.load(typ_arg)]

    return builder.call(fn, call_args)


@lower_builtin("h5file.keys", pio_api.h5file_type)
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
def h5g_get_num_objs_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64),
                            [h5file_lir_type])
    fn = builder.module.get_or_insert_function(fnty, name="h5g_get_num_objs")
    return builder.call(fn, args)


@lower_builtin(pio_api.h5g_get_objname_by_idx, h5file_type, types.int64)
def h5g_get_objname_by_idx_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [h5file_lir_type, lir.IntType(64)])
    fn = builder.module.get_or_insert_function(
        fnty, name="h5g_get_objname_by_idx")
    return builder.call(fn, args)

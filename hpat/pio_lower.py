from numba import types, cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
from hpat import pio_api
import h5py
from llvmlite import ir as lir
import hio
import llvmlite.binding as ll
ll.add_symbol('hpat_h5_open', hio.hpat_h5_open)
ll.add_symbol('hpat_h5_size', hio.hpat_h5_size)
ll.add_symbol('hpat_h5_read', hio.hpat_h5_read)

#@lower_builtin(h5py.File, types.string, types.string)
#@lower_builtin(h5py.File, types.string, types.Const)
#@lower_builtin(h5py.File, types.Const, types.string)
@lower_builtin(h5py.File, types.Const, types.Const, types.int64)
def h5_open(context, builder, sig, args):
    # works for constant strings only
    # TODO: extend to string variables
    arg1, arg2, _ = sig.args
    val1 = context.insert_const_string(builder.module, arg1.value)
    val2 = context.insert_const_string(builder.module, arg2.value)
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_open")
    return builder.call(fn, [val1, val2, args[2]])

@lower_builtin(pio_api.h5size, types.int32, types.Const, types.int32)
def h5_size(context, builder, sig, args):
    # works for constant string only
    # TODO: extend to string variables
    arg1, arg2, args3 = sig.args
    val2 = context.insert_const_string(builder.module, arg2.value)
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(32), lir.IntType(8).as_pointer(), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_size")
    return builder.call(fn, [args[0], val2, args[2]])

@lower_builtin(pio_api.h5read, types.int32, types.Const, types.int32,
    types.containers.UniTuple, types.containers.UniTuple, types.int64,
    types.npytypes.Array)
def h5_read(context, builder, sig, args):
    # insert the dset_name string arg
    dset_name_arg = sig.args[1]
    val2 = context.insert_const_string(builder.module, dset_name_arg.value)
    # extra last arg type for type enum
    arg_typs = [lir.IntType(32), lir.IntType(8).as_pointer(), lir.IntType(32),
        lir.IntType(64).as_pointer(), lir.IntType(64).as_pointer(),
        lir.IntType(64), lir.IntType(8).as_pointer(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_read")
    out = make_array(sig.args[6])(context, builder, args[6])
    # store size vars array struct to pointer
    count_ptr = cgutils.alloca_once(builder, args[3].type)
    builder.store(args[3], count_ptr)
    size_ptr = cgutils.alloca_once(builder, args[4].type)
    builder.store(args[4], size_ptr)
    # store an int to specify data type
    typ_enum = _h5_typ_table[sig.args[6].dtype]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))
    call_args = [args[0], val2, args[2],
        builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
        builder.bitcast(size_ptr, lir.IntType(64).as_pointer()), args[5],
        builder.bitcast(out.data, lir.IntType(8).as_pointer()),
        builder.load(typ_arg)]

    return builder.call(fn, call_args)

@lower_builtin(pio_api.h5close, types.int32)
def h5_close(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_close")
    return builder.call(fn, args)

_h5_typ_table = {
    types.int8:0,
    types.uint8:1,
    types.int32:2,
    types.int64:3,
    types.float32:4,
    types.float64:5
    }

@lower_builtin(pio_api.h5create_dset, types.int32, types.Const,
    types.containers.UniTuple, types.Const)
def h5_create_dset(context, builder, sig, args):
    # insert the dset_name string arg
    dset_name_arg = sig.args[1]
    val2 = context.insert_const_string(builder.module, dset_name_arg.value)

    # extra last arg type for type enum
    arg_typs = [lir.IntType(32), lir.IntType(8).as_pointer(), lir.IntType(32),
        lir.IntType(64).as_pointer(),
        lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_create_dset")

    ndims = sig.args[2].count
    ndims_arg = lir.Constant(lir.IntType(32), ndims)

    # store size vars array struct to pointer
    count_ptr = cgutils.alloca_once(builder, args[2].type)
    builder.store(args[2], count_ptr)

    # store an int to specify data type
    typ_enum = _h5_str_typ_table[sig.args[3].value]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))

    call_args = [args[0], val2, ndims_arg,
        builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
        builder.load(typ_arg)]

    return builder.call(fn, call_args)

_h5_str_typ_table = {
    'i1':0,
    'u1':1,
    'i4':2,
    'i8':3,
    'f4':4,
    'f8':5
    }

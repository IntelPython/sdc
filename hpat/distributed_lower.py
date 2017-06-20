from numba import types, cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
import hpat
from hpat import distributed_api
import time
from llvmlite import ir as lir
import hdist
import llvmlite.binding as ll
ll.add_symbol('hpat_dist_get_rank', hdist.hpat_dist_get_rank)
ll.add_symbol('hpat_dist_get_size', hdist.hpat_dist_get_size)
ll.add_symbol('hpat_dist_get_end', hdist.hpat_dist_get_end)
ll.add_symbol('hpat_dist_get_node_portion', hdist.hpat_dist_get_node_portion)
ll.add_symbol('hpat_dist_get_time', hdist.hpat_dist_get_time)
ll.add_symbol('hpat_dist_reduce_i4', hdist.hpat_dist_reduce_i4)
ll.add_symbol('hpat_dist_reduce_i8', hdist.hpat_dist_reduce_i8)
ll.add_symbol('hpat_dist_reduce_f4', hdist.hpat_dist_reduce_f4)
ll.add_symbol('hpat_dist_reduce_f8', hdist.hpat_dist_reduce_f8)
ll.add_symbol('hpat_dist_arr_reduce', hdist.hpat_dist_arr_reduce)

@lower_builtin(distributed_api.get_rank)
def dist_get_rank(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_rank")
    return builder.call(fn, [])

@lower_builtin(distributed_api.get_size)
def dist_get_size(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_size")
    return builder.call(fn, [])

@lower_builtin(distributed_api.get_end, types.int64, types.int64, types.int32, types.int32)
def dist_get_end(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64), lir.IntType(64),
                                            lir.IntType(32), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_end")
    return builder.call(fn, [args[0], args[1], args[2], args[3]])

@lower_builtin(distributed_api.get_node_portion, types.int64, types.int64, types.int32, types.int32)
def dist_get_portion(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64), lir.IntType(64),
                                            lir.IntType(32), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_node_portion")
    return builder.call(fn, [args[0], args[1], args[2], args[3]])

@lower_builtin(distributed_api.dist_reduce, types.int64)
@lower_builtin(distributed_api.dist_reduce, types.int32)
@lower_builtin(distributed_api.dist_reduce, types.float32)
@lower_builtin(distributed_api.dist_reduce, types.float64)
def lower_dist_reduce(context, builder, sig, args):
    ltyp = args[0].type
    fnty = lir.FunctionType(ltyp, [ltyp])
    typ_map = {types.int32:"i4", types.int64:"i8", types.float32:"f4", types.float64:"f8"}
    typ_str = typ_map[sig.args[0]]
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_reduce_{}".format(typ_str))
    return builder.call(fn, [args[0]])

@lower_builtin(distributed_api.dist_arr_reduce, types.npytypes.Array)
def lower_dist_arr_reduce(context, builder, sig, args):
    # store an int to specify data type
    typ_enum = hpat.pio_lower._h5_typ_table[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))
    ndims = sig.args[0].ndim

    out = make_array(sig.args[0])(context, builder, args[0])
    # store size vars array struct to pointer
    size_ptr = cgutils.alloca_once(builder, out.shape.type)
    builder.store(out.shape, size_ptr)
    size_arg = builder.bitcast(size_ptr, lir.IntType(64).as_pointer())

    ndim_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), sig.args[0].ndim))
    call_args = [builder.bitcast(out.data, lir.IntType(8).as_pointer()),
                size_arg, builder.load(ndim_arg), builder.load(typ_arg)]

    # array, shape, ndim, extra last arg type for type enum
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64).as_pointer(),
        lir.IntType(32), lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_arr_reduce")
    return builder.call(fn, call_args)

@lower_builtin(time.time)
def dist_get_time(context, builder, sig, args):
    fnty = lir.FunctionType(lir.DoubleType(), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_time")
    return builder.call(fn, [])

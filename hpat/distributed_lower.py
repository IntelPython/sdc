from numba import types, cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
import numba.targets.arrayobj
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed
from numba.typing.builtins import IndexValueType
import numpy as np
import hpat
from hpat import distributed_api
import time
from llvmlite import ir as lir
import hdist
import llvmlite.binding as ll
ll.add_symbol('hpat_dist_get_rank', hdist.hpat_dist_get_rank)
ll.add_symbol('hpat_dist_get_size', hdist.hpat_dist_get_size)
ll.add_symbol('hpat_dist_get_start', hdist.hpat_dist_get_start)
ll.add_symbol('hpat_dist_get_end', hdist.hpat_dist_get_end)
ll.add_symbol('hpat_dist_get_node_portion', hdist.hpat_dist_get_node_portion)
ll.add_symbol('hpat_dist_get_time', hdist.hpat_dist_get_time)
ll.add_symbol('hpat_get_time', hdist.hpat_get_time)
ll.add_symbol('hpat_barrier', hdist.hpat_barrier)
ll.add_symbol('hpat_dist_reduce', hdist.hpat_dist_reduce)
ll.add_symbol('hpat_dist_arr_reduce', hdist.hpat_dist_arr_reduce)
ll.add_symbol('hpat_dist_exscan_i4', hdist.hpat_dist_exscan_i4)
ll.add_symbol('hpat_dist_exscan_i8', hdist.hpat_dist_exscan_i8)
ll.add_symbol('hpat_dist_exscan_f4', hdist.hpat_dist_exscan_f4)
ll.add_symbol('hpat_dist_exscan_f8', hdist.hpat_dist_exscan_f8)
ll.add_symbol('hpat_dist_irecv', hdist.hpat_dist_irecv)
ll.add_symbol('hpat_dist_isend', hdist.hpat_dist_isend)
ll.add_symbol('hpat_dist_wait', hdist.hpat_dist_wait)
ll.add_symbol('hpat_dist_get_item_pointer', hdist.hpat_dist_get_item_pointer)
ll.add_symbol('hpat_get_dummy_ptr', hdist.hpat_get_dummy_ptr)


_h5_typ_table = {
    types.int8:0,
    types.uint8:1,
    types.int32:2,
    types.int64:3,
    types.float32:4,
    types.float64:5
    }

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

@lower_builtin(distributed_api.get_start, types.int64, types.int32, types.int32)
def dist_get_start(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64),
                                            lir.IntType(32), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_start")
    return builder.call(fn, [args[0], args[1], args[2]])

@lower_builtin(distributed_api.get_end, types.int64, types.int32, types.int32)
def dist_get_end(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64),
                                            lir.IntType(32), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_end")
    return builder.call(fn, [args[0], args[1], args[2]])

@lower_builtin(distributed_api.get_node_portion, types.int64, types.int32, types.int32)
def dist_get_portion(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64),
                                            lir.IntType(32), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_node_portion")
    return builder.call(fn, [args[0], args[1], args[2]])

@lower_builtin(distributed_api.dist_reduce, types.int64, types.int32)
@lower_builtin(distributed_api.dist_reduce, types.int32, types.int32)
@lower_builtin(distributed_api.dist_reduce, types.float32, types.int32)
@lower_builtin(distributed_api.dist_reduce, types.float64, types.int32)
@lower_builtin(distributed_api.dist_reduce, IndexValueType, types.int32)
def lower_dist_reduce(context, builder, sig, args):
    val_typ = args[0].type
    op_typ = args[1].type

    target_typ = sig.args[0]
    if isinstance(target_typ, IndexValueType):
        target_typ = target_typ.val_typ
        supported_typs = [types.int32, types.float32, types.float64]
        import sys
        if not sys.platform.startswith('win'):
            # long is 4 byte on Windows
            supported_typs.append(types.int64)
        if target_typ not in supported_typs:  # pragma: no cover
            raise TypeError("argmin/argmax not supported for type {}".format(
                                                                    target_typ))

    in_ptr = cgutils.alloca_once(builder, val_typ)
    out_ptr = cgutils.alloca_once(builder, val_typ)
    builder.store(args[0], in_ptr)
    # cast to char *
    in_ptr = builder.bitcast(in_ptr, lir.IntType(8).as_pointer())
    out_ptr = builder.bitcast(out_ptr, lir.IntType(8).as_pointer())

    typ_enum = _h5_typ_table[target_typ]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32),
                                                                    typ_enum))

    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer(),
                        lir.IntType(8).as_pointer(), op_typ, lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_reduce")
    builder.call(fn, [in_ptr, out_ptr, args[1], builder.load(typ_arg)])
    # cast back to value type
    out_ptr = builder.bitcast(out_ptr, val_typ.as_pointer())
    return builder.load(out_ptr)


@lower_builtin(distributed_api.dist_reduce, types.npytypes.Array, types.int32)
def lower_dist_arr_reduce(context, builder, sig, args):

    op_typ = args[1].type

    # store an int to specify data type
    typ_enum = _h5_typ_table[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))
    ndims = sig.args[0].ndim

    out = make_array(sig.args[0])(context, builder, args[0])
    # store size vars array struct to pointer
    size_ptr = cgutils.alloca_once(builder, out.shape.type)
    builder.store(out.shape, size_ptr)
    size_arg = builder.bitcast(size_ptr, lir.IntType(64).as_pointer())

    ndim_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), sig.args[0].ndim))
    call_args = [builder.bitcast(out.data, lir.IntType(8).as_pointer()),
                size_arg, builder.load(ndim_arg), args[1], builder.load(typ_arg)]

    # array, shape, ndim, extra last arg type for type enum
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64).as_pointer(),
        lir.IntType(32), op_typ, lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_arr_reduce")
    builder.call(fn, call_args)
    res = out._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@lower_builtin(time.time)
def dist_get_time(context, builder, sig, args):
    fnty = lir.FunctionType(lir.DoubleType(), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_get_time")
    return builder.call(fn, [])

@lower_builtin(distributed_api.dist_time)
def dist_get_dist_time(context, builder, sig, args):
    fnty = lir.FunctionType(lir.DoubleType(), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_time")
    return builder.call(fn, [])

@lower_builtin(distributed_api.barrier)
def dist_barrier(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_barrier")
    return builder.call(fn, [])

@lower_builtin(distributed_api.dist_cumsum, types.npytypes.Array, types.npytypes.Array)
def lower_dist_cumsum(context, builder, sig, args):

    dtype = sig.args[0].dtype
    zero = dtype(0)

    def cumsum_impl(in_arr, out_arr):  # pragma: no cover
        c = zero
        for v in np.nditer(in_arr):
            c += v.item()
        prefix_var = distributed_api.dist_exscan(c)
        for i in range(in_arr.size):
            prefix_var += in_arr[i]
            out_arr[i] = prefix_var
        return 0

    res = context.compile_internal(builder, cumsum_impl, sig, args,
                                    locals=dict(c=dtype,
                                    prefix_var=dtype))
    return res


@lower_builtin(distributed_api.dist_exscan, types.int64)
@lower_builtin(distributed_api.dist_exscan, types.int32)
@lower_builtin(distributed_api.dist_exscan, types.float32)
@lower_builtin(distributed_api.dist_exscan, types.float64)
def lower_dist_exscan(context, builder, sig, args):
    ltyp = args[0].type
    fnty = lir.FunctionType(ltyp, [ltyp])
    typ_map = {types.int32:"i4", types.int64:"i8", types.float32:"f4", types.float64:"f8"}
    typ_str = typ_map[sig.args[0]]
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_exscan_{}".format(typ_str))
    return builder.call(fn, [args[0]])


@lower_builtin(distributed_api.irecv, types.npytypes.Array, types.int32,
types.int32, types.int32, types.boolean)
def lower_dist_irecv(context, builder, sig, args):
    # store an int to specify data type
    typ_enum = _h5_typ_table[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))

    out = make_array(sig.args[0])(context, builder, args[0])

    call_args = [builder.bitcast(out.data, lir.IntType(8).as_pointer()),
                args[1], builder.load(typ_arg),
                args[2], args[3], args[4]]

    # array, size, extra arg type for type enum
    # pe, tag, cond
    arg_typs = [lir.IntType(8).as_pointer(),
        lir.IntType(32), lir.IntType(32), lir.IntType(32), lir.IntType(32),
        lir.IntType(1)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_irecv")
    return builder.call(fn, call_args)

@lower_builtin(distributed_api.isend, types.npytypes.Array, types.int32,
types.int32, types.int32, types.boolean)
def lower_dist_isend(context, builder, sig, args):
    # store an int to specify data type
    typ_enum = _h5_typ_table[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))

    out = make_array(sig.args[0])(context, builder, args[0])

    call_args = [builder.bitcast(out.data, lir.IntType(8).as_pointer()),
                args[1], builder.load(typ_arg),
                args[2], args[3], args[4]]

    # array, size, extra arg type for type enum
    # pe, tag, cond
    arg_typs = [lir.IntType(8).as_pointer(),
        lir.IntType(32), lir.IntType(32), lir.IntType(32), lir.IntType(32),
        lir.IntType(1)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_isend")
    return builder.call(fn, call_args)

@lower_builtin(distributed_api.wait, types.int32, types.boolean)
def lower_dist_wait(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(32), lir.IntType(1)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_wait")
    return builder.call(fn, args)

# @lower_builtin(distributed_api.dist_setitem, types.Array, types.Any, types.Any,
#     types.intp, types.intp)
# def dist_setitem_array(context, builder, sig, args):
#     """add check for access to be in processor bounds of the array"""
#     # TODO: replace array shape if array is small
#     #  (processor chuncks smaller than setitem range causes index normalization)
#     # remove start and count args to call regular get_item_pointer2
#     count = args.pop()
#     start = args.pop()
#     sig.args = tuple([sig.args[0], sig.args[1], sig.args[2]])
#     regular_get_item_pointer2 = cgutils.get_item_pointer2
#     # add bounds check for distributed access,
#     # return a dummy pointer if out of bounds
#     def dist_get_item_pointer2(builder, data, shape, strides, layout, inds,
#                       wraparound=False):
#         # get local index or -1 if out of bounds
#         fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64), lir.IntType(64), lir.IntType(64)])
#         fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_item_pointer")
#         first_ind = builder.call(fn, [inds[0], start, count])
#         inds = tuple([first_ind, *inds[1:]])
#         # regular local pointer with new indices
#         in_ptr = regular_get_item_pointer2(builder, data, shape, strides, layout, inds, wraparound)
#         ret_ptr = cgutils.alloca_once(builder, in_ptr.type)
#         builder.store(in_ptr, ret_ptr)
#         not_inbound = builder.icmp_signed('==', first_ind, lir.Constant(lir.IntType(64), -1))
#         # get dummy pointer
#         dummy_fnty  = lir.FunctionType(lir.IntType(8).as_pointer(), [])
#         dummy_fn = builder.module.get_or_insert_function(dummy_fnty, name="hpat_get_dummy_ptr")
#         dummy_ptr = builder.bitcast(builder.call(dummy_fn, []), in_ptr.type)
#         with builder.if_then(not_inbound, likely=True):
#             builder.store(dummy_ptr, ret_ptr)
#         return builder.load(ret_ptr)
#
#     # replace inner array access call for setitem generation
#     cgutils.get_item_pointer2 = dist_get_item_pointer2
#     numba.targets.arrayobj.setitem_array(context, builder, sig, args)
#     cgutils.get_item_pointer2 = regular_get_item_pointer2
#     return lir.Constant(lir.IntType(32), 0)

# find overlapping range of an input range (start:stop) and a chunk range
# (chunk_start:chunk_start+chunk_count). Inputs are assumed positive.
# output is set to empty range of local range goes out of bounds
@numba.njit
def _get_local_range(start, stop, chunk_start, chunk_count):  # pragma: no cover
    assert start >= 0 and stop > 0
    new_start = max(start, chunk_start)
    new_stop = min(stop, chunk_start + chunk_count)
    loc_start = new_start - chunk_start
    loc_stop = new_stop - chunk_start
    if loc_start < 0 or loc_stop < 0:
        loc_start = 1
        loc_stop = 0
    return loc_start, loc_stop

@numba.njit
def _set_if_in_range(A, val, index, chunk_start, chunk_count):  # pragma: no cover
    if index >= chunk_start and index < chunk_start+chunk_count:
        A[index] = val

@numba.njit
def _root_rank_select(old_val, new_val):  # pragma: no cover
    if distributed_api.get_rank() == 0:
        return old_val
    return new_val

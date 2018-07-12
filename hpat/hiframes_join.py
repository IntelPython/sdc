from __future__ import print_function, division, absolute_import

import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.extending import overload
from numba.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes)
import hpat
from hpat import distributed, distributed_analysis
from hpat.utils import debug_prints, alloc_arr_tup, empty_like_type
from hpat.distributed_analysis import Distribution
from hpat.hiframes_sort import (
    alloc_shuffle_metadata, data_alloc_shuffle_metadata, alltoallv,
    alltoallv_tup, finalize_shuffle_meta, finalize_data_shuffle_meta,
    update_shuffle_meta, update_data_shuffle_meta,
    )
from hpat.str_arr_ext import (string_array_type, to_string_list,
                              cp_str_list_to_array, str_list_to_array,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, del_str, num_total_chars,
                              getitem_str_offset, copy_str_arr_slice, setitem_string_array)
from hpat.hiframes_api import str_copy
from hpat.timsort import copyElement_tup, getitem_arr_tup
import numpy as np


class Join(ir.Stmt):
    def __init__(self, df_out, left_df, right_df, left_key, right_key, df_vars, loc):
        self.df_out = df_out
        self.left_df = left_df
        self.right_df = right_df
        self.left_key = left_key
        self.right_key = right_key
        self.df_out_vars = df_vars[self.df_out]
        self.left_vars = df_vars[left_df]
        self.right_vars = df_vars[right_df]
        # needs df columns for type inference stage
        self.df_vars = df_vars
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)

        in_cols = ""
        for (c, v) in self.left_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_left_str = "{}{{{}}}".format(self.left_df, in_cols)

        in_cols = ""
        for (c, v) in self.right_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_right_str = "{}{{{}}}".format(self.right_df, in_cols)
        return "join [{}={}]: {} , {}, {}".format(self.left_key,
                                                  self.right_key, df_out_str, df_left_str, df_right_str)


def join_array_analysis(join_node, equiv_set, typemap, array_analysis):
    post = []
    # empty join nodes should be deleted in remove dead
    assert len(join_node.df_out_vars) > 0, "empty join in array analysis"

    # arrays of left_df and right_df have same size in first dimension
    all_shapes = []
    for _, col_var in (list(join_node.left_vars.items())
                       + list(join_node.right_vars.items())):
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        col_shape = equiv_set.get_shape(col_var)
        all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    all_shapes = []
    for _, col_var in join_node.df_out_vars.items():
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        (shape, c_post) = array_analysis._gen_shape_call(
            equiv_set, col_var, typ.ndim, None)
        equiv_set.insert_equiv(col_var, shape)
        post.extend(c_post)
        all_shapes.append(shape[0])
        equiv_set.define(col_var)

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.array_analysis.array_analysis_extensions[Join] = join_array_analysis


def join_distributed_analysis(join_node, array_dists):

    # input columns have same distribution
    in_dist = Distribution.OneD
    for _, col_var in (list(join_node.left_vars.items())
                       + list(join_node.right_vars.items())):
        in_dist = Distribution(
            min(in_dist.value, array_dists[col_var.name].value))

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in join_node.df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for _, col_var in join_node.df_out_vars.items():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        in_dist = out_dist

    # assign input distributions
    for _, col_var in (list(join_node.left_vars.items())
                       + list(join_node.right_vars.items())):
        array_dists[col_var.name] = in_dist

    return


distributed_analysis.distributed_analysis_extensions[Join] = join_distributed_analysis


def join_typeinfer(join_node, typeinferer):
    # TODO: consider keys with same name, cols with suffix
    for col_name, col_var in (list(join_node.left_vars.items())
                              + list(join_node.right_vars.items())):
        out_col_var = join_node.df_out_vars[col_name]
        typeinferer.constraints.append(typeinfer.Propagate(dst=out_col_var.name,
                                                           src=col_var.name, loc=join_node.loc))
    return


typeinfer.typeinfer_extensions[Join] = join_typeinfer


def visit_vars_join(join_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting join vars for:", join_node)
        print("cbdata: ", sorted(cbdata.items()))

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = visit_vars_inner(
            join_node.left_vars[col_name], callback, cbdata)
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = visit_vars_inner(
            join_node.right_vars[col_name], callback, cbdata)
    # output
    for col_name in list(join_node.df_out_vars.keys()):
        join_node.df_out_vars[col_name] = visit_vars_inner(
            join_node.df_out_vars[col_name], callback, cbdata)


# add call to visit Join variable
ir_utils.visit_vars_extensions[Join] = visit_vars_join


def remove_dead_join(join_node, lives, arg_aliases, alias_map, func_ir, typemap):
    # if an output column is dead, the related input column is not needed
    # anymore in the join
    dead_cols = []
    left_key_dead = False
    right_key_dead = False
    # TODO: remove output of dead keys

    for col_name, col_var in join_node.df_out_vars.items():
        if col_var.name not in lives:
            if col_name == join_node.left_key:
                left_key_dead = True
            elif col_name == join_node.right_key:
                right_key_dead = True
            else:
                dead_cols.append(col_name)

    for cname in dead_cols:
        assert cname in join_node.left_vars or cname in join_node.right_vars
        join_node.left_vars.pop(cname, None)
        join_node.right_vars.pop(cname, None)
        join_node.df_out_vars.pop(cname)

    # remove empty join node
    if len(join_node.df_out_vars) == 0:
        return None

    return join_node


ir_utils.remove_dead_extensions[Join] = remove_dead_join


def join_usedefs(join_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # input columns are used
    use_set.update({v.name for v in join_node.left_vars.values()})
    use_set.update({v.name for v in join_node.right_vars.values()})

    # output columns are defined
    def_set.update({v.name for v in join_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Join] = join_usedefs


def get_copies_join(join_node, typemap):
    # join doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in join_node.df_out_vars.values())
    return set(), kill_set


ir_utils.copy_propagate_extensions[Join] = get_copies_join


def apply_copies_join(join_node, var_dict, name_var_table,
                      typemap, calltypes, save_copies):
    """apply copy propagate in join node"""

    # left
    for col_name in list(join_node.left_vars.keys()):
        join_node.left_vars[col_name] = replace_vars_inner(
            join_node.left_vars[col_name], var_dict)
    # right
    for col_name in list(join_node.right_vars.keys()):
        join_node.right_vars[col_name] = replace_vars_inner(
            join_node.right_vars[col_name], var_dict)
    # output
    for col_name in list(join_node.df_out_vars.keys()):
        join_node.df_out_vars[col_name] = replace_vars_inner(
            join_node.df_out_vars[col_name], var_dict)

    return


ir_utils.apply_copy_propagate_extensions[Join] = apply_copies_join


def join_distributed_run(join_node, array_dists, typemap, calltypes, typingctx, targetctx):
    parallel = True
    for v in (list(join_node.left_vars.values())
              + list(join_node.right_vars.values())
              + list(join_node.df_out_vars.values())):
        if (array_dists[v.name] != distributed.Distribution.OneD
                and array_dists[v.name] != distributed.Distribution.OneD_Var):
            parallel = False

    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    loc = join_node.loc
    # get column variables
    left_key_var = join_node.left_vars[join_node.left_key]
    right_key_var = join_node.right_vars[join_node.right_key]

    left_other_col_vars = [v for (n, v) in sorted(join_node.left_vars.items())
                           if n != join_node.left_key]
    right_other_col_vars = [v for (n, v) in sorted(join_node.right_vars.items())
                            if n != join_node.right_key]
    # get column types
    left_other_col_typ = [typemap[v.name] for v in left_other_col_vars]
    right_other_col_typ = [typemap[v.name] for v in right_other_col_vars]
    arg_typs = tuple([typemap[left_key_var.name], typemap[right_key_var.name]]
                     + left_other_col_typ + right_other_col_typ)
    # arg names of non-key columns
    left_other_names = ["t1_c" + str(i)
                        for i in range(len(left_other_col_vars))]
    right_other_names = ["t2_c" + str(i)
                         for i in range(len(right_other_col_vars))]
    # all arg names
    left_arg_names = ['t1_key'] + left_other_names
    right_arg_names = ['t2_key'] + right_other_names

    func_text = "def f(t1_key, t2_key,{}{}{}):\n".format(
                ",".join(left_other_names),
                ("," if len(left_other_names) != 0 else ""),
                ",".join(right_other_names))

    func_text += "    data_left = ({}{})\n".format(",".join(left_other_names),
                                                "," if len(left_other_names) != 0 else "")
    func_text += "    data_right = ({}{})\n".format(",".join(right_other_names),
                                                "," if len(right_other_names) != 0 else "")

    if parallel:
        func_text += "    t1_key, data_left = parallel_join(t1_key, data_left)\n"
        #func_text += "    print(t2_key, data_right)\n"
        func_text += "    t2_key, data_right = parallel_join(t2_key, data_right)\n"
        #func_text += "    print(t2_key, data_right)\n"
        local_left_data = "t1_key" + (", " if len(left_other_names) != 0 else "") + ",".join(["data_left[{}]".format(i) for i in range(len(left_other_names))])
        local_right_data = "t2_key" + (", " if len(right_other_names) != 0 else "") + ",".join(["data_right[{}]".format(i) for i in range(len(right_other_names))])
    else:
        local_left_data = ",".join(left_arg_names)
        local_right_data = ",".join(right_arg_names)


    # local sort
    func_text += "    local_sort_f1(t1_key, data_left)\n"
    func_text += "    local_sort_f2(t2_key, data_right)\n"

    # align output variables for local merge
    # add keys first (TODO: remove dead keys)
    merge_out = [join_node.df_out_vars[join_node.left_key]]
    merge_out.append(join_node.df_out_vars[join_node.right_key])
    merge_out += [join_node.df_out_vars[n] for (n, v) in sorted(join_node.left_vars.items())
                  if n != join_node.left_key]
    merge_out += [join_node.df_out_vars[n] for (n, v) in sorted(join_node.right_vars.items())
                  if n != join_node.right_key]
    out_names = ["t3_c" + str(i) for i in range(len(merge_out))]

    func_text += "    out_t1_key, out_t2_key, out_data_left, out_data_right = hpat.hiframes_join.local_merge_new(t1_key, t2_key, data_left, data_right)\n"

    for i in range(len(left_other_names)):
        func_text += "    left_{} = out_data_left[{}]\n".format(i, i)

    for i in range(len(right_other_names)):
        func_text += "    right_{} = out_data_right[{}]\n".format(i, i)

    func_text += "    {} = out_t1_key\n".format(out_names[0])
    func_text += "    {} = out_t2_key\n".format(out_names[1])

    for i in range(len(left_other_names)):
        func_text += "    {} = left_{}\n".format(out_names[i+2], i)

    for i in range(len(right_other_names)):
        func_text += "    {} = right_{}\n".format(out_names[i+2+len(left_other_names)], i)

    # func_text += "    {} = hpat.hiframes_join.local_merge({}, {}, {})\n".format(
    #     ",".join(out_names), len(left_arg_names),
    #     local_left_data, local_right_data)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    join_impl = loc_vars['f']

    left_data_tup_typ = types.Tuple([typemap[v.name] for v in left_other_col_vars])
    _local_sort_f1 = hpat.hiframes_sort.get_local_sort_func(typemap[left_key_var.name], left_data_tup_typ)
    right_data_tup_typ = types.Tuple([typemap[v.name] for v in right_other_col_vars])
    _local_sort_f2 = hpat.hiframes_sort.get_local_sort_func(typemap[right_key_var.name], right_data_tup_typ)

    f_block = compile_to_numba_ir(join_impl,
                                  {'hpat': hpat, 'np': np,
                                  'to_string_list': to_string_list,
                                  'cp_str_list_to_array': cp_str_list_to_array,
                                  'local_sort_f1': _local_sort_f1,
                                  'local_sort_f2': _local_sort_f2,
                                  'parallel_join': parallel_join},
                                  typingctx, arg_typs,
                                  typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [left_key_var, right_key_var]
                      + left_other_col_vars + right_other_col_vars)

    nodes = f_block.body[:-3]
    for i in range(len(merge_out)):
        nodes[-len(merge_out) + i].target = merge_out[i]

    return nodes


distributed.distributed_run_extensions[Join] = join_distributed_run


@numba.njit
def parallel_join(key_arr, data):
    # alloc shuffle meta
    n_pes = hpat.distributed_api.get_size()
    shuffle_meta = alloc_shuffle_metadata(key_arr, n_pes, False)
    data_shuffle_meta = data_alloc_shuffle_metadata(data, n_pes, False)

    # calc send/recv counts
    for i in range(len(key_arr)):
        val = key_arr[i]
        node_id = hash(val) % n_pes
        update_shuffle_meta(shuffle_meta, node_id, i, val, False)
        update_data_shuffle_meta(data_shuffle_meta, node_id, i, data, False)

    finalize_shuffle_meta(key_arr, shuffle_meta)
    finalize_data_shuffle_meta(data, data_shuffle_meta, shuffle_meta)

    # write send buffers
    for i in range(len(key_arr)):
        val = key_arr[i]
        node_id = hash(val) % n_pes
        write_send_buff(shuffle_meta, node_id, val)
        write_data_send_buff(data_shuffle_meta, node_id, i, data, shuffle_meta)
        # update last since it is reused in data
        shuffle_meta.tmp_offset[node_id] += 1

    # shuffle
    alltoallv(key_arr, shuffle_meta)
    out_data = alltoallv_tup(data, data_shuffle_meta, shuffle_meta)

    return shuffle_meta.out_arr, out_data

def write_send_buff(shuffle_meta, node_id, val):
    return

@overload(write_send_buff)
def write_send_buff_overload(meta_t, node_id_t, val_t):
    arr_t = meta_t.struct['out_arr']
    if isinstance(arr_t, types.Array):
        def write_impl(shuffle_meta, node_id, val):
            # TODO: refactor to use only tmp_offset
            ind = shuffle_meta.send_disp[node_id] + shuffle_meta.tmp_offset[node_id]
            shuffle_meta.send_buff[ind] = val

        return write_impl
    assert arr_t == string_array_type
    def write_str_impl(shuffle_meta, node_id, val):
        n_chars = len(val)
        # offset buff
        ind = shuffle_meta.send_disp[node_id] + shuffle_meta.tmp_offset[node_id]
        shuffle_meta.send_arr_lens[ind] = n_chars
        # data buff
        indc = shuffle_meta.send_disp_chars[node_id] + shuffle_meta.tmp_offset_chars[node_id]
        str_copy(shuffle_meta.send_arr_chars, indc, val.c_str(), n_chars)
        shuffle_meta.tmp_offset_chars[node_id] += n_chars
        del_str(val)

    return write_str_impl


def write_data_send_buff(data_shuffle_meta, node_id, i, data, key_meta):
    return

@overload(write_data_send_buff)
def write_data_send_buff_overload(meta_t, node_id_t, ind_t, data_t, key_meta_t):
    func_text = "def f(meta_tup, node_id, ind, data, key_meta):\n"
    for i, typ in enumerate(data_t.types):
        func_text += "  val_{} = data[{}][ind]\n".format(i, i)
        func_text += "  ind_{} = key_meta.send_disp[node_id] + key_meta.tmp_offset[node_id]\n".format(i)
        if isinstance(typ, types.Array):
            func_text += "  meta_tup[{}].send_buff[ind_{}] = val_{}\n".format(i, i, i)
        else:
            # TODO: fix
            assert typ == string_array_type
            func_text += "  n_chars_{} = len(val_{})\n".format(i, i)
            func_text += "  meta_tup[{}].send_arr_lens[ind] = n_chars_{}\n".format(i, i)
            func_text += "  indc_{} = meta_tup[{}].send_disp_chars[node_id] + meta_tup[{}].tmp_offset_chars[node_id]\n".format(i, i, i)
            func_text += "  str_copy(meta_tup[{}].send_arr_chars, indc_{}, val_{}.c_str(), n_chars_{})\n".format(i, i, i, i)
            func_text += "  meta_tup[{}].tmp_offset_chars[node_id] += n_chars_{}\n".format(i, i)
            func_text += "  del_str(val_{})\n".format(i)

    func_text += "  return\n"
    loc_vars = {}
    exec(func_text, {'del_str': del_str, 'str_copy': str_copy}, loc_vars)
    write_impl = loc_vars['f']
    return write_impl

from numba.typing.templates import (
    signature, AbstractTemplate, infer_global, infer)
from numba.extending import (register_model, models, lower_builtin)
from numba import cgutils

# a native buffer pointer managed explicity (e.g. deleted manually)


class CBufferType(types.Opaque):
    def __init__(self):
        super(CBufferType, self).__init__(name='CBufferType')


c_buffer_type = CBufferType()
# ctypes_int32_typ = types.ArrayCTypes(types.Array(types.int32, 1, 'C'))

register_model(CBufferType)(models.OpaqueModel)


def get_sendrecv_counts():
    return 0


def shuffle_data():
    return 0


def sort():
    return 0


def local_merge():
    return 0


@infer_global(get_sendrecv_counts)
class SendRecvCountTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        out_typ = types.Tuple([c_buffer_type, c_buffer_type, c_buffer_type,
                               c_buffer_type, types.intp])
        return signature(out_typ, *args)


@infer_global(shuffle_data)
class ShuffleTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.int32, *args)


@infer_global(sort)
class SortTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.int32, *args)


@infer_global(local_merge)
class LocalMergeTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        out_typ = types.Tuple(args[1:])
        return signature(out_typ, *args)


LocalMergeTyper.support_literals = True

from llvmlite import ir as lir
import llvmlite.binding as ll
from numba.targets.arrayobj import make_array
from hpat.distributed_lower import _h5_typ_table
import chiframes
ll.add_symbol('get_join_sendrecv_counts', chiframes.get_join_sendrecv_counts)
ll.add_symbol('timsort', chiframes.timsort)
import hdist
ll.add_symbol('c_alltoallv', hdist.c_alltoallv)

@numba.njit
def send_recv_counts_new(key_arr):
    n_pes = hpat.distributed_api.get_size()
    send_counts = np.zeros(n_pes, np.int32)
    recv_counts = np.empty(n_pes, np.int32)
    for i in range(len(key_arr)):
        # TODO: delete string
        node_id = hash(key_arr[i]) % n_pes
        send_counts[node_id] += 1
    hpat.distributed_api.alltoall(send_counts, recv_counts, 1)
    return send_counts, recv_counts


@numba.njit
def calc_disp(arr):
    disp = np.empty_like(arr)
    disp[0] = 0
    for i in range(1, len(arr)):
        disp[i] = disp[i-1] + arr[i-1]
    return disp

@lower_builtin(get_sendrecv_counts, types.Array)
def lower_get_sendrecv_counts(context, builder, sig, args):
    # prepare buffer args
    pointer_to_cbuffer_typ = lir.IntType(8).as_pointer().as_pointer()
    send_counts = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
    recv_counts = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
    send_disp = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
    recv_disp = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())

    # prepare key array args
    key_arr = make_array(sig.args[0])(context, builder, args[0])
    # XXX: assuming key arr is 1D
    assert key_arr.shape.type.count == 1
    arr_len = builder.extract_value(key_arr.shape, 0)
    # TODO: extend to other key types
    assert sig.args[0].dtype == types.intp
    key_typ_enum = _h5_typ_table[sig.args[0].dtype]
    key_typ_arg = builder.load(cgutils.alloca_once_value(builder,
                                                         lir.Constant(lir.IntType(32), key_typ_enum)))
    key_arr_data = builder.bitcast(key_arr.data, lir.IntType(8).as_pointer())

    call_args = [send_counts, recv_counts, send_disp, recv_disp, arr_len,
                 key_typ_arg, key_arr_data]

    fnty = lir.FunctionType(lir.IntType(64), [pointer_to_cbuffer_typ] * 4
                            + [lir.IntType(64), lir.IntType(32), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty,
                                               name="get_join_sendrecv_counts")
    total_size = builder.call(fn, call_args)
    items = [builder.load(send_counts), builder.load(recv_counts),
             builder.load(send_disp), builder.load(recv_disp), total_size]
    out_tuple_typ = types.Tuple([c_buffer_type, c_buffer_type, c_buffer_type,
                                 c_buffer_type, types.intp])
    return context.make_tuple(builder, out_tuple_typ, items)

@lower_builtin(shuffle_data, types.Array, types.Array, types.Array,
              types.Array, types.VarArg(types.Any))
def lower_shuffle_arr(context, builder, sig, args):
    args[0] = make_array(sig.args[0])(context, builder, args[0]).data
    args[1] = make_array(sig.args[1])(context, builder, args[1]).data
    args[2] = make_array(sig.args[2])(context, builder, args[2]).data
    args[3] = make_array(sig.args[3])(context, builder, args[3]).data
    sig.args = (c_buffer_type, c_buffer_type, c_buffer_type, c_buffer_type, *sig.args[4:])
    return lower_shuffle(context, builder, sig, args)

@lower_builtin(shuffle_data, c_buffer_type, c_buffer_type, c_buffer_type,
               c_buffer_type, types.VarArg(types.Any))
def lower_shuffle(context, builder, sig, args):
    # assuming there are 4 buffer arguments, column vars, send arrs, recv arrs
    assert (len(args) - 4) % 3 == 0
    num_cols = (len(args) - 4) // 3
    send_counts, recv_counts, send_disp, recv_disp = args[:4]

    send_counts = builder.bitcast(send_counts, lir.IntType(8).as_pointer())
    recv_counts = builder.bitcast(recv_counts, lir.IntType(8).as_pointer())
    send_disp = builder.bitcast(send_disp, lir.IntType(8).as_pointer())
    recv_disp = builder.bitcast(recv_disp, lir.IntType(8).as_pointer())

    col_names = ["c" + str(i) for i in range(num_cols)]
    send_names = ["send_c" + str(i) for i in range(num_cols)]
    # create send buffer building function
    func_text = "def f(send_disp, {}, {}):\n".format(
        ",".join(col_names), ",".join(send_names))
    func_text += "    n_pes = hpat.distributed_api.get_size()\n"
    func_text += "    tmp_count = np.zeros(n_pes, dtype=np.int64)\n"
    func_text += "    for i in range(len(c0)):\n"
    func_text += "        node_id = c0[i] % n_pes\n"
    func_text += "        ind = send_disp[node_id] + tmp_count[node_id]\n"
    func_text += "        send_c0[ind] = c0[i]\n"
    for i in range(1, num_cols):
        func_text += "        send_c{}[ind] = c{}[i]\n".format(i, i)
    func_text += "        tmp_count[node_id] = tmp_count[node_id] + 1\n"
    #func_text += "        hpat.cprint(node_id)\n"

    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'np': np}, loc_vars)
    f = loc_vars['f']
    f_args = [send_disp] + args[4:4 + 2 * num_cols]
    f_sig = signature(types.void, *((c_buffer_type,) +
                                    sig.args[4:4 + 2 * num_cols]))
    context.compile_internal(builder, f, f_sig, f_args)
    # generate alltoallv calls
    for i in range(0, num_cols):
        arr_typ = sig.args[4 + i]
        send_arg = args[4 + num_cols + i]
        recv_arg = args[4 + 2 * num_cols + i]
        gen_alltoallv(context, builder, arr_typ, send_arg, recv_arg, send_counts,
                      recv_counts, send_disp, recv_disp)

    return lir.Constant(lir.IntType(32), 0)


def gen_alltoallv(context, builder, arr_typ, send_arg, recv_arg, send_counts,
                  recv_counts, send_disp, recv_disp):
    #
    typ_enum = _h5_typ_table[arr_typ.dtype]
    typ_arg = builder.load(cgutils.alloca_once_value(builder,
                                                     lir.Constant(lir.IntType(32), typ_enum)))
    send_data = make_array(arr_typ)(context, builder, send_arg).data
    recv_data = make_array(arr_typ)(context, builder, recv_arg).data
    send_data = builder.bitcast(send_data, lir.IntType(8).as_pointer())
    recv_data = builder.bitcast(recv_data, lir.IntType(8).as_pointer())

    call_args = [send_data, recv_data, send_counts,
                 recv_counts, send_disp, recv_disp, typ_arg]

    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(
        8).as_pointer()] * 6 + [lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="c_alltoallv")
    builder.call(fn, call_args)


@lower_builtin(sort, types.VarArg(types.Any))
def lower_sort(context, builder, sig, args):
    #
    key_arr = make_array(sig.args[0])(context, builder, args[0])
    key_data = builder.bitcast(key_arr.data, lir.IntType(8).as_pointer())
    # XXX: assuming key arr is 1D
    assert key_arr.shape.type.count == 1
    arr_len = builder.extract_value(key_arr.shape, 0)
    num_other_cols = len(args) - 1
    # build array of other column arrays arg
    other_arrs = cgutils.alloca_once(
        builder, lir.IntType(8).as_pointer(), num_other_cols)
    for i in range(num_other_cols):
        ptr = cgutils.gep_inbounds(builder, other_arrs, i)
        arr = make_array(sig.args[i + 1])(context, builder, args[i + 1])
        arr_data = builder.bitcast(arr.data, lir.IntType(8).as_pointer())
        builder.store(arr_data, ptr)

    call_args = [key_data, arr_len, other_arrs,
                 lir.Constant(lir.IntType(64), num_other_cols)]
    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer(),
                                             lir.IntType(64), lir.IntType(8).as_pointer().as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="timsort")
    builder.call(fn, call_args)
    return lir.Constant(lir.IntType(32), 0)


def ensure_capacity(arr, new_size):
    new_arr = arr
    curr_len = len(arr)
    if curr_len < new_size:
        new_len = 2 * curr_len
        new_arr = np.empty(new_len, arr.dtype)
        new_arr[:curr_len] = arr
    return new_arr

@overload(ensure_capacity)
def ensure_capacity_overload(arr_t, new_size_t):
    if isinstance(arr_t, types.Array):
        return ensure_capacity
    assert isinstance(arr_t, (types.Tuple, types.UniTuple))
    count = arr_t.count

    func_text = "def f(data, new_size):\n"
    func_text += "  return ({}{})\n".format(','.join(["ensure_capacity(data[{}], new_size)".format(
        i) for i in range(count)]),
        "," if count == 1 else "")  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {'ensure_capacity': ensure_capacity}, loc_vars)
    alloc_impl = loc_vars['f']
    return alloc_impl


@numba.njit
def ensure_capacity_str(arr, new_size, n_chars):
    # new_size is right after write index
    new_arr = arr
    curr_len = len(arr)
    curr_num_chars = num_total_chars(arr)

    # TODO: corner case test
    #print("new alloc", new_size, curr_len, getitem_str_offset(arr, new_size-1), n_chars, curr_num_chars)
    if curr_len < new_size or getitem_str_offset(arr, new_size-1) + n_chars > curr_num_chars:
        new_len = 2 * curr_len
        new_num_chars = 2 * curr_num_chars
        new_arr = pre_alloc_string_array(new_len, new_num_chars)
        copy_str_arr_slice(new_arr, arr, new_size-1)

    return new_arr


def trim_arr_tup(data, new_size):  # pragma: no cover
    return data

@overload(trim_arr_tup)
def trim_arr_tup_overload(data_t, new_size_t):
    assert isinstance(data_t, (types.Tuple, types.UniTuple))
    count = data_t.count

    func_text = "def f(data, new_size):\n"
    func_text += "  return ({}{})\n".format(','.join(["trim_arr(data[{}], new_size)".format(
        i) for i in range(count)]),
        "," if count == 1 else "")  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {'trim_arr': trim_arr}, loc_vars)
    alloc_impl = loc_vars['f']
    return alloc_impl


# @numba.njit
# def copy_merge_data(left_key, data_left, data_right, left_ind, right_ind,
#                         out_left_key, out_data_left, out_data_right, out_ind):
#     out_left_key = ensure_capacity(out_left_key, out_ind+1)
#     out_data_left = ensure_capacity(out_data_left, out_ind+1)
#     out_data_right = ensure_capacity(out_data_right, out_ind+1)

#     out_left_key[out_ind] = left_key[left_ind]
#     copyElement_tup(data_left, left_ind, out_data_left, out_ind)
#     copyElement_tup(data_right, right_ind, out_data_right, out_ind)
#     return out_left_key, out_data_left, out_data_right

def copy_elem_buff(arr, ind, val):  # pragma: no cover
    new_arr = ensure_capacity(arr, ind+1)
    new_arr[ind] = val
    return new_arr

@overload(copy_elem_buff)
def copy_elem_buff_overload(arr_t, ind_t, val_t):
    if isinstance(arr_t, types.Array):
        return copy_elem_buff

    assert arr_t == string_array_type
    def copy_elem_buff_str(arr, ind, val):
        new_arr = ensure_capacity_str(arr, ind+1, len(val))
        #new_arr[ind] = val
        setitem_string_array(get_offset_ptr(new_arr), get_data_ptr(new_arr), val, ind)
        return new_arr

    return copy_elem_buff_str

def copy_elem_buff_tup(arr, ind, val):  # pragma: no cover
    return arr

@overload(copy_elem_buff_tup)
def copy_elem_buff_tup_overload(data_t, ind_t, val_t):
    assert isinstance(data_t, (types.Tuple, types.UniTuple))
    count = data_t.count

    func_text = "def f(data, ind, val):\n"
    for i in range(count):
        func_text += "  arr_{} = copy_elem_buff(data[{}], ind, val[{}])\n".format(i, i, i)
    func_text += "  return ({}{})\n".format(
        ','.join(["arr_{}".format(i) for i in range(count)]),
        "," if count == 1 else "")

    loc_vars = {}
    exec(func_text, {'copy_elem_buff': copy_elem_buff}, loc_vars)
    cp_impl = loc_vars['f']
    return cp_impl

def trim_arr(arr, size):  # pragma: no cover
    return arr[:size]

@overload(trim_arr)
def trim_arr_overload(arr_t, size_t):
    if isinstance(arr_t, types.Array):
        return trim_arr

    assert arr_t == string_array_type
    def trim_arr_str(arr, size):
        # print("trim size", size, arr[size-1], getitem_str_offset(arr, size))
        new_arr = pre_alloc_string_array(size, np.int64(getitem_str_offset(arr, size)))
        copy_str_arr_slice(new_arr, arr, size)
        return new_arr

    return trim_arr_str

@numba.njit
def local_merge_new(left_key, right_key, data_left, data_right):
    curr_size = 101 + min(len(left_key), len(right_key)) // 10
    out_left_key = empty_like_type(curr_size, left_key)
    out_data_left = alloc_arr_tup(curr_size, data_left)
    out_data_right = alloc_arr_tup(curr_size, data_right)

    out_ind = 0
    left_ind = 0
    right_ind = 0

    while left_ind < len(left_key) and right_ind < len(right_key):
        if left_key[left_ind] == right_key[right_ind]:
            out_left_key = copy_elem_buff(out_left_key, out_ind, left_key[left_ind])
            l_data_val = getitem_arr_tup(data_left, left_ind)
            out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
            r_data_val = getitem_arr_tup(data_right, right_ind)
            out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)

            out_ind += 1
            left_run = left_ind + 1
            while left_run < len(left_key) and left_key[left_run] == right_key[right_ind]:
                out_left_key = copy_elem_buff(out_left_key, out_ind, left_key[left_run])
                l_data_val = getitem_arr_tup(data_left, left_run)
                out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
                r_data_val = getitem_arr_tup(data_right, right_ind)
                out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)

                out_ind += 1
                left_run += 1
            right_run = right_ind + 1
            while right_run < len(right_key) and right_key[right_run] == left_key[left_ind]:
                out_left_key = copy_elem_buff(out_left_key, out_ind, left_key[left_ind])
                l_data_val = getitem_arr_tup(data_left, left_ind)
                out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
                r_data_val = getitem_arr_tup(data_right, right_run)
                out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)

                out_ind += 1
                right_run += 1
            left_ind += 1
            right_ind += 1
        elif left_key[left_ind] < right_key[right_ind]:
            left_ind += 1
        else:
            right_ind += 1

    #out_left_key = out_left_key[:out_ind]
    out_left_key = trim_arr(out_left_key, out_ind)

    out_right_key = out_left_key.copy()
    out_data_left = trim_arr_tup(out_data_left, out_ind)
    out_data_right = trim_arr_tup(out_data_right, out_ind)

    return out_left_key, out_right_key, out_data_left, out_data_right

@lower_builtin(local_merge, types.Const, types.VarArg(types.Any))
def lower_local_merge(context, builder, sig, args):
    #
    num_left_cols = sig.args[0].value
    num_right_cols = len(args) - num_left_cols - 1
    left_other_names = ["t1_c" + str(i) for i in range(num_left_cols - 1)]
    right_other_names = ["t2_c" + str(i) for i in range(num_right_cols - 1)]

    # create merge function
    func_text = "def f(left_key, right_key, {}{} {}):\n".format(
        ",".join(left_other_names),
        ("," if len(left_other_names) != 0 else ""),
        ",".join(right_other_names))
    # initialize output arrays with a heuristic starting size
    func_text += "    curr_size = 101 + min(len(left_key), len(right_key)) // 10\n"
    func_text += "    out_left_key = np.empty(curr_size, left_key.dtype)\n"
    func_text += "    out_right_key = np.empty(curr_size, right_key.dtype)\n"
    for v in (left_other_names + right_other_names):
        func_text += "    out_{} = np.empty(curr_size, {}.dtype)\n".format(v, v)
    func_text += "    out_ind = 0\n"
    func_text += "    left_ind = 0\n"
    func_text += "    right_ind = 0\n"
    func_text += "    while left_ind < len(left_key) and right_ind < len(right_key):\n"
    func_text += "        if left_key[left_ind] == right_key[right_ind]:\n"
    func_text += _set_merge_output("            ", left_other_names,
                                   right_other_names, "left_ind", "right_ind")
    func_text += "            left_run = left_ind + 1\n"
    func_text += "            while left_run < len(left_key) and left_key[left_run] == right_key[right_ind]:\n"
    func_text += _set_merge_output("                ", left_other_names,
                                   right_other_names, "left_run", "right_ind")
    func_text += "                left_run += 1\n"
    func_text += "            right_run = right_ind + 1\n"
    func_text += "            while right_run < len(right_key) and right_key[right_run] == left_key[left_ind]:\n"
    func_text += _set_merge_output("                ", left_other_names,
                                   right_other_names, "left_ind", "right_run")
    func_text += "                right_run += 1\n"
    func_text += "            left_ind += 1\n"
    func_text += "            right_ind += 1\n"
    func_text += "        elif left_key[left_ind] < right_key[right_ind]:\n"
    func_text += "            left_ind += 1\n"
    func_text += "        else:\n"
    func_text += "            right_ind += 1\n"
    # shrink to size
    func_text += "    out_left_key = out_left_key[:out_ind]\n"
    #func_text += "    out_right_key = out_right_key[:out_ind]\n"
    func_text += "    out_right_key = out_left_key.copy()\n"
    for v in (left_other_names + right_other_names):
        func_text += "    out_{} = out_{}[:out_ind]\n".format(v, v)
    # return output
    out_left_other_names = ["out_" + v for v in left_other_names]
    out_right_other_names = ["out_" + v for v in right_other_names]
    func_text += "    return out_left_key,{}{} out_right_key,{}\n".format(
        ",".join(out_left_other_names),
        ("," if len(out_left_other_names) != 0 else ""),
        ",".join(out_right_other_names))

    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'np': np}, loc_vars)
    f = loc_vars['f']
    # args: left key, right key, left other cols, right other cols
    f_args = [args[1], args[num_left_cols + 1]] + \
        args[2:num_left_cols + 1] + args[num_left_cols + 2:]
    f_sig = signature(sig.return_type, *sig.args[1:])
    return context.compile_internal(builder, f, f_sig, f_args)


def _set_merge_output(indent, left_other_names, right_other_names, left_ind, right_ind):
    func_text = ""
    #func_text += indent + "hpat.cprint({}[-1])\n".format(out)
    func_text += indent + "if out_ind >= curr_size:\n"
    func_text += indent + "    new_size = 2 * curr_size\n"
    for v in ['left_key', 'right_key'] + right_other_names + left_other_names:
        out = "out_" + v
        func_text += indent + \
            "    new_{} = np.empty(new_size, {}.dtype)\n".format(out, out)
        func_text += indent + "    new_{}[:curr_size] = {}\n".format(out, out)
        func_text += indent + "    {} = new_{}\n".format(out, out)
    func_text += indent + "    curr_size = new_size\n"

    func_text += indent + \
        "{}[out_ind] = {}\n".format(
            "out_left_key", "left_key[{}]".format(left_ind))
    # func_text += indent + \
    #     "{}[out_ind] = {}\n".format(
    #         "out_right_key", "right_key[{}]".format(left_ind))

    for v in left_other_names:
        func_text += indent + \
            "{}[out_ind] = {}\n".format(
                "out_" + v, v + "[{}]".format(left_ind))
    for v in right_other_names:
        func_text += indent + \
            "{}[out_ind] = {}\n".format(
                "out_" + v, v + "[{}]".format(right_ind))
    func_text += indent + "out_ind += 1\n"
    return func_text


@infer
class GetItemCBuf(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        c_buff, idx = args
        if isinstance(c_buff, CBufferType):
            if isinstance(idx, types.Integer):
                return signature(types.int32, c_buff, idx)


@lower_builtin('getitem', c_buffer_type, types.intp)
def c_buffer_type_getitem(context, builder, sig, args):
    base_ptr = builder.bitcast(args[0], lir.IntType(32).as_pointer())
    return builder.load(builder.gep(base_ptr, [args[1]], inbounds=True))

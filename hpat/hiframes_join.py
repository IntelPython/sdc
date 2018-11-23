from __future__ import print_function, division, absolute_import
import operator
from collections import defaultdict
import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.extending import overload
from numba.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes,
                            mk_unique_var)
import hpat
from hpat import distributed, distributed_analysis
from hpat.utils import debug_prints, alloc_arr_tup, empty_like_type
from hpat.distributed_analysis import Distribution
from hpat.hiframes_sort import (
    alloc_shuffle_metadata, data_alloc_shuffle_metadata, alltoallv,
    alltoallv_tup, finalize_shuffle_meta, finalize_data_shuffle_meta,
    update_shuffle_meta, update_data_shuffle_meta, alloc_pre_shuffle_metadata,
    _get_keys_tup, _get_data_tup)
from hpat.str_arr_ext import (string_array_type, to_string_list,
                              cp_str_list_to_array, str_list_to_array,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, del_str, num_total_chars,
                              getitem_str_offset, copy_str_arr_slice,
                              setitem_string_array, str_copy_ptr,
                              setitem_str_offset)
from hpat.str_ext import string_type
from hpat.timsort import copyElement_tup, getitem_arr_tup, setitem_arr_tup
from hpat.shuffle_utils import getitem_arr_tup_single, val_to_tup
import numpy as np


class Join(ir.Stmt):
    def __init__(self, df_out, left_df, right_df, left_keys, right_keys, df_vars, how, loc):
        self.df_out = df_out
        self.left_df = left_df
        self.right_df = right_df
        self.left_keys = left_keys
        self.right_keys = right_keys
        self.df_out_vars = df_vars[self.df_out].copy()
        self.left_vars = df_vars[left_df].copy()
        self.right_vars = df_vars[right_df].copy()
        self.how = how
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
        return "join [{}={}]: {} , {}, {}".format(
            self.left_keys, self.right_keys, df_out_str, df_left_str,
            df_right_str)


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

    # TODO: can columns of the same input table have diffrent dists?
    # left and right inputs can have 1D or 1D_Var seperately (q26 case)
    # input columns have same distribution
    left_dist = Distribution.OneD
    right_dist = Distribution.OneD
    for col_var in join_node.left_vars.values():
        left_dist = Distribution(
            min(left_dist.value, array_dists[col_var.name].value))

    for col_var in join_node.right_vars.values():
        right_dist = Distribution(
            min(right_dist.value, array_dists[col_var.name].value))

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for col_var in join_node.df_out_vars.values():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, left_dist.value))
    out_dist = Distribution(min(out_dist.value, right_dist.value))
    for col_var in join_node.df_out_vars.values():
        array_dists[col_var.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        left_dist = out_dist
        right_dist = out_dist

    # assign input distributions
    for col_var in join_node.left_vars.values():
        array_dists[col_var.name] = left_dist

    for col_var in join_node.right_vars.values():
        array_dists[col_var.name] = right_dist

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
    if not hpat.hiframes_api.enable_hiframes_remove_dead:
        return join_node
    # if an output column is dead, the related input column is not needed
    # anymore in the join
    dead_cols = []
    left_key_dead = False
    right_key_dead = False
    # TODO: remove output of dead keys

    for col_name, col_var in join_node.df_out_vars.items():
        if col_var.name not in lives:
            if col_name in join_node.left_keys:
                left_key_dead = True
            elif col_name in join_node.right_keys:
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

def build_join_definitions(join_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in join_node.df_out_vars.values():
        definitions[col_var.name].append(join_node)

    return definitions

ir_utils.build_defs_extensions[Join] = build_join_definitions


def join_distributed_run(join_node, array_dists, typemap, calltypes, typingctx, targetctx, dist_pass):
    parallel = True
    for v in (list(join_node.left_vars.values())
              + list(join_node.right_vars.values())
              + list(join_node.df_out_vars.values())):
        if (array_dists[v.name] != distributed.Distribution.OneD
                and array_dists[v.name] != distributed.Distribution.OneD_Var):
            parallel = False

    # TODO: rebalance if output distributions are 1D instead of 1D_Var
    loc = join_node.loc
    n_keys = len(join_node.left_keys)
    # get column variables
    left_key_vars = tuple(join_node.left_vars[c] for c in join_node.left_keys)
    right_key_vars = tuple(join_node.right_vars[c] for c in join_node.right_keys)

    left_other_col_vars = tuple(v for (n, v) in sorted(join_node.left_vars.items())
                           if n not in join_node.left_keys)
    right_other_col_vars = tuple(v for (n, v) in sorted(join_node.right_vars.items())
                            if n not in join_node.right_keys)
    # get column types
    arg_vars = (left_key_vars + right_key_vars
                + left_other_col_vars + right_other_col_vars)
    arg_typs = tuple(typemap[v.name] for v in arg_vars)
    scope = arg_vars[0].scope

    # arg names of non-key columns
    left_other_names = tuple("t1_c" + str(i)
                        for i in range(len(left_other_col_vars)))
    right_other_names = tuple("t2_c" + str(i)
                         for i in range(len(right_other_col_vars)))

    left_key_names = tuple("t1_key" + str(i) for i in range(n_keys))
    right_key_names = tuple("t2_key" + str(i) for i in range(n_keys))

    func_text = "def f({}, {},{}{}{}):\n".format(
                ",".join(left_key_names),
                ",".join(right_key_names),
                ",".join(left_other_names),
                ("," if len(left_other_names) != 0 else ""),
                ",".join(right_other_names))

    func_text += "    t1_keys = ({},)\n".format(",".join(left_key_names))
    func_text += "    t2_keys = ({},)\n".format(",".join(right_key_names))

    func_text += "    data_left = ({}{})\n".format(",".join(left_other_names),
                                                "," if len(left_other_names) != 0 else "")
    func_text += "    data_right = ({}{})\n".format(",".join(right_other_names),
                                                "," if len(right_other_names) != 0 else "")

    if parallel:
        if join_node.how == 'asof':
            # only the right key needs to be aligned
            func_text += "    t2_keys, data_right = parallel_asof_comm(t1_keys, t2_keys, data_right)\n"
        else:
            func_text += "    t1_keys, data_left = parallel_join(t1_keys, data_left)\n"
            func_text += "    t2_keys, data_right = parallel_join(t2_keys, data_right)\n"
            #func_text += "    print(t2_key, data_right)\n"

    if join_node.how != 'asof':
        # asof key is already sorted, TODO: add error checking
        # local sort
        func_text += "    local_sort_f1(t1_keys, data_left)\n"
        func_text += "    local_sort_f2(t2_keys, data_right)\n"

    # align output variables for local merge
    # add keys first (TODO: remove dead keys)
    out_l_key_vars = tuple(join_node.df_out_vars[c] for c in join_node.left_keys)
    out_r_key_vars = tuple(join_node.df_out_vars[c] for c in join_node.right_keys)
    # create dummy variable if right key is not actually returned
    # using the same output left key causes errors for asof case
    if join_node.left_keys == join_node.right_keys:
        out_r_key_vars = tuple(ir.Var(scope, mk_unique_var('dummy_k'), loc)
                                                        for _ in range(n_keys))
        for v, w in zip(out_r_key_vars, out_l_key_vars):
            typemap[v.name] = typemap[w.name]

    merge_out = out_l_key_vars + out_r_key_vars
    merge_out += tuple(join_node.df_out_vars[n] for (n, v) in sorted(join_node.left_vars.items())
                  if n not in join_node.left_keys)
    merge_out += tuple(join_node.df_out_vars[n] for (n, v) in sorted(join_node.right_vars.items())
                  if n not in join_node.right_keys)
    out_names = ["t3_c" + str(i) for i in range(len(merge_out))]

    if join_node.how == 'asof':
        func_text += ("    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
        " = hpat.hiframes_join.local_merge_asof(t1_keys, t2_keys, data_left, data_right)\n")
    else:
        func_text += ("    out_t1_keys, out_t2_keys, out_data_left, out_data_right"
        " = hpat.hiframes_join.local_merge_new(t1_keys, t2_keys, data_left, data_right, {}, {})\n".format(
            join_node.how in ('left', 'outer'), join_node.how == 'outer'))

    for i in range(len(left_other_names)):
        func_text += "    left_{} = out_data_left[{}]\n".format(i, i)

    for i in range(len(right_other_names)):
        func_text += "    right_{} = out_data_right[{}]\n".format(i, i)

    for i in range(n_keys):
        func_text += "    t1_keys_{} = out_t1_keys[{}]\n".format(i, i)

    for i in range(n_keys):
        func_text += "    t2_keys_{} = out_t2_keys[{}]\n".format(i, i)

    for i in range(n_keys):
        func_text += "    {} = t1_keys_{}\n".format(out_names[i], i)
    for i in range(n_keys):
        func_text += "    {} = t2_keys_{}\n".format(out_names[n_keys + i], i)

    for i in range(len(left_other_names)):
        func_text += "    {} = left_{}\n".format(out_names[i+2*n_keys], i)

    for i in range(len(right_other_names)):
        func_text += "    {} = right_{}\n".format(out_names[i+2*n_keys+len(left_other_names)], i)

    # func_text += "    {} = hpat.hiframes_join.local_merge({}, {}, {})\n".format(
    #     ",".join(out_names), len(left_arg_names),
    #     local_left_data, local_right_data)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    join_impl = loc_vars['f']

    # print(func_text)

    left_keys_tup_typ = types.Tuple([typemap[v.name] for v in left_key_vars])
    left_data_tup_typ = types.Tuple([typemap[v.name] for v in left_other_col_vars])
    _local_sort_f1 = hpat.hiframes_sort.get_local_sort_func(left_keys_tup_typ, left_data_tup_typ)
    right_keys_tup_typ = types.Tuple([typemap[v.name] for v in right_key_vars])
    right_data_tup_typ = types.Tuple([typemap[v.name] for v in right_other_col_vars])
    _local_sort_f2 = hpat.hiframes_sort.get_local_sort_func(right_keys_tup_typ, right_data_tup_typ)

    f_block = compile_to_numba_ir(join_impl,
                                  {'hpat': hpat, 'np': np,
                                  'to_string_list': to_string_list,
                                  'cp_str_list_to_array': cp_str_list_to_array,
                                  'local_sort_f1': _local_sort_f1,
                                  'local_sort_f2': _local_sort_f2,
                                  'parallel_join': parallel_join,
                                  'parallel_asof_comm': parallel_asof_comm},
                                  typingctx, arg_typs,
                                  typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, arg_vars)

    nodes = f_block.body[:-3]
    for i in range(len(merge_out)):
        nodes[-len(merge_out) + i].target = merge_out[i]

    return nodes


distributed.distributed_run_extensions[Join] = join_distributed_run


@numba.njit
def parallel_join(key_arrs, data):
    # alloc shuffle meta
    n_pes = hpat.distributed_api.get_size()
    pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, data, n_pes, False)


    # calc send/recv counts
    for i in range(len(key_arrs[0])):
        val = getitem_arr_tup_single(key_arrs, i)
        node_id = hash(val) % n_pes
        update_shuffle_meta(pre_shuffle_meta, node_id, i, val_to_tup(val),
            getitem_arr_tup(data, i), False)

    shuffle_meta = finalize_shuffle_meta(key_arrs, data, pre_shuffle_meta,
                                          n_pes, False)

    # write send buffers
    for i in range(len(key_arrs[0])):
        val = getitem_arr_tup_single(key_arrs, i)
        node_id = hash(val) % n_pes
        write_send_buff(shuffle_meta, node_id, i, val_to_tup(val), data)
        # update last since it is reused in data
        shuffle_meta.tmp_offset[node_id] += 1

    # shuffle
    recvs = alltoallv_tup(key_arrs + data, shuffle_meta)
    out_keys = _get_keys_tup(recvs, key_arrs)
    out_data = _get_data_tup(recvs, key_arrs)

    return out_keys, out_data

@numba.njit
def parallel_asof_comm(left_key_arrs, right_key_arrs, right_data):
    # align the left and right intervals
    # allgather the boundaries of all left intervals and calculate overlap
    # rank = hpat.distributed_api.get_rank()
    n_pes = hpat.distributed_api.get_size()
    # TODO: multiple keys
    bnd_starts = np.empty(n_pes, left_key_arrs[0].dtype)
    bnd_ends = np.empty(n_pes, left_key_arrs[0].dtype)
    hpat.distributed_api.allgather(bnd_starts, left_key_arrs[0][0])
    hpat.distributed_api.allgather(bnd_ends, left_key_arrs[0][-1])

    send_counts = np.zeros(n_pes, np.int32)
    send_disp = np.zeros(n_pes, np.int32)
    recv_counts = np.zeros(n_pes, np.int32)
    my_start = right_key_arrs[0][0]
    my_end = right_key_arrs[0][-1]

    offset = -1
    i = 0
    # ignore no overlap processors (end of their interval is before current)
    while i < n_pes-1 and bnd_ends[i] < my_start:
        i += 1
    while i < n_pes and bnd_starts[i] <= my_end:
        offset, count = _count_overlap(right_key_arrs[0], bnd_starts[i], bnd_ends[i])
        # one extra element in case first value is needed for start of boundary
        if offset != 0:
            offset -= 1
            count += 1
        send_counts[i] = count
        send_disp[i] = offset
        i += 1
    # one extra element in case last value is need for start of boundary
    # TODO: see if next processor provides the value
    while i < n_pes:
        send_counts[i] = 1
        send_disp[i] = len(right_key_arrs[0]) - 1
        i += 1

    hpat.distributed_api.alltoall(send_counts, recv_counts, 1)
    n_total_recv = recv_counts.sum()
    out_r_keys = np.empty(n_total_recv, right_key_arrs[0].dtype)
    # TODO: support string
    out_r_data = alloc_arr_tup(n_total_recv, right_data)
    recv_disp = hpat.hiframes_join.calc_disp(recv_counts)
    hpat.distributed_api.alltoallv(right_key_arrs[0], out_r_keys, send_counts,
                                   recv_counts, send_disp, recv_disp)
    hpat.distributed_api.alltoallv_tup(right_data, out_r_data, send_counts,
                                   recv_counts, send_disp, recv_disp)

    return (out_r_keys,), out_r_data

@numba.njit
def _count_overlap(r_key_arr, start, end):
    # TODO: use binary search
    count = 0
    offset = 0
    j = 0
    while j < len(r_key_arr) and r_key_arr[j] < start:
        offset += 1
        j += 1
    while j < len(r_key_arr) and start <= r_key_arr[j] <= end:
        j += 1
        count += 1
    return offset, count




def write_send_buff(shuffle_meta, node_id, i, val, data):
    return i

@overload(write_send_buff)
def write_data_buff_overload(meta_t, node_id_t, i_t, val_t, data_t):
    func_text = "def f(meta, node_id, i, val, data):\n"
    func_text += "  w_ind = meta.send_disp[node_id] + meta.tmp_offset[node_id]\n"
    n_keys = len(val_t.types)
    n_str = 0
    for i, typ in enumerate(val_t.types + data_t.types):
        val = ("val[{}]".format(i) if i < n_keys
               else "data[{}][i]".format(i - n_keys))
        func_text += "  val_{} = {}\n".format(i, val)
        if not typ in (string_type, string_array_type):
            func_text += "  meta.send_buff_tup[{}][w_ind] = val_{}\n".format(i, i)
        else:
            func_text += "  n_chars_{} = len(val_{})\n".format(i, i)
            func_text += "  meta.send_arr_lens_tup[{}][w_ind] = n_chars_{}\n".format(n_str, i)
            func_text += "  indc_{} = meta.send_disp_char_tup[{}][node_id] + meta.tmp_offset_char_tup[{}][node_id]\n".format(i, n_str, n_str)
            func_text += "  str_copy_ptr(meta.send_arr_chars_tup[{}], indc_{}, val_{}.c_str(), n_chars_{})\n".format(n_str, i, i, i)
            func_text += "  meta.tmp_offset_char_tup[{}][node_id] += n_chars_{}\n".format(n_str, i)
            # func_text += "  del_str(val_{})\n".format(i)
            n_str += 1

    func_text += "  return w_ind\n"

    # print(func_text)

    loc_vars = {}
    exec(func_text, {'del_str': del_str, 'str_copy_ptr': str_copy_ptr}, loc_vars)
    write_impl = loc_vars['f']
    return write_impl




# def write_send_buff(shuffle_meta, node_id, val):
#     return 0

# @overload(write_send_buff)
# def write_send_buff_overload(meta_t, node_id_t, val_t):
#     arr_t = meta_t.struct['out_arr']
#     if isinstance(arr_t, types.Array):
#         def write_impl(shuffle_meta, node_id, val):
#             # TODO: refactor to use only tmp_offset
#             ind = shuffle_meta.send_disp[node_id] + shuffle_meta.tmp_offset[node_id]
#             shuffle_meta.send_buff[ind] = val
#             return ind

#         return write_impl
#     assert arr_t == string_array_type
#     def write_str_impl(shuffle_meta, node_id, val):
#         n_chars = len(val)
#         # offset buff
#         ind = shuffle_meta.send_disp[node_id] + shuffle_meta.tmp_offset[node_id]
#         shuffle_meta.send_arr_lens[ind] = n_chars
#         # data buff
#         indc = shuffle_meta.send_disp_char[node_id] + shuffle_meta.tmp_offset_char[node_id]
#         str_copy_ptr(shuffle_meta.send_arr_chars, indc, val.c_str(), n_chars)
#         shuffle_meta.tmp_offset_char[node_id] += n_chars
#         #del_str(val)
#         return ind

#     return write_str_impl


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
            func_text += "  meta_tup[{}].send_arr_lens[ind_{}] = n_chars_{}\n".format(i, i, i)
            func_text += "  indc_{} = meta_tup[{}].send_disp_char[node_id] + meta_tup[{}].tmp_offset_char[node_id]\n".format(i, i, i)
            func_text += "  str_copy_ptr(meta_tup[{}].send_arr_chars, indc_{}, val_{}.c_str(), n_chars_{})\n".format(i, i, i, i)
            func_text += "  meta_tup[{}].tmp_offset_char[node_id] += n_chars_{}\n".format(i, i)
            func_text += "  del_str(val_{})\n".format(i)

    func_text += "  return\n"
    loc_vars = {}
    exec(func_text, {'del_str': del_str, 'str_copy_ptr': str_copy_ptr}, loc_vars)
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
from hpat.utils import _numba_to_c_type_map
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
    key_typ_enum = _numba_to_c_type_map[sig.args[0].dtype]
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
    typ_enum = _numba_to_c_type_map[arr_typ.dtype]
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
        new_num_chars = 2 * curr_num_chars + n_chars
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

#     out_left_key[out_ind] = left_keys[left_ind]
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

def setnan_elem_buff(arr, ind):  # pragma: no cover
    new_arr = ensure_capacity(arr, ind+1)
    setitem_arr_nan(new_arr, ind)
    return new_arr

@overload(setnan_elem_buff)
def setnan_elem_buff_overload(arr_t, ind_t):
    if isinstance(arr_t, types.Array):
        return setnan_elem_buff

    assert arr_t == string_array_type
    def setnan_elem_buff_str(arr, ind):
        new_arr = ensure_capacity_str(arr, ind+1, 0)
        # TODO: set actual nan for str
        # TODO: why doesn't setitem_str_offset work
        #setitem_arr_nan(new_arr, ind)
        #setitem_str_offset(arr, ind+1, getitem_str_offset(arr, ind))
        setitem_string_array(get_offset_ptr(new_arr), get_data_ptr(new_arr), '', ind)
        #print(getitem_str_offset(arr, ind), getitem_str_offset(arr, ind+1))
        return new_arr

    return setnan_elem_buff_str

def setnan_elem_buff_tup(arr, ind):  # pragma: no cover
    return arr

@overload(setnan_elem_buff_tup)
def setnan_elem_buff_tup_overload(data_t, ind_t):
    assert isinstance(data_t, (types.Tuple, types.UniTuple))
    count = data_t.count

    func_text = "def f(data, ind):\n"
    for i in range(count):
        func_text += "  arr_{} = setnan_elem_buff(data[{}], ind)\n".format(i, i)
    func_text += "  return ({}{})\n".format(
        ','.join(["arr_{}".format(i) for i in range(count)]),
        "," if count == 1 else "")

    loc_vars = {}
    exec(func_text, {'setnan_elem_buff': setnan_elem_buff}, loc_vars)
    cp_impl = loc_vars['f']
    return cp_impl


@numba.njit
def local_merge_new(left_keys, right_keys, data_left, data_right, is_left=False,
                                                               is_outer=False):
    curr_size = 101 + min(len(left_keys[0]), len(right_keys[0])) // 10
    out_left_key = alloc_arr_tup(curr_size, left_keys)
    out_data_left = alloc_arr_tup(curr_size, data_left)
    out_data_right = alloc_arr_tup(curr_size, data_right)

    out_ind = 0
    left_ind = 0
    right_ind = 0

    while left_ind < len(left_keys[0]) and right_ind < len(right_keys[0]):
        if getitem_arr_tup(left_keys, left_ind) == getitem_arr_tup(right_keys, right_ind):
            key = getitem_arr_tup(left_keys, left_ind)
            # catesian product in case of duplicate keys on either side
            left_run = left_ind
            while left_run < len(left_keys[0]) and getitem_arr_tup(left_keys, left_run) == key:
                right_run = right_ind
                while right_run < len(right_keys[0]) and getitem_arr_tup(right_keys, right_run) == key:
                    out_left_key = copy_elem_buff_tup(out_left_key, out_ind, key)
                    l_data_val = getitem_arr_tup(data_left, left_run)
                    out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
                    r_data_val = getitem_arr_tup(data_right, right_run)
                    out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)
                    out_ind += 1
                    right_run += 1
                left_run += 1
            left_ind = left_run
            right_ind = right_run
        elif getitem_arr_tup(left_keys, left_ind) < getitem_arr_tup(right_keys, right_ind):
            if is_left:
                out_left_key = copy_elem_buff_tup(out_left_key, out_ind, getitem_arr_tup(left_keys, left_ind))
                l_data_val = getitem_arr_tup(data_left, left_ind)
                out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
                out_data_right = setnan_elem_buff_tup(out_data_right, out_ind)
                out_ind += 1
            left_ind += 1
        else:
            if is_outer:
                # TODO: support separate keys?
                out_left_key = copy_elem_buff_tup(out_left_key, out_ind, getitem_arr_tup(right_keys, right_ind))
                out_data_left = setnan_elem_buff_tup(out_data_left, out_ind)
                r_data_val = getitem_arr_tup(data_right, right_ind)
                out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)
                out_ind += 1
            right_ind += 1

    if is_left and left_ind < len(left_keys[0]):
        while left_ind < len(left_keys[0]):
            out_left_key = copy_elem_buff_tup(out_left_key, out_ind, getitem_arr_tup(left_keys, left_ind))
            l_data_val = getitem_arr_tup(data_left, left_ind)
            out_data_left = copy_elem_buff_tup(out_data_left, out_ind, l_data_val)
            out_data_right = setnan_elem_buff_tup(out_data_right, out_ind)
            out_ind += 1
            left_ind += 1

    if is_outer and right_ind < len(right_keys[0]):
        while right_ind < len(right_keys[0]):
            out_left_key = copy_elem_buff_tup(out_left_key, out_ind, getitem_arr_tup(right_keys, right_ind))
            out_data_left = setnan_elem_buff_tup(out_data_left, out_ind)
            r_data_val = getitem_arr_tup(data_right, right_ind)
            out_data_right = copy_elem_buff_tup(out_data_right, out_ind, r_data_val)
            out_ind += 1
            right_ind += 1

    #out_left_key = out_left_key[:out_ind]
    out_left_key = trim_arr_tup(out_left_key, out_ind)

    out_right_key = copy_arr_tup(out_left_key)
    out_data_left = trim_arr_tup(out_data_left, out_ind)
    out_data_right = trim_arr_tup(out_data_right, out_ind)

    return out_left_key, out_right_key, out_data_left, out_data_right


@numba.njit
def local_merge_asof(left_keys, right_keys, data_left, data_right):
    # adapted from pandas/_libs/join_func_helper.pxi
    l_size = len(left_keys[0])
    r_size = len(right_keys[0])

    out_left_keys = alloc_arr_tup(l_size, left_keys)
    out_right_keys = alloc_arr_tup(l_size, right_keys)
    out_data_left = alloc_arr_tup(l_size, data_left)
    out_data_right = alloc_arr_tup(l_size, data_right)

    left_ind = 0
    right_ind = 0

    for left_ind in range(l_size):
        # restart right_ind if it went negative in a previous iteration
        if right_ind < 0:
            right_ind = 0

        # find last position in right whose value is less than left's
        while right_ind < r_size and getitem_arr_tup(right_keys, right_ind) <= getitem_arr_tup(left_keys, left_ind):
            right_ind += 1

        right_ind -= 1

        setitem_arr_tup(out_left_keys, left_ind, getitem_arr_tup(left_keys, left_ind))
        # TODO: copy_tup
        setitem_arr_tup(out_data_left, left_ind, getitem_arr_tup(data_left, left_ind))

        if right_ind >= 0:
            setitem_arr_tup(out_right_keys, left_ind, getitem_arr_tup(right_keys, right_ind))
            setitem_arr_tup(out_data_right, left_ind, getitem_arr_tup(data_right, right_ind))
        else:
            setitem_arr_tup_nan(out_right_keys, left_ind)
            setitem_arr_tup_nan(out_data_right, left_ind)

    return out_left_keys, out_right_keys, out_data_left, out_data_right



@lower_builtin(local_merge, types.Literal, types.VarArg(types.Any))
def lower_local_merge(context, builder, sig, args):
    #
    num_left_cols = sig.args[0].literal_value
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


@infer_global(operator.getitem)
class GetItemCBuf(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        c_buff, idx = args
        if isinstance(c_buff, CBufferType):
            if isinstance(idx, types.Integer):
                return signature(types.int32, c_buff, idx)


@lower_builtin(operator.getitem, c_buffer_type, types.intp)
def c_buffer_type_getitem(context, builder, sig, args):
    base_ptr = builder.bitcast(args[0], lir.IntType(32).as_pointer())
    return builder.load(builder.gep(base_ptr, [args[1]], inbounds=True))


def setitem_arr_nan(arr, ind):
    arr[ind] = np.nan

@overload(setitem_arr_nan)
def setitem_arr_nan_overload(arr_t, ind_t):
    if isinstance(arr_t.dtype, types.Float):
        return setitem_arr_nan
    if isinstance(arr_t.dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = arr_t.dtype('NaT')
        def _setnan_impl(arr, ind):
            arr[ind] = nat
        return _setnan_impl
    # TODO: support strings, bools, etc.
    # XXX: set NA values in bool arrays to False
    # FIXME: replace with proper NaN
    if arr_t.dtype == types.bool_:
        def b_set(a, i):
            a[i] = False
        return b_set
    return lambda a, i: None

def setitem_arr_tup_nan(arr_tup, ind):  # pragma: no cover
    for arr in arr_tup:
        arr[ind] = np.nan

@overload(setitem_arr_tup_nan)
def setitem_arr_tup_nan_overload(arr_tup_t, ind_t):
    count = arr_tup_t.count

    func_text = "def f(arr_tup, ind):\n"
    for i in range(count):
        func_text += "  setitem_arr_nan(arr_tup[{}], ind)\n".format(i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {'setitem_arr_nan': setitem_arr_nan}, loc_vars)
    impl = loc_vars['f']
    return impl

def copy_arr_tup(arrs):
    return tuple(a.copy() for a in arrs)

@overload(copy_arr_tup)
def copy_arr_tup_overload(arrs_t):
    count = arrs_t.count
    func_text = "def f(arrs):\n"
    func_text += "  return ({},)\n".format(",".join("arrs[{}].copy()".format(i) for i in range(count)))

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl

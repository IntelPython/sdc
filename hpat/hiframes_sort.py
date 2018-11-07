import numpy as np
import math
from collections import namedtuple
import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes,
                            mk_unique_var)
from numba.typing import signature
from numba.extending import overload
import hpat
import hpat.timsort
from hpat.timsort import getitem_arr_tup
from hpat.utils import _numba_to_c_type_map
from hpat import distributed, distributed_analysis
from hpat.distributed_api import Reduce_Type
from hpat.distributed_analysis import Distribution
from hpat.utils import (debug_prints, empty_like_type, get_ctypes_ptr,
    gen_getitem)
from hpat.str_arr_ext import (string_array_type, to_string_list,
                              cp_str_list_to_array, str_list_to_array,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, del_str, num_total_chars)
from hpat.str_ext import string_type
MIN_SAMPLES = 1000000
#MIN_SAMPLES = 100
samplePointsPerPartitionHint = 20
MPI_ROOT = 0


class Sort(ir.Stmt):
    def __init__(self, df_in, df_out, key_arr, out_key_arr, df_in_vars,
                                                    df_out_vars, inplace, loc):
        # for printing only
        self.df_in = df_in
        self.df_out = df_out
        self.key_arr = key_arr
        self.out_key_arr = out_key_arr
        self.df_in_vars = df_in_vars
        self.df_out_vars = df_out_vars
        self.inplace = inplace
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        in_cols = ""
        for (c, v) in self.df_in_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)
        return "sort: [key: {}] {} [key: {}] {}".format(
            self.key_arr.name, df_in_str, self.out_key_arr.name, df_out_str)


def sort_array_analysis(sort_node, equiv_set, typemap, array_analysis):

    # arrays of input df have same size in first dimension as key array
    col_shape = equiv_set.get_shape(sort_node.key_arr)
    all_shapes = []
    if (typemap[sort_node.key_arr.name] != string_array_type
            and col_shape is not None):
        all_shapes = [col_shape[0]]
    for col_var in sort_node.df_in_vars.values():
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        col_shape = equiv_set.get_shape(col_var)
        if col_shape is not None:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # arrays of output have the same shape (not necessarily the same as input
    # arrays in the parallel case, TODO: fix)
    col_shape = equiv_set.get_shape(sort_node.out_key_arr)
    all_shapes = []
    if (typemap[sort_node.out_key_arr.name] != string_array_type
            and col_shape is not None):
        all_shapes = [col_shape[0]]
    for col_var in sort_node.df_out_vars.values():
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        col_shape = equiv_set.get_shape(col_var)
        if col_shape is not None:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], []


numba.array_analysis.array_analysis_extensions[Sort] = sort_array_analysis


def sort_distributed_analysis(sort_node, array_dists):

    # input columns have same distribution
    in_dist = array_dists[sort_node.key_arr.name]
    for col_var in sort_node.df_in_vars.values():
        in_dist = Distribution(
            min(in_dist.value, array_dists[col_var.name].value))

    # output is 1D_Var due to shuffle, has to meet input dist
    # TODO: set to input dist in inplace case
    out_dist = Distribution(min(in_dist.value, Distribution.OneD_Var.value))
    if sort_node.out_key_arr.name in array_dists:
        out_dist = Distribution(min(out_dist.value,
                                array_dists[sort_node.out_key_arr.name].value))
    for col_var in sort_node.df_out_vars.values():
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value))

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        in_dist = out_dist

    # set dists
    for col_var in sort_node.df_in_vars.values():
        array_dists[col_var.name] = in_dist
    array_dists[sort_node.key_arr.name] = in_dist

    for col_var in sort_node.df_out_vars.values():
        array_dists[col_var.name] = out_dist
    array_dists[sort_node.out_key_arr.name] = out_dist

    # TODO: handle rebalance
    # assert not (in_dist == Distribution.OneD and out_dist == Distribution.OneD_Var)
    return


distributed_analysis.distributed_analysis_extensions[Sort] = sort_distributed_analysis

def sort_typeinfer(sort_node, typeinferer):
    # input and output arrays have the same type
    typeinferer.constraints.append(typeinfer.Propagate(
        dst=sort_node.out_key_arr.name, src=sort_node.key_arr.name,
        loc=sort_node.loc))
    for col_name, col_var in sort_node.df_in_vars.items():
        out_col_var = sort_node.df_out_vars[col_name]
        typeinferer.constraints.append(typeinfer.Propagate(
            dst=out_col_var.name, src=col_var.name, loc=sort_node.loc))
    return

typeinfer.typeinfer_extensions[Sort] = sort_typeinfer


def visit_vars_sort(sort_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting sort vars for:", sort_node)
        print("cbdata: ", sorted(cbdata.items()))

    sort_node.key_arr = visit_vars_inner(
        sort_node.key_arr, callback, cbdata)
    sort_node.out_key_arr = visit_vars_inner(
        sort_node.out_key_arr, callback, cbdata)

    for col_name in list(sort_node.df_in_vars.keys()):
        sort_node.df_in_vars[col_name] = visit_vars_inner(
            sort_node.df_in_vars[col_name], callback, cbdata)

    for col_name in list(sort_node.df_out_vars.keys()):
        sort_node.df_out_vars[col_name] = visit_vars_inner(
            sort_node.df_out_vars[col_name], callback, cbdata)

# add call to visit sort variable
ir_utils.visit_vars_extensions[Sort] = visit_vars_sort


def remove_dead_sort(sort_node, lives, arg_aliases, alias_map, func_ir, typemap):
    # TODO: remove this feature
    if not hpat.hiframes_api.enable_hiframes_remove_dead:
        return sort_node

    # TODO: arg aliases for inplace case?
    dead_cols = []

    for col_name, col_var in sort_node.df_out_vars.items():
        if col_var.name not in lives:
            dead_cols.append(col_name)

    for cname in dead_cols:
        sort_node.df_in_vars.pop(cname)
        sort_node.df_out_vars.pop(cname)

    # remove empty sort node
    if (len(sort_node.df_out_vars) == 0
            and sort_node.out_key_arr.name not in lives):
        return None

    return sort_node


ir_utils.remove_dead_extensions[Sort] = remove_dead_sort


def sort_usedefs(sort_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # key array and input columns are used
    use_set.add(sort_node.key_arr.name)
    use_set.update({v.name for v in sort_node.df_in_vars.values()})

    # output arrays are defined
    if not sort_node.inplace:
        def_set.add(sort_node.out_key_arr.name)
        def_set.update({v.name for v in sort_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Sort] = sort_usedefs


def get_copies_sort(sort_node, typemap):
    # sort doesn't generate copies, it just kills the output columns
    kill_set = set()
    if not sort_node.inplace:
        kill_set = set(v.name for v in sort_node.df_out_vars.values())
        kill_set.add(sort_node.out_key_arr.name)
    return set(), kill_set

ir_utils.copy_propagate_extensions[Sort] = get_copies_sort


def apply_copies_sort(sort_node, var_dict, name_var_table,
                        typemap, calltypes, save_copies):
    """apply copy propagate in sort node"""
    sort_node.key_arr = replace_vars_inner(sort_node.key_arr, var_dict)
    sort_node.out_key_arr = replace_vars_inner(sort_node.out_key_arr, var_dict)

    for col_name in list(sort_node.df_in_vars.keys()):
        sort_node.df_in_vars[col_name] = replace_vars_inner(
            sort_node.df_in_vars[col_name], var_dict)

    for col_name in list(sort_node.df_out_vars.keys()):
        sort_node.df_out_vars[col_name] = replace_vars_inner(
            sort_node.df_out_vars[col_name], var_dict)

    return

ir_utils.apply_copy_propagate_extensions[Sort] = apply_copies_sort


def sort_distributed_run(sort_node, array_dists, typemap, calltypes, typingctx,
                                                         targetctx, dist_pass):
    parallel = True
    in_vars = list(sort_node.df_in_vars.values())
    out_vars = list(sort_node.df_out_vars.values())
    for v in [sort_node.key_arr, sort_node.out_key_arr] + in_vars + out_vars:
        if (array_dists[v.name] != distributed.Distribution.OneD
                and array_dists[v.name] != distributed.Distribution.OneD_Var):
            parallel = False

    loc = sort_node.loc
    scope = sort_node.key_arr.scope
    # copy arrays when not inplace
    nodes = []
    key_arr = sort_node.key_arr
    if not sort_node.inplace:
        key_arr = _copy_array_nodes(key_arr, nodes, typingctx, typemap,
                                                                     calltypes)
        new_in_vars = []
        for v in in_vars:
            v_cp = _copy_array_nodes(v, nodes, typingctx, typemap, calltypes)
            new_in_vars.append(v_cp)
        in_vars = new_in_vars


    col_name_args = ', '.join(["c"+str(i) for i in range(len(in_vars))])
    # TODO: use *args
    func_text = "def f(key_arr, {}):\n".format(col_name_args)
    func_text += "  data = ({}{})\n".format(col_name_args,
        "," if len(in_vars) == 1 else "")  # single value needs comma to become tuple
    func_text += "  local_sort_f(key_arr, data)\n"
    func_text += "  return key_arr, data\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sort_impl = loc_vars['f']

    key_typ = typemap[key_arr.name]
    data_tup_typ = types.Tuple([typemap[v.name] for v in in_vars])
    _local_sort_f = get_local_sort_func(key_typ, data_tup_typ)

    f_block = compile_to_numba_ir(sort_impl,
                                    {'hpat': hpat,
                                    'local_sort_f': _local_sort_f,
                                    'to_string_list': to_string_list,
                                    'cp_str_list_to_array': cp_str_list_to_array},
                                    typingctx,
                                    tuple([key_typ] + list(data_tup_typ.types)),
                                    typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [key_arr] + in_vars)
    nodes += f_block.body[:-2]
    ret_var = nodes[-1].target
    # get key
    key_arr = ir.Var(scope, mk_unique_var(key_arr.name), loc)
    typemap[key_arr.name] = key_typ
    gen_getitem(key_arr, ret_var, 0, calltypes, nodes)
    # get data tup
    data_tup_var = ir.Var(scope, mk_unique_var('sort_data'), loc)
    typemap[data_tup_var.name] = data_tup_typ
    gen_getitem(data_tup_var, ret_var, 1, calltypes, nodes)

    if not parallel:
        nodes.append(ir.Assign(key_arr, sort_node.out_key_arr, loc))
        for i, var in enumerate(out_vars):
            gen_getitem(var, data_tup_var, i, calltypes, nodes)
        return nodes

    # parallel case
    def par_sort_impl(key_arr, data):
        out_key, out_data = parallel_sort(key_arr, data)
        # TODO: use k-way merge instead of sort
        # sort output
        local_sort_f(out_key, out_data)
        return out_key, out_data

    f_block = compile_to_numba_ir(par_sort_impl,
                                    {'hpat': hpat,
                                    'parallel_sort': parallel_sort,
                                    'to_string_list': to_string_list,
                                    'cp_str_list_to_array': cp_str_list_to_array,
                                    'local_sort_f': _local_sort_f},
                                    typingctx,
                                    (key_typ, data_tup_typ),
                                    typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [key_arr, data_tup_var])
    nodes += f_block.body[:-2]
    ret_var = nodes[-1].target
    # get output key
    gen_getitem(sort_node.out_key_arr, ret_var, 0, calltypes, nodes)
    # get data tup
    data_tup = ir.Var(scope, mk_unique_var('sort_data'), loc)
    typemap[data_tup.name] = data_tup_typ
    gen_getitem(data_tup, ret_var, 1, calltypes, nodes)

    for i, var in enumerate(out_vars):
        gen_getitem(var, data_tup, i, calltypes, nodes)

    # TODO: handle 1D balance for inplace case

    return nodes


distributed.distributed_run_extensions[Sort] = sort_distributed_run

def _copy_array_nodes(var, nodes, typingctx, typemap, calltypes):
    def _impl(arr):
        return arr.copy()

    f_block = compile_to_numba_ir(_impl, {}, typingctx, (typemap[var.name],),
                                    typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [var])
    nodes += f_block.body[:-2]
    return nodes[-1].target

def to_string_list_typ(typ):
    if typ == string_array_type:
        return types.List(hpat.str_ext.string_type)

    if isinstance(typ, (types.Tuple, types.UniTuple)):
        new_typs = []
        for i in range(typ.count):
            new_typs.append(to_string_list_typ(typ.types[i]))
        return types.Tuple(new_typs)

    return typ

def get_local_sort_func(key_typ, data_tup_typ):
    sort_state_spec = [
        ('key_arr', to_string_list_typ(key_typ)),
        ('aLength', numba.intp),
        ('minGallop', numba.intp),
        ('tmpLength', numba.intp),
        ('tmp', to_string_list_typ(key_typ)),
        ('stackSize', numba.intp),
        ('runBase', numba.int64[:]),
        ('runLen', numba.int64[:]),
        ('data', to_string_list_typ(data_tup_typ)),
        ('tmp_data', to_string_list_typ(data_tup_typ)),
    ]
    SortStateCL = numba.jitclass(sort_state_spec)(hpat.timsort.SortState)

    # XXX: make sure function is not using old SortState
    local_sort.__globals__['SortState'] = SortStateCL
    _local_sort_f = numba.njit(local_sort)
    _local_sort_f.compile(signature(types.none, key_typ, data_tup_typ))
    return _local_sort_f


def local_sort(key_arr, data):
    # convert StringArray to list(string) to enable swapping in sort
    l_key_arr = to_string_list(key_arr)
    l_data = to_string_list(data)
    n_out = len(key_arr)
    sort_state_o = SortState(l_key_arr, n_out, l_data)
    hpat.timsort.sort(sort_state_o, l_key_arr, 0, n_out, l_data)
    cp_str_list_to_array(key_arr, l_key_arr)
    cp_str_list_to_array(data, l_data)


@numba.njit
def parallel_sort(key_arr, data):
    n_local = len(key_arr)
    n_total = hpat.distributed_api.dist_reduce(n_local, np.int32(Reduce_Type.Sum.value))

    n_pes = hpat.distributed_api.get_size()
    my_rank = hpat.distributed_api.get_rank()

    # similar to Spark's sample computation Partitioner.scala
    sampleSize = min(samplePointsPerPartitionHint * n_pes, MIN_SAMPLES)

    fraction = min(sampleSize / max(n_total, 1), 1.0)
    n_loc_samples = min(math.ceil(fraction * n_local), n_local)
    inds = np.random.randint(0, n_local, n_loc_samples)
    samples = key_arr[inds]
    # print(sampleSize, fraction, n_local, n_loc_samples, len(samples))

    all_samples = hpat.distributed_api.gatherv(samples)
    all_samples = to_string_list(all_samples)
    bounds = empty_like_type(n_pes-1, all_samples)

    if my_rank == MPI_ROOT:
        all_samples.sort()
        n_samples = len(all_samples)
        step = math.ceil(n_samples / n_pes)
        for i in range(n_pes - 1):
            bounds[i] = all_samples[min((i + 1) * step, n_samples - 1)]
        # print(bounds)

    bounds = str_list_to_array(bounds)
    bounds = hpat.distributed_api.prealloc_str_for_bcast(bounds)
    hpat.distributed_api.bcast(bounds)

    # calc send/recv counts
    key_arrs = (key_arr,)
    pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, data, n_pes, True)
    node_id = 0
    for i in range(n_local):
        val = key_arr[i]
        if node_id < (n_pes - 1) and val >= bounds[node_id]:
            node_id += 1
        update_shuffle_meta(pre_shuffle_meta, node_id, i, (val,),
            getitem_arr_tup(data, i), True)

    shuffle_meta = finalize_shuffle_meta(key_arrs, data, pre_shuffle_meta,
                                          n_pes, True)

    # shuffle
    recvs = alltoallv_tup(key_arrs + data, shuffle_meta)
    out_key, = _get_keys_tup(recvs, key_arrs)
    out_data = _get_data_tup(recvs, key_arrs)

    return out_key, out_data

def alloc_shuffle_metadata():
    pass
def data_alloc_shuffle_metadata():
    pass

def finalize_data_shuffle_meta():
    pass

def update_data_shuffle_meta():
    pass


########## metadata required for shuffle
# send_counts -> pre, single
# recv_counts -> single
# send_buff
# out_arr
# n_send  -> single
# n_out  -> single
# send_disp -> single
# recv_disp -> single
# tmp_offset -> single
############### string arrays
# send_counts_char -> pre
# recv_counts_char
# send_arr_lens -> pre
# send_arr_chars
# send_disp_char
# recv_disp_char
# tmp_offset_char
#### dummy array to key reference count alive, since ArrayCTypes can't be
#### passed to jitclass TODO: update
# send_arr_chars_arr


PreShuffleMeta = namedtuple('PreShuffleMeta',
    'send_counts, send_counts_char_tup, send_arr_lens_tup')

ShuffleMeta = namedtuple('ShuffleMeta',
    ('send_counts, recv_counts, n_send, n_out, send_disp, recv_disp, '
    'tmp_offset, send_buff_tup, out_arr_tup, send_counts_char_tup, '
    'recv_counts_char_tup, send_arr_lens_tup, send_arr_chars_tup, '
    'send_disp_char_tup, recv_disp_char_tup, tmp_offset_char_tup, '
    'send_arr_chars_arr_tup'))


# before shuffle, 'send_counts' is needed as well as
# 'send_counts_char' and 'send_arr_lens' for every string type
def alloc_pre_shuffle_metadata(arr, data, n_pes, is_contig):
    return PreShuffleMeta(np.zeros(n_pes, np.int32), ())

@overload(alloc_pre_shuffle_metadata)
def alloc_pre_shuffle_metadata_overload(key_arrs_t, data_t, n_pes_t, is_contig_t):

    func_text = "def f(key_arrs, data, n_pes, is_contig):\n"
    # send_counts
    func_text += "  send_counts = np.zeros(n_pes, np.int32)\n"

    # send_counts_char, send_arr_lens for strings
    n_keys = len(key_arrs_t.types)
    n_str = 0
    for i, typ in enumerate(key_arrs_t.types + data_t.types):
        if typ == string_array_type:
            func_text += ("  arr = key_arrs[{}]\n".format(i) if i < n_keys
                else "  arr = data[{}]".format(i - n_keys))
            func_text += "  send_counts_char_{} = np.zeros(n_pes, np.int32)\n".format(n_str)
            func_text += "  send_arr_lens_{} = np.empty(1, np.uint32)\n".format(n_str)
            # needs allocation since written in update before finalize
            func_text += "  if is_contig:\n"
            func_text += "    send_arr_lens_{} = np.empty(len(arr), np.uint32)\n".format(n_str)
            n_str += 1

    count_char_tup = ", ".join("send_counts_char_{}".format(i)
                                                        for i in range(n_str))
    lens_tup = ", ".join("send_arr_lens_{}".format(i) for i in range(n_str))
    extra_comma = "," if n_str == 1 else ""
    func_text += "  return PreShuffleMeta(send_counts, ({}{}), ({}{}))\n".format(
        count_char_tup, extra_comma, lens_tup, extra_comma)

    # print(func_text)

    loc_vars = {}
    exec(func_text, {'np':np, 'PreShuffleMeta': PreShuffleMeta}, loc_vars)
    alloc_impl = loc_vars['f']
    return alloc_impl



# 'send_counts' is updated, and 'send_counts_char' and 'send_arr_lens'
# for every string type
def update_shuffle_meta(pre_shuffle_meta, node_id, ind, val, data, is_contig=True):
    pre_shuffle_meta.send_counts[node_id] += 1

@overload(update_shuffle_meta)
def update_shuffle_meta_overload(meta_t, node_id_t, ind_t, val_t, data_t, is_contig_t=None):
    func_text = "def f(pre_shuffle_meta, node_id, ind, val, data, is_contig=True):\n"
    func_text += "  pre_shuffle_meta.send_counts[node_id] += 1\n"
    n_keys = len(val_t.types)
    n_str = 0
    for i, typ in enumerate(val_t.types + data_t.types):
        if typ in (string_type, string_array_type):
            val_or_data = 'val[{}]'.format(i) if i < n_keys else 'data[{}]'.format(i - n_keys)
            func_text += "  n_chars = len({})\n".format(val_or_data)
            func_text += "  pre_shuffle_meta.send_counts_char_tup[{}][node_id] += n_chars\n".format(n_str)
            func_text += "  if is_contig:\n"
            func_text += "    pre_shuffle_meta.send_arr_lens_tup[{}][ind] = n_chars\n".format(n_str)
            n_str += 1

    # print(func_text)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    update_impl = loc_vars['f']
    return update_impl



def finalize_shuffle_meta(arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):
    return ShuffleMeta()

@overload(finalize_shuffle_meta)
def finalize_shuffle_meta_overload(key_arrs_t, data_t, pre_shuffle_meta_t, n_pes_t, is_contig_t, init_vals_t=None):

    func_text = "def f(key_arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):\n"
    # common metas: send_counts, recv_counts, tmp_offset, n_out, n_send, send_disp, recv_disp
    func_text += "  send_counts = pre_shuffle_meta.send_counts\n"
    func_text += "  recv_counts = np.empty(n_pes, np.int32)\n"
    func_text += "  tmp_offset = np.zeros(n_pes, np.int32)\n"  # for non-contig
    func_text += "  hpat.distributed_api.alltoall(send_counts, recv_counts, 1)\n"
    func_text += "  n_out = recv_counts.sum()\n"
    func_text += "  n_send = send_counts.sum()\n"
    func_text += "  send_disp = hpat.hiframes_join.calc_disp(send_counts)\n"
    func_text += "  recv_disp = hpat.hiframes_join.calc_disp(recv_counts)\n"

    n_keys = len(key_arrs_t.types)
    n_all = len(key_arrs_t.types + data_t.types)
    n_str = 0

    for i, typ in enumerate(key_arrs_t.types + data_t.types):
        func_text += ("  arr = key_arrs[{}]\n".format(i) if i < n_keys
                      else "  arr = data[{}]\n".format(i - n_keys))
        if isinstance(typ, types.Array):
            func_text += "  out_arr_{} = np.empty(n_out, arr.dtype)\n".format(i)
            func_text += "  send_buff_{} = arr\n".format(i)
            func_text += "  if not is_contig:\n"
            if i >= n_keys and init_vals_t is not None:
                func_text += "    send_buff_{} = np.full(n_send, init_vals[{}], arr.dtype)\n".format(i, i - n_keys)
            else:
                func_text += "    send_buff_{} = np.empty(n_send, arr.dtype)\n".format(i)
        else:
            assert typ == string_array_type
            # send_buff is None for strings
            func_text += "  send_buff_{} = None\n".format(i)
            # send/recv counts
            func_text += "  send_counts_char_{} = pre_shuffle_meta.send_counts_char_tup[{}]\n".format(n_str, n_str)
            func_text += "  recv_counts_char_{} = np.empty(n_pes, np.int32)\n".format(n_str)
            func_text += ("  hpat.distributed_api.alltoall("
                "send_counts_char_{}, recv_counts_char_{}, 1)\n").format(n_str, n_str)
            # alloc output
            func_text += "  n_all_chars = recv_counts_char_{}.sum()\n".format(n_str)
            func_text += "  out_arr_{} = pre_alloc_string_array(n_out, n_all_chars)\n".format(i)
            # send/recv disp
            func_text += ("  send_disp_char_{} = hpat.hiframes_join."
                "calc_disp(send_counts_char_{})\n").format(n_str, n_str)
            func_text += ("  recv_disp_char_{} = hpat.hiframes_join."
                "calc_disp(recv_counts_char_{})\n").format(n_str, n_str)

            # tmp_offset_char, send_arr_lens
            func_text += "  tmp_offset_char_{} = np.zeros(n_pes, np.int32)\n".format(n_str)
            func_text += "  send_arr_lens_{} = pre_shuffle_meta.send_arr_lens_tup[{}]\n".format(n_str, n_str)
            # send char arr
            # TODO: arr refcount if arr is not stored somewhere?
            func_text += "  send_arr_chars_arr_{} = np.empty(1, np.uint8)\n".format(i)
            func_text += "  send_arr_chars_{} = get_ctypes_ptr(get_data_ptr(arr))\n".format(i)
            func_text += "  if not is_contig:\n"
            func_text += "    s_n_all_chars = send_counts_char_{}.sum()\n".format(n_str)
            func_text += "    send_arr_chars_arr_{} = np.empty(s_n_all_chars, np.uint8)\n".format(n_str)
            func_text += "    send_arr_chars_{} = get_ctypes_ptr(send_arr_chars_arr_{}.ctypes)\n".format(n_str, n_str)
            n_str += 1


    send_buffs = ", ".join("send_buff_{}".format(i) for i in range(n_all))
    out_arrs = ", ".join("out_arr_{}".format(i) for i in range(n_all))
    all_comma = "," if n_all == 1 else ""
    send_counts_chars = ", ".join("send_counts_char_{}".format(i) for i in range(n_str))
    recv_counts_chars = ", ".join("recv_counts_char_{}".format(i) for i in range(n_str))
    send_arr_lens = ", ".join("send_arr_lens_{}".format(i) for i in range(n_str))
    send_arr_chars = ", ".join("send_arr_chars_{}".format(i) for i in range(n_str))
    send_disp_chars = ", ".join("send_disp_char_{}".format(i) for i in range(n_str))
    recv_disp_chars = ", ".join("recv_disp_char_{}".format(i) for i in range(n_str))
    tmp_offset_chars = ", ".join("tmp_offset_char_{}".format(i) for i in range(n_str))
    send_arr_chars_arrs = ", ".join("send_arr_chars_arr_{}".format(i) for i in range(n_str))
    str_comma = "," if n_str == 1 else ""


    func_text += ('  return ShuffleMeta(send_counts, recv_counts, n_send, '
        'n_out, send_disp, recv_disp, tmp_offset, ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), )\n').format(
            send_buffs, all_comma, out_arrs, all_comma, send_counts_chars, str_comma, recv_counts_chars, str_comma,
            send_arr_lens, str_comma, send_arr_chars, str_comma, send_disp_chars, str_comma, recv_disp_chars, str_comma,
            tmp_offset_chars, str_comma, send_arr_chars_arrs, str_comma
        )

    # print(func_text)

    loc_vars = {}
    exec(func_text, {'np': np, 'hpat': hpat,
         'pre_alloc_string_array': pre_alloc_string_array,
         'num_total_chars': num_total_chars,
         'get_data_ptr': get_data_ptr,
         'ShuffleMeta': ShuffleMeta,
         'get_ctypes_ptr': get_ctypes_ptr}, loc_vars)
    finalize_impl = loc_vars['f']
    return finalize_impl



def alltoallv(arr, m):
    return

@overload(alltoallv)
def alltoallv_impl(arr_t, metadata_t):
    if isinstance(arr_t, types.Array):
        def a2av_impl(arr, metadata):
            hpat.distributed_api.alltoallv(
                metadata.send_buff, metadata.out_arr, metadata.send_counts,
                metadata.recv_counts, metadata.send_disp, metadata.recv_disp)
        return a2av_impl

    assert arr_t == string_array_type
    int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
    char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])
    def a2av_str_impl(arr, metadata):
        # TODO: increate refcount?
        offset_ptr = get_offset_ptr(metadata.out_arr)
        hpat.distributed_api.c_alltoallv(
            metadata.send_arr_lens.ctypes, offset_ptr, metadata.send_counts.ctypes,
            metadata.recv_counts.ctypes, metadata.send_disp.ctypes, metadata.recv_disp.ctypes, int32_typ_enum)
        hpat.distributed_api.c_alltoallv(
            metadata.send_arr_chars, get_data_ptr(metadata.out_arr), metadata.send_counts_char.ctypes,
            metadata.recv_counts_char.ctypes, metadata.send_disp_char.ctypes, metadata.recv_disp_char.ctypes, char_typ_enum)
        convert_len_arr_to_offset(offset_ptr, metadata.n_out)
    return a2av_str_impl



def alltoallv_tup(arrs, shuffle_meta):
    return arrs

@overload(alltoallv_tup)
def alltoallv_tup_overload(arrs_t, shuffle_meta_t):
    func_text = "def f(arrs, meta):\n"
    n_str = 0
    for i, typ in enumerate(arrs_t.types):
        if isinstance(typ, types.Array):
            func_text += ("  hpat.distributed_api.alltoallv("
                "meta.send_buff_tup[{}], meta.out_arr_tup[{}], meta.send_counts,"
                "meta.recv_counts, meta.send_disp, meta.recv_disp)\n").format(i, i)
        else:
            assert typ == string_array_type
            func_text += "  offset_ptr_{} = get_offset_ptr(meta.out_arr_tup[{}])\n".format(i, i)

            func_text += ("  hpat.distributed_api.c_alltoallv("
                "meta.send_arr_lens_tup[{}].ctypes, offset_ptr_{}, meta.send_counts.ctypes, "
                "meta.recv_counts.ctypes, meta.send_disp.ctypes, "
                "meta.recv_disp.ctypes, int32_typ_enum)\n").format(n_str, i)

            func_text += ("  hpat.distributed_api.c_alltoallv("
                "meta.send_arr_chars_tup[{}], get_data_ptr(meta.out_arr_tup[{}]), meta.send_counts_char_tup[{}].ctypes,"
                "meta.recv_counts_char_tup[{}].ctypes, meta.send_disp_char_tup[{}].ctypes,"
                "meta.recv_disp_char_tup[{}].ctypes, char_typ_enum)\n").format(n_str, i, n_str, n_str, n_str, n_str)

            func_text += "  convert_len_arr_to_offset(offset_ptr_{}, meta.n_out)\n".format(i)
            n_str += 1

    func_text += "  return ({}{})\n".format(
        ','.join(['meta.out_arr_tup[{}]'.format(i) for i in range(arrs_t.count)]),
        "," if arrs_t.count == 1 else "")

    int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
    char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])
    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'get_offset_ptr': get_offset_ptr,
         'get_data_ptr': get_data_ptr, 'int32_typ_enum': int32_typ_enum,
         'char_typ_enum': char_typ_enum,
         'convert_len_arr_to_offset': convert_len_arr_to_offset}, loc_vars)
    a2a_impl = loc_vars['f']
    return a2a_impl


def _get_keys_tup(recvs, key_arrs):
    return recvs[:len(key_arrs)]

@overload(_get_keys_tup)
def _get_keys_tup_overload(recvs_t, key_arrs_t):
    n_keys = len(key_arrs_t.types)
    func_text = "def f(recvs, key_arrs):\n"
    res = ",".join("recvs[{}]".format(i) for i in range(n_keys))
    func_text += "  return ({}{})\n".format(res, "," if n_keys==1 else "")
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl


def _get_data_tup(recvs, key_arrs):
    return recvs[len(key_arrs):]

@overload(_get_data_tup)
def _get_data_tup_overload(recvs_t, key_arrs_t):
    n_keys = len(key_arrs_t.types)
    n_all = len(recvs_t.types)
    n_data = n_all - n_keys
    func_text = "def f(recvs, key_arrs):\n"
    res = ",".join("recvs[{}]".format(i) for i in range(n_keys, n_all))
    func_text += "  return ({}{})\n".format(res, "," if n_data==1 else "")
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl

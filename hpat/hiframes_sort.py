import numpy as np
import math
from collections import namedtuple
import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes)
from numba.typing import signature
from numba.extending import overload
import hpat
import hpat.timsort
from hpat import distributed, distributed_analysis
from hpat.distributed_api import Reduce_Type, _h5_typ_table
from hpat.distributed_analysis import Distribution
from hpat.utils import debug_prints, empty_like_type
from hpat.str_arr_ext import (string_array_type, to_string_list,
                              cp_str_list_to_array, str_list_to_array,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, del_str, num_total_chars)

MIN_SAMPLES = 1000000
#MIN_SAMPLES = 100
samplePointsPerPartitionHint = 20
MPI_ROOT = 0


class Sort(ir.Stmt):
    def __init__(self, df_in, key_arr, df_vars, loc):
        self.df_in = df_in
        self.df_vars = df_vars
        self.key_arr = key_arr
        self.loc = loc

    def __repr__(self):  # pragma: no cover
        in_cols = ""
        for (c, v) in self.df_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        return "sort: [key: {}] {}".format(self.key_arr, df_in_str)


def sort_array_analysis(sort_node, equiv_set, typemap, array_analysis):

    # arrays of input df have same size in first dimension as key array
    col_shape = equiv_set.get_shape(sort_node.key_arr)
    if typemap[sort_node.key_arr.name] == string_array_type:
        all_shapes = []
    else:
        all_shapes = [col_shape[0]]
    for col_var in sort_node.df_vars.values():
        typ = typemap[col_var.name]
        if typ == string_array_type:
            continue
        col_shape = equiv_set.get_shape(col_var)
        all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], []


numba.array_analysis.array_analysis_extensions[Sort] = sort_array_analysis


def sort_distributed_analysis(sort_node, array_dists):

    # input columns have same distribution
    in_dist = array_dists[sort_node.key_arr.name]
    for col_var in sort_node.df_vars.values():
        in_dist = Distribution(
            min(in_dist.value, array_dists[col_var.name].value))

    # set dists
    for col_var in sort_node.df_vars.values():
        array_dists[col_var.name] = in_dist
    array_dists[sort_node.key_arr.name] = in_dist
    return


distributed_analysis.distributed_analysis_extensions[Sort] = sort_distributed_analysis

def sort_typeinfer(sort_node, typeinferer):
    # no need for inference since sort just uses arrays without creating any
    return

typeinfer.typeinfer_extensions[Sort] = sort_typeinfer


def visit_vars_sort(sort_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting sort vars for:", sort_node)
        print("cbdata: ", sorted(cbdata.items()))

    sort_node.key_arr = visit_vars_inner(
        sort_node.key_arr, callback, cbdata)

    for col_name in list(sort_node.df_vars.keys()):
        sort_node.df_vars[col_name] = visit_vars_inner(
            sort_node.df_vars[col_name], callback, cbdata)

# add call to visit sort variable
ir_utils.visit_vars_extensions[Sort] = visit_vars_sort


def remove_dead_sort(sort_node, lives, arg_aliases, alias_map, func_ir, typemap):
    #
    dead_cols = []

    for col_name, col_var in sort_node.df_vars.items():
        if col_var.name not in lives:
            dead_cols.append(col_name)

    for cname in dead_cols:
        sort_node.df_vars.pop(cname)

    # remove empty sort node
    if len(sort_node.df_vars) == 0 and sort_node.key_arr.name not in lives:
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
    use_set.update({v.name for v in sort_node.df_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Sort] = sort_usedefs


def get_copies_sort(sort_node, typemap):
    # sort doesn't generate copies
    return set(), set()

ir_utils.copy_propagate_extensions[Sort] = get_copies_sort


def apply_copies_sort(sort_node, var_dict, name_var_table,
                        typemap, calltypes, save_copies):
    """apply copy propagate in sort node"""
    sort_node.key_arr = replace_vars_inner(sort_node.key_arr, var_dict)

    for col_name in list(sort_node.df_vars.keys()):
        sort_node.df_vars[col_name] = replace_vars_inner(
            sort_node.df_vars[col_name], var_dict)

    return

ir_utils.apply_copy_propagate_extensions[Sort] = apply_copies_sort


def sort_distributed_run(sort_node, array_dists, typemap, calltypes, typingctx, targetctx):
    parallel = True
    data_vars = list(sort_node.df_vars.values())
    for v in [sort_node.key_arr] + data_vars:
        if (array_dists[v.name] != distributed.Distribution.OneD
                and array_dists[v.name] != distributed.Distribution.OneD_Var):
            parallel = False

    key_arr = sort_node.key_arr

    col_name_args = ', '.join(["c"+str(i) for i in range(len(data_vars))])
    # TODO: use *args
    func_text = "def f(key_arr, {}):\n".format(col_name_args)
    func_text += "  data = ({}{})\n".format(col_name_args,
        "," if len(data_vars) == 1 else "")  # single value needs comma to become tuple
    func_text += "  local_sort_f(key_arr, data)\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sort_impl = loc_vars['f']

    key_typ = typemap[key_arr.name]
    data_tup_typ = types.Tuple([typemap[v.name] for v in sort_node.df_vars.values()])
    _local_sort_f = get_local_sort_func(key_typ, data_tup_typ)

    f_block = compile_to_numba_ir(sort_impl,
                                    {'hpat': hpat,
                                    'local_sort_f': _local_sort_f,
                                    'to_string_list': to_string_list,
                                    'cp_str_list_to_array': cp_str_list_to_array},
                                    typingctx,
                                    tuple([key_typ] + list(data_tup_typ.types)),
                                    typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [sort_node.key_arr] + data_vars)
    nodes = f_block.body[:-3]

    if not parallel:
        return nodes

    # parallel case
    # TODO: refactor with previous call, use *args?
    # get data variable tuple
    func_text = "def f({}):\n".format(col_name_args)
    func_text += "  data = ({}{})\n".format(col_name_args,
        "," if len(data_vars) == 1 else "")  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    tup_impl = loc_vars['f']
    f_block = compile_to_numba_ir(tup_impl,
                                    {},
                                    typingctx,
                                    list(data_tup_typ.types),
                                    typemap, calltypes).blocks.popitem()[1]

    replace_arg_nodes(f_block, data_vars)
    nodes += f_block.body[:-3]
    data_tup_var = nodes[-1].target

    def par_sort_impl(key_arr, data):
        out, out_data = parallel_sort(key_arr, data)
        # TODO: use k-way merge instead of sort
        # sort output
        local_sort_f(out, out_data)
        res_data = out_data
        res = out

    f_block = compile_to_numba_ir(par_sort_impl,
                                    {'hpat': hpat,
                                    'parallel_sort': parallel_sort,
                                    'to_string_list': to_string_list,
                                    'cp_str_list_to_array': cp_str_list_to_array,
                                    'local_sort_f': _local_sort_f},
                                    typingctx,
                                    (key_typ, data_tup_typ),
                                    typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [sort_node.key_arr, data_tup_var])
    nodes += f_block.body[:-3]
    # set vars since new arrays are created after communication
    data_tup = nodes[-2].target
    # key
    nodes.append(ir.Assign(nodes[-1].target, sort_node.key_arr, sort_node.key_arr.loc))

    for i, var in enumerate(data_vars):
        getitem = ir.Expr.static_getitem(data_tup, i, None, var.loc)
        calltypes[getitem] = None
        nodes.append(ir.Assign(getitem, var, var.loc))

    return nodes


distributed.distributed_run_extensions[Sort] = sort_distributed_run


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
    shuffle_meta = alloc_shuffle_metadata(key_arr, n_pes, True)
    data_shuffle_meta = data_alloc_shuffle_metadata(data, n_pes, True)
    node_id = 0
    for i in range(n_local):
        val = key_arr[i]
        if node_id < (n_pes - 1) and val >= bounds[node_id]:
            node_id += 1
        update_shuffle_meta(shuffle_meta, node_id, i, val)
        update_data_shuffle_meta(data_shuffle_meta, node_id, i, data)

    finalize_shuffle_meta(key_arr, shuffle_meta)
    finalize_data_shuffle_meta(data, data_shuffle_meta, shuffle_meta)

    # shuffle
    alltoallv(key_arr, shuffle_meta)
    out_data = alltoallv_tup(data, data_shuffle_meta, shuffle_meta)

    return shuffle_meta.out_arr, out_data

# ShuffleMeta = namedtuple('ShuffleMeta',
#     ['send_counts', 'recv_counts', 'out_arr', 'n_out', 'send_disp', 'recv_disp', 'send_counts_char',
#     'recv_counts_char', 'send_arr_lens', 'send_arr_chars'])

class ShuffleMeta:
    def __init__(self, send_counts, recv_counts, send_buff, out_arr, n_out, send_disp, recv_disp, tmp_offset, send_counts_char,
            recv_counts_char, send_arr_lens, send_arr_chars, send_disp_char, recv_disp_char, tmp_offset_chars):
        self.send_counts = send_counts
        self.recv_counts = recv_counts
        self.send_buff = send_buff
        self.out_arr = out_arr
        self.n_out = n_out
        self.send_disp = send_disp
        self.recv_disp = recv_disp
        self.tmp_offset = tmp_offset
        # string arrays
        self.send_counts_char = send_counts_char
        self.recv_counts_char = recv_counts_char
        self.send_arr_lens = send_arr_lens
        self.send_arr_chars = send_arr_chars
        self.send_disp_char = send_disp_char
        self.recv_disp_char = recv_disp_char
        self.tmp_offset_chars = tmp_offset_chars


def update_shuffle_meta(shuffle_meta, node_id, ind, val, is_contig=True):
    shuffle_meta.send_counts[node_id] += 1

@overload(update_shuffle_meta)
def update_shuffle_meta_overload(meta_t, node_id_t, ind_t, val_t, is_contig_t=None):
    arr_t = meta_t.struct['out_arr']
    if isinstance(arr_t, types.Array):
        def update_impl(shuffle_meta, node_id, ind, val, is_contig=True):
            shuffle_meta.send_counts[node_id] += 1
        return update_impl
    assert arr_t == string_array_type
    def update_str_impl(shuffle_meta, node_id, ind, val, is_contig=True):
        n_chars = len(val)
        shuffle_meta.send_counts[node_id] += 1
        shuffle_meta.send_counts_char[node_id] += n_chars
        if is_contig:
            shuffle_meta.send_arr_lens[ind] = n_chars
        del_str(val)

    return update_str_impl

def alloc_shuffle_metadata(arr, n_pes, contig):
    return ShuffleMeta(np.zeros(1), np.zeros(1), arr, arr, n_pes, np.zeros(1),
        np.zeros(1), np.zeros(1), None, None, None, None, None, None, None)

@overload(alloc_shuffle_metadata)
def alloc_shuffle_metadata_overload(arr_t, n_pes_t, is_contig_t):
    if isinstance(arr_t, types.Array):
        ShuffleMetaCL = get_shuffle_meta_class(arr_t)
        def shuff_meta_impl(arr, n_pes, is_contig):
            send_counts = np.zeros(n_pes, np.int32)
            recv_counts = np.empty(n_pes, np.int32)
            send_buff = arr
            tmp_offset = send_counts  # dummy
            if not is_contig:
                send_buff = np.empty_like(arr)
                tmp_offset = np.zeros(n_pes, np.int32)
            # arr as out_arr placeholder, send/recv counts as placeholder for type inference
            return ShuffleMetaCL(
                send_counts, recv_counts, send_buff, arr, 0, send_counts, recv_counts, tmp_offset,
                None, None, None, None, None, None, None)
        return shuff_meta_impl

    assert arr_t == string_array_type
    ShuffleMetaCL = get_shuffle_meta_class(arr_t)
    def shuff_meta_str_impl(arr, n_pes, is_contig):
        send_counts = np.zeros(n_pes, np.int32)
        recv_counts = np.empty(n_pes, np.int32)
        send_counts_char = np.zeros(n_pes, np.int32)
        recv_counts_char = np.empty(n_pes, np.int32)
        send_arr_lens = np.empty(len(arr), np.uint32)
        send_arr_chars = get_data_ptr(arr)
        tmp_offset = send_counts  # dummy
        tmp_offset_chars = send_counts  # dummy

        if not is_contig:
            n_all_chars = num_total_chars(arr)
            send_arr_chars = np.empty(n_all_chars, np.uint8).ctypes
            tmp_offset = np.zeros(n_pes, np.int32)
            tmp_offset_chars = np.zeros(n_pes, np.int32)
        # arr as out_arr placeholder, send/recv counts as placeholder for type inference
        return ShuffleMetaCL(
            send_counts, recv_counts, None, arr, 0, send_counts, recv_counts, tmp_offset,
            send_counts_char, recv_counts_char, send_arr_lens,
            send_arr_chars, send_counts_char, recv_counts_char, tmp_offset_chars)
    return shuff_meta_str_impl

def finalize_shuffle_meta(arr, shuffle_meta):
    return

@overload(finalize_shuffle_meta)
def finalize_shuffle_meta_overload(arr_t, shuffle_meta_t):
    if isinstance(arr_t, types.Array):
        def finalize_impl(arr, shuffle_meta):
            hpat.distributed_api.alltoall(shuffle_meta.send_counts, shuffle_meta.recv_counts, 1)
            shuffle_meta.n_out = shuffle_meta.recv_counts.sum()
            shuffle_meta.out_arr = np.empty(shuffle_meta.n_out, arr.dtype)
            shuffle_meta.send_disp = hpat.hiframes_join.calc_disp(shuffle_meta.send_counts)
            shuffle_meta.recv_disp = hpat.hiframes_join.calc_disp(shuffle_meta.recv_counts)
        return finalize_impl

    assert arr_t == string_array_type
    def finalize_str_impl(arr, shuffle_meta):
        hpat.distributed_api.alltoall(shuffle_meta.send_counts, shuffle_meta.recv_counts, 1)
        hpat.distributed_api.alltoall(shuffle_meta.send_counts_char, shuffle_meta.recv_counts_char, 1)
        shuffle_meta.n_out = shuffle_meta.recv_counts.sum()
        n_all_chars = shuffle_meta.recv_counts_char.sum()
        shuffle_meta.out_arr = pre_alloc_string_array(shuffle_meta.n_out, n_all_chars)
        shuffle_meta.send_disp = hpat.hiframes_join.calc_disp(shuffle_meta.send_counts)
        shuffle_meta.recv_disp = hpat.hiframes_join.calc_disp(shuffle_meta.recv_counts)
        shuffle_meta.send_disp_char = hpat.hiframes_join.calc_disp(shuffle_meta.send_counts_char)
        shuffle_meta.recv_disp_char = hpat.hiframes_join.calc_disp(shuffle_meta.recv_counts_char)

    return finalize_str_impl


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
    int32_typ_enum = np.int32(_h5_typ_table[types.int32])
    char_typ_enum = np.int32(_h5_typ_table[types.uint8])
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

def get_shuffle_meta_class(arr_t):
    count_arr_typ = types.Array(types.int32, 1, 'C')
    if isinstance(arr_t, types.Array):
        spec = [
                ('send_counts', count_arr_typ),
                ('recv_counts', count_arr_typ),
                ('send_buff', arr_t),
                ('out_arr', arr_t),
                ('n_out', types.intp),
                ('send_disp', count_arr_typ),
                ('recv_disp', count_arr_typ),
                ('tmp_offset', count_arr_typ),
                ('send_counts_char', types.none),
                ('recv_counts_char', types.none),
                ('send_arr_lens', types.none),
                ('send_arr_chars', types.none),
                ('send_disp_char', types.none),
                ('recv_disp_char', types.none),
                ('tmp_offset_chars', types.none),
            ]
    else:
        spec = [
            ('send_counts', count_arr_typ),
            ('recv_counts', count_arr_typ),
            ('send_buff', types.none),
            ('out_arr', arr_t),
            ('n_out', types.intp),
            ('send_disp', count_arr_typ),
            ('recv_disp', count_arr_typ),
            ('tmp_offset', count_arr_typ),
            ('send_counts_char', count_arr_typ),
            ('recv_counts_char', count_arr_typ),
            ('send_arr_lens', types.Array(types.uint32, 1, 'C')),
            ('send_arr_chars', types.voidptr),
            ('send_disp_char', count_arr_typ),
            ('recv_disp_char', count_arr_typ),
            ('tmp_offset_chars', count_arr_typ),
        ]

    ShuffleMetaCL = numba.jitclass(spec)(ShuffleMeta)
    return ShuffleMetaCL


########  meta data for string data column handling  #########


def data_alloc_shuffle_metadata(arr, n_pes, is_contig):
    return ShuffleMeta(np.zeros(1), np.zeros(1), arr, arr, n_pes, np.zeros(1),
        np.zeros(1), np.zeros(1), None, None, None, None, None, None, None)

@overload(data_alloc_shuffle_metadata)
def data_alloc_shuffle_metadata_overload(data_t, n_pes_t, is_contig_t):
    count = data_t.count
    spec_null = [
        ('send_counts', types.none),
        ('recv_counts', types.none),
        ('send_buff', types.none),
        ('out_arr', types.none),
        ('n_out', types.none),
        ('send_disp', types.none),
        ('recv_disp', types.none),
        ('tmp_offset', types.none),
        ('send_counts_char', types.none),
        ('recv_counts_char', types.none),
        ('send_arr_lens', types.none),
        ('send_arr_chars', types.none),
        ('send_disp_char', types.none),
        ('recv_disp_char', types.none),
        ('tmp_offset_chars', types.none),
    ]
    count_arr_typ = types.Array(types.int32, 1, 'C')
    spec_str = [
        ('send_counts', types.none),
        ('recv_counts', types.none),
        ('send_buff', types.none),
        ('out_arr', string_array_type),
        ('n_out', types.none),
        ('send_disp', types.none),
        ('recv_disp', types.none),
        ('tmp_offset', types.none),
        ('send_counts_char', count_arr_typ),
        ('recv_counts_char', count_arr_typ),
        ('send_arr_lens', types.Array(types.uint32, 1, 'C')),
        ('send_arr_chars', types.voidptr),
        ('send_disp_char', count_arr_typ),
        ('recv_disp_char', count_arr_typ),
        ('tmp_offset_chars', count_arr_typ),
    ]
    ShuffleMetaStr = numba.jitclass(spec_str)(ShuffleMeta)

    glbls = {'ShuffleMetaStr': ShuffleMetaStr, 'np': np, 'get_data_ptr': get_data_ptr}
    for i, typ in enumerate(data_t.types):
        if isinstance(typ, types.Array):
            spec_null[2] = ('send_buff', typ)
            spec_null[3] = ('out_arr', typ)
            ShuffleMetaCL = numba.jitclass(spec_null.copy())(ShuffleMeta)
            glbls['ShuffleMeta_{}'.format(i)] = ShuffleMetaCL

    func_text = "def f(data, n_pes, is_contig):\n"
    for i in range(count):
        typ = data_t.types[i]
        func_text += "  arr = data[{}]\n".format(i)
        if isinstance(typ, types.Array):
            func_text += "  send_buff = arr\n"
            func_text += "  if not is_contig:\n"
            func_text += "    send_buff = np.empty_like(arr)\n"
            func_text += ("  meta_{} = ShuffleMeta_{}(None, None, send_buff, arr, None, None,"
                " None, None, None, None, None, None, None, None, None)\n").format(i, i)
        else:
            assert typ == string_array_type
            func_text += "  send_counts_char = np.zeros(n_pes, np.int32)\n"
            func_text += "  recv_counts_char = np.empty(n_pes, np.int32)\n"
            func_text += "  send_arr_lens = np.empty(len(arr), np.uint32)\n"
            func_text += "  send_arr_chars = get_data_ptr(arr)\n"
            func_text += "  tmp_offset_chars = send_counts_char\n"
            func_text += "  if not is_contig:\n"
            func_text += "    n_all_chars = num_total_chars(arr)\n"
            func_text += "    send_arr_chars = np.empty(n_all_chars, np.uint8).ctypes\n"
            func_text += "    tmp_offset_chars = np.zeros(n_pes, np.int32)\n"
            func_text += ("  meta_{} = ShuffleMetaStr(None, None, None, arr, None, "
                "None, None, None, send_counts_char, recv_counts_char, send_arr_lens,"
                " send_arr_chars, send_counts_char, recv_counts_char, send_arr_chars)\n").format(i)
    func_text += "  return ({}{})\n".format(
        ','.join(['meta_{}'.format(i) for i in range(count)]),
        "," if count == 1 else "")

    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    alloc_impl = loc_vars['f']
    return alloc_impl

def update_data_shuffle_meta(shuffle_meta, node_id, ind, data, is_contig=True):
    return

@overload(update_data_shuffle_meta)
def update_data_shuffle_meta_overload(meta_t, node_id_t, ind_t, data_t, is_contig=None):
    func_text = "def f(meta_tup, node_id, ind, data, is_contig=True):\n"
    for i, typ in enumerate(data_t.types):
        if typ == string_array_type:
            func_text += "  val_{} = data[{}][ind]\n".format(i, i)
            func_text += "  n_chars_{} = len(val_{})\n".format(i, i)
            func_text += "  del_str(val_{})\n".format(i)
            func_text += "  meta_tup[{}].send_counts_char[node_id] += n_chars_{}\n".format(i, i)
            func_text += "  if is_contig:\n"
            func_text += "    meta_tup[{}].send_arr_lens[ind] = n_chars_{}\n".format(i, i)

    func_text += "  return\n"
    loc_vars = {}
    exec(func_text, {'del_str': del_str}, loc_vars)
    update_impl = loc_vars['f']
    return update_impl

def finalize_data_shuffle_meta(data, shuffle_meta, key_meta):
    return

@overload(finalize_data_shuffle_meta)
def finalize_data_shuffle_meta_overload(data_t, shuffle_meta_t, key_meta_t):
    func_text = "def f(data, meta_tup, key_meta):\n"
    for i, typ in enumerate(data_t.types):
        if isinstance(typ, types.Array):
            func_text += "  meta_tup[{}].out_arr = np.empty(key_meta.n_out, np.{})\n".format(i, typ.dtype)
        else:
            assert typ == string_array_type
            func_text += ("  hpat.distributed_api.alltoall("
                "meta_tup[{}].send_counts_char, meta_tup[{}].recv_counts_char, 1)\n").format(i, i)
            func_text += "  n_all_chars_{} = meta_tup[{}].recv_counts_char.sum()\n".format(i, i)
            func_text += "  meta_tup[{}].out_arr = pre_alloc_string_array(key_meta.n_out, n_all_chars_{})\n".format(i, i)
            func_text += ("  meta_tup[{}].send_disp_char = hpat.hiframes_join."
                "calc_disp(meta_tup[{}].send_counts_char)\n").format(i, i)
            func_text += ("  meta_tup[{}].recv_disp_char = hpat.hiframes_join."
                "calc_disp(meta_tup[{}].recv_counts_char)\n").format(i, i)

    func_text += "  return\n"
    loc_vars = {}
    exec(func_text, {'np': np, 'hpat': hpat,
         'pre_alloc_string_array': pre_alloc_string_array}, loc_vars)
    finalize_impl = loc_vars['f']
    return finalize_impl

def alltoallv_tup(data, data_shuffle_meta, shuffle_meta):
    return data

@overload(alltoallv_tup)
def alltoallv_tup_overload(data_t, data_shuffle_meta_t, shuffle_meta_t):
    func_text = "def f(data, meta_tup, key_meta):\n"
    for i, typ in enumerate(data_t.types):
        if isinstance(typ, types.Array):
            func_text += ("  hpat.distributed_api.alltoallv("
                "meta_tup[{}].send_buff, meta_tup[{}].out_arr, key_meta.send_counts,"
                "key_meta.recv_counts, key_meta.send_disp, key_meta.recv_disp)\n").format(i, i)
        else:
            assert typ == string_array_type
            func_text += "  offset_ptr_{} = get_offset_ptr(meta_tup[{}].out_arr)\n".format(i, i)

            func_text += ("  hpat.distributed_api.c_alltoallv("
                "meta_tup[{}].send_arr_lens.ctypes, offset_ptr_{}, key_meta.send_counts.ctypes, "
                "key_meta.recv_counts.ctypes, key_meta.send_disp.ctypes, "
                "key_meta.recv_disp.ctypes, int32_typ_enum)\n").format(i, i)

            func_text += ("  hpat.distributed_api.c_alltoallv("
                "meta_tup[{}].send_arr_chars, get_data_ptr(meta_tup[{}].out_arr), meta_tup[{}].send_counts_char.ctypes,"
                "meta_tup[{}].recv_counts_char.ctypes, meta_tup[{}].send_disp_char.ctypes,"
                "meta_tup[{}].recv_disp_char.ctypes, char_typ_enum)\n").format(i, i, i, i, i, i)

            func_text += "  convert_len_arr_to_offset(offset_ptr_{}, key_meta.n_out)\n".format(i)

    func_text += "  return ({}{})\n".format(
        ','.join(['meta_tup[{}].out_arr'.format(i) for i in range(data_t.count)]),
        "," if data_t.count == 1 else "")

    int32_typ_enum = np.int32(_h5_typ_table[types.int32])
    char_typ_enum = np.int32(_h5_typ_table[types.uint8])
    loc_vars = {}
    exec(func_text, {'hpat': hpat, 'get_offset_ptr': get_offset_ptr,
         'get_data_ptr': get_data_ptr, 'int32_typ_enum': int32_typ_enum,
         'char_typ_enum': char_typ_enum,
         'convert_len_arr_to_offset': convert_len_arr_to_offset}, loc_vars)
    a2a_impl = loc_vars['f']
    return a2a_impl

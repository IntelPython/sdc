import numpy as np
import math
import numba
from numba import typeinfer, ir, ir_utils, config, types
from numba.ir_utils import (visit_vars_inner, replace_vars_inner,
                            compile_to_numba_ir, replace_arg_nodes)
from numba.typing import signature
import hpat
import hpat.timsort
from hpat import distributed, distributed_analysis
from hpat.distributed_api import Reduce_Type
from hpat.distributed_analysis import Distribution
from hpat.utils import debug_prints
from hpat.str_arr_ext import string_array_type

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
    # TODO: string key
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
    key_typ = typemap[key_arr.name]
    data_tup_typ = types.Tuple([typemap[v.name] for v in sort_node.df_vars.values()])

    sort_state_spec = [
        ('key_arr', key_typ),
        ('aLength', numba.intp),
        ('minGallop', numba.intp),
        ('tmpLength', numba.intp),
        ('tmp', key_typ),
        ('stackSize', numba.intp),
        ('runBase', numba.int64[:]),
        ('runLen', numba.int64[:]),
        ('data', data_tup_typ),
        ('tmp_data', data_tup_typ),
    ]

    col_name_args = ', '.join(["c"+str(i) for i in range(len(data_vars))])
    # TODO: use *args
    func_text = "def f(key_arr, {}):\n".format(col_name_args)
    func_text += "  data = ({}{})\n".format(col_name_args,
        "," if len(data_vars) == 1 else "")  # single value needs comma to become tuple
    func_text += "  _sort_len = len(key_arr)\n"
    func_text += "  sort_state = SortState(key_arr, _sort_len, data)\n"
    func_text += "  hpat.timsort.sort(sort_state, key_arr, 0, _sort_len, data)\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sort_impl = loc_vars['f']

    SortStateCL = numba.jitclass(sort_state_spec)(hpat.timsort.SortState)

    f_block = compile_to_numba_ir(sort_impl,
                                    {'hpat': hpat, 'SortState': SortStateCL},
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
        out = parallel_sort(key_arr, data)
        key_arr = out
        # TODO: use k-way merge instead of sort
        # sort output
        n_out = len(key_arr)
        sort_state_o = SortState(key_arr, n_out, data)
        hpat.timsort.sort(sort_state_o, key_arr, 0, n_out, data)

    f_block = compile_to_numba_ir(par_sort_impl,
                                    {'hpat': hpat, 'SortState': SortStateCL,
                                    'parallel_sort': parallel_sort},
                                    typingctx,
                                    (key_typ, data_tup_typ),
                                    typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [sort_node.key_arr, data_tup_var])
    nodes += f_block.body[:-3]
    return nodes


distributed.distributed_run_extensions[Sort] = sort_distributed_run


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
    bounds = np.empty(n_pes-1, key_arr.dtype)

    if my_rank == MPI_ROOT:
        all_samples.sort()
        n_samples = len(all_samples)
        step = math.ceil(n_samples / n_pes)
        for i in range(n_pes - 1):
            bounds[i] = all_samples[min((i + 1) * step, n_samples - 1)]
        # print(bounds)

    hpat.distributed_api.bcast(bounds)

    # calc send/recv counts
    send_counts = np.zeros(n_pes, np.int32)
    recv_counts = np.empty(n_pes, np.int32)
    node_id = 0
    for i in range(n_local):
        if node_id < (n_pes - 1) and key_arr[i] >= bounds[node_id]:
            node_id += 1
        send_counts[node_id] += 1
    hpat.distributed_api.alltoall(send_counts, recv_counts, 1)

    # shuffle
    n_out = recv_counts.sum()
    out_key_arr = np.empty(n_out, key_arr.dtype)
    send_disp = hpat.hiframes_join.calc_disp(send_counts)
    recv_disp = hpat.hiframes_join.calc_disp(recv_counts)
    hpat.distributed_api.alltoallv(key_arr, out_key_arr, send_counts, recv_counts, send_disp, recv_disp)

    return out_key_arr

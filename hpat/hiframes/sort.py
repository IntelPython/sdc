import numpy as np
import math
from collections import defaultdict
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

from hpat.shuffle_utils import (alltoallv, alltoallv_tup,
    finalize_shuffle_meta, update_shuffle_meta,  alloc_pre_shuffle_metadata,
    _get_keys_tup, _get_data_tup)

from hpat.str_arr_ext import (string_array_type, to_string_list,
                              cp_str_list_to_array, str_list_to_array,
                              get_offset_ptr, get_data_ptr, convert_len_arr_to_offset,
                              pre_alloc_string_array, num_total_chars)
from hpat.str_ext import string_type


MIN_SAMPLES = 1000000
#MIN_SAMPLES = 100
samplePointsPerPartitionHint = 20
MPI_ROOT = 0


class Sort(ir.Stmt):
    def __init__(self, df_in, df_out, key_arrs, out_key_arrs, df_in_vars,
                                    df_out_vars, inplace, loc, ascending=True):
        # for printing only
        self.df_in = df_in
        self.df_out = df_out
        self.key_arrs = key_arrs
        self.out_key_arrs = out_key_arrs
        self.df_in_vars = df_in_vars
        self.df_out_vars = df_out_vars
        self.inplace = inplace
        # HACK make sure ascending is boolean (seen error for none in CI)
        # TODO: fix source of issue
        if not isinstance(ascending, bool):
            ascending = True
        self.ascending = ascending
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
            ", ".join(v.name for v in self.key_arrs), df_in_str,
            ", ".join(v.name for v in self.out_key_arrs), df_out_str)


def sort_array_analysis(sort_node, equiv_set, typemap, array_analysis):

    # arrays of input df have same size in first dimension
    all_shapes = []
    in_arrs = sort_node.key_arrs + list(sort_node.df_in_vars.values())
    for col_var in in_arrs:
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
    out_arrs = sort_node.out_key_arrs + list(sort_node.df_out_vars.values())
    for col_var in out_arrs:
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

    in_arrs = sort_node.key_arrs + list(sort_node.df_in_vars.values())
    out_arrs = sort_node.out_key_arrs + list(sort_node.df_out_vars.values())
    # input columns have same distribution
    in_dist = Distribution.OneD
    for col_var in in_arrs:
        in_dist = Distribution(
            min(in_dist.value, array_dists[col_var.name].value))

    # output is 1D_Var due to shuffle, has to meet input dist
    # TODO: set to input dist in inplace case
    out_dist = Distribution(min(in_dist.value, Distribution.OneD_Var.value))
    for col_var in out_arrs:
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value))

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        in_dist = out_dist

    # set dists
    for col_var in in_arrs:
        array_dists[col_var.name] = in_dist

    for col_var in out_arrs:
        array_dists[col_var.name] = out_dist

    # TODO: handle rebalance
    # assert not (in_dist == Distribution.OneD and out_dist == Distribution.OneD_Var)
    return


distributed_analysis.distributed_analysis_extensions[Sort] = sort_distributed_analysis

def sort_typeinfer(sort_node, typeinferer):
    # input and output arrays have the same type
    for in_key, out_key in zip(sort_node.key_arrs, sort_node.out_key_arrs):
        typeinferer.constraints.append(typeinfer.Propagate(
            dst=out_key.name, src=in_key.name,
            loc=sort_node.loc))
    for col_name, col_var in sort_node.df_in_vars.items():
        out_col_var = sort_node.df_out_vars[col_name]
        typeinferer.constraints.append(typeinfer.Propagate(
            dst=out_col_var.name, src=col_var.name, loc=sort_node.loc))
    return

typeinfer.typeinfer_extensions[Sort] = sort_typeinfer

def build_sort_definitions(sort_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    # output arrays are defined
    if not sort_node.inplace:
        for col_var in (sort_node.out_key_arrs
                + list(sort_node.df_out_vars.values())):
            definitions[col_var.name].append(sort_node)

    return definitions

ir_utils.build_defs_extensions[Sort] = build_sort_definitions

def visit_vars_sort(sort_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting sort vars for:", sort_node)
        print("cbdata: ", sorted(cbdata.items()))

    for i in range(len(sort_node.key_arrs)):
        sort_node.key_arrs[i] = visit_vars_inner(
            sort_node.key_arrs[i], callback, cbdata)
        sort_node.out_key_arrs[i] = visit_vars_inner(
            sort_node.out_key_arrs[i], callback, cbdata)

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
    if not hpat.hiframes.api.enable_hiframes_remove_dead:
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
            and all(v.name not in lives for v in sort_node.out_key_arrs)):
        return None

    return sort_node


ir_utils.remove_dead_extensions[Sort] = remove_dead_sort


def sort_usedefs(sort_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # key array and input columns are used
    use_set.update({v.name for v in sort_node.key_arrs})
    use_set.update({v.name for v in sort_node.df_in_vars.values()})

    # output arrays are defined
    if not sort_node.inplace:
        def_set.update({v.name for v in sort_node.out_key_arrs})
        def_set.update({v.name for v in sort_node.df_out_vars.values()})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Sort] = sort_usedefs


def get_copies_sort(sort_node, typemap):
    # sort doesn't generate copies, it just kills the output columns
    kill_set = set()
    if not sort_node.inplace:
        kill_set = set(v.name for v in sort_node.df_out_vars.values())
        kill_set.update({v.name for v in sort_node.out_key_arrs})
    return set(), kill_set

ir_utils.copy_propagate_extensions[Sort] = get_copies_sort


def apply_copies_sort(sort_node, var_dict, name_var_table,
                        typemap, calltypes, save_copies):
    """apply copy propagate in sort node"""
    for i in range(len(sort_node.key_arrs)):
        sort_node.key_arrs[i] = replace_vars_inner(sort_node.key_arrs[i], var_dict)
        sort_node.out_key_arrs[i] = replace_vars_inner(sort_node.out_key_arrs[i], var_dict)

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
    for v in sort_node.key_arrs + sort_node.out_key_arrs + in_vars + out_vars:
        if (array_dists[v.name] != distributed.Distribution.OneD
                and array_dists[v.name] != distributed.Distribution.OneD_Var):
            parallel = False

    loc = sort_node.loc
    scope = sort_node.key_arrs[0].scope
    # copy arrays when not inplace
    nodes = []
    key_arrs = sort_node.key_arrs
    if not sort_node.inplace:
        new_keys = []
        for v in key_arrs:
            new_key = _copy_array_nodes(v, nodes, typingctx, typemap, calltypes)
            new_keys.append(new_key)
        key_arrs = new_keys
        new_in_vars = []
        for v in in_vars:
            v_cp = _copy_array_nodes(v, nodes, typingctx, typemap, calltypes)
            new_in_vars.append(v_cp)
        in_vars = new_in_vars

    key_name_args = ', '.join("key"+str(i) for i in range(len(key_arrs)))
    col_name_args = ', '.join(["c"+str(i) for i in range(len(in_vars))])
    # TODO: use *args
    func_text = "def f({}, {}):\n".format(key_name_args, col_name_args)
    func_text += "  key_arrs = ({},)\n".format(key_name_args)
    func_text += "  data = ({}{})\n".format(col_name_args,
        "," if len(in_vars) == 1 else "")  # single value needs comma to become tuple
    func_text += "  hpat.hiframes.sort.local_sort(key_arrs, data, {})\n".format(sort_node.ascending)
    func_text += "  return key_arrs, data\n"

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    sort_impl = loc_vars['f']

    key_typ = types.Tuple([typemap[v.name] for v in key_arrs])
    data_tup_typ = types.Tuple([typemap[v.name] for v in in_vars])

    f_block = compile_to_numba_ir(sort_impl,
                                    {'hpat': hpat,
                                    'to_string_list': to_string_list,
                                    'cp_str_list_to_array': cp_str_list_to_array},
                                    typingctx,
                                    tuple(list(key_typ.types) + list(data_tup_typ.types)),
                                    typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, key_arrs + in_vars)
    nodes += f_block.body[:-2]
    ret_var = nodes[-1].target
    # get key tup
    key_arrs_tup_var = ir.Var(scope, mk_unique_var('key_data'), loc)
    typemap[key_arrs_tup_var.name] = key_typ
    gen_getitem(key_arrs_tup_var, ret_var, 0, calltypes, nodes)
    # get data tup
    data_tup_var = ir.Var(scope, mk_unique_var('sort_data'), loc)
    typemap[data_tup_var.name] = data_tup_typ
    gen_getitem(data_tup_var, ret_var, 1, calltypes, nodes)

    if not parallel:
        for i, var in enumerate(sort_node.out_key_arrs):
            gen_getitem(var, key_arrs_tup_var, i, calltypes, nodes)
        for i, var in enumerate(out_vars):
            gen_getitem(var, data_tup_var, i, calltypes, nodes)
        return nodes

    ascending_var = ir.Var(scope, mk_unique_var('ascending'), loc)
    typemap[ascending_var.name] = types.bool_
    nodes.append(
        ir.Assign(ir.Const(sort_node.ascending, loc), ascending_var, loc))

    # parallel case
    def par_sort_impl(key_arrs, data, ascending):
        out_key, out_data = parallel_sort(key_arrs, data, ascending)
        # TODO: use k-way merge instead of sort
        # sort output
        hpat.hiframes.sort.local_sort(out_key, out_data, ascending)
        return out_key, out_data

    f_block = compile_to_numba_ir(par_sort_impl,
                                    {'hpat': hpat,
                                    'parallel_sort': parallel_sort,
                                    'to_string_list': to_string_list,
                                    'cp_str_list_to_array': cp_str_list_to_array},
                                    typingctx,
                                    (key_typ, data_tup_typ, types.bool_),
                                    typemap, calltypes).blocks.popitem()[1]
    replace_arg_nodes(f_block, [key_arrs_tup_var, data_tup_var, ascending_var])
    nodes += f_block.body[:-2]
    ret_var = nodes[-1].target
    # get output key
    key_tup = ir.Var(scope, mk_unique_var('sort_keys'), loc)
    typemap[key_tup.name] = key_typ
    gen_getitem(key_tup, ret_var, 0, calltypes, nodes)
    # get data tup
    data_tup = ir.Var(scope, mk_unique_var('sort_data'), loc)
    typemap[data_tup.name] = data_tup_typ
    gen_getitem(data_tup, ret_var, 1, calltypes, nodes)

    for i, var in enumerate(sort_node.out_key_arrs):
        gen_getitem(var, key_tup, i, calltypes, nodes)

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


# TODO: fix cache issue
@numba.njit(no_cpython_wrapper=True, cache=False)
def local_sort(key_arrs, data, ascending=True):
    # convert StringArray to list(string) to enable swapping in sort
    l_key_arrs = to_string_list(key_arrs)
    l_data = to_string_list(data)
    n_out = len(key_arrs[0])
    hpat.timsort.sort(l_key_arrs, 0, n_out, l_data)
    if not ascending:
        hpat.timsort.reverseRange(l_key_arrs, 0, n_out, l_data)
    cp_str_list_to_array(key_arrs, l_key_arrs)
    cp_str_list_to_array(data, l_data)


@numba.njit(no_cpython_wrapper=True, cache=True)
def parallel_sort(key_arrs, data, ascending=True):
    n_local = len(key_arrs[0])
    n_total = hpat.distributed_api.dist_reduce(n_local, np.int32(Reduce_Type.Sum.value))

    n_pes = hpat.distributed_api.get_size()
    my_rank = hpat.distributed_api.get_rank()

    # similar to Spark's sample computation Partitioner.scala
    sampleSize = min(samplePointsPerPartitionHint * n_pes, MIN_SAMPLES)

    fraction = min(sampleSize / max(n_total, 1), 1.0)
    n_loc_samples = min(math.ceil(fraction * n_local), n_local)
    inds = np.random.randint(0, n_local, n_loc_samples)
    samples = key_arrs[0][inds]
    # print(sampleSize, fraction, n_local, n_loc_samples, len(samples))

    all_samples = hpat.distributed_api.gatherv(samples)
    all_samples = to_string_list(all_samples)
    bounds = empty_like_type(n_pes-1, all_samples)

    if my_rank == MPI_ROOT:
        all_samples.sort()
        if not ascending:
            all_samples = all_samples[::-1]
        n_samples = len(all_samples)
        step = math.ceil(n_samples / n_pes)
        for i in range(n_pes - 1):
            bounds[i] = all_samples[min((i + 1) * step, n_samples - 1)]
        # print(bounds)

    bounds = str_list_to_array(bounds)
    bounds = hpat.distributed_api.prealloc_str_for_bcast(bounds)
    hpat.distributed_api.bcast(bounds)

    # calc send/recv counts
    pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, data, n_pes, True)
    node_id = 0
    for i in range(n_local):
        val = key_arrs[0][i]
        # TODO: refactor
        if node_id < (n_pes - 1) and (ascending and val >= bounds[node_id]
                                or (not ascending) and val <= bounds[node_id]):
            node_id += 1
        update_shuffle_meta(pre_shuffle_meta, node_id, i, (val,),
            getitem_arr_tup(data, i), True)

    shuffle_meta = finalize_shuffle_meta(key_arrs, data, pre_shuffle_meta,
                                          n_pes, True)

    # shuffle
    recvs = alltoallv_tup(key_arrs + data, shuffle_meta)
    out_key = _get_keys_tup(recvs, key_arrs)
    out_data = _get_data_tup(recvs, key_arrs)

    return out_key, out_data

# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


from __future__ import print_function, division, absolute_import

import operator
from collections import namedtuple, defaultdict
from functools import reduce
import copy
import numpy as np
import numba
from numba import typeinfer, ir, ir_utils, config, types, compiler, typed_passes
from numba.ir_utils import (visit_vars_inner, replace_vars_inner, remove_dead,
                            compile_to_numba_ir, replace_arg_nodes,
                            replace_vars_stmt, find_callname, guard,
                            mk_unique_var, find_topo_order, is_getitem,
                            build_definitions, remove_dels, get_ir_of_code,
                            get_definition, find_callname, get_name_var_table,
                            replace_var_names)
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks, Parfor
from numba.analysis import compute_use_defs
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.extending import overload, lower_builtin
import hpat
from hpat.utils import (is_call_assign, is_var_assign, is_assign, debug_prints,
                        alloc_arr_tup, empty_like_type)
from hpat import distributed, distributed_analysis
from hpat.distributed_analysis import Distribution
from hpat.utils import _numba_to_c_type_map, unliteral_all
from hpat.str_ext import string_type
from hpat.set_ext import num_total_chars_set_string, build_set
from hpat.str_arr_ext import (string_array_type, pre_alloc_string_array,
                              get_offset_ptr, get_data_ptr)

from hpat.hiframes.join import write_send_buff
from hpat.timsort import getitem_arr_tup
from hpat.shuffle_utils import (getitem_arr_tup_single, val_to_tup, alltoallv,
                                alltoallv_tup, finalize_shuffle_meta, update_shuffle_meta,
                                alloc_pre_shuffle_metadata, _get_keys_tup, _get_data_tup)


AggFuncStruct = namedtuple('AggFuncStruct',
                           ['var_typs', 'init_func', 'update_all_func', 'combine_all_func',
                            'eval_all_func'])

supported_agg_funcs = ['sum', 'count', 'mean',
                       'min', 'max', 'prod', 'var', 'std', 'agg', 'aggregate']


def get_agg_func(func_ir, func_name, rhs):
    from hpat.hiframes.series_kernels import series_replace_funcs
    if func_name == 'var':
        return _column_var_impl_linear
    if func_name == 'std':
        return _column_std_impl_linear
    if func_name in supported_agg_funcs[:-2]:
        func = series_replace_funcs[func_name]
        # returning generic function
        # TODO: support type-specific funcs e.g. for dt64
        if isinstance(func, dict):
            func = func[types.float64]
        return func

    assert func_name in ['agg', 'aggregate']
    # agg case
    # error checking: make sure there is function input only
    if len(rhs.args) != 1:
        raise ValueError("agg expects 1 argument")
    agg_func = guard(get_definition, func_ir, rhs.args[0])
    if agg_func is None or not (isinstance(agg_func, ir.Expr)
                                and agg_func.op == 'make_function'):
        raise ValueError("lambda for map not found")

    def agg_func_wrapper(A):
        return A
    agg_func_wrapper.__code__ = agg_func.code
    agg_func = agg_func_wrapper
    return agg_func


# type(dtype) is called by np.full (used in agg_typer)
@infer_global(type)
class TypeDt64(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1 and isinstance(args[0], (types.NPDatetime, types.NPTimedelta)):
            classty = types.DType(args[0])
            return signature(classty, *args)


# combine function takes the reduce vars in reverse order of their user
@numba.njit
def _var_combine(ssqdm_a, mean_a, nobs_a, ssqdm_b, mean_b, nobs_b):  # pragma: no cover
    nobs = (nobs_a + nobs_b)
    mean_x = (nobs_a * mean_a + nobs_b * mean_b) / nobs
    delta = mean_b - mean_a
    M2 = ssqdm_a + ssqdm_b + delta * delta * nobs_a * nobs_b / nobs
    return M2, mean_x, nobs

# XXX: njit doesn't work when hpat.jit() is used for agg_func in hiframes
# @numba.njit


def __special_combine(*args):
    return


@infer_global(__special_combine)
class SpecialCombineTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.void, *unliteral_all(args))


@lower_builtin(__special_combine, types.VarArg(types.Any))
def lower_special_combine(context, builder, sig, args):
    return context.get_dummy_value()


# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def _column_var_impl_linear(A):  # pragma: no cover
    nobs = 0
    mean_x = 0.0
    ssqdm_x = 0.0
    N = len(A)
    for i in numba.parfor.internal_prange(N):
        hpat.hiframes.aggregate.__special_combine(
            ssqdm_x, mean_x, nobs, hpat.hiframes.aggregate._var_combine)
        val = A[i]
        if not np.isnan(val):
            nobs += 1
            delta = val - mean_x
            mean_x += delta / nobs
            # TODO: Pandas formula is better or Welford?
            # ssqdm_x += ((nobs - 1) * delta ** 2) / nobs
            delta2 = val - mean_x
            ssqdm_x += delta * delta2
    return hpat.hiframes.rolling.calc_var(2, nobs, mean_x, ssqdm_x)

# TODO: avoid code duplication


def _column_std_impl_linear(A):  # pragma: no cover
    nobs = 0
    mean_x = 0.0
    ssqdm_x = 0.0
    N = len(A)
    for i in numba.parfor.internal_prange(N):
        hpat.hiframes.aggregate.__special_combine(
            ssqdm_x, mean_x, nobs, hpat.hiframes.aggregate._var_combine)
        val = A[i]
        if not np.isnan(val):
            nobs += 1
            delta = val - mean_x
            mean_x += delta / nobs
            # TODO: Pandas formula is better or Welford?
            # ssqdm_x += ((nobs - 1) * delta ** 2) / nobs
            delta2 = val - mean_x
            ssqdm_x += delta * delta2
    v = hpat.hiframes.rolling.calc_var(2, nobs, mean_x, ssqdm_x)
    return v**0.5


class Aggregate(ir.Stmt):
    def __init__(self, df_out, df_in, key_names, out_key_vars, df_out_vars,
                 df_in_vars, key_arrs, agg_func, tp_vars, loc,
                 pivot_arr=None, pivot_values=None,
                 is_crosstab=False):
        # name of output dataframe (just for printing purposes)
        self.df_out = df_out
        # name of input dataframe (just for printing purposes)
        self.df_in = df_in
        # key name (for printing)
        self.key_names = key_names
        self.out_key_vars = out_key_vars

        self.df_out_vars = df_out_vars
        self.df_in_vars = df_in_vars
        self.key_arrs = key_arrs

        self.agg_func = agg_func
        # XXX update tp_vars in copy propagate etc.?
        self.out_typer_vars = tp_vars

        self.loc = loc
        # pivot_table handling
        self.pivot_arr = pivot_arr
        self.pivot_values = pivot_values
        self.is_crosstab = is_crosstab

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)
        in_cols = ""
        for (c, v) in self.df_in_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        pivot = ("pivot {}:{}".format(self.pivot_arr.name, self.pivot_values)
                 if self.pivot_arr is not None else "")
        key_names = ",".join(self.key_names)
        key_arrnames = ",".join([v.name for v in self.key_arrs])
        return "aggregate: {} = {} [key: {}:{}] {}".format(
            df_out_str, df_in_str, key_names, key_arrnames, pivot)


def aggregate_typeinfer(aggregate_node, typeinferer):
    for out_name, out_var in aggregate_node.df_out_vars.items():
        if aggregate_node.pivot_arr is not None:
            tp_var = list(aggregate_node.out_typer_vars.values())[0]
        else:
            tp_var = aggregate_node.out_typer_vars[out_name]

        typeinferer.constraints.append(
            typeinfer.Propagate(
                dst=out_var.name, src=tp_var.name, loc=aggregate_node.loc))

    # return key case
    if aggregate_node.out_key_vars is not None:
        for in_key, out_key in zip(aggregate_node.key_arrs, aggregate_node.out_key_vars):
            typeinferer.constraints.append(typeinfer.Propagate(
                dst=out_key.name, src=in_key.name,
                loc=aggregate_node.loc))

    return


typeinfer.typeinfer_extensions[Aggregate] = aggregate_typeinfer


def aggregate_usedefs(aggregate_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    if aggregate_node.out_typer_vars is not None:
        # typer vars are used before typing (hiframes_typed should set None)
        for v in aggregate_node.out_typer_vars.values():
            use_set.add(v.name)
    # key array and input columns are used
    use_set.update({v.name for v in aggregate_node.key_arrs})
    use_set.update({v.name for v in aggregate_node.df_in_vars.values()})

    if aggregate_node.pivot_arr is not None:
        use_set.add(aggregate_node.pivot_arr.name)

    # output columns are defined
    def_set.update({v.name for v in aggregate_node.df_out_vars.values()})

    # return key is defined
    if aggregate_node.out_key_vars is not None:
        def_set.update({v.name for v in aggregate_node.out_key_vars})

    return numba.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.analysis.ir_extension_usedefs[Aggregate] = aggregate_usedefs


def remove_dead_aggregate(aggregate_node, lives, arg_aliases, alias_map, func_ir, typemap):
    #
    if not hpat.hiframes.api.enable_hiframes_remove_dead:
        return aggregate_node

    dead_cols = []

    for col_name, col_var in aggregate_node.df_out_vars.items():
        if col_var.name not in lives:
            dead_cols.append(col_name)

    for cname in dead_cols:
        aggregate_node.df_out_vars.pop(cname)
        if aggregate_node.pivot_arr is None:
            aggregate_node.df_in_vars.pop(cname)
            if aggregate_node.out_typer_vars is not None:
                aggregate_node.out_typer_vars.pop(cname)
        else:
            aggregate_node.pivot_values.remove(cname)

    out_key_vars = aggregate_node.out_key_vars
    if out_key_vars is not None and all(v.name not in lives for v in out_key_vars):
        aggregate_node.out_key_vars = None

    # TODO: test agg remove
    # remove empty aggregate node
    if (len(aggregate_node.df_out_vars) == 0
            and aggregate_node.out_key_vars is None):
        return None

    return aggregate_node


ir_utils.remove_dead_extensions[Aggregate] = remove_dead_aggregate


def get_copies_aggregate(aggregate_node, typemap):
    # aggregate doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in aggregate_node.df_out_vars.values())
    if aggregate_node.out_key_vars is not None:
        kill_set.update({v.name for v in aggregate_node.out_key_vars})
    return set(), kill_set


ir_utils.copy_propagate_extensions[Aggregate] = get_copies_aggregate


def apply_copies_aggregate(aggregate_node, var_dict, name_var_table,
                           typemap, calltypes, save_copies):
    """apply copy propagate in aggregate node"""
    for i in range(len(aggregate_node.key_arrs)):
        aggregate_node.key_arrs[i] = replace_vars_inner(
            aggregate_node.key_arrs[i], var_dict)

    for col_name in list(aggregate_node.df_in_vars.keys()):
        aggregate_node.df_in_vars[col_name] = replace_vars_inner(
            aggregate_node.df_in_vars[col_name], var_dict)
    for col_name in list(aggregate_node.df_out_vars.keys()):
        aggregate_node.df_out_vars[col_name] = replace_vars_inner(
            aggregate_node.df_out_vars[col_name], var_dict)

    if aggregate_node.out_key_vars is not None:
        for i in range(len(aggregate_node.out_key_vars)):
            aggregate_node.out_key_vars[i] = replace_vars_inner(
                aggregate_node.out_key_vars[i], var_dict)

    if aggregate_node.pivot_arr is not None:
        aggregate_node.pivot_arr = replace_vars_inner(
            aggregate_node.pivot_arr, var_dict)

    return


ir_utils.apply_copy_propagate_extensions[Aggregate] = apply_copies_aggregate


def visit_vars_aggregate(aggregate_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting aggregate vars for:", aggregate_node)
        print("cbdata: ", sorted(cbdata.items()))

    for i in range(len(aggregate_node.key_arrs)):
        aggregate_node.key_arrs[i] = visit_vars_inner(
            aggregate_node.key_arrs[i], callback, cbdata)

    for col_name in list(aggregate_node.df_in_vars.keys()):
        aggregate_node.df_in_vars[col_name] = visit_vars_inner(
            aggregate_node.df_in_vars[col_name], callback, cbdata)
    for col_name in list(aggregate_node.df_out_vars.keys()):
        aggregate_node.df_out_vars[col_name] = visit_vars_inner(
            aggregate_node.df_out_vars[col_name], callback, cbdata)

    if aggregate_node.out_key_vars is not None:
        for i in range(len(aggregate_node.out_key_vars)):
            aggregate_node.out_key_vars[i] = visit_vars_inner(
                aggregate_node.out_key_vars[i], callback, cbdata)

    if aggregate_node.pivot_arr is not None:
        aggregate_node.pivot_arr = visit_vars_inner(
            aggregate_node.pivot_arr, callback, cbdata)


# add call to visit aggregate variable
ir_utils.visit_vars_extensions[Aggregate] = visit_vars_aggregate


def aggregate_array_analysis(aggregate_node, equiv_set, typemap,
                             array_analysis):
    # empty aggregate nodes should be deleted in remove dead
    assert len(aggregate_node.df_in_vars) > 0 or aggregate_node.out_key_vars is not None or aggregate_node.is_crosstab, (
        "empty aggregate in array analysis")

    # arrays of input df have same size in first dimension as key array
    # string array doesn't have shape in array analysis
    all_shapes = []
    for key_arr in aggregate_node.key_arrs:
        key_typ = typemap[key_arr.name]
        if key_typ != string_array_type:
            col_shape = equiv_set.get_shape(key_arr)
            all_shapes.append(col_shape[0])

    if aggregate_node.pivot_arr is not None:
        pivot_typ = typemap[aggregate_node.pivot_arr.name]
        if pivot_typ != string_array_type:
            col_shape = equiv_set.get_shape(aggregate_node.pivot_arr)
            all_shapes.append(col_shape[0])

    for _, col_var in aggregate_node.df_in_vars.items():
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
    post = []
    all_shapes = []
    out_vars = list(aggregate_node.df_out_vars.values())
    if aggregate_node.out_key_vars is not None:
        out_vars.extend(aggregate_node.out_key_vars)

    for col_var in out_vars:
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


numba.array_analysis.array_analysis_extensions[Aggregate] = aggregate_array_analysis


def aggregate_distributed_analysis(aggregate_node, array_dists):
    # input columns have same distribution
    in_dist = Distribution.OneD
    for _, col_var in aggregate_node.df_in_vars.items():
        in_dist = Distribution(
            min(in_dist.value, array_dists[col_var.name].value))

    # key arrays
    for key_arr in aggregate_node.key_arrs:
        in_dist = Distribution(
            min(in_dist.value, array_dists[key_arr.name].value))

    # pivot case
    if aggregate_node.pivot_arr is not None:
        in_dist = Distribution(
            min(in_dist.value, array_dists[aggregate_node.pivot_arr.name].value))
        array_dists[aggregate_node.pivot_arr.name] = in_dist

    for _, col_var in aggregate_node.df_in_vars.items():
        array_dists[col_var.name] = in_dist
    for key_arr in aggregate_node.key_arrs:
        array_dists[key_arr.name] = in_dist

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for _, col_var in aggregate_node.df_out_vars.items():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value))

    if aggregate_node.out_key_vars is not None:
        for col_var in aggregate_node.out_key_vars:
            if col_var.name in array_dists:
                out_dist = Distribution(
                    min(out_dist.value, array_dists[col_var.name].value))

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for _, col_var in aggregate_node.df_out_vars.items():
        array_dists[col_var.name] = out_dist

    if aggregate_node.out_key_vars is not None:
        for cvar in aggregate_node.out_key_vars:
            array_dists[cvar.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        for key_arr in aggregate_node.key_arrs:
            array_dists[key_arr.name] = out_dist
        # pivot case
        if aggregate_node.pivot_arr is not None:
            array_dists[aggregate_node.pivot_arr.name] = out_dist
        for _, col_var in aggregate_node.df_in_vars.items():
            array_dists[col_var.name] = out_dist

    return


distributed_analysis.distributed_analysis_extensions[Aggregate] = aggregate_distributed_analysis


def build_agg_definitions(agg_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in agg_node.df_out_vars.values():
        definitions[col_var.name].append(agg_node)

    if agg_node.out_key_vars is not None:
        for cvar in agg_node.out_key_vars:
            definitions[cvar.name].append(agg_node)

    return definitions


ir_utils.build_defs_extensions[Aggregate] = build_agg_definitions


def __update_redvars():
    pass


@infer_global(__update_redvars)
class UpdateDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.void, *args)


def __combine_redvars():
    pass


@infer_global(__combine_redvars)
class CombineDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.void, *args)


def __eval_res():
    pass


@infer_global(__eval_res)
class EvalDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        # takes the output array as first argument to know the output dtype
        return signature(args[0].dtype, *args)


def agg_distributed_run(agg_node, array_dists, typemap, calltypes, typingctx, targetctx, dist_pass):
    parallel = True
    for v in (list(agg_node.df_in_vars.values())
              + list(agg_node.df_out_vars.values()) + agg_node.key_arrs):
        if (array_dists[v.name] != distributed.Distribution.OneD
                and array_dists[v.name] != distributed.Distribution.OneD_Var):
            parallel = False
        # TODO: check supported types
        # if (typemap[v.name] != types.Array(types.intp, 1, 'C')
        #         and typemap[v.name] != types.Array(types.float64, 1, 'C')):
        #     raise ValueError(
        #         "Only int64 and float64 columns are currently supported in aggregate")
        # if (typemap[left_key_var.name] != types.Array(types.intp, 1, 'C')
        #     or typemap[right_key_var.name] != types.Array(types.intp, 1, 'C')):
        # raise ValueError("Only int64 keys are currently supported in aggregate")

    # TODO: rebalance if output distributions are 1D instead of 1D_Var

    # TODO: handle key column being part of output

    key_typs = tuple(typemap[v.name] for v in agg_node.key_arrs)
    # get column variables
    in_col_vars = [v for (n, v) in agg_node.df_in_vars.items()]
    out_col_vars = [v for (n, v) in agg_node.df_out_vars.items()]
    # get column types
    in_col_typs = tuple(typemap[v.name] for v in in_col_vars)
    out_col_typs = tuple(typemap[v.name] for v in out_col_vars)
    pivot_typ = types.none if agg_node.pivot_arr is None else typemap[agg_node.pivot_arr.name]
    arg_typs = tuple(key_typs + in_col_typs + (pivot_typ,))

    agg_func_struct = get_agg_func_struct(
        agg_node.agg_func, in_col_typs, out_col_typs, typingctx, targetctx,
        pivot_typ, agg_node.pivot_values, agg_node.is_crosstab)

    return_key = agg_node.out_key_vars is not None
    out_typs = [t.dtype for t in out_col_typs]

    top_level_func = gen_top_level_agg_func(
        agg_node.key_names, return_key, agg_func_struct.var_typs, out_typs,
        agg_node.df_in_vars.keys(), agg_node.df_out_vars.keys(), parallel)

    f_block = compile_to_numba_ir(top_level_func,
                                  {'hpat': hpat, 'np': np,
                                   'agg_seq_iter': agg_seq_iter,
                                   'parallel_agg': parallel_agg,
                                   '__update_redvars': agg_func_struct.update_all_func,
                                   '__init_func': agg_func_struct.init_func,
                                   '__combine_redvars': agg_func_struct.combine_all_func,
                                   '__eval_res': agg_func_struct.eval_all_func,
                                   'dt64_dtype': np.dtype('datetime64[ns]'),
                                   },
                                  typingctx, arg_typs,
                                  typemap, calltypes).blocks.popitem()[1]

    nodes = []
    if agg_node.pivot_arr is None:
        scope = agg_node.key_arrs[0].scope
        loc = agg_node.loc
        none_var = ir.Var(scope, mk_unique_var("dummy_none"), loc)
        typemap[none_var.name] = types.none
        nodes.append(ir.Assign(ir.Const(None, loc), none_var, loc))
        in_col_vars.append(none_var)
    else:
        in_col_vars.append(agg_node.pivot_arr)

    replace_arg_nodes(f_block, agg_node.key_arrs + in_col_vars)

    tuple_assign = f_block.body[-3]
    assert (is_assign(tuple_assign) and isinstance(tuple_assign.value, ir.Expr)
            and tuple_assign.value.op == 'build_tuple')
    nodes += f_block.body[:-3]

    out_vars = list(agg_node.df_out_vars.values())
    if return_key:
        out_vars += agg_node.out_key_vars

    for i, var in enumerate(out_vars):
        out_var = tuple_assign.value.items[i]
        nodes.append(ir.Assign(out_var, var, var.loc))

    return nodes


distributed.distributed_run_extensions[Aggregate] = agg_distributed_run


@numba.njit
def parallel_agg(key_arrs, data_redvar_dummy, out_dummy_tup, data_in, init_vals,
                 __update_redvars, __combine_redvars, __eval_res, return_key, pivot_arr):  # pragma: no cover
    # alloc shuffle meta
    n_pes = hpat.distributed_api.get_size()
    pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, data_redvar_dummy, n_pes, False)

    # calc send/recv counts
    key_set = get_key_set(key_arrs)
    for i in range(len(key_arrs[0])):
        val = getitem_arr_tup_single(key_arrs, i)
        if val not in key_set:
            key_set.add(val)
            node_id = hash(val) % n_pes
            # data isn't computed here yet so pass empty tuple
            update_shuffle_meta(pre_shuffle_meta, node_id, i, val_to_tup(val), (), False)

    shuffle_meta = finalize_shuffle_meta(key_arrs, data_redvar_dummy, pre_shuffle_meta, n_pes, False, init_vals)

    agg_parallel_local_iter(key_arrs, data_in, shuffle_meta, data_redvar_dummy, __update_redvars, pivot_arr)

    recvs = alltoallv_tup(key_arrs + data_redvar_dummy, shuffle_meta)
    # print(data_shuffle_meta[0].out_arr)
    key_arrs = _get_keys_tup(recvs, key_arrs)
    reduce_recvs = _get_data_tup(recvs, key_arrs)
    out_arrs = agg_parallel_combine_iter(key_arrs, reduce_recvs, out_dummy_tup,
                                         init_vals, __combine_redvars, __eval_res, return_key, data_in, pivot_arr)
    return out_arrs

    # key_arr = shuffle_meta.out_arr
    # n_uniq_keys = len(set(key_arr))
    # out_key = __agg_func(n_uniq_keys, 0, key_arr)
    # return (out_key,)


@numba.njit
def agg_parallel_local_iter(key_arrs, data_in, shuffle_meta, data_redvar_dummy,
                            __update_redvars, pivot_arr):  # pragma: no cover
    # _init_val_0 = np.int64(0)
    # redvar_0_arr = np.full(n_uniq_keys, _init_val_0, np.int64)
    # _init_val_1 = np.int64(0)
    # redvar_1_arr = np.full(n_uniq_keys, _init_val_1, np.int64)
    # out_key = np.empty(n_uniq_keys, np.float64)
    n_pes = hpat.distributed_api.get_size()
    # hpat.dict_ext.init_dict_float64_int64()
    # key_write_map = get_key_dict(key_arrs[0])
    key_write_map, byte_v = get_key_dict(key_arrs)

    redvar_arrs = get_shuffle_data_send_buffs(shuffle_meta, key_arrs, data_redvar_dummy)

    for i in range(len(key_arrs[0])):
        # k = key_arrs[0][i]
        k = _getitem_keys(key_arrs, i, byte_v)
        if k not in key_write_map:
            # k is byte_vec but we need tuple value for hashing
            val = getitem_arr_tup_single(key_arrs, i)
            node_id = hash(val) % n_pes
            w_ind = write_send_buff(shuffle_meta, node_id, i, val_to_tup(val), ())
            shuffle_meta.tmp_offset[node_id] += 1
            key_write_map[k] = w_ind
        else:
            w_ind = key_write_map[k]
        __update_redvars(redvar_arrs, data_in, w_ind, i, pivot_arr)
        #redvar_arrs[0][w_ind], redvar_arrs[1][w_ind] = __update_redvars(redvar_arrs[0][w_ind], redvar_arrs[1][w_ind], data_in[0][i])
    hpat.dict_ext.byte_vec_free(byte_v)
    return


@numba.njit
def agg_parallel_combine_iter(key_arrs, reduce_recvs, out_dummy_tup, init_vals,
                              __combine_redvars, __eval_res, return_key, data_in, pivot_arr):  # pragma: no cover
    key_set = _build_set_tup(key_arrs)
    n_uniq_keys = len(key_set)
    out_arrs = alloc_agg_output(n_uniq_keys, out_dummy_tup, key_set, data_in,
                                return_key)
    # out_arrs = alloc_arr_tup(n_uniq_keys, out_dummy_tup)
    local_redvars = alloc_arr_tup(n_uniq_keys, reduce_recvs, init_vals)

    # key_write_map = get_key_dict(key_arrs[0])
    key_write_map, byte_v = get_key_dict(key_arrs)
    curr_write_ind = 0
    for i in range(len(key_arrs[0])):
        # k = key_arrs[0][i]
        k = _getitem_keys(key_arrs, i, byte_v)
        if k not in key_write_map:
            w_ind = curr_write_ind
            curr_write_ind += 1
            key_write_map[k] = w_ind
            if return_key:
                _set_out_keys(out_arrs, w_ind, key_arrs, i, k)
                # setitem_array_with_str(out_arrs[-1], w_ind, k)
                # out_arrs[-1][w_ind] = k
        else:
            w_ind = key_write_map[k]
        __combine_redvars(local_redvars, reduce_recvs, w_ind, i, pivot_arr)
    for j in range(n_uniq_keys):
        __eval_res(local_redvars, out_arrs, j)

    hpat.dict_ext.byte_vec_free(byte_v)
    return out_arrs


@numba.njit
def agg_seq_iter(key_arrs, redvar_dummy_tup, out_dummy_tup, data_in, init_vals,
                 __update_redvars, __eval_res, return_key, pivot_arr):  # pragma: no cover
    key_set = _build_set_tup(key_arrs)
    n_uniq_keys = len(key_set)
    out_arrs = alloc_agg_output(n_uniq_keys, out_dummy_tup, key_set, data_in,
                                return_key)
    # out_arrs = alloc_arr_tup(n_uniq_keys, out_dummy_tup)
    local_redvars = alloc_arr_tup(n_uniq_keys, redvar_dummy_tup, init_vals)

    key_write_map, byte_v = get_key_dict(key_arrs)
    curr_write_ind = 0
    for i in range(len(key_arrs[0])):
        #k = key_arrs[0][i]
        k = _getitem_keys(key_arrs, i, byte_v)
        if k not in key_write_map:
            w_ind = curr_write_ind
            curr_write_ind += 1
            key_write_map[k] = w_ind
            if return_key:
                _set_out_keys(out_arrs, w_ind, key_arrs, i, k)
                # setitem_array_with_str(out_arrs[-1], w_ind, k)
                # out_arrs[-1][w_ind] = k
        else:
            w_ind = key_write_map[k]
        __update_redvars(local_redvars, data_in, w_ind, i, pivot_arr)
    for j in range(n_uniq_keys):
        __eval_res(local_redvars, out_arrs, j)

    hpat.dict_ext.byte_vec_free(byte_v)
    return out_arrs


def get_shuffle_data_send_buffs(sh, karrs, data):  # pragma: no cover
    return ()


@overload(get_shuffle_data_send_buffs)
def get_shuffle_data_send_buffs_overload(meta, key_arrs, data):
    n_keys = len(key_arrs.types)
    count = len(data.types)

    func_text = "def send_buff_impl(meta, key_arrs, data):\n"
    func_text += "  return ({}{})\n".format(','.join(["meta.send_buff_tup[{}]".format(
        i) for i in range(n_keys, count + n_keys)]),
        "," if count == 1 else "")  # single value needs comma to become tuple

    # print(func_text)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    send_buff_impl = loc_vars['send_buff_impl']
    return send_buff_impl


def get_key_dict(arr):  # pragma: no cover
    return dict()


@overload(get_key_dict)
def get_key_dict_overload(arr):
    """returns dictionary and possibly a byte_vec for multi-key case
    """
    # get byte_vec dict for multi-key case
    if isinstance(arr, types.BaseTuple) and len(arr.types) != 1:
        n_bytes = 0
        context = numba.targets.registry.cpu_target.target_context
        for t in arr.types:
            n_bytes += context.get_abi_sizeof(context.get_data_type(t.dtype))

        def _impl(arr):
            b_v = hpat.dict_ext.byte_vec_init(n_bytes, 0)
            b_dict = hpat.dict_ext.dict_byte_vec_int64_init()
            return b_dict, b_v
        return _impl

    # regular scalar keys
    dtype = arr.types[0].dtype
    func_text = "def k_dict_impl(arr):\n"
    func_text += "  b_v = hpat.dict_ext.byte_vec_init(1, 0)\n"
    func_text += "  return hpat.dict_ext.dict_{}_int64_init(), b_v\n".format(dtype)
    loc_vars = {}
    exec(func_text, {'hpat': hpat}, loc_vars)
    k_dict_impl = loc_vars['k_dict_impl']
    return k_dict_impl


def _getitem_keys(key_arrs, i, b_v):
    return key_arrs[i]


@overload(_getitem_keys)
def _getitem_keys_overload(arrs, ind, b_v):
    if isinstance(arrs, types.BaseTuple) and len(arrs.types) != 1:
        func_text = "def getitem_impl(arrs, ind, b_v):\n"
        offset = 0
        context = numba.targets.registry.cpu_target.target_context
        for i, t in enumerate(arrs.types):
            n_bytes = context.get_abi_sizeof(context.get_data_type(t.dtype))
            func_text += "  arr_ptr = arrs[{}].ctypes.data + ind * {}\n".format(i, n_bytes)
            func_text += "  hpat.dict_ext.byte_vec_set(b_v, {}, arr_ptr, {})\n".format(offset, n_bytes)
            offset += n_bytes

        func_text += "  return b_v\n"
        loc_vars = {}
        exec(func_text, {'hpat': hpat}, loc_vars)
        getitem_impl = loc_vars['getitem_impl']
        return getitem_impl

    return lambda arrs, ind, b_v: arrs[0][ind]


def _set_out_keys(out_arrs, w_ind, key_arrs, i, k):
    setitem_array_with_str(out_arrs[-1], w_ind, k)


@overload(_set_out_keys)
def _set_out_keys_overload(out_arrs, w_ind, key_arrs, i, k):
    if isinstance(key_arrs, types.BaseTuple):
        n_keys = len(key_arrs.types)
        n_outs = len(out_arrs.types)
        key_start = n_outs - n_keys

        func_text = "def set_keys_impl(out_arrs, w_ind, key_arrs, i, k):\n"
        for i in range(n_keys):
            func_text += "  setitem_array_with_str(out_arrs[{}], w_ind, key_arrs[{}][i])\n".format(
                key_start + i, i)

        # print(func_text)
        loc_vars = {}
        exec(func_text, {'setitem_array_with_str': setitem_array_with_str}, loc_vars)
        set_keys_impl = loc_vars['set_keys_impl']
        return set_keys_impl

    return _set_out_keys


def get_key_set(arr):  # pragma: no cover
    return set()


@overload(get_key_set)
def get_key_set_overload(arr):
    if arr == string_array_type or (isinstance(arr, types.BaseTuple)
                                    and len(arr.types) == 1 and arr.types[0] == string_array_type):
        return lambda arr: hpat.set_ext.init_set_string()

    if isinstance(arr, types.BaseTuple):
        def get_set_tup(arr):
            s = set()
            v = getitem_arr_tup_single(arr, 0)
            s.add(v)
            s.remove(v)
            return s
        return get_set_tup

    # hack to return set with specified type
    def get_set(arr):
        s = set()
        s.add(arr[0])
        s.remove(arr[0])
        return s

    return get_set


def alloc_agg_output(n_uniq_keys, out_dummy_tup, key_set, data_in, return_key):  # pragma: no cover
    return out_dummy_tup


@overload(alloc_agg_output)
def alloc_agg_output_overload(n_uniq_keys, out_dummy_tup, key_set,
                              data_in, return_key):

    # return key is either True or None
    if return_key == types.boolean:
        # TODO: handle pivot_table/crosstab with return key
        dtype = key_set.dtype
        key_types = list(dtype.types) if isinstance(dtype, types.BaseTuple) else [dtype]
        n_keys = len(key_types)
        assert out_dummy_tup.count == data_in.count + n_keys

        func_text = "def out_alloc_f(n_uniq_keys, out_dummy_tup, key_set, data_in, return_key):\n"
        for i in range(data_in.count):
            func_text += "  c_{} = empty_like_type(n_uniq_keys, out_dummy_tup[{}])\n".format(i, i)

        # string special case
        # TODO: handle strings in multi-key case
        if key_types == [string_type]:
            # TODO: handle unicode length
            func_text += "  num_total_chars = num_total_chars_set_string(key_set)\n"
            func_text += "  out_key_0 = pre_alloc_string_array(n_uniq_keys, num_total_chars)\n"
        else:
            for i, key_typ in enumerate(key_types):
                func_text += "  out_key_{} = np.empty(n_uniq_keys, np.{})\n".format(i, key_typ)

        func_text += "  return ({}{}{},)\n".format(
            ", ".join(["c_{}".format(i) for i in range(data_in.count)]),
            "," if data_in.count != 0 else "",
            ", ".join(["out_key_{}".format(i) for i in range(n_keys)]))

        loc_vars = {}
        # print(func_text)
        exec(func_text, {'empty_like_type': empty_like_type, 'np': np,
                         'pre_alloc_string_array': pre_alloc_string_array,
                         'num_total_chars_set_string': num_total_chars_set_string}, loc_vars)
        alloc_impl = loc_vars['out_alloc_f']
        return alloc_impl

    assert return_key == types.none

    def no_key_out_alloc(n_uniq_keys, out_dummy_tup, key_set, data_in, return_key):
        return alloc_arr_tup(n_uniq_keys, out_dummy_tup)

    return no_key_out_alloc

# TODO: fix BaseContext.get_function() used in is_true()
# @overload(bool)
# def bool_none_overload(v_t):
#     if v_t == types.none:
#         return lambda a: False


@infer_global(bool)
class BoolNoneTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        val_t = args[0]
        if val_t == types.none:
            return signature(types.boolean, *args)


@lower_builtin(bool, types.none)
def lower_column_mean_impl(context, builder, sig, args):
    res = context.compile_internal(builder, lambda a: False, sig, args)
    return res  # impl_ret_untracked(context, builder, sig.return_type, res)


def setitem_array_with_str(arr, i, v):  # pragma: no cover
    return


@overload(setitem_array_with_str)
def setitem_array_with_str_overload(arr, i, val):
    if arr == string_array_type:
        def setitem_str_arr(arr, i, val):
            arr[i] = val
        return setitem_str_arr

    # return_key == False case where val could be string resulting in typing
    # issue, no need to set
    if val == string_type:
        return lambda arr, i, val: None

    def setitem_impl(arr, i, val):
        arr[i] = val

    return setitem_impl


def _get_np_dtype(t):
    if t == types.NPDatetime('ns'):
        return "dt64_dtype"
    return "np.{}".format(t)


def gen_top_level_agg_func(key_names, return_key, red_var_typs, out_typs,
                           in_col_names, out_col_names, parallel):
    """create the top level aggregation function by generating text
    """

    # arg names
    in_names = tuple("in_{}".format(c) for c in in_col_names)
    out_names = tuple("out_{}".format(c) for c in out_col_names)
    key_args = ", ".join("key_{}".format(
        _sanitize_varname(c)) for c in key_names)

    in_args = ", ".join(in_names)
    if in_args != '':
        in_args = ", " + in_args

    # pass None instead of False to enable static specialization in
    # alloc_agg_output()
    return_key_p = "True" if return_key else "None"

    func_text = "def agg_top({}{}, pivot_arr):\n".format(key_args, in_args)
    func_text += "    data_redvar_dummy = ({}{})\n".format(
        ",".join(["np.empty(1, {})".format(_get_np_dtype(t)) for t in red_var_typs]),
        "," if len(red_var_typs) == 1 else "")
    func_text += "    out_dummy_tup = ({}{}{})\n".format(
        ",".join(["np.empty(1, {})".format(_get_np_dtype(t)) for t in out_typs]),
        "," if len(out_typs) != 0 else "",
        "{},".format(key_args) if return_key else "")
    func_text += "    data_in = ({}{})\n".format(",".join(in_names),
                                                 "," if len(in_names) == 1 else "")
    func_text += "    init_vals = __init_func()\n"

    out_keys = tuple("out_key_{}".format(
        _sanitize_varname(c)) for c in key_names)
    out_tup = ", ".join(out_names + out_keys if return_key else out_names)

    if parallel:
        func_text += ("    ({},) = parallel_agg(({},), data_redvar_dummy, "
                      "out_dummy_tup, data_in, init_vals, __update_redvars, "
                      "__combine_redvars, __eval_res, {}, pivot_arr)\n").format(out_tup, key_args, return_key_p)
    else:
        func_text += ("    ({},) = agg_seq_iter(({},), data_redvar_dummy, "
                      "out_dummy_tup, data_in, init_vals, __update_redvars, "
                      "__eval_res, {}, pivot_arr)\n").format(out_tup, key_args, return_key_p)

    func_text += "    return ({},)\n".format(out_tup)

    # print(func_text)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    agg_top = loc_vars['agg_top']
    return agg_top


def compile_to_optimized_ir(func, arg_typs, typingctx):
    state = namedtuple('State',
                       ['typingctx', 'targetctx', 'args', 'func_ir', 'typemap', 'return_type',
                        'calltypes', 'metadata'])

    # XXX are outside function's globals needed?
    code = func.code if hasattr(func, 'code') else func.__code__
    state.func_ir = get_ir_of_code({'numba': numba, 'np': np, 'hpat': hpat}, code)

    state.typingctx = typingctx
    state.args = arg_typs
    state.locals = {}
    state.metadata = {}

    # rename all variables to avoid conflict (init and eval nodes)
    var_table = get_name_var_table(state.func_ir.blocks)
    new_var_dict = {}
    for name, _ in var_table.items():
        new_var_dict[name] = mk_unique_var(name)
    replace_var_names(state.func_ir.blocks, new_var_dict)
    state.func_ir._definitions = build_definitions(state.func_ir.blocks)

    assert state.func_ir.arg_count == 1, "agg function should have one input"
    input_name = state.func_ir.arg_names[0]
    df_pass = hpat.hiframes.hiframes_untyped.HiFramesPass()
    df_pass.run_pass(state)
    remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir)
    state.typemap, return_type, state.calltypes = typed_passes.type_inference_stage(
        typingctx, state.func_ir, arg_typs, None)

    options = numba.targets.cpu.ParallelOptions(True)
    flags = compiler.Flags()
    state.targetctx = numba.targets.cpu.CPUContext(typingctx)

    preparfor_pass = numba.parfor.PreParforPass(state.func_ir, state.typemap, state.calltypes, state.typingctx, options)
    preparfor_pass.run()
    state.func_ir._definitions = build_definitions(state.func_ir.blocks)
    df_t_pass = hpat.hiframes.hiframes_typed.HiFramesTypedPass()
    df_t_pass.run_pass(state)
    numba.rewrites.rewrite_registry.apply('after-inference', state)
    parfor_pass = numba.parfor.ParforPass(state.func_ir, state.typemap,
                                          state.calltypes, return_type, state.typingctx,
                                          options, flags)
    parfor_pass.run()
    remove_dels(state.func_ir.blocks)
    # make sure eval nodes are after the parfor for easier extraction
    # TODO: extract an eval func more robustly
    numba.parfor.maximize_fusion(state.func_ir, state.func_ir.blocks, state.typemap, False)
    return state.func_ir, state


def get_agg_func_struct(agg_func, in_col_types, out_col_typs, typingctx,
                        targetctx, pivot_typ, pivot_values, is_crosstab):
    """find initialization, update, combine and final evaluation code of the
    aggregation function. Currently assuming that the function is single block
    and has one parfor.
    """
    all_reduce_vars = []
    all_vartypes = []
    all_init_nodes = []
    all_eval_funcs = []
    all_update_funcs = []
    all_combine_funcs = []
    typemap = {}
    calltypes = {}
    # offsets of reduce vars
    curr_offset = 0
    redvar_offsets = [0]

    if is_crosstab and len(in_col_types) == 0:
        # use dummy int input type for crosstab since doesn't have input
        in_col_types = [types.Array(types.intp, 1, 'C')]

    for in_col_typ in in_col_types:
        f_ir, pm = compile_to_optimized_ir(
            agg_func, tuple([in_col_typ]), typingctx)

        f_ir._definitions = build_definitions(f_ir.blocks)
        # TODO: support multiple top-level blocks
        assert len(f_ir.blocks) == 1 and 0 in f_ir.blocks, ("only simple functions"
                                                            " with one block supported for aggregation")
        block = f_ir.blocks[0]

        # find and ignore arg and size/shape nodes for input arr
        block_body, arr_var = _rm_arg_agg_block(block, pm.typemap)

        parfor_ind = -1
        for i, stmt in enumerate(block_body):
            if isinstance(stmt, numba.parfor.Parfor):
                assert parfor_ind == -1, "only one parfor for aggregation function"
                parfor_ind = i

        parfor = block_body[parfor_ind]
        remove_dels(parfor.loop_body)
        remove_dels({0: parfor.init_block})

        init_nodes = block_body[:parfor_ind] + parfor.init_block.body
        eval_nodes = block_body[parfor_ind + 1:]

        redvars, var_to_redvar = get_parfor_reductions(parfor, parfor.params,
                                                       pm.calltypes)

        # find reduce variables given their names
        reduce_vars = [0] * len(redvars)
        for stmt in init_nodes:
            if is_assign(stmt) and stmt.target.name in redvars:
                ind = redvars.index(stmt.target.name)
                reduce_vars[ind] = stmt.target
        var_types = [pm.typemap[v] for v in redvars]

        combine_func = gen_combine_func(f_ir, parfor, redvars, var_to_redvar,
                                        var_types, arr_var, pm, typingctx, targetctx)

        init_nodes = _mv_read_only_init_vars(init_nodes, parfor, eval_nodes)
        # remove len(arr) for string arrays (not handled by array analysis)
        if pm.typemap[arr_var.name] == string_array_type:
            init_nodes = _rm_len_str_arr(init_nodes, arr_var, f_ir)

        # XXX: update mutates parfor body
        update_func = gen_update_func(parfor, redvars, var_to_redvar, var_types,
                                      arr_var, in_col_typ, pm, typingctx, targetctx)

        eval_func = gen_eval_func(f_ir, eval_nodes, reduce_vars, var_types, pm, typingctx, targetctx)

        all_reduce_vars += reduce_vars
        all_vartypes += var_types
        all_init_nodes += init_nodes
        all_eval_funcs.append(eval_func)
        typemap.update(pm.typemap)
        calltypes.update(pm.calltypes)
        all_update_funcs.append(update_func)
        all_combine_funcs.append(combine_func)
        curr_offset += len(redvars)
        redvar_offsets.append(curr_offset)

    all_vartypes = (all_vartypes * len(pivot_values) if pivot_values is not None
                    else all_vartypes)
    all_reduce_vars = (all_reduce_vars * len(pivot_values) if pivot_values is not None
                       else all_reduce_vars)

    init_func = gen_init_func(all_init_nodes, all_reduce_vars, all_vartypes,
                              typingctx, targetctx)
    update_all_func = gen_all_update_func(all_update_funcs, all_vartypes,
                                          in_col_types, redvar_offsets, typingctx, targetctx, pivot_typ,
                                          pivot_values, is_crosstab)
    combine_all_func = gen_all_combine_func(all_combine_funcs, all_vartypes,
                                            redvar_offsets, typingctx, targetctx, pivot_typ, pivot_values)
    eval_all_func = gen_all_eval_func(all_eval_funcs, all_vartypes,
                                      redvar_offsets, out_col_typs, typingctx, targetctx, pivot_values)

    return AggFuncStruct(all_vartypes, init_func,
                         update_all_func, combine_all_func, eval_all_func)


def _mv_read_only_init_vars(init_nodes, parfor, eval_nodes):
    """move stmts that are only used in the parfor body to the beginning of
    parfor body. For example, in test_agg_seq_str, B='aa' should be moved.
    """
    # get parfor body usedefs
    use_defs = compute_use_defs(parfor.loop_body)
    parfor_uses = set()
    for s in use_defs.usemap.values():
        parfor_uses |= s
    parfor_defs = set()
    for s in use_defs.defmap.values():
        parfor_defs |= s

    # get uses of eval nodes
    dummy_block = ir.Block(ir.Scope(None, parfor.loc), parfor.loc)
    dummy_block.body = eval_nodes
    e_use_defs = compute_use_defs({0: dummy_block})
    e_uses = e_use_defs.usemap[0]

    # find stmts that are only used in parfor body
    i_uses = set()  # variables used later in init nodes
    new_init_nodes = []
    const_nodes = []
    for stmt in reversed(init_nodes):
        stmt_uses = {v.name for v in stmt.list_vars()}
        if is_assign(stmt):
            v = stmt.target.name
            stmt_uses.remove(v)
            # v is only used in parfor body
            if (v in parfor_uses and v not in i_uses and v not in e_uses
                    and v not in parfor_defs):
                const_nodes.append(stmt)
                i_uses |= stmt_uses
                continue
        i_uses |= stmt_uses
        new_init_nodes.append(stmt)

    const_nodes.reverse()
    new_init_nodes.reverse()

    first_body_label = min(parfor.loop_body.keys())
    first_block = parfor.loop_body[first_body_label]
    first_block.body = const_nodes + first_block.body
    return new_init_nodes


def _rm_len_str_arr(init_nodes, arr_var, f_ir):
    """remove len(arr_var) for init_nodes. len() still exists for string
    arrays since array analysis doesn't handle them.
    """
    new_init_nodes = []

    for stmt in reversed(init_nodes):
        if (is_call_assign(stmt)
                and find_callname(f_ir, stmt.value) == ('len', 'builtins')
                and stmt.value.args[0].name == arr_var.name):
            continue
        new_init_nodes.append(stmt)

    new_init_nodes.reverse()
    return new_init_nodes


def gen_init_func(init_nodes, reduce_vars, var_types, typingctx, targetctx):

    # parallelaccelerator adds functions that check the size of input array
    # these calls need to be removed
    _checker_calls = (numba.parfor.max_checker, numba.parfor.min_checker,
                      numba.parfor.argmax_checker, numba.parfor.argmin_checker)
    checker_vars = set()
    cleaned_init_nodes = []
    for stmt in init_nodes:
        if (is_assign(stmt) and isinstance(stmt.value, ir.Global)
                and stmt.value.value in _checker_calls):
            checker_vars.add(stmt.target.name)
        elif is_call_assign(stmt) and stmt.value.func.name in checker_vars:
            pass  # remove call
        else:
            cleaned_init_nodes.append(stmt)

    init_nodes = cleaned_init_nodes

    return_typ = types.Tuple(var_types)

    def dummy_f(): return None
    f_ir = compile_to_numba_ir(dummy_f, {})
    block = list(f_ir.blocks.values())[0]
    loc = block.loc

    # return initialized reduce vars as tuple
    tup_var = ir.Var(block.scope, mk_unique_var("init_tup"), loc)
    tup_assign = ir.Assign(ir.Expr.build_tuple(reduce_vars, loc), tup_var, loc)
    block.body = block.body[-2:]
    block.body = init_nodes + [tup_assign] + block.body
    block.body[-2].value.value = tup_var

    # compile implementation to binary (Dispatcher)
    init_all_func = compiler.compile_ir(typingctx, targetctx, f_ir, (), return_typ, compiler.DEFAULT_FLAGS, {})

    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](dummy_f)
    imp_dis.add_overload(init_all_func)
    return imp_dis


def gen_all_update_func(update_funcs, reduce_var_types, in_col_types,
                        redvar_offsets, typingctx, targetctx, pivot_typ, pivot_values,
                        is_crosstab):

    num_cols = len(in_col_types)
    if pivot_values is not None:
        assert num_cols == 1

    reduce_arrs_tup_typ = types.Tuple([types.Array(t, 1, 'C') for t in reduce_var_types])
    col_tup_typ = types.Tuple(in_col_types)
    arg_typs = (reduce_arrs_tup_typ, col_tup_typ, types.intp, types.intp, pivot_typ)

    # redvar_arrs[0][w_ind], redvar_arrs[1][w_ind] = __update_redvars(
    #              redvar_arrs[0][w_ind], redvar_arrs[1][w_ind], data_in[0][i])

    num_redvars = redvar_offsets[num_cols]

    func_text = "def update_all_f(redvar_arrs, data_in, w_ind, i, pivot_arr):\n"
    if pivot_values is not None:
        func_text += "  pv = pivot_arr[i]\n"
        for j, pv in enumerate(pivot_values):
            el = "el" if j != 0 else ""
            func_text += "  {}if pv == '{}':\n".format(el, pv)  # TODO: non-string pivot
            init_offset = num_redvars * j
            redvar_access = ", ".join(["redvar_arrs[{}][w_ind]".format(i)
                                       for i in range(init_offset + redvar_offsets[0], init_offset + redvar_offsets[1])])
            data_access = "data_in[0][i]"
            if is_crosstab:  # TODO: crosstab with values arg
                data_access = "0"
            func_text += "    {} = update_vars_0({}, {})\n".format(redvar_access, redvar_access, data_access)
    else:
        for j in range(num_cols):
            redvar_access = ", ".join(["redvar_arrs[{}][w_ind]".format(i)
                                       for i in range(redvar_offsets[j], redvar_offsets[j + 1])])
            func_text += "  {} = update_vars_{}({},  data_in[{}][i])\n".format(redvar_access, j, redvar_access, j)
    func_text += "  return\n"
    # print(func_text)

    glbs = {}
    for i, f in enumerate(update_funcs):
        glbs['update_vars_{}'.format(i)] = f
    loc_vars = {}
    exec(func_text, glbs, loc_vars)
    update_all_f = loc_vars['update_all_f']

    f_ir = compile_to_numba_ir(update_all_f, glbs)

    # compile implementation to binary (Dispatcher)
    update_all_func = compiler.compile_ir(typingctx, targetctx, f_ir, arg_typs, types.none, compiler.DEFAULT_FLAGS, {})

    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](update_all_f)
    imp_dis.add_overload(update_all_func)
    return imp_dis


def gen_all_combine_func(combine_funcs, reduce_var_types, redvar_offsets,
                         typingctx, targetctx, pivot_typ, pivot_values):

    reduce_arrs_tup_typ = types.Tuple([types.Array(t, 1, 'C')
                                       for t in reduce_var_types])
    arg_typs = (reduce_arrs_tup_typ, reduce_arrs_tup_typ, types.intp,
                types.intp, pivot_typ)

    num_cols = len(redvar_offsets) - 1
    num_redvars = redvar_offsets[num_cols]

    #       redvar_0_arr[w_ind], redvar_1_arr[w_ind] = __combine_redvars_0(
    #             redvar_0_arr[w_ind], redvar_1_arr[w_ind], in_c0[i], in_c1[i])
    #       redvar_2_arr[w_ind], redvar_3_arr[w_ind] = __combine_redvars_1(
    #             redvar_2_arr[w_ind], redvar_3_arr[w_ind], in_c2[i], in_c3[i])

    func_text = "def combine_all_f(redvar_arrs, recv_arrs, w_ind, i, pivot_arr):\n"

    if pivot_values is not None:
        assert num_cols == 1
        for k in range(len(pivot_values)):
            init_offset = num_redvars * k
            redvar_access = ", ".join(["redvar_arrs[{}][w_ind]".format(i)
                                       for i in range(init_offset + redvar_offsets[0], init_offset + redvar_offsets[1])])
            recv_access = ", ".join(["recv_arrs[{}][i]".format(i)
                                     for i in range(init_offset + redvar_offsets[0], init_offset + redvar_offsets[1])])
            func_text += "  {} = combine_vars_0({}, {})\n".format(
                redvar_access, redvar_access, recv_access)
    else:
        for j in range(num_cols):
            redvar_access = ", ".join(["redvar_arrs[{}][w_ind]".format(i)
                                       for i in range(redvar_offsets[j], redvar_offsets[j + 1])])
            recv_access = ", ".join(["recv_arrs[{}][i]".format(i)
                                     for i in range(redvar_offsets[j], redvar_offsets[j + 1])])
            func_text += "  {} = combine_vars_{}({}, {})\n".format(
                redvar_access, j, redvar_access, recv_access)
    func_text += "  return\n"
    # print(func_text)
    glbs = {}
    for i, f in enumerate(combine_funcs):
        glbs['combine_vars_{}'.format(i)] = f
    loc_vars = {}
    exec(func_text, glbs, loc_vars)
    combine_all_f = loc_vars['combine_all_f']

    f_ir = compile_to_numba_ir(combine_all_f, glbs)

    # compile implementation to binary (Dispatcher)
    combine_all_func = compiler.compile_ir(typingctx, targetctx, f_ir, arg_typs, types.none, compiler.DEFAULT_FLAGS, {})

    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](combine_all_f)
    imp_dis.add_overload(combine_all_func)
    return imp_dis


def gen_all_eval_func(eval_funcs, reduce_var_types, redvar_offsets,
                      out_col_typs, typingctx, targetctx, pivot_values):

    reduce_arrs_tup_typ = types.Tuple([types.Array(t, 1, 'C') for t in reduce_var_types])
    out_col_typs = types.Tuple(out_col_typs)
    arg_typs = (reduce_arrs_tup_typ, out_col_typs, types.intp)

    num_cols = len(redvar_offsets) - 1

    #       out_c0[j] = __eval_res_0(redvar_0_arr[j], redvar_1_arr[j])
    #       out_c1[j] = __eval_res_1(redvar_2_arr[j], redvar_3_arr[j])

    num_redvars = redvar_offsets[num_cols]

    func_text = "def eval_all_f(redvar_arrs, out_arrs, j):\n"

    if pivot_values is not None:
        assert num_cols == 1
        for j in range(len(pivot_values)):
            init_offset = num_redvars * j
            redvar_access = ", ".join(["redvar_arrs[{}][j]".format(i) for i in range(
                init_offset + redvar_offsets[0], init_offset + redvar_offsets[1])])
            func_text += "  out_arrs[{}][j] = eval_vars_0({})\n".format(j, redvar_access)
    else:
        for j in range(num_cols):
            redvar_access = ", ".join(["redvar_arrs[{}][j]".format(i)
                                       for i in range(redvar_offsets[j], redvar_offsets[j + 1])])
            func_text += "  out_arrs[{}][j] = eval_vars_{}({})\n".format(j, j, redvar_access)
    func_text += "  return\n"
    # print(func_text)
    glbs = {}
    for i, f in enumerate(eval_funcs):
        glbs['eval_vars_{}'.format(i)] = f
    loc_vars = {}
    exec(func_text, glbs, loc_vars)
    eval_all_f = loc_vars['eval_all_f']

    f_ir = compile_to_numba_ir(eval_all_f, glbs)

    # compile implementation to binary (Dispatcher)
    eval_all_func = compiler.compile_ir(typingctx, targetctx, f_ir, arg_typs, types.none, compiler.DEFAULT_FLAGS, {})

    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](eval_all_f)
    imp_dis.add_overload(eval_all_func)
    return imp_dis


def gen_eval_func(f_ir, eval_nodes, reduce_vars, var_types, pm, typingctx, targetctx):

    # eval func takes reduce vars and produces final result
    num_red_vars = len(var_types)
    in_names = ["in{}".format(i) for i in range(num_red_vars)]
    return_typ = pm.typemap[eval_nodes[-1].value.name]

    # TODO: non-numeric return
    zero = return_typ(0)
    func_text = "def agg_eval({}):\n return _zero\n".format(", ".join(in_names))

    # print(func_text)
    loc_vars = {}
    exec(func_text, {'_zero': zero}, loc_vars)
    agg_eval = loc_vars['agg_eval']

    arg_typs = tuple(var_types)
    f_ir = compile_to_numba_ir(agg_eval, {'numba': numba, 'hpat': hpat, 'np': np, '_zero': zero},  # TODO: add outside globals
                               typingctx, arg_typs,
                               pm.typemap, pm.calltypes)

    # TODO: support multi block eval funcs
    block = list(f_ir.blocks.values())[0]

    # assign inputs to reduce vars used in computation
    assign_nodes = []
    for i, v in enumerate(reduce_vars):
        assign_nodes.append(ir.Assign(block.body[i].target, v, v.loc))
    block.body = block.body[:num_red_vars] + assign_nodes + eval_nodes

    # compile implementation to binary (Dispatcher)
    eval_func = compiler.compile_ir(typingctx, targetctx, f_ir, arg_typs, return_typ, compiler.DEFAULT_FLAGS, {})

    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](agg_eval)
    imp_dis.add_overload(eval_func)
    return imp_dis


def gen_combine_func(f_ir, parfor, redvars, var_to_redvar, var_types, arr_var,
                     pm, typingctx, targetctx):
    num_red_vars = len(redvars)
    redvar_in_names = ["v{}".format(i) for i in range(num_red_vars)]
    in_names = ["in{}".format(i) for i in range(num_red_vars)]

    func_text = "def agg_combine({}):\n".format(", ".join(redvar_in_names + in_names))

    blocks = wrap_parfor_blocks(parfor)
    topo_order = find_topo_order(blocks)
    topo_order = topo_order[1:]  # ignore init block
    unwrap_parfor_blocks(parfor)

    special_combines = {}
    ignore_redvar_inds = []

    for label in topo_order:
        bl = parfor.loop_body[label]
        for stmt in bl.body:
            if is_call_assign(stmt) and (guard(find_callname, f_ir, stmt.value)
                                         == ('__special_combine', 'hpat.hiframes.aggregate')):
                args = stmt.value.args
                l_argnames = []
                r_argnames = []
                for v in args[:-1]:
                    ind = redvars.index(v.name)
                    ignore_redvar_inds.append(ind)
                    l_argnames.append("v{}".format(ind))
                    r_argnames.append("in{}".format(ind))
                comb_name = "__special_combine__{}".format(len(special_combines))
                func_text += "    ({},) = {}({})\n".format(
                    ", ".join(l_argnames), comb_name, ", ".join(l_argnames + r_argnames))
                dummy_call = ir.Expr.call(args[-1], [], (), bl.loc)
                sp_func = guard(find_callname, f_ir, dummy_call)
                # XXX: only var supported for now
                # TODO: support general functions
                assert sp_func == ('_var_combine', 'hpat.hiframes.aggregate')
                sp_func = hpat.hiframes.aggregate._var_combine
                special_combines[comb_name] = sp_func

            # reduction variables
            if is_assign(stmt) and stmt.target.name in redvars:
                red_var = stmt.target.name
                ind = redvars.index(red_var)
                if ind in ignore_redvar_inds:
                    continue
                if len(f_ir._definitions[red_var]) == 2:
                    # 0 is the actual func since init_block is traversed later
                    # in parfor.py:3039, TODO: make this detection more robust
                    # XXX trying both since init_prange doesn't work for min
                    var_def = f_ir._definitions[red_var][0]
                    func_text += _match_reduce_def(var_def, f_ir, ind)
                    var_def = f_ir._definitions[red_var][1]
                    func_text += _match_reduce_def(var_def, f_ir, ind)

    func_text += "    return {}".format(", ".join(["v{}".format(i)
                                                   for i in range(num_red_vars)]))
    # print(func_text)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    agg_combine = loc_vars['agg_combine']

    # reduction variable types for new input and existing values
    arg_typs = tuple(2 * var_types)

    glbs = {'numba': numba, 'hpat': hpat, 'np': np}
    glbs.update(special_combines)
    f_ir = compile_to_numba_ir(agg_combine, glbs,  # TODO: add outside globals
                               typingctx, arg_typs,
                               pm.typemap, pm.calltypes)

    block = list(f_ir.blocks.values())[0]

    return_typ = pm.typemap[block.body[-1].value.name]
    # compile implementation to binary (Dispatcher)
    combine_func = compiler.compile_ir(typingctx, targetctx, f_ir, arg_typs, return_typ, compiler.DEFAULT_FLAGS, {})

    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](agg_combine)
    imp_dis.add_overload(combine_func)
    return imp_dis


def _match_reduce_def(var_def, f_ir, ind):
    func_text = ""
    while isinstance(var_def, ir.Var):
        var_def = guard(get_definition, f_ir, var_def)
    # TODO: support other reductions
    if (isinstance(var_def, ir.Expr)
            and var_def.op == 'inplace_binop'
            and var_def.fn in ('+=', operator.iadd)):
        func_text = "    v{} += in{}\n".format(ind, ind)
    if (isinstance(var_def, ir.Expr) and var_def.op == 'call'):
        fdef = guard(find_callname, f_ir, var_def)
        if fdef == ('min', 'builtins'):
            func_text = "    v{} = min(v{}, in{})\n".format(ind, ind, ind)
        if fdef == ('max', 'builtins'):
            func_text = "    v{} = max(v{}, in{})\n".format(ind, ind, ind)
    return func_text


def gen_update_func(parfor, redvars, var_to_redvar, var_types, arr_var,
                    in_col_typ, pm, typingctx, targetctx):
    num_red_vars = len(redvars)
    var_types = [pm.typemap[v] for v in redvars]

    num_in_vars = 1

    # create input value variable for each reduction variable
    in_vars = []
    for i in range(num_in_vars):
        in_var = ir.Var(arr_var.scope, "$input{}".format(i), arr_var.loc)
        in_vars.append(in_var)

    # replace X[i] with input value
    index_var = parfor.loop_nests[0].index_variable
    red_ir_vars = [0] * num_red_vars
    for bl in parfor.loop_body.values():
        new_body = []
        for stmt in bl.body:
            # remove extra index assignment i = parfor_index for isna(A, i)
            if is_var_assign(stmt) and stmt.value.name == index_var.name:
                continue
            if is_getitem(stmt) and stmt.value.value.name == arr_var.name:
                stmt.value = in_vars[0]
            # XXX replace hpat.hiframes.api.isna(A, i) for now
            # TODO: handle actual NA
            # for test_agg_seq_count_str test
            if (is_call_assign(stmt) and guard(find_callname, pm.func_ir, stmt.value)
                    == ('isna', 'hpat.hiframes.api')
                    and stmt.value.args[0].name == arr_var.name):
                stmt.value = ir.Const(False, stmt.target.loc)
            # store reduction variables
            if is_assign(stmt) and stmt.target.name in redvars:
                ind = redvars.index(stmt.target.name)
                red_ir_vars[ind] = stmt.target
            new_body.append(stmt)
        bl.body = new_body

    redvar_in_names = ["v{}".format(i) for i in range(num_red_vars)]
    in_names = ["in{}".format(i) for i in range(num_in_vars)]

    func_text = "def agg_update({}):\n".format(", ".join(redvar_in_names + in_names))
    func_text += "    __update_redvars()\n"
    func_text += "    return {}".format(", ".join(["v{}".format(i)
                                                   for i in range(num_red_vars)]))

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    agg_update = loc_vars['agg_update']

    # XXX input column type can be different than reduction variable type
    arg_typs = tuple(var_types + [in_col_typ.dtype] * num_in_vars)

    f_ir = compile_to_numba_ir(agg_update, {'__update_redvars': __update_redvars},  # TODO: add outside globals
                               typingctx, arg_typs,
                               pm.typemap, pm.calltypes)

    f_ir._definitions = build_definitions(f_ir.blocks)

    body = f_ir.blocks.popitem()[1].body
    return_typ = pm.typemap[body[-1].value.name]

    blocks = wrap_parfor_blocks(parfor)
    topo_order = find_topo_order(blocks)
    topo_order = topo_order[1:]  # ignore init block
    unwrap_parfor_blocks(parfor)

    f_ir.blocks = parfor.loop_body
    first_block = f_ir.blocks[topo_order[0]]
    last_block = f_ir.blocks[topo_order[-1]]

    # arg assigns
    initial_assigns = body[:(num_red_vars + num_in_vars)]
    if num_red_vars > 1:
        # return nodes: build_tuple, cast, return
        return_nodes = body[-3:]
        assert (is_assign(return_nodes[0])
                and isinstance(return_nodes[0].value, ir.Expr)
                and return_nodes[0].value.op == 'build_tuple')
    else:
        # return nodes: cast, return
        return_nodes = body[-2:]

    # assign input reduce vars
    # redvar_i = v_i
    for i in range(num_red_vars):
        arg_var = body[i].target
        node = ir.Assign(arg_var, red_ir_vars[i], arg_var.loc)
        initial_assigns.append(node)

    # assign input value vars
    # redvar_in_i = in_i
    for i in range(num_red_vars, num_red_vars + num_in_vars):
        arg_var = body[i].target
        node = ir.Assign(arg_var, in_vars[i - num_red_vars], arg_var.loc)
        initial_assigns.append(node)

    first_block.body = initial_assigns + first_block.body

    # assign ouput reduce vars
    # v_i = red_var_i
    after_assigns = []
    for i in range(num_red_vars):
        arg_var = body[i].target
        node = ir.Assign(red_ir_vars[i], arg_var, arg_var.loc)
        after_assigns.append(node)

    last_block.body += after_assigns + return_nodes

    # TODO: simplify f_ir
    # compile implementation to binary (Dispatcher)
    agg_impl_func = compiler.compile_ir(typingctx, targetctx, f_ir, arg_typs, return_typ, compiler.DEFAULT_FLAGS, {})

    imp_dis = numba.targets.registry.dispatcher_registry['cpu'](agg_update)
    imp_dis.add_overload(agg_impl_func)
    return imp_dis


def _rm_arg_agg_block(block, typemap):
    block_body = []
    arr_var = None
    for i, stmt in enumerate(block.body):
        if is_assign(stmt) and isinstance(stmt.value, ir.Arg):
            arr_var = stmt.target
            arr_typ = typemap[arr_var.name]
            # string arrays don't have shape generated by array analysis
            if arr_typ == string_array_type:
                block_body += block.body[i + 1:]
                break
            assert isinstance(arr_typ, types.Array), "array type expected"
            # XXX assuming shape/size nodes are right after arg
            shape_nd = block.body[i + 1]
            assert (is_assign(shape_nd) and isinstance(shape_nd.value, ir.Expr)
                    and shape_nd.value.op == 'getattr' and shape_nd.value.attr == 'shape'
                    and shape_nd.value.value.name == arr_var.name)
            shape_vr = shape_nd.target
            size_nd = block.body[i + 2]
            assert (is_assign(size_nd) and isinstance(size_nd.value, ir.Expr)
                    and size_nd.value.op == 'static_getitem'
                    and size_nd.value.value.name == shape_vr.name)
            # ignore size/shape vars
            block_body += block.body[i + 3:]
            break
        block_body.append(stmt)

    return block_body, arr_var


# adapted from numba/parfor.py
def get_parfor_reductions(parfor, parfor_params, calltypes,
                          reduce_varnames=None, param_uses=None, var_to_param=None):
    """find variables that are updated using their previous values and an array
    item accessed with parfor index, e.g. s = s+A[i]
    """
    if reduce_varnames is None:
        reduce_varnames = []

    # for each param variable, find what other variables are used to update it
    # also, keep the related nodes
    if param_uses is None:
        param_uses = defaultdict(list)
    if var_to_param is None:
        var_to_param = {}

    blocks = wrap_parfor_blocks(parfor)
    topo_order = find_topo_order(blocks)
    topo_order = topo_order[1:]  # ignore init block
    unwrap_parfor_blocks(parfor)

    for label in reversed(topo_order):
        for stmt in reversed(parfor.loop_body[label].body):
            if (isinstance(stmt, ir.Assign)
                    and (stmt.target.name in parfor_params
                         or stmt.target.name in var_to_param)):
                lhs = stmt.target.name
                rhs = stmt.value
                cur_param = lhs if lhs in parfor_params else var_to_param[lhs]
                used_vars = []
                if isinstance(rhs, ir.Var):
                    used_vars = [rhs.name]
                elif isinstance(rhs, ir.Expr):
                    used_vars = [v.name for v in stmt.value.list_vars()]
                param_uses[cur_param].extend(used_vars)
                for v in used_vars:
                    var_to_param[v] = cur_param
            if isinstance(stmt, Parfor):
                # recursive parfors can have reductions like test_prange8
                get_parfor_reductions(stmt, parfor_params, calltypes,
                                      reduce_varnames, param_uses, var_to_param)

    for param, used_vars in param_uses.items():
        # a parameter is a reduction variable if its value is used to update it
        # check reduce_varnames since recursive parfors might have processed
        # param already
        if param in used_vars and param not in reduce_varnames:
            reduce_varnames.append(param)

    return reduce_varnames, var_to_param


def _build_set_tup(arr_tup):
    return build_set(arr_tup[0])


@overload(_build_set_tup)
def _build_set_tup_overload(arr_tup):
    # TODO: support string in tuple set
    if isinstance(arr_tup, types.BaseTuple) and len(arr_tup.types) != 1:
        def _impl(arr_tup):
            n = len(arr_tup[0])
            s = set()
            for i in range(n):
                s.add(getitem_arr_tup(arr_tup, i))
            return s
        return _impl
    return _build_set_tup


def _sanitize_varname(varname):
    return varname.replace('$', '_').replace('.', '_')

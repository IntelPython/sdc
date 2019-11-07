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


"""
.. module:: distributed.py

The description of the entire module will be here.
Supported and unsupported list can also be added here
"""
from __future__ import print_function, division, absolute_import
import operator
import types as pytypes  # avoid confusion with numba.types
import copy
import warnings
from collections import defaultdict
import numpy as np

import numba
from numba import ir, types, typing, config, numpy_support, ir_utils, postproc
from numba.ir_utils import (
    mk_unique_var,
    replace_vars_inner,
    find_topo_order,
    dprint_func_ir,
    remove_dead,
    mk_alloc,
    get_global_func_typ,
    get_name_var_table,
    get_call_table,
    get_tuple_table,
    remove_dels,
    compile_to_numba_ir,
    replace_arg_nodes,
    guard,
    get_definition,
    require,
    GuardException,
    find_callname,
    build_definitions,
    find_build_sequence,
    find_const,
    is_get_setitem)
from numba.inline_closurecall import inline_closure_call
from numba.typing import signature
from numba.parfor import (
    Parfor,
    lower_parfor_sequential,
    get_parfor_reductions,
    get_parfor_params,
    wrap_parfor_blocks,
    unwrap_parfor_blocks)

from numba.compiler_machinery import FunctionPass, register_pass

import hpat
import hpat.utils
from hpat import distributed_api, distributed_lower
from hpat.io.pio_api import h5file_type, h5group_type
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type
from hpat.distributed_api import Reduce_Type
from hpat.distributed_analysis import Distribution, DistributedAnalysis
from hpat.utils import (
    is_alloc_callname,
    is_whole_slice,
    is_array_container,
    get_slice_step,
    is_array,
    is_np_array,
    find_build_tuple,
    debug_prints,
    ReplaceFunc,
    gen_getitem,
    is_call,
    is_const_slice,
    update_globals)
from hpat.hiframes.pd_dataframe_ext import DataFrameType

distributed_run_extensions = {}

# analysis data for debugging
dist_analysis = None
fir_text = None


@register_pass(mutates_CFG=True, analysis_only=False)
class DistributedPass(FunctionPass):
    """The summary of the class should be here for example below is the summary line for this class

    This is an adapter for a new numba passes interface. Numba pass must be stateless. This class wraps statefull DistributedPassImpl
    """

    _name = "distributed_pass"

    def __init__(self):
        pass

    def run_pass(self, state):
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXX')
        # state.func_ir.dump()
        # print('YYYYYYYYYYYYYYYYYYYYYYYYYYY')
        res = DistributedPassImpl(state).run_pass()
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXX')
        # state.func_ir.dump()
        # print('YYYYYYYYYYYYYYYYYYYYYYYYYYY')
        return res

class DistributedPassImpl(object):
    """The summary of the class should be here for example below is the summary line for this class

    This class analyzes program and transforms to distributed
    """

    def __init__(self, state):
        self._dist_analysis = None
        self._T_arrs = None  # set of transposed arrays (taken from analysis)

        self._rank_var = None  # will be set in run
        self._size_var = None
        self._g_dist_var = None
        self._set1_var = None  # variable set to 1
        self._set0_var = None  # variable set to 0
        self._array_starts = {}
        self._array_counts = {}

        # keep shape attr calls on parallel arrays like X.shape
        self._shape_attrs = {}
        # keep array sizes of parallel arrays to handle shape attrs
        self._array_sizes = {}
        # save output of converted 1DVar array len() variables
        # which are global sizes, in order to recover local
        # size for 1DVar allocs and parfors
        self.oneDVar_len_vars = {}

        self.state = state

    def run_pass(self):
        remove_dels(self.state.func_ir.blocks)
        dprint_func_ir(self.state.func_ir, "starting distributed pass")
        self.state.func_ir._definitions = build_definitions(self.state.func_ir.blocks)
        dist_analysis_pass = DistributedAnalysis(
            self.state.func_ir, self.state.typemap, self.state.calltypes, self.state.typingctx,
            self.state.metadata)
        self._dist_analysis = dist_analysis_pass.run()
        # dprint_func_ir(self.state.func_ir, "after analysis distributed")

        self._T_arrs = dist_analysis_pass._T_arrs
        self._parallel_accesses = dist_analysis_pass._parallel_accesses
        if debug_prints():  # pragma: no cover
            print("distributions: ", self._dist_analysis)

        self._gen_dist_inits()
        self.state.func_ir._definitions = build_definitions(self.state.func_ir.blocks)
        self.state.func_ir.blocks = self._run_dist_pass(self.state.func_ir.blocks, 0)
        self.state.func_ir.blocks = self._dist_prints(self.state.func_ir.blocks)
        remove_dead(self.state.func_ir.blocks, self.state.func_ir.arg_names, self.state.func_ir, self.state.typemap)
        dprint_func_ir(self.state.func_ir, "after distributed pass")
        lower_parfor_sequential(
            self.state.typingctx, self.state.func_ir, self.state.typemap, self.state.calltypes)
        if hpat.multithread_mode:
            # parfor params need to be updated for multithread_mode since some
            # new variables like alloc_start are introduced by distributed pass
            # and are used in later parfors
            parfor_ids = get_parfor_params(
                self.state.func_ir.blocks, True, defaultdict(list))
        post_proc = postproc.PostProcessor(self.state.func_ir)
        post_proc.run()

        # save data for debug and test
        global dist_analysis, fir_text
        dist_analysis = self._dist_analysis
        import io
        str_io = io.StringIO()
        self.state.func_ir.dump(str_io)
        fir_text = str_io.getvalue()
        str_io.close()

        return True

    def _run_dist_pass(self, blocks, depth):
        """This function does something"""
        topo_order = find_topo_order(blocks)
        namevar_table = get_name_var_table(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            new_body = []
            replaced = False
            for i, inst in enumerate(block.body):
                out_nodes = None
                if type(inst) in distributed_run_extensions:
                    f = distributed_run_extensions[type(inst)]
                    out_nodes = f(inst, self._dist_analysis.array_dists,
                                  self.state.typemap, self.state.calltypes, self.state.typingctx,
                                  self.state.targetctx, self)
                elif isinstance(inst, Parfor):
                    print('********** process parfor **********')
                    out_nodes = self._run_parfor(inst, namevar_table, depth)
                    # run dist pass recursively
                    p_blocks = wrap_parfor_blocks(inst)
                    # build_definitions(p_blocks, self.state.func_ir._definitions)
                    self._run_dist_pass(p_blocks, depth + 1)
                    unwrap_parfor_blocks(inst)
                    print('********** process parfor end **********')
                elif isinstance(inst, ir.Assign):
                    lhs = inst.target.name
                    rhs = inst.value
                    if isinstance(rhs, ir.Expr):
                        out_nodes = self._run_expr(inst, namevar_table)
                    elif isinstance(rhs, ir.Var) and (self._is_1D_arr(rhs.name)
                                                      and not is_array_container(self.state.typemap, rhs.name)):
                        self._array_starts[lhs] = self._array_starts[rhs.name]
                        self._array_counts[lhs] = self._array_counts[rhs.name]
                        self._array_sizes[lhs] = self._array_sizes[rhs.name]
                    elif isinstance(rhs, ir.Arg):
                        out_nodes = self._run_arg(inst)
                elif isinstance(inst, (ir.StaticSetItem, ir.SetItem)):
                    if isinstance(inst, ir.SetItem):
                        index = inst.index
                    else:
                        index = inst.index_var
                    out_nodes = self._run_getsetitem(inst.target,
                                                     index, inst, inst)
                elif isinstance(inst, ir.Return):
                    out_nodes = self._gen_barrier() + [inst]

                if out_nodes is None:
                    new_body.append(inst)
                elif isinstance(out_nodes, list):
                    new_body += out_nodes
                elif isinstance(out_nodes, ReplaceFunc):
                    rp_func = out_nodes
                    if rp_func.pre_nodes is not None:
                        new_body.extend(rp_func.pre_nodes)
                    # inline_closure_call expects a call assignment
                    dummy_call = ir.Expr.call(
                        ir.Var(block.scope, "dummy", inst.loc),
                        rp_func.args, (), inst.loc)
                    if isinstance(inst, ir.Assign):
                        # replace inst.value to a call with target args
                        # as expected by inline_closure_call
                        inst.value = dummy_call
                    else:
                        # replace inst with dummy assignment
                        # for cases like SetItem
                        loc = block.loc
                        dummy_var = ir.Var(
                            block.scope, mk_unique_var("r_dummy"), loc)
                        block.body[i] = ir.Assign(dummy_call, dummy_var, loc)
                    block.body = new_body + block.body[i:]
                    # TODO: use Parfor loop blocks when replacing funcs in
                    # parfor loop body
                    update_globals(rp_func.func, rp_func.glbls)
                    inline_closure_call(self.state.func_ir, rp_func.glbls,
                                        block, len(new_body), rp_func.func, self.state.typingctx,
                                        rp_func.arg_types,
                                        self.state.typemap, self.state.calltypes, work_list)
                    replaced = True
                    break
                else:
                    assert False, "invalid dist pass out nodes"

            if not replaced:
                blocks[label].body = new_body
                # print('******************** new_body')
                # print('\n'.join([str(a) for a in new_body]))
                # print('********************')

        return blocks

    def _run_expr(self, inst, namevar_table):
        lhs = inst.target.name
        rhs = inst.value
        nodes = [inst]
        if rhs.op == 'call':
            return self._run_call(inst)
        # we save array start/count for data pointer to enable
        # file read
        if (rhs.op == 'getattr' and rhs.attr == 'ctypes'
                and (self._is_1D_arr(rhs.value.name))):
            arr_name = rhs.value.name
            self._array_starts[lhs] = self._array_starts[arr_name]
            self._array_counts[lhs] = self._array_counts[arr_name]
            self._array_sizes[lhs] = self._array_sizes[arr_name]
        if (rhs.op == 'getattr'
                and (self._is_1D_arr(rhs.value.name)
                     or self._is_1D_Var_arr(rhs.value.name))
                and rhs.attr == 'size'):
            return self._run_array_size(inst.target, rhs.value)
        if (rhs.op == 'static_getitem'
                and rhs.value.name in self._shape_attrs):
            arr = self._shape_attrs[rhs.value.name]
            ndims = self.state.typemap[arr].ndim
            if arr not in self._T_arrs and rhs.index == 0:
                # return parallel size
                if self._is_1D_arr(arr):
                    # XXX hack for array container case, TODO: handle properly
                    if arr not in self._array_sizes:
                        arr_var = namevar_table[arr]
                        nodes = self._gen_1D_Var_len(arr_var)
                        nodes[-1].target = inst.target
                        return nodes
                    inst.value = self._array_sizes[arr][rhs.index]
                else:
                    assert self._is_1D_Var_arr(arr)
                    arr_var = namevar_table[arr]
                    nodes = self._gen_1D_Var_len(arr_var)
                    nodes[-1].target = inst.target
                    # save output of converted 1DVar array len() variables
                    # which are global sizes, in order to recover local
                    # size for 1DVar allocs and parfors
                    self.oneDVar_len_vars[inst.target.name] = arr_var
                    return nodes
            # last dimension of transposed arrays is partitioned
            if arr in self._T_arrs and rhs.index == ndims - 1:
                assert not self._is_1D_Var_arr(
                    arr), "1D_Var arrays cannot transpose"
                inst.value = self._array_sizes[arr][rhs.index]
        if rhs.op in ['getitem', 'static_getitem']:
            if rhs.op == 'getitem':
                index = rhs.index
            else:
                index = rhs.index_var
            return self._run_getsetitem(rhs.value, index, rhs, inst)
        if (rhs.op == 'getattr'
                and (self._is_1D_arr(rhs.value.name)
                     or self._is_1D_Var_arr(rhs.value.name))
                and rhs.attr == 'shape'):
            # XXX: return a new tuple using sizes here?
            self._shape_attrs[lhs] = rhs.value.name
        if (rhs.op == 'getattr'
                and self._is_1D_arr(rhs.value.name)
                and rhs.attr == 'T'):
            assert lhs in self._T_arrs
            orig_arr = rhs.value.name
            self._array_starts[lhs] = copy.copy(
                self._array_starts[orig_arr]).reverse()
            self._array_counts[lhs] = copy.copy(
                self._array_counts[orig_arr]).reverse()
            self._array_sizes[lhs] = copy.copy(
                self._array_sizes[orig_arr]).reverse()
        if (rhs.op == 'exhaust_iter'
                and rhs.value.name in self._shape_attrs):
            self._shape_attrs[lhs] = self._shape_attrs[rhs.value.name]
        if rhs.op == 'inplace_binop' and self._is_1D_arr(rhs.lhs.name):
            self._array_starts[lhs] = self._array_starts[rhs.lhs.name]
            self._array_counts[lhs] = self._array_counts[rhs.lhs.name]
            self._array_sizes[lhs] = self._array_sizes[rhs.lhs.name]

        return nodes

    def _gen_1D_Var_len(self, arr):
        def f(A, op):  # pragma: no cover
            c = len(A)
            res = hpat.distributed_api.dist_reduce(c, op)
        f_block = compile_to_numba_ir(f, {'hpat': hpat}, self.state.typingctx,
                                      (self.state.typemap[arr.name], types.int32),
                                      self.state.typemap, self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(
            f_block, [arr, ir.Const(Reduce_Type.Sum.value, arr.loc)])
        nodes = f_block.body[:-3]  # remove none return
        return nodes

    def _gen_dist_inits(self):
        # add initializations
        topo_order = find_topo_order(self.state.func_ir.blocks)
        first_block = self.state.func_ir.blocks[topo_order[0]]
        # set scope and loc of generated code to the first variable in block
        scope = first_block.scope
        loc = first_block.loc
        out = []
        self._set1_var = ir.Var(scope, mk_unique_var("$const_parallel"), loc)
        self.state.typemap[self._set1_var.name] = types.int64
        set1_assign = ir.Assign(ir.Const(1, loc), self._set1_var, loc)
        out.append(set1_assign)
        self._set0_var = ir.Var(scope, mk_unique_var("$const_parallel"), loc)
        self.state.typemap[self._set0_var.name] = types.int64
        set0_assign = ir.Assign(ir.Const(0, loc), self._set0_var, loc)
        out.append(set0_assign)
        # g_dist_var = Global(hpat.distributed_api)
        g_dist_var = ir.Var(scope, mk_unique_var("$distributed_g_var"), loc)
        self._g_dist_var = g_dist_var
        self.state.typemap[g_dist_var.name] = types.misc.Module(hpat.distributed_api)
        g_dist = ir.Global('distributed_api', hpat.distributed_api, loc)
        g_dist_assign = ir.Assign(g_dist, g_dist_var, loc)
        # attr call: rank_attr = getattr(g_dist_var, get_rank)
        rank_attr_call = ir.Expr.getattr(g_dist_var, "get_rank", loc)
        rank_attr_var = ir.Var(scope, mk_unique_var("$get_rank_attr"), loc)
        self.state.typemap[rank_attr_var.name] = get_global_func_typ(
            distributed_api.get_rank)
        rank_attr_assign = ir.Assign(rank_attr_call, rank_attr_var, loc)
        # rank_var = hpat.distributed_api.get_rank()
        rank_var = ir.Var(scope, mk_unique_var("$rank"), loc)
        self.state.typemap[rank_var.name] = types.int32
        rank_call = ir.Expr.call(rank_attr_var, [], (), loc)
        self.state.calltypes[rank_call] = self.state.typemap[rank_attr_var.name].get_call_type(
            self.state.typingctx, [], {})
        rank_assign = ir.Assign(rank_call, rank_var, loc)
        self._rank_var = rank_var
        out += [g_dist_assign, rank_attr_assign, rank_assign]

        # attr call: size_attr = getattr(g_dist_var, get_size)
        size_attr_call = ir.Expr.getattr(g_dist_var, "get_size", loc)
        size_attr_var = ir.Var(scope, mk_unique_var("$get_size_attr"), loc)
        self.state.typemap[size_attr_var.name] = get_global_func_typ(
            distributed_api.get_size)
        size_attr_assign = ir.Assign(size_attr_call, size_attr_var, loc)
        # size_var = hpat.distributed_api.get_size()
        size_var = ir.Var(scope, mk_unique_var("$dist_size"), loc)
        self.state.typemap[size_var.name] = types.int32
        size_call = ir.Expr.call(size_attr_var, [], (), loc)
        self.state.calltypes[size_call] = self.state.typemap[size_attr_var.name].get_call_type(
            self.state.typingctx, [], {})
        size_assign = ir.Assign(size_call, size_var, loc)
        self._size_var = size_var
        out += [size_attr_assign, size_assign]
        first_block.body = out + first_block.body

    def _run_call(self, assign):
        lhs = assign.target.name
        rhs = assign.value
        func_var = rhs.func.name
        scope = assign.target.scope
        loc = assign.target.loc
        out = [assign]

        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.state.func_ir, rhs, self.state.typemap)
        if fdef is None:
            # FIXME: since parfors are transformed and then processed
            # recursively, some funcs don't have definitions. The generated
            # arrays should be assigned REP and the var definitions added.
            # warnings.warn(
            #     "function call couldn't be found for distributed pass")
            return out
        else:
            func_name, func_mod = fdef

        # divide 1D alloc
        # XXX allocs should be matched before going to _run_call_np
        if self._is_1D_arr(lhs) and is_alloc_callname(func_name, func_mod):
            # XXX for pre_alloc_string_array(n, nc), we assume nc is local
            # value (updated only in parfor like _str_replace_regex_impl)
            size_var = rhs.args[0]
            out, new_size_var = self._run_alloc(size_var, lhs, scope, loc)
            # empty_inferred is tuple for some reason
            rhs.args = list(rhs.args)
            rhs.args[0] = new_size_var
            out.append(assign)
            return out

        # fix 1D_Var allocs in case global len of another 1DVar is used
        if self._is_1D_Var_arr(lhs) and is_alloc_callname(func_name, func_mod):
            size_var = rhs.args[0]
            out, new_size_var = self._fix_1D_Var_alloc(
                size_var, lhs, scope, loc)
            # empty_inferred is tuple for some reason
            rhs.args = list(rhs.args)
            rhs.args[0] = new_size_var
            out.append(assign)
            return out

        # numpy direct functions
        if isinstance(func_mod, str) and func_mod == 'numpy':
            return self._run_call_np(lhs, func_name, assign, rhs.args)

        # array.func calls
        if isinstance(func_mod, ir.Var) and is_np_array(self.state.typemap, func_mod.name):
            return self._run_call_array(lhs, func_mod, func_name, assign, rhs.args)

        # df.func calls
        if isinstance(func_mod, ir.Var) and isinstance(self.state.typemap[func_mod.name], DataFrameType):
            return self._run_call_df(lhs, func_mod, func_name, assign, rhs.args)

        # string_array.func_calls
        if (self._is_1D_arr(lhs) and isinstance(func_mod, ir.Var)
                and self.state.typemap[func_mod.name] == string_array_type):
            if func_name == 'copy':
                self._array_starts[lhs] = self._array_starts[func_mod.name]
                self._array_counts[lhs] = self._array_counts[func_mod.name]
                self._array_sizes[lhs] = self._array_sizes[func_mod.name]

        if fdef == ('permutation', 'numpy.random'):
            if self.state.typemap[rhs.args[0].name] == types.int64:
                self._array_sizes[lhs] = [rhs.args[0]]
                return self._run_permutation_int(assign, rhs.args)

        # len(A) if A is 1D
        if fdef == ('len', 'builtins') and rhs.args and self._is_1D_arr(rhs.args[0].name):
            arr = rhs.args[0].name
            assign.value = self._array_sizes[arr][0]

        # len(A) if A is 1D_Var
        if fdef == ('len', 'builtins') and rhs.args and self._is_1D_Var_arr(rhs.args[0].name):
            arr_var = rhs.args[0]
            out = self._gen_1D_Var_len(arr_var)
            out[-1].target = assign.target
            self.oneDVar_len_vars[assign.target.name] = arr_var

        if (hpat.config._has_h5py and (func_mod == 'hpat.io.pio_api'
                                       and func_name in ('h5read', 'h5write', 'h5read_filter'))
                and self._is_1D_arr(rhs.args[5].name)):
            # TODO: make create_dataset/create_group collective
            arr = rhs.args[5].name
            ndims = len(self._array_starts[arr])
            starts_var = ir.Var(scope, mk_unique_var("$h5_starts"), loc)
            self.state.typemap[starts_var.name] = types.UniTuple(
                types.int64, ndims)
            start_tuple_call = ir.Expr.build_tuple(
                self._array_starts[arr], loc)
            starts_assign = ir.Assign(start_tuple_call, starts_var, loc)
            rhs.args[2] = starts_var
            counts_var = ir.Var(scope, mk_unique_var("$h5_counts"), loc)
            self.state.typemap[counts_var.name] = types.UniTuple(
                types.int64, ndims)
            count_tuple_call = ir.Expr.build_tuple(
                self._array_counts[arr], loc)
            counts_assign = ir.Assign(count_tuple_call, counts_var, loc)
            out = [starts_assign, counts_assign, assign]
            rhs.args[3] = counts_var
            rhs.args[4] = self._set1_var
            # set parallel arg in file open
            file_varname = rhs.args[0].name
            self._file_open_set_parallel(file_varname)

        if hpat.config._has_h5py and (func_mod == 'hpat.io.pio_api'
                                      and func_name == 'get_filter_read_indices'):
            #
            out += self._gen_1D_Var_len(assign.target)
            size_var = out[-1].target
            self._array_sizes[lhs] = [size_var]
            g_out, start_var, count_var = self._gen_1D_div(
                size_var, scope, loc, "$alloc", "get_node_portion",
                distributed_api.get_node_portion)
            self._array_starts[lhs] = [start_var]
            self._array_counts[lhs] = [count_var]
            out += g_out

        if (hpat.config._has_pyarrow
                and fdef == ('read_parquet', 'hpat.io.parquet_pio')
                and self._is_1D_arr(rhs.args[2].name)):
            arr = rhs.args[2].name
            assert len(self._array_starts[arr]) == 1, "only 1D arrs in parquet"
            start_var = self._array_starts[arr][0]
            count_var = self._array_counts[arr][0]
            rhs.args += [start_var, count_var]

            def f(fname, cindex, arr, out_dtype, start, count):  # pragma: no cover
                return hpat.io.parquet_pio.read_parquet_parallel(fname, cindex,
                                                                 arr, out_dtype, start, count)

            return self._replace_func(f, rhs.args)

        if (hpat.config._has_pyarrow
                and fdef == ('read_parquet_str', 'hpat.io.parquet_pio')
                and self._is_1D_arr(lhs)):
            arr = lhs
            size_var = rhs.args[2]
            assert self.state.typemap[size_var.name] == types.intp
            self._array_sizes[arr] = [size_var]
            out, start_var, count_var = self._gen_1D_div(size_var, scope, loc,
                                                         "$alloc", "get_node_portion", distributed_api.get_node_portion)
            self._array_starts[lhs] = [start_var]
            self._array_counts[lhs] = [count_var]
            rhs.args[2] = start_var
            rhs.args.append(count_var)

            def f(fname, cindex, start, count):  # pragma: no cover
                return hpat.io.parquet_pio.read_parquet_str_parallel(fname, cindex,
                                                                     start, count)

            f_block = compile_to_numba_ir(f, {'hpat': hpat}, self.state.typingctx,
                                          (self.state.typemap[rhs.args[0].name], types.intp,
                                           types.intp, types.intp),
                                          self.state.typemap, self.state.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, rhs.args)
            out += f_block.body[:-2]
            out[-1].target = assign.target

        # TODO: fix numba.extending
        if hpat.config._has_xenon and (fdef == ('read_xenon_col', 'numba.extending')
                                       and self._is_1D_arr(rhs.args[3].name)):
            arr = rhs.args[3].name
            assert len(self._array_starts[arr]) == 1, "only 1D arrs in Xenon"
            start_var = self._array_starts[arr][0]
            count_var = self._array_counts[arr][0]
            rhs.args += [start_var, count_var]

            def f(connect_tp, dset_tp, col_id_tp, column_tp, schema_arr_tp, start, count):  # pragma: no cover
                return hpat.io.xenon_ext.read_xenon_col_parallel(
                    connect_tp, dset_tp, col_id_tp, column_tp, schema_arr_tp, start, count)

            return self._replace_func(f, rhs.args)

        if hpat.config._has_xenon and (fdef == ('read_xenon_str', 'numba.extending')
                                       and self._is_1D_arr(lhs)):
            arr = lhs
            size_var = rhs.args[3]
            assert self.state.typemap[size_var.name] == types.intp
            self._array_sizes[arr] = [size_var]
            out, start_var, count_var = self._gen_1D_div(size_var, scope, loc,
                                                         "$alloc", "get_node_portion", distributed_api.get_node_portion)
            self._array_starts[lhs] = [start_var]
            self._array_counts[lhs] = [count_var]
            rhs.args.remove(size_var)
            rhs.args.append(start_var)
            rhs.args.append(count_var)

            def f(connect_tp, dset_tp, col_id_tp, schema_arr_tp, start_tp, count_tp):  # pragma: no cover
                return hpat.io.xenon_ext.read_xenon_str_parallel(
                    connect_tp, dset_tp, col_id_tp, schema_arr_tp, start_tp, count_tp)

            f_block = compile_to_numba_ir(f,
                                          {'hpat': hpat},
                                          self.state.typingctx,
                                          (hpat.io.xenon_ext.xe_connect_type,
                                           hpat.io.xenon_ext.xe_dset_type,
                                           types.intp,
                                           self.state.typemap[rhs.args[3].name],
                                              types.intp,
                                              types.intp),
                                          self.state.typemap,
                                          self.state.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, rhs.args)
            out += f_block.body[:-2]
            out[-1].target = assign.target

        if (hpat.config._has_ros
                and fdef == ('read_ros_images_inner', 'hpat.ros')
                and self._is_1D_arr(rhs.args[0].name)):
            arr = rhs.args[0].name
            assert len(self._array_starts[arr]) == 4, "only 4D arrs in ros"
            start_var = self._array_starts[arr][0]
            count_var = self._array_counts[arr][0]
            rhs.args += [start_var, count_var]

            def f(arr, bag, start, count):  # pragma: no cover
                return hpat.ros.read_ros_images_inner_parallel(arr, bag,
                                                               start, count)

            return self._replace_func(f, rhs.args)

        if (func_mod == 'hpat.hiframes.api' and func_name in (
                'to_arr_from_series', 'ts_series_to_arr_typ',
                'to_date_series_type', 'init_series')
                and self._is_1D_arr(rhs.args[0].name)):
            # TODO: handle index
            in_arr = rhs.args[0].name
            self._array_starts[lhs] = self._array_starts[in_arr]
            self._array_counts[lhs] = self._array_counts[in_arr]
            self._array_sizes[lhs] = self._array_sizes[in_arr]

        if (fdef == ('init_dataframe', 'hpat.hiframes.pd_dataframe_ext')
                and self._is_1D_arr(rhs.args[0].name)):
            in_arr = rhs.args[0].name
            self._array_starts[lhs] = self._array_starts[in_arr]
            self._array_counts[lhs] = self._array_counts[in_arr]
            self._array_sizes[lhs] = self._array_sizes[in_arr]

        if (fdef == ('compute_split_view', 'hpat.hiframes.split_impl')
                and self._is_1D_arr(rhs.args[0].name)):
            in_arr = rhs.args[0].name
            self._array_starts[lhs] = self._array_starts[in_arr]
            self._array_counts[lhs] = self._array_counts[in_arr]
            self._array_sizes[lhs] = self._array_sizes[in_arr]

        if (fdef == ('get_split_view_index', 'hpat.hiframes.split_impl')
                and self._is_1D_arr(rhs.args[0].name)):
            arr = rhs.args[0]
            index_var = rhs.args[1]
            sub_nodes = self._get_ind_sub(
                index_var, self._array_starts[arr.name][0])
            out = sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            out.append(assign)
            return out

        if (fdef == ('setitem_str_arr_ptr', 'hpat.str_arr_ext')
                and self._is_1D_arr(rhs.args[0].name)):
            arr = rhs.args[0]
            index_var = rhs.args[1]
            sub_nodes = self._get_ind_sub(
                index_var, self._array_starts[arr.name][0])
            out = sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            out.append(assign)
            return out

        if (fdef == ('str_arr_item_to_numeric', 'hpat.str_arr_ext')
                and self._is_1D_arr(rhs.args[0].name)):
            # TODO: test parallel
            arr = rhs.args[0]
            index_var = rhs.args[1]
            sub_nodes = self._get_ind_sub(
                index_var, self._array_starts[arr.name][0])
            out = sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            # input string array
            arr = rhs.args[2]
            index_var = rhs.args[3]
            sub_nodes = self._get_ind_sub(
                index_var, self._array_starts[arr.name][0])
            out += sub_nodes
            rhs.args[3] = sub_nodes[-1].target
            out.append(assign)
            return out

        if fdef == ('isna', 'hpat.hiframes.api') and self._is_1D_arr(rhs.args[0].name):
            # fix index in call to isna
            arr = rhs.args[0]
            ind = rhs.args[1]
            out = self._get_ind_sub(ind, self._array_starts[arr.name][0])
            rhs.args[1] = out[-1].target
            out.append(assign)

        if fdef == ('rolling_fixed', 'hpat.hiframes.rolling') and (
                self._is_1D_arr(rhs.args[0].name)
                or self._is_1D_Var_arr(rhs.args[0].name)):
            in_arr = rhs.args[0].name
            if self._is_1D_arr(in_arr):
                self._array_starts[lhs] = self._array_starts[in_arr]
                self._array_counts[lhs] = self._array_counts[in_arr]
                self._array_sizes[lhs] = self._array_sizes[in_arr]
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.state.typemap[true_var.name] = types.boolean
            rhs.args[3] = true_var
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if fdef == ('rolling_variable', 'hpat.hiframes.rolling') and (
                self._is_1D_arr(rhs.args[0].name)
                or self._is_1D_Var_arr(rhs.args[0].name)):
            in_arr = rhs.args[0].name
            if self._is_1D_arr(in_arr):
                self._array_starts[lhs] = self._array_starts[in_arr]
                self._array_counts[lhs] = self._array_counts[in_arr]
                self._array_sizes[lhs] = self._array_sizes[in_arr]
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.state.typemap[true_var.name] = types.boolean
            rhs.args[4] = true_var
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if (func_mod == 'hpat.hiframes.rolling'
            and func_name in ('shift', 'pct_change')
            and (self._is_1D_arr(rhs.args[0].name)
                 or self._is_1D_Var_arr(rhs.args[0].name))):
            in_arr = rhs.args[0].name
            if self._is_1D_arr(in_arr):
                self._array_starts[lhs] = self._array_starts[in_arr]
                self._array_counts[lhs] = self._array_counts[in_arr]
                self._array_sizes[lhs] = self._array_sizes[in_arr]
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.state.typemap[true_var.name] = types.boolean
            rhs.args[2] = true_var
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if fdef == ('quantile', 'hpat.hiframes.api') and (self._is_1D_arr(rhs.args[0].name)
                                                          or self._is_1D_Var_arr(rhs.args[0].name)):
            arr = rhs.args[0].name
            if arr in self._array_sizes:
                assert len(self._array_sizes[arr]
                           ) == 1, "only 1D arrs in quantile"
                size_var = self._array_sizes[arr][0]
            else:
                size_var = self._set0_var
            rhs.args += [size_var]

            def f(arr, q, size):
                return hpat.hiframes.api.quantile_parallel(arr, q, size)
            return self._replace_func(f, rhs.args)

        if fdef == (
            'nunique', 'hpat.hiframes.api') and (
            self._is_1D_arr(
                rhs.args[0].name) or self._is_1D_Var_arr(
                rhs.args[0].name)):

            def f(arr):
                return hpat.hiframes.api.nunique_parallel(arr)

            return self._replace_func(f, rhs.args)

        if fdef == (
            'unique', 'hpat.hiframes.api') and (
            self._is_1D_arr(
                rhs.args[0].name) or self._is_1D_Var_arr(
                rhs.args[0].name)):

            def f(arr):
                return hpat.hiframes.api.unique_parallel(arr)

            return self._replace_func(f, rhs.args)

        if fdef == (
            'nlargest', 'hpat.hiframes.api') and (
            self._is_1D_arr(
                rhs.args[0].name) or self._is_1D_Var_arr(
                rhs.args[0].name)):

            def f(arr, k, i, f):
                return hpat.hiframes.api.nlargest_parallel(arr, k, i, f)

            return self._replace_func(f, rhs.args)

        if fdef == (
            'median', 'hpat.hiframes.api') and (
            self._is_1D_arr(
                rhs.args[0].name) or self._is_1D_Var_arr(
                rhs.args[0].name)):

            def f(arr):
                return hpat.hiframes.api.median(arr, True)

            return self._replace_func(f, rhs.args)

        if fdef == ('convert_rec_to_tup', 'hpat.hiframes.api'):
            # optimize Series back to back map pattern with tuples
            # TODO: create another optimization pass?
            arg_def = guard(get_definition, self.state.func_ir, rhs.args[0])
            if (is_call(arg_def) and guard(find_callname, self.state.func_ir, arg_def)
                    == ('convert_tup_to_rec', 'hpat.hiframes.api')):
                assign.value = arg_def.args[0]
            return out

        if fdef == ('dist_return', 'hpat.distributed_api'):
            # always rebalance returned distributed arrays
            # TODO: need different flag for 1D_Var return (distributed_var)?
            # TODO: rebalance strings?
            # return [assign]  # self._run_call_rebalance_array(lhs, assign, rhs.args)
            assign.value = rhs.args[0]
            return [assign]

        if ((fdef == ('get_series_data', 'hpat.hiframes.api')
             or fdef == ('get_series_index', 'hpat.hiframes.api')
             or fdef == ('get_dataframe_data', 'hpat.hiframes.pd_dataframe_ext'))):
            out = [assign]
            arr = assign.target
            # gen len() using 1D_Var reduce approach.
            # TODO: refactor to avoid reduction for 1D
            # arr_typ = self.state.typemap[arr.name]
            ndim = 1
            out += self._gen_1D_Var_len(arr)
            total_length = out[-1].target
            div_nodes, start_var, count_var = self._gen_1D_div(
                total_length, arr.scope, arr.loc, "$input", "get_node_portion", distributed_api.get_node_portion)
            out += div_nodes

            # XXX: get sizes in lower dimensions
            self._array_starts[lhs] = [-1] * ndim
            self._array_counts[lhs] = [-1] * ndim
            self._array_sizes[lhs] = [-1] * ndim
            self._array_starts[lhs][0] = start_var
            self._array_counts[lhs][0] = count_var
            self._array_sizes[lhs][0] = total_length

            return out

        if fdef == ('threaded_return', 'hpat.distributed_api'):
            assign.value = rhs.args[0]
            return [assign]

        if fdef == ('rebalance_array', 'hpat.distributed_api'):
            return self._run_call_rebalance_array(lhs, assign, rhs.args)

        # output of mnb.predict is 1D with same size as 1st dimension of input
        # TODO: remove ml module and use new DAAL API
        if func_name == 'predict':
            getattr_call = guard(get_definition, self.state.func_ir, func_var)
            if (getattr_call and self.state.typemap[getattr_call.value.name] == hpat.ml.naive_bayes.mnb_type):
                in_arr = rhs.args[0].name
                self._array_starts[lhs] = [self._array_starts[in_arr][0]]
                self._array_counts[lhs] = [self._array_counts[in_arr][0]]
                self._array_sizes[lhs] = [self._array_sizes[in_arr][0]]

        if fdef == ('file_read', 'hpat.io.np_io') and rhs.args[1].name in self._array_starts:
            _fname = rhs.args[0]
            _data_ptr = rhs.args[1]
            _start = self._array_starts[_data_ptr.name][0]
            _count = self._array_counts[_data_ptr.name][0]

            def f(fname, data_ptr, start, count):  # pragma: no cover
                return hpat.io.np_io.file_read_parallel(fname, data_ptr, start, count)
            return self._replace_func(f, [_fname, _data_ptr, _start, _count])

        return out

    def _run_call_np(self, lhs, func_name, assign, args):
        """transform np.func() calls
        """
        # allocs are handled separately
        is_1D_bool = (self._is_1D_Var_arr(lhs) or self._is_1D_arr(lhs))
        err_str = "allocation calls handled separately 'empty', 'zeros', 'ones', 'full' etc."
        assert not (is_1D_bool and func_name in hpat.utils.np_alloc_callnames), err_str

        out = [assign]
        scope = assign.target.scope
        loc = assign.loc

        # numba doesn't support np.reshape() form yet
        # if func_name == 'reshape':
        #     size_var = args[1]
        #     # handle reshape like new allocation
        #     out, new_size_var = self._run_alloc(size_var, lhs)
        #     args[1] = new_size_var
        #     out.append(assign)

        if (func_name == 'array' and is_array(self.state.typemap, args[0].name) and self._is_1D_arr(args[0].name)):
            in_arr = args[0].name
            self._array_starts[lhs] = self._array_starts[in_arr]
            self._array_counts[lhs] = self._array_counts[in_arr]
            self._array_sizes[lhs] = self._array_sizes[in_arr]

        # output array has same properties (starts etc.) as input array
        if (func_name in ['cumsum', 'cumprod', 'empty_like', 'zeros_like', 'ones_like',
                          'full_like', 'copy', 'ravel', 'ascontiguousarray'] and self._is_1D_arr(args[0].name)):
            if func_name == 'ravel':
                assert self.state.typemap[args[0].name].ndim == 1, "only 1D ravel supported"
            in_arr = args[0].name
            self._array_starts[lhs] = self._array_starts[in_arr]
            self._array_counts[lhs] = self._array_counts[in_arr]
            self._array_sizes[lhs] = self._array_sizes[in_arr]

        if (func_name in ['cumsum', 'cumprod'] and self._is_1D_arr(args[0].name)):
            in_arr = args[0].name
            in_arr_var = args[0]
            lhs_var = assign.target
            # allocate output array
            # TODO: compute inplace if input array is dead
            out = mk_alloc(
                self.state.typemap,
                self.state.calltypes,
                lhs_var,
                tuple(
                    self._array_sizes[in_arr]),
                self.state.typemap[in_arr].dtype,
                scope,
                loc)
            # generate distributed call
            dist_attr_var = ir.Var(scope, mk_unique_var("$dist_attr"), loc)
            dist_func_name = "dist_" + func_name
            dist_func = getattr(distributed_api, dist_func_name)
            dist_attr_call = ir.Expr.getattr(self._g_dist_var, dist_func_name, loc)
            self.state.typemap[dist_attr_var.name] = get_global_func_typ(dist_func)
            dist_func_assign = ir.Assign(dist_attr_call, dist_attr_var, loc)
            err_var = ir.Var(scope, mk_unique_var("$dist_err_var"), loc)
            self.state.typemap[err_var.name] = types.int32
            dist_call = ir.Expr.call(dist_attr_var, [in_arr_var, lhs_var], (), loc)
            self.state.calltypes[dist_call] = self.state.typemap[dist_attr_var.name].get_call_type(
                self.state.typingctx, [self.state.typemap[in_arr], self.state.typemap[lhs]], {})
            dist_assign = ir.Assign(dist_call, err_var, loc)
            return out + [dist_func_assign, dist_assign]

        # sum over the first axis is distributed, A.sum(0)
        if func_name == 'sum' and len(args) == 2:
            axis_def = guard(get_definition, self.state.func_ir, args[1])
            if isinstance(axis_def, ir.Const) and axis_def.value == 0:
                reduce_op = Reduce_Type.Sum
                reduce_var = assign.target
                return out + self._gen_reduce(reduce_var, reduce_op, scope, loc)

        if func_name == 'dot':
            return self._run_call_np_dot(lhs, assign, args)

        if func_name == 'stack' and self._is_1D_arr(lhs):
            # TODO: generalize
            in_arrs, _ = guard(find_build_sequence, self.state.func_ir, args[0])
            arr0 = in_arrs[0].name
            self._array_starts[lhs] = [self._array_starts[arr0][0], None]
            self._array_counts[lhs] = [self._array_counts[arr0][0], None]
            self._array_sizes[lhs] = [self._array_sizes[arr0][0], None]

        return out

    def _run_call_array(self, lhs, arr, func_name, assign, args):
        #
        out = [assign]
        if func_name in ('astype', 'copy') and self._is_1D_arr(lhs):
            self._array_starts[lhs] = self._array_starts[arr.name]
            self._array_counts[lhs] = self._array_counts[arr.name]
            self._array_sizes[lhs] = self._array_sizes[arr.name]

        # HACK support A.reshape(n, 1) for 1D_Var
        if func_name == 'reshape' and self._is_1D_Var_arr(arr.name):
            assert len(args) == 2 and guard(find_const, self.state.func_ir, args[1]) == 1
            assert args[0].name in self.oneDVar_len_vars
            size_var = args[0]
            out, new_size_var = self._fix_1D_Var_alloc(size_var, lhs, assign.target.scope, assign.loc)
            # empty_inferred is tuple for some reason
            assign.value.args = list(args)
            assign.value.args[0] = new_size_var
            out.append(assign)
            return out

        if func_name == 'reshape' and self._is_1D_arr(arr.name):
            return self._run_reshape(assign, arr, args)

        if func_name == 'transpose' and self._is_1D_arr(lhs):
            # Currently only 1D arrays are supported
            assert self._is_1D_arr(arr.name)
            ndim = self.state.typemap[arr.name].ndim
            self._array_starts[lhs] = [-1] * ndim
            self._array_counts[lhs] = [-1] * ndim
            self._array_sizes[lhs] = [-1] * ndim
            self._array_starts[lhs][0] = self._array_starts[arr.name][0]
            self._array_counts[lhs][0] = self._array_counts[arr.name][0]
            self._array_sizes[lhs][0] = self._array_sizes[arr.name][0]

        # TODO: refactor
        # TODO: add unittest
        if func_name == 'tofile':
            if self._is_1D_arr(arr.name):
                _fname = args[0]
                _start = self._array_starts[arr.name][0]
                _count = self._array_counts[arr.name][0]

                def f(fname, arr, start, count):  # pragma: no cover
                    return hpat.io.np_io.file_write_parallel(fname, arr, start, count)

                return self._replace_func(f, [_fname, arr, _start, _count])

            if self._is_1D_Var_arr(arr.name):
                _fname = args[0]

                def f(fname, arr):  # pragma: no cover
                    count = len(arr)
                    start = hpat.distributed_api.dist_exscan(count)
                    return hpat.io.np_io.file_write_parallel(fname, arr, start, count)

                return self._replace_func(f, [_fname, arr])

        return out

    def _run_call_df(self, lhs, df, func_name, assign, args):
        if func_name == 'to_csv' and self._is_1D_arr(df.name):
            # set index to proper range if None
            # avoid header for non-zero ranks
            # write to string then parallel file write
            # df.to_csv(fname) ->
            # l = len(df)
            # index_start = dist_exscan(l)
            # df2 = df(index=range(index_start, index_start+l))
            # header = header and is_root  # only first line has header
            # str_out = df2.to_csv(None, header=header)
            # hpat.io.np_io._file_write_parallel(fname, str_out)

            df_typ = self.state.typemap[df.name]
            rhs = assign.value
            fname = args[0]

            # update df index and get to_csv from new df
            nodes = self._fix_parallel_df_index(df)
            new_df = nodes[-1].target
            new_df_typ = self.state.typemap[new_df.name]
            new_to_csv = ir.Expr.getattr(new_df, 'to_csv', new_df.loc)
            new_func = ir.Var(new_df.scope, mk_unique_var('func'), new_df.loc)
            self.state.typemap[new_func.name] = self.state.typingctx.resolve_getattr(
                new_df_typ, 'to_csv')
            nodes.append(ir.Assign(new_to_csv, new_func, new_df.loc))
            rhs.func = new_func

            # # header = header and is_root
            kws = dict(rhs.kws)
            true_var = ir.Var(assign.target.scope, mk_unique_var('true'), rhs.loc)
            self.state.typemap[true_var.name] = types.bool_
            nodes.append(
                ir.Assign(ir.Const(True, new_df.loc), true_var, new_df.loc))
            header_var = self._get_arg(
                'to_csv', rhs.args, kws, 5, 'header', true_var)
            nodes += self._gen_is_root_and_cond(header_var)
            header_var = nodes[-1].target
            if len(rhs.args) > 5:
                rhs.args[5] = header_var
            else:
                kws['header'] = header_var
                rhs.kws = kws

            # fix to_csv() type to have None as 1st arg
            call_type = self.state.calltypes.pop(rhs)
            arg_typs = list((types.none,) + call_type.args[1:])
            arg_typs[5] = types.bool_
            arg_typs = tuple(arg_typs)
            # self.state.calltypes[rhs] = self.state.typemap[rhs.func.name].get_call_type(
            #      self.state.typingctx, arg_typs, {})
            self.state.calltypes[rhs] = numba.typing.Signature(
                string_type, arg_typs, new_df_typ,
                call_type.pysig)

            # None as 1st arg
            none_var = ir.Var(assign.target.scope, mk_unique_var('none'), rhs.loc)
            self.state.typemap[none_var.name] = types.none
            none_assign = ir.Assign(ir.Const(None, rhs.loc), none_var, rhs.loc)
            nodes.append(none_assign)
            rhs.args[0] = none_var

            # str_out = df.to_csv(None)
            str_out = ir.Var(assign.target.scope, mk_unique_var('write_csv'), rhs.loc)
            self.state.typemap[str_out.name] = string_type
            new_assign = ir.Assign(rhs, str_out, rhs.loc)
            nodes.append(new_assign)

            # print_node = ir.Print([str_out], None, rhs.loc)
            # self.state.calltypes[print_node] = signature(types.none, string_type)
            # nodes.append(print_node)

            # TODO: fix lazy IO load
            from . import hio
            import llvmlite.binding as ll
            ll.add_symbol('file_write_parallel', hio.file_write_parallel)
            # HACK use the string in a dummy function to avoid refcount issues
            # TODO: fix string data reference count
            dummy_use = numba.njit(lambda a: None)

            def f(fname, str_out):  # pragma: no cover
                count = len(str_out)
                start = hpat.distributed_api.dist_exscan(count)
                hpat.io.np_io._file_write_parallel(
                    fname._data, str_out._data, start, count, 1)
                dummy_use(str_out)

            return self._replace_func(
                f, [fname, str_out], pre_nodes=nodes)

        return [assign]

    def _gen_is_root_and_cond(self, cond_var):
        def f(cond):
            return cond & (hpat.distributed_api.get_rank() == 0)
        f_block = compile_to_numba_ir(f, {'hpat': hpat},
                                      self.state.typingctx,
                                      (self.state.typemap[cond_var.name],),
                                      self.state.typemap,
                                      self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [cond_var])
        nodes = f_block.body[:-2]
        return nodes

    def _fix_parallel_df_index(self, df):
        def f(df):  # pragma: no cover
            length = len(df)
            start = hpat.distributed_api.dist_exscan(length)
            ind = np.arange(start, start + length)
            df2 = hpat.hiframes.pd_dataframe_ext.set_df_index(df, ind)
            return df2

        f_block = compile_to_numba_ir(f, {'hpat': hpat, 'np': np},
                                      self.state.typingctx,
                                      (self.state.typemap[df.name],),
                                      self.state.typemap,
                                      self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [df])
        nodes = f_block.body[:-2]
        return nodes

    def _run_permutation_int(self, assign, args):
        lhs = assign.target
        n = args[0]

        def f(lhs, n):
            hpat.distributed_lower.dist_permutation_int(lhs, n)

        f_block = compile_to_numba_ir(f, {'hpat': hpat},
                                      self.state.typingctx,
                                      (self.state.typemap[lhs.name], types.intp),
                                      self.state.typemap,
                                      self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [lhs, n])
        f_block.body = [assign] + f_block.body
        return f_block.body[:-3]

    # Returns an IR node that defines a variable holding the size of |dtype|.
    def dtype_size_assign_ir(self, dtype, scope, loc):
        context = numba.targets.cpu.CPUContext(self.state.typingctx)
        dtype_size = context.get_abi_sizeof(context.get_data_type(dtype))
        dtype_size_var = ir.Var(scope, mk_unique_var("dtype_size"), loc)
        self.state.typemap[dtype_size_var.name] = types.intp
        return ir.Assign(ir.Const(dtype_size, loc), dtype_size_var, loc)

    def _run_permutation_array_index(self, lhs, rhs, idx):
        scope, loc = lhs.scope, lhs.loc
        dtype = self.state.typemap[lhs.name].dtype
        out = mk_alloc(self.state.typemap, self.state.calltypes, lhs,
                       (self._array_counts[lhs.name][0],
                        *self._array_sizes[lhs.name][1:]), dtype, scope, loc)

        def f(lhs, lhs_len, dtype_size, rhs, idx, idx_len):
            hpat.distributed_lower.dist_permutation_array_index(
                lhs, lhs_len, dtype_size, rhs, idx, idx_len)

        f_block = compile_to_numba_ir(f, {'hpat': hpat},
                                      self.state.typingctx,
                                      (self.state.typemap[lhs.name],
                                       types.intp,
                                       types.intp,
                                       self.state.typemap[rhs.name],
                                       self.state.typemap[idx.name],
                                       types.intp),
                                      self.state.typemap,
                                      self.state.calltypes).blocks.popitem()[1]
        dtype_ir = self.dtype_size_assign_ir(dtype, scope, loc)
        out.append(dtype_ir)
        replace_arg_nodes(f_block, [lhs, self._array_sizes[lhs.name][0],
                                    dtype_ir.target, rhs, idx,
                                    self._array_sizes[idx.name][0]])
        f_block.body = out + f_block.body
        return f_block.body[:-3]

    def _run_reshape(self, assign, in_arr, args):
        lhs = assign.target
        scope = assign.target.scope
        loc = assign.target.loc
        if len(args) == 1:
            new_shape = args[0]
        else:
            # reshape can take list of ints
            new_shape = args
        # TODO: avoid alloc and copy if no communication necessary
        # get new local shape in reshape and set start/count vars like new allocation
        out, new_local_shape_var = self._run_alloc(new_shape, lhs.name, scope, loc)
        # get actual tuple for mk_alloc
        if len(args) != 1:
            sh_list = guard(find_build_tuple, self.state.func_ir, new_local_shape_var)
            assert sh_list is not None, "invalid shape in reshape"
            new_local_shape_var = tuple(sh_list)
        dtype = self.state.typemap[in_arr.name].dtype
        out += mk_alloc(self.state.typemap, self.state.calltypes, lhs,
                        new_local_shape_var, dtype, scope, loc)

        def f(lhs, in_arr, new_0dim_global_len, old_0dim_global_len, dtype_size):  # pragma: no cover
            hpat.distributed_lower.dist_oneD_reshape_shuffle(
                lhs, in_arr, new_0dim_global_len, old_0dim_global_len, dtype_size)

        f_block = compile_to_numba_ir(f, {'hpat': hpat},
                                      self.state.typingctx,
                                      (self.state.typemap[lhs.name], self.state.typemap[in_arr.name],
                                       types.intp, types.intp, types.intp),
                                      self.state.typemap, self.state.calltypes).blocks.popitem()[1]
        dtype_ir = self.dtype_size_assign_ir(dtype, scope, loc)
        out.append(dtype_ir)
        replace_arg_nodes(f_block, [lhs, in_arr, self._array_sizes[lhs.name][0],
                                    self._array_sizes[in_arr.name][0],
                                    dtype_ir.target])
        out += f_block.body[:-3]
        return out
        # if len(args) == 1:
        #     args[0] = new_size_var
        # else:
        #     args[0] = self._tuple_table[new_size_var.name][0]
        # out.append(assign)

    def _run_call_rebalance_array(self, lhs, assign, args):
        out = [assign]
        if not self._is_1D_Var_arr(args[0].name):
            if self._is_1D_arr(args[0].name):
                in_1d_arr = args[0].name
                self._array_starts[lhs] = self._array_starts[in_1d_arr]
                self._array_counts[lhs] = self._array_counts[in_1d_arr]
                self._array_sizes[lhs] = self._array_sizes[in_1d_arr]
            else:
                warnings.warn("array {} is not 1D_Block_Var".format(args[0].name))
            return out

        arr = args[0]
        ndim = self.state.typemap[arr.name].ndim
        out = self._gen_1D_Var_len(arr)
        total_length = out[-1].target
        div_nodes, start_var, count_var = self._gen_1D_div(
            total_length, arr.scope, arr.loc, "$rebalance", "get_node_portion",
            distributed_api.get_node_portion)
        out += div_nodes

        # XXX: get sizes in lower dimensions
        self._array_starts[lhs] = [-1] * ndim
        self._array_counts[lhs] = [-1] * ndim
        self._array_sizes[lhs] = [-1] * ndim
        self._array_starts[lhs][0] = start_var
        self._array_counts[lhs][0] = count_var
        self._array_sizes[lhs][0] = total_length

        def f(arr, count):  # pragma: no cover
            b_arr = hpat.distributed_api.rebalance_array_parallel(arr, count)

        f_block = compile_to_numba_ir(f, {'hpat': hpat}, self.state.typingctx,
                                      (self.state.typemap[arr.name], types.intp),
                                      self.state.typemap, self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr, count_var])
        out += f_block.body[:-3]
        out[-1].target = assign.target
        return out

    def _run_call_np_dot(self, lhs, assign, args):
        out = [assign]
        arg0 = args[0].name
        arg1 = args[1].name
        ndim0 = self.state.typemap[arg0].ndim
        ndim1 = self.state.typemap[arg1].ndim
        t0 = arg0 in self._T_arrs
        t1 = arg1 in self._T_arrs

        # reduction across dataset
        if self._is_1D_arr(arg0) and self._is_1D_arr(arg1):
            dprint("run dot dist reduce:", arg0, arg1)
            reduce_op = Reduce_Type.Sum
            reduce_var = assign.target
            out += self._gen_reduce(reduce_var, reduce_op, reduce_var.scope,
                                    reduce_var.loc)

        # assign starts/counts/sizes data structures for output array
        if ndim0 == 2 and ndim1 == 1 and not t0 and self._is_1D_arr(arg0):
            # special case were arg1 vector is treated as column vector
            # samples dot weights: np.dot(X,w)
            # output is 1D array same size as dim 0 of X
            assert self.state.typemap[lhs].ndim == 1
            assert self._is_1D_arr(lhs)
            self._array_starts[lhs] = [self._array_starts[arg0][0]]
            self._array_counts[lhs] = [self._array_counts[arg0][0]]
            self._array_sizes[lhs] = [self._array_sizes[arg0][0]]
            dprint("run dot case 1 Xw:", arg0, arg1)
        if ndim0 == 2 and ndim1 == 2 and not t0 and not t1:
            # samples dot weights: np.dot(X,w)
            assert self._is_1D_arr(lhs)
            # first dimension is same as X
            # second dimension not needed
            self._array_starts[lhs] = [self._array_starts[arg0][0], -1]
            self._array_counts[lhs] = [self._array_counts[arg0][0], -1]
            self._array_sizes[lhs] = [self._array_sizes[arg0][0], -1]
            dprint("run dot case 4 Xw:", arg0, arg1)

        return out

    def _run_alloc(self, size_var, lhs, scope, loc):
        """ divides array sizes and assign its sizes/starts/counts attributes
        returns generated nodes and the new size variable to enable update of
        the alloc call.
        """
        out = []
        new_size_var = None

        # size is single int var
        if isinstance(size_var, ir.Var) and isinstance(self.state.typemap[size_var.name], types.Integer):
            self._array_sizes[lhs] = [size_var]
            out, start_var, end_var = self._gen_1D_div(size_var, scope, loc,
                                                       "$alloc", "get_node_portion", distributed_api.get_node_portion)
            self._array_starts[lhs] = [start_var]
            self._array_counts[lhs] = [end_var]
            new_size_var = end_var
            return out, new_size_var

        # tuple variable of ints
        if isinstance(size_var, ir.Var):
            # see if size_var is a 1D array's shape
            # it is already the local size, no need to transform
            var_def = guard(get_definition, self.state.func_ir, size_var)
            oned_varnames = set(v for v in self._dist_analysis.array_dists
                                if self._dist_analysis.array_dists[v] == Distribution.OneD)
            if (isinstance(var_def, ir.Expr) and var_def.op == 'getattr'
                    and var_def.attr == 'shape' and var_def.value.name in oned_varnames):
                prev_arr = var_def.value.name
                self._array_starts[lhs] = self._array_starts[prev_arr]
                self._array_counts[lhs] = self._array_counts[prev_arr]
                self._array_sizes[lhs] = self._array_sizes[prev_arr]
                return out, size_var

            # size should be either int or tuple of ints
            #assert size_var.name in self._tuple_table
            # self._tuple_table[size_var.name]
            size_list = self._get_tuple_varlist(size_var, out)
            size_list = [ir_utils.convert_size_to_var(s, self.state.typemap, scope, loc, out)
                         for s in size_list]
        # tuple of int vars
        else:
            assert isinstance(size_var, (tuple, list))
            size_list = list(size_var)

        self._array_sizes[lhs] = size_list
        gen_nodes, start_var, end_var = self._gen_1D_div(size_list[0], scope, loc,
                                                         "$alloc", "get_node_portion", distributed_api.get_node_portion)
        out += gen_nodes
        ndims = len(size_list)
        new_size_list = copy.copy(size_list)
        new_size_list[0] = end_var
        tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
        self.state.typemap[tuple_var.name] = types.containers.UniTuple(
            types.intp, ndims)
        tuple_call = ir.Expr.build_tuple(new_size_list, loc)
        tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
        out.append(tuple_assign)
        self.state.func_ir._definitions[tuple_var.name] = [tuple_call]
        self._array_starts[lhs] = [self._set0_var] * ndims
        self._array_starts[lhs][0] = start_var
        self._array_counts[lhs] = new_size_list
        new_size_var = tuple_var
        return out, new_size_var

    def _fix_1D_Var_alloc(self, size_var, lhs, scope, loc):
        """ OneD_Var allocs use sizes of other OneD_var variables,
        so find the local size of those variables (since we transform
        to use global size)
        """
        out = []
        new_size_var = None

        # size is single int var
        if isinstance(size_var, ir.Var) and isinstance(self.state.typemap[size_var.name], types.Integer):
            # array could be allocated inside 1D_Var nodes like sort
            if size_var.name not in self.oneDVar_len_vars:
                return [], size_var
            # assert size_var.name in self.oneDVar_len_vars, "invalid 1DVar alloc"
            arr_var = self.oneDVar_len_vars[size_var.name]

            def f(oneD_var_arr):  # pragma: no cover
                arr_len = len(oneD_var_arr)
            f_block = compile_to_numba_ir(f, {'hpat': hpat}, self.state.typingctx,
                                          (self.state.typemap[arr_var.name],),
                                          self.state.typemap, self.state.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [arr_var])
            out = f_block.body[:-3]  # remove none return
            new_size_var = out[-1].target
            return out, new_size_var

        # tuple variable of ints
        if isinstance(size_var, ir.Var):
            # see if size_var is a 1D_Var array's shape
            # it is already the local size, no need to transform
            var_def = guard(get_definition, self.state.func_ir, size_var)
            oned_var_varnames = set(v for v in self._dist_analysis.array_dists
                                    if self._dist_analysis.array_dists[v] == Distribution.OneD_Var)
            if (isinstance(var_def, ir.Expr) and var_def.op == 'getattr'
                    and var_def.attr == 'shape' and var_def.value.name in oned_var_varnames):
                return out, size_var
            # size should be either int or tuple of ints
            #assert size_var.name in self._tuple_table
            # self._tuple_table[size_var.name]
            size_list = self._get_tuple_varlist(size_var, out)
            size_list = [ir_utils.convert_size_to_var(s, self.state.typemap, scope, loc, out)
                         for s in size_list]
        # tuple of int vars
        else:
            assert isinstance(size_var, (tuple, list))
            size_list = list(size_var)

        arr_var = self.oneDVar_len_vars[size_list[0].name]

        def f(oneD_var_arr):  # pragma: no cover
            arr_len = len(oneD_var_arr)
        f_block = compile_to_numba_ir(f, {'hpat': hpat}, self.state.typingctx,
                                      (self.state.typemap[arr_var.name],),
                                      self.state.typemap, self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr_var])
        nodes = f_block.body[:-3]  # remove none return
        new_size_var = nodes[-1].target
        out += nodes

        ndims = len(size_list)
        new_size_list = copy.copy(size_list)
        new_size_list[0] = new_size_var
        tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
        self.state.typemap[tuple_var.name] = types.containers.UniTuple(
            types.intp, ndims)
        tuple_call = ir.Expr.build_tuple(new_size_list, loc)
        tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
        out.append(tuple_assign)
        self.state.func_ir._definitions[tuple_var.name] = [tuple_call]
        return out, tuple_var

    # new_body += self._run_1D_array_shape(
    #                                inst.target, rhs.value)
    # def _run_1D_array_shape(self, lhs, arr):
    #     """return shape tuple with global size of 1D/1D_Var arrays
    #     """
    #     nodes = []
    #     if self._is_1D_arr(arr.name):
    #         dim1_size = self._array_sizes[arr.name][0]
    #     else:
    #         assert self._is_1D_Var_arr(arr.name)
    #         nodes += self._gen_1D_Var_len(arr)
    #         dim1_size = nodes[-1].target
    #
    #     ndim = self._get_arr_ndim(arr.name)
    #
    #     func_text = "def f(arr, dim1):\n"
    #     func_text += "    s = (dim1, {})\n".format(
    #         ",".join(["arr.shape[{}]".format(i) for i in range(1, ndim)]))
    #     loc_vars = {}
    #     exec(func_text, {}, loc_vars)
    #     f = loc_vars['f']
    #
    #     f_ir = compile_to_numba_ir(f, {'np': np}, self.state.typingctx,
    #                                (self.state.typemap[arr.name], types.intp),
    #                                self.state.typemap, self.state.calltypes)
    #     f_block = f_ir.blocks.popitem()[1]
    #     replace_arg_nodes(f_block, [arr, dim1_size])
    #     nodes += f_block.body[:-3]
    #     nodes[-1].target = lhs
    #     return nodes

    def _run_array_size(self, lhs, arr):
        # get total size by multiplying all dimension sizes
        nodes = []
        if self._is_1D_arr(arr.name):
            dim1_size = self._array_sizes[arr.name][0]
        else:
            assert self._is_1D_Var_arr(arr.name)
            nodes += self._gen_1D_Var_len(arr)
            dim1_size = nodes[-1].target

        def f(arr, dim1):  # pragma: no cover
            sizes = np.array(arr.shape)
            sizes[0] = dim1
            s = sizes.prod()

        f_ir = compile_to_numba_ir(f, {'np': np}, self.state.typingctx,
                                   (self.state.typemap[arr.name], types.intp),
                                   self.state.typemap, self.state.calltypes)
        f_block = f_ir.blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr, dim1_size])
        nodes += f_block.body[:-3]
        nodes[-1].target = lhs
        return nodes

    def _run_getsetitem(self, arr, index_var, node, full_node):
        out = [full_node]
        # 1D_Var arrays need adjustment for 1D_Var parfors as well
        if ((self._is_1D_arr(arr.name) or (self._is_1D_Var_arr(arr.name) and arr.name in self._array_starts))
                and ((arr.name, index_var.name) in self._parallel_accesses)):
            scope = index_var.scope
            loc = index_var.loc
            #ndims = self._get_arr_ndim(arr.name)
            # if ndims==1:
            # multi-dimensional array could be indexed with 1D index
            if isinstance(self.state.typemap[index_var.name], types.Integer):
                sub_nodes = self._get_ind_sub(
                    index_var, self._array_starts[arr.name][0])
                out = sub_nodes
                _set_getsetitem_index(node, sub_nodes[-1].target)
            else:
                index_list = guard(find_build_tuple, self.state.func_ir, index_var)
                assert index_list is not None
                sub_nodes = self._get_ind_sub(
                    index_list[0], self._array_starts[arr.name][0])
                out = sub_nodes
                new_index_list = copy.copy(index_list)
                new_index_list[0] = sub_nodes[-1].target
                tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
                self.state.typemap[tuple_var.name] = self.state.typemap[index_var.name]
                tuple_call = ir.Expr.build_tuple(new_index_list, loc)
                tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
                out.append(tuple_assign)
                _set_getsetitem_index(node, tuple_var)

            out.append(full_node)

        elif self._is_1D_arr(arr.name) and isinstance(node, (ir.StaticSetItem, ir.SetItem)):
            is_multi_dim = False
            # we only consider 1st dimension for multi-dim arrays
            inds = guard(find_build_tuple, self.state.func_ir, index_var)
            if inds is not None:
                index_var = inds[0]
                is_multi_dim = True

            # no need for transformation for whole slices
            if guard(is_whole_slice, self.state.typemap, self.state.func_ir, index_var):
                return out

            # TODO: support multi-dim slice setitem like X[a:b, c:d]
            assert not is_multi_dim
            start = self._array_starts[arr.name][0]
            count = self._array_counts[arr.name][0]

            if isinstance(self.state.typemap[index_var.name], types.Integer):
                def f(A, val, index, chunk_start, chunk_count):  # pragma: no cover
                    hpat.distributed_lower._set_if_in_range(
                        A, val, index, chunk_start, chunk_count)

                return self._replace_func(
                    f, [arr, node.value, index_var, start, count])

            assert isinstance(self.state.typemap[index_var.name],
                              types.misc.SliceType), "slice index expected"

            # convert setitem with global range to setitem with local range
            # that overlaps with the local array chunk
            def f(A, val, start, stop, chunk_start, chunk_count):  # pragma: no cover
                loc_start, loc_stop = hpat.distributed_lower._get_local_range(
                    start, stop, chunk_start, chunk_count)
                A[loc_start:loc_stop] = val

            slice_call = get_definition(self.state.func_ir, index_var)
            slice_start = slice_call.args[0]
            slice_stop = slice_call.args[1]
            return self._replace_func(
                f, [arr, node.value, slice_start, slice_stop, start, count])
            # print_node = ir.Print([start_var, end_var], None, loc)
            # self.state.calltypes[print_node] = signature(types.none, types.int64, types.int64)
            # out.append(print_node)
            #
            # setitem_attr_var = ir.Var(scope, mk_unique_var("$setitem_attr"), loc)
            # setitem_attr_call = ir.Expr.getattr(self._g_dist_var, "dist_setitem", loc)
            # self.state.typemap[setitem_attr_var.name] = get_global_func_typ(
            #                                 distributed_api.dist_setitem)
            # setitem_assign = ir.Assign(setitem_attr_call, setitem_attr_var, loc)
            # out = [setitem_assign]
            # setitem_call = ir.Expr.call(setitem_attr_var,
            #                     [arr, index_var, node.value, start, count], (), loc)
            # self.state.calltypes[setitem_call] = self.state.typemap[setitem_attr_var.name].get_call_type(
            #     self.state.typingctx, [self.state.typemap[arr.name],
            #     self.state.typemap[index_var.name], self.state.typemap[node.value.name],
            #     types.intp, types.intp], {})
            # err_var = ir.Var(scope, mk_unique_var("$setitem_err_var"), loc)
            # self.state.typemap[err_var.name] = types.int32
            # setitem_assign = ir.Assign(setitem_call, err_var, loc)
            # out.append(setitem_assign)

        elif self._is_1D_arr(arr.name) and node.op in ['getitem', 'static_getitem']:
            is_multi_dim = False
            lhs = full_node.target

            # we only consider 1st dimension for multi-dim arrays
            inds = guard(find_build_tuple, self.state.func_ir, index_var)
            if inds is not None:
                index_var = inds[0]
                is_multi_dim = True

            arr_def = guard(get_definition, self.state.func_ir, index_var)
            if isinstance(arr_def, ir.Expr) and arr_def.op == 'call':
                fdef = guard(find_callname, self.state.func_ir, arr_def, self.state.typemap)
                if fdef == ('permutation', 'numpy.random'):
                    self._array_starts[lhs.name] = self._array_starts[arr.name]
                    self._array_counts[lhs.name] = self._array_counts[arr.name]
                    self._array_sizes[lhs.name] = self._array_sizes[arr.name]
                    out = self._run_permutation_array_index(lhs, arr, index_var)

            # no need for transformation for whole slices
            if guard(is_whole_slice, self.state.typemap, self.state.func_ir, index_var):
                # A = X[:,3]
                self._array_starts[lhs.name] = [self._array_starts[arr.name][0]]
                self._array_counts[lhs.name] = [self._array_counts[arr.name][0]]
                self._array_sizes[lhs.name] = [self._array_sizes[arr.name][0]]

            # strided whole slice
            # e.g. A = X[::2,5]
            elif guard(is_whole_slice, self.state.typemap, self.state.func_ir, index_var, accept_stride=True):
                # FIXME: we use rebalance array to handle the output array
                # TODO: convert to neighbor exchange
                # on each processor, the slice has to start from an offset:
                # |step-(start%step)|
                in_arr = full_node.value.value
                start = self._array_starts[in_arr.name][0]
                step = get_slice_step(self.state.typemap, self.state.func_ir, index_var)

                def f(A, start, step):
                    offset = abs(step - (start % step)) % step
                    B = A[offset::step]

                f_block = compile_to_numba_ir(f, {}, self.state.typingctx,
                                              (self.state.typemap[in_arr.name], types.intp, types.intp),
                                              self.state.typemap, self.state.calltypes).blocks.popitem()[1]
                replace_arg_nodes(f_block, [in_arr, start, step])
                out = f_block.body[:-3]  # remove none return
                imb_arr = out[-1].target

                # call rebalance
                self._dist_analysis.array_dists[imb_arr.name] = Distribution.OneD_Var
                out += self._run_call_rebalance_array(lhs.name, full_node, [imb_arr])
                out[-1].target = lhs

            elif self._is_REP(lhs.name) and guard(
                    is_const_slice, self.state.typemap, self.state.func_ir, index_var):
                # cases like S.head()
                # bcast if all in rank 0, otherwise gatherv
                in_arr = full_node.value.value
                start = self._array_starts[in_arr.name][0]
                count = self._array_counts[in_arr.name][0]
                return self._replace_func(
                    lambda arr, slice_index, start, count: hpat.distributed_api.const_slice_getitem(
                        arr, slice_index, start, count), [in_arr, index_var, start, count])

        return out

    def _run_parfor(self, parfor, namevar_table, depth):
        # stencil_accesses, neighborhood = get_stencil_accesses(
        #     parfor, self.state.typemap)

        dist_depth = 0

        if depth > dist_depth:
            # Do not distribute
            if depth == (dist_depth + 1):
                parfor.no_sequential_lowering = True
            return [parfor]

        # Thread and 1D parfors turn to gufunc in multithread mode
        if (hpat.multithread_mode
                and self._dist_analysis.parfor_dists[parfor.id]
                != Distribution.REP):
            parfor.no_sequential_lowering = True

        if self._dist_analysis.parfor_dists[parfor.id] == Distribution.OneD_Var:
            return self._run_parfor_1D_Var(parfor, namevar_table)

        if self._dist_analysis.parfor_dists[parfor.id] != Distribution.OneD:
            if True or debug_prints():  # pragma: no cover
                print("parfor " + str(parfor.id) + " not parallelized.")
            return [parfor]

        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        range_size = parfor.loop_nests[0].stop
        out = []

        # return range to original size of array
        # if stencil_accesses:
        #     #right_length = neighborhood[1][0]
        #     left_length, right_length = self._get_stencil_border_length(
        #         neighborhood)
        #     if right_length:
        #         new_range_size = ir.Var(
        #             scope, mk_unique_var("new_range_size"), loc)
        #         self.state.typemap[new_range_size.name] = types.intp
        #         index_const = ir.Var(scope, mk_unique_var("index_const"), loc)
        #         self.state.typemap[index_const.name] = types.intp
        #         out.append(
        #             ir.Assign(ir.Const(right_length, loc), index_const, loc))
        #         calc_call = ir.Expr.binop('+', range_size, index_const, loc)
        #         self.state.calltypes[calc_call] = ir_utils.find_op_typ('+',
        #                                                          [types.intp, types.intp])
        #         out.append(ir.Assign(calc_call, new_range_size, loc))
        #         range_size = new_range_size

        div_nodes, start_var, end_var = self._gen_1D_div(range_size, scope, loc,
                                                         "$loop", "get_end", distributed_api.get_end)
        out += div_nodes
        # print_node = ir.Print([start_var, end_var, range_size], None, loc)
        # self.state.calltypes[print_node] = signature(types.none, types.int64, types.int64, types.intp)
        # out.append(print_node)

        parfor.loop_nests[0].start = start_var
        parfor.loop_nests[0].stop = end_var

        # if stencil_accesses:
        #     # TODO assuming single array in stencil
        #     arr_set = set(stencil_accesses.values())
        #     arr = arr_set.pop()
        #     assert not arr_set  # only one array
        #     self._run_parfor_stencil(parfor, out, start_var, end_var,
        #                              neighborhood, namevar_table[arr])
        # else:
        #     out.append(parfor)
        out.append(parfor)

        init_reduce_nodes, reduce_nodes = self._gen_parfor_reductions(
            parfor, namevar_table)
        parfor.init_block.body += init_reduce_nodes
        out += reduce_nodes
        return out

    def _run_parfor_1D_Var(self, parfor, namevar_table):
        # recover range of 1DVar parfors coming from converted 1DVar array len()
        prepend = []
        for l in parfor.loop_nests:
            if l.stop.name in self.oneDVar_len_vars:
                arr_var = self.oneDVar_len_vars[l.stop.name]

                def f(A):  # pragma: no cover
                    arr_len = len(A)
                f_block = compile_to_numba_ir(f, {'hpat': hpat}, self.state.typingctx,
                                              (self.state.typemap[arr_var.name],),
                                              self.state.typemap, self.state.calltypes).blocks.popitem()[1]
                replace_arg_nodes(f_block, [arr_var])
                nodes = f_block.body[:-3]  # remove none return
                l.stop = nodes[-1].target
                prepend += nodes

        # see if parfor index is used in compute other than array access
        # (e.g. argmin)
        l_nest = parfor.loop_nests[0]
        ind_varname = l_nest.index_variable.name
        ind_used = False
        for block in parfor.loop_body.values():
            for stmt in block.body:
                if not is_get_setitem(stmt) and ind_varname in (v.name for v in stmt.list_vars()):
                    ind_used = True
                    dprint("index of 1D_Var pafor {} used in {}".format(
                        parfor.id, stmt))
                break

        # fix parfor start and stop bounds using ex_scan on ranges
        if ind_used:
            scope = l_nest.index_variable.scope
            loc = l_nest.index_variable.loc
            if isinstance(l_nest.start, int):
                start_var = ir.Var(scope, mk_unique_var("loop_start"), loc)
                self.state.typemap[start_var.name] = types.intp
                prepend.append(ir.Assign(
                    ir.Const(l_nest.start, loc), start_var, loc))
                l_nest.start = start_var

            def _fix_ind_bounds(start, stop):
                prefix = hpat.distributed_api.dist_exscan(stop - start)
                # rank = hpat.distributed_api.get_rank()
                # print(rank, prefix, start, stop)
                return start + prefix, stop + prefix

            f_block = compile_to_numba_ir(_fix_ind_bounds, {'hpat': hpat},
                                          self.state.typingctx, (types.intp, types.intp), self.state.typemap,
                                          self.state.calltypes).blocks.popitem()[1]
            replace_arg_nodes(f_block, [l_nest.start, l_nest.stop])
            nodes = f_block.body[:-2]
            ret_var = nodes[-1].target
            gen_getitem(l_nest.start, ret_var, 0, self.state.calltypes, nodes)
            gen_getitem(l_nest.stop, ret_var, 1, self.state.calltypes, nodes)
            prepend += nodes

            array_accesses = ir_utils.get_array_accesses(parfor.loop_body)
            for (arr, index) in array_accesses:
                if self._index_has_par_index(index, ind_varname):
                    self._array_starts[arr] = [l_nest.start]

        init_reduce_nodes, reduce_nodes = self._gen_parfor_reductions(
            parfor, namevar_table)
        parfor.init_block.body += init_reduce_nodes
        out = prepend + [parfor] + reduce_nodes
        return out

    def _run_arg(self, assign):
        rhs = assign.value
        out = [assign]

        if rhs.name not in self.state.metadata['distributed']:
            return None

        arr = assign.target
        typ = self.state.typemap[arr.name]
        if is_array_container(self.state.typemap, arr.name):
            return None

        # TODO: comprehensive support for Series vars
        from hpat.hiframes.pd_series_ext import SeriesType
        if isinstance(typ, (SeriesType, hpat.hiframes.pd_dataframe_ext.DataFrameType)):
            return None

        # gen len() using 1D_Var reduce approach.
        # TODO: refactor to avoid reduction
        ndim = self.state.typemap[arr.name].ndim
        out += self._gen_1D_Var_len(arr)
        total_length = out[-1].target
        div_nodes, start_var, count_var = self._gen_1D_div(
            total_length, arr.scope, arr.loc, "$input", "get_node_portion",
            distributed_api.get_node_portion)
        out += div_nodes

        # XXX: get sizes in lower dimensions
        self._array_starts[arr.name] = [-1] * ndim
        self._array_counts[arr.name] = [-1] * ndim
        self._array_sizes[arr.name] = [-1] * ndim
        self._array_starts[arr.name][0] = start_var
        self._array_counts[arr.name][0] = count_var
        self._array_sizes[arr.name][0] = total_length
        return out

    def _index_has_par_index(self, index, other_index):
        if index == other_index:
            return True
        # multi-dim case
        tup_list = guard(find_build_tuple, self.state.func_ir, index)
        if tup_list is not None:
            index_tuple = [var.name for var in tup_list]
            if index_tuple[0] == index:
                return True
        return False

    def _gen_parfor_reductions(self, parfor, namevar_table):
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        pre = []
        out = []
        _, reductions = get_parfor_reductions(
            parfor, parfor.params, self.state.calltypes)

        # print('aaaaaaaaaaaaaaaaaaa')
        # parfor.dump()
        # print('aaaaaaaaaaaaaaaaaaa')
        for reduce_varname, (init_val, reduce_nodes) in reductions.items():
            # print(len(reduce_nodes))
            # print('\n'.join([str(a) for a in reduce_nodes]))
            reduce_op = guard(self._get_reduce_op, reduce_nodes)
            # TODO: initialize reduction vars (arrays)
            reduce_var = namevar_table[reduce_varname]
            pre += self._gen_init_reduce(reduce_var, reduce_op)
            out += self._gen_reduce(reduce_var, reduce_op, scope, loc)

        return pre, out

    # def _get_var_const_val(self, var):
    #     if isinstance(var, int):
    #         return var
    #     node = guard(get_definition, self.state.func_ir, var)
    #     if isinstance(node, ir.Const):
    #         return node.value
    #     if isinstance(node, ir.Expr):
    #         if node.op == 'unary' and node.fn == '-':
    #             return -self._get_var_const_val(node.value)
    #         if node.op == 'binop':
    #             lhs = self._get_var_const_val(node.lhs)
    #             rhs = self._get_var_const_val(node.rhs)
    #             if node.fn == '+':
    #                 return lhs + rhs
    #             if node.fn == '-':
    #                 return lhs - rhs
    #             if node.fn == '//':
    #                 return lhs // rhs
    #     return None

    def _gen_1D_div(self, size_var, scope, loc, prefix, end_call_name, end_call):
        div_nodes = []
        if isinstance(size_var, int):
            new_size_var = ir.Var(
                scope, mk_unique_var(prefix + "_size_var"), loc)
            self.state.typemap[new_size_var.name] = types.int64
            size_assign = ir.Assign(ir.Const(size_var, loc), new_size_var, loc)
            div_nodes.append(size_assign)
            size_var = new_size_var

        # attr call: start_attr = getattr(g_dist_var, get_start)
        start_attr_call = ir.Expr.getattr(self._g_dist_var, "get_start", loc)
        start_attr_var = ir.Var(scope, mk_unique_var("$get_start_attr"), loc)
        self.state.typemap[start_attr_var.name] = get_global_func_typ(
            distributed_api.get_start)
        start_attr_assign = ir.Assign(start_attr_call, start_attr_var, loc)

        # start_var = get_start(size, rank, pes)
        start_var = ir.Var(scope, mk_unique_var(prefix + "_start_var"), loc)
        self.state.typemap[start_var.name] = types.int64
        start_expr = ir.Expr.call(start_attr_var, [size_var,
                                                   self._size_var, self._rank_var], (), loc)
        self.state.calltypes[start_expr] = self.state.typemap[start_attr_var.name].get_call_type(
            self.state.typingctx, [types.int64, types.int32, types.int32], {})
        start_assign = ir.Assign(start_expr, start_var, loc)

        # attr call: end_attr = getattr(g_dist_var, get_end)
        end_attr_call = ir.Expr.getattr(self._g_dist_var, end_call_name, loc)
        end_attr_var = ir.Var(scope, mk_unique_var("$get_end_attr"), loc)
        self.state.typemap[end_attr_var.name] = get_global_func_typ(end_call)
        end_attr_assign = ir.Assign(end_attr_call, end_attr_var, loc)

        end_var = ir.Var(scope, mk_unique_var(prefix + "_end_var"), loc)
        self.state.typemap[end_var.name] = types.int64
        end_expr = ir.Expr.call(end_attr_var, [size_var, self._size_var, self._rank_var], (), loc)
        self.state.calltypes[end_expr] = self.state.typemap[end_attr_var.name].get_call_type(
            self.state.typingctx, [types.int64, types.int32, types.int32], {})
        end_assign = ir.Assign(end_expr, end_var, loc)
        div_nodes += [start_attr_assign, start_assign, end_attr_assign, end_assign]
        return div_nodes, start_var, end_var

    def _get_ind_sub(self, ind_var, start_var):
        if (isinstance(ind_var, slice)
                or isinstance(self.state.typemap[ind_var.name],
                              types.misc.SliceType)):
            return self._get_ind_sub_slice(ind_var, start_var)
        # gen sub
        f_ir = compile_to_numba_ir(lambda ind, start: ind - start, {}, self.state.typingctx,
                                   (types.intp, types.intp), self.state.typemap, self.state.calltypes)
        block = f_ir.blocks.popitem()[1]
        replace_arg_nodes(block, [ind_var, start_var])
        return block.body[:-2]

    def _get_ind_sub_slice(self, slice_var, offset_var):
        if isinstance(slice_var, slice):
            f_text = """def f(offset):
                return slice({} - offset, {} - offset)
            """.format(slice_var.start, slice_var.stop)
            loc = {}
            exec(f_text, {}, loc)
            f = loc['f']
            args = [offset_var]
            arg_typs = (types.intp,)
        else:
            def f(old_slice, offset):  # pragma: no cover
                return slice(old_slice.start - offset, old_slice.stop - offset)
            args = [slice_var, offset_var]
            slice_type = self.state.typemap[slice_var.name]
            arg_typs = (slice_type, types.intp,)
        _globals = self.state.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(f, _globals, self.state.typingctx, arg_typs,
                                   self.state.typemap, self.state.calltypes)
        _, block = f_ir.blocks.popitem()
        replace_arg_nodes(block, args)
        return block.body[:-2]  # ignore return nodes

    def _dist_prints(self, blocks):
        new_blocks = {}
        for (block_label, block) in blocks.items():
            scope = block.scope
            i = _find_first_print(block.body)
            while i != -1:
                inst = block.body[i]
                loc = inst.loc
                # split block across print
                prev_block = ir.Block(scope, loc)
                new_blocks[block_label] = prev_block
                block_label = ir_utils.next_label()
                print_label = ir_utils.next_label()

                prev_block.body = block.body[:i]
                rank_comp_var = ir.Var(scope, mk_unique_var("$rank_comp"), loc)
                self.state.typemap[rank_comp_var.name] = types.boolean
                comp_expr = ir.Expr.binop(operator.eq, self._rank_var, self._set0_var, loc)
                expr_typ = self.state.typingctx.resolve_function_type(operator.eq, (types.int32, types.int64), {})
                #expr_typ = find_op_typ(operator.eq, [types.int32, types.int64])
                self.state.calltypes[comp_expr] = expr_typ
                comp_assign = ir.Assign(comp_expr, rank_comp_var, loc)
                prev_block.body.append(comp_assign)
                print_branch = ir.Branch(rank_comp_var, print_label, block_label, loc)
                prev_block.body.append(print_branch)

                print_block = ir.Block(scope, loc)
                print_block.body.append(inst)
                print_block.body.append(ir.Jump(block_label, loc))
                new_blocks[print_label] = print_block

                block.body = block.body[i + 1:]
                i = _find_first_print(block.body)
            new_blocks[block_label] = block
        return new_blocks

    def _file_open_set_parallel(self, file_varname):
        var = file_varname
        while True:
            var_def = get_definition(self.state.func_ir, var)
            require(isinstance(var_def, ir.Expr))
            if var_def.op == 'call':
                fdef = find_callname(self.state.func_ir, var_def)
                if (fdef[0] in ('create_dataset', 'create_group')
                        and isinstance(fdef[1], ir.Var)
                        and self.state.typemap[fdef[1].name] in (h5file_type, h5group_type)):
                    self._file_open_set_parallel(fdef[1].name)
                    return
                else:
                    assert fdef == ('File', 'h5py')
                    var_def.args[2] = self._set1_var
                    return
            # TODO: handle control flow
            require(var_def.op in ('getitem', 'static_getitem'))
            var = var_def.value.name

        # for label, block in self.state.func_ir.blocks.items():
        #     for stmt in block.body:
        #         if (isinstance(stmt, ir.Assign)
        #                 and stmt.target.name == file_varname):
        #             rhs = stmt.value
        #             assert isinstance(rhs, ir.Expr) and rhs.op == 'call'
        #             call_name = self._call_table[rhs.func.name][0]
        #             if call_name == 'h5create_group':
        #                 # if read/write call is on a group, find its actual file
        #                 f_varname = rhs.args[0].name
        #                 self._file_open_set_parallel(f_varname)
        #                 return
        #             else:
        #                 assert call_name == 'File'
        #                 rhs.args[2] = self._set1_var

    def _gen_barrier(self):
        def f():  # pragma: no cover
            return hpat.distributed_api.barrier()

        f_blocks = compile_to_numba_ir(f, {'hpat': hpat}, self.state.typingctx, {},
                                       self.state.typemap, self.state.calltypes).blocks
        block = f_blocks[min(f_blocks.keys())]
        return block.body[:-2]  # remove return

    def _gen_reduce(self, reduce_var, reduce_op, scope, loc):
        op_var = ir.Var(scope, mk_unique_var("$reduce_op"), loc)
        self.state.typemap[op_var.name] = types.int32
        op_assign = ir.Assign(ir.Const(reduce_op.value, loc), op_var, loc)

        def f(val, op):  # pragma: no cover
            hpat.distributed_api.dist_reduce(val, op)

        f_ir = compile_to_numba_ir(f,
                                   {'hpat': hpat},
                                   self.state.typingctx,
                                   (self.state.typemap[reduce_var.name], types.int32),
                                   self.state.typemap,
                                   self.state.calltypes)
        _, block = f_ir.blocks.popitem()

        replace_arg_nodes(block, [reduce_var, op_var])
        dist_reduce_nodes = [op_assign] + block.body[:-3]
        dist_reduce_nodes[-1].target = reduce_var
        # print('*****************************')
        # print('\n'.join([str(a) for a in dist_reduce_nodes]))
        # print('*****************************')
        return dist_reduce_nodes

    def _get_reduce_op(self, reduce_nodes):
        require(len(reduce_nodes) == 2)
        require(isinstance(reduce_nodes[0], ir.Assign))
        require(isinstance(reduce_nodes[1], ir.Assign))
        require(isinstance(reduce_nodes[0].value, ir.Expr))
        require(isinstance(reduce_nodes[1].value, ir.Var))
        rhs = reduce_nodes[0].value

        if rhs.op == 'inplace_binop':
            if rhs.fn in ('+=', operator.iadd):
                return Reduce_Type.Sum
            if rhs.fn in ('|=', operator.ior):
                return Reduce_Type.Or
            if rhs.fn in ('*=', operator.imul):
                return Reduce_Type.Prod

        if rhs.op == 'call':
            func = find_callname(self.state.func_ir, rhs, self.state.typemap)
            if func == ('min', 'builtins'):
                if isinstance(self.state.typemap[rhs.args[0].name], numba.typing.builtins.IndexValueType):
                    return Reduce_Type.Argmin
                return Reduce_Type.Min
            if func == ('max', 'builtins'):
                if isinstance(self.state.typemap[rhs.args[0].name], numba.typing.builtins.IndexValueType):
                    return Reduce_Type.Argmax
                return Reduce_Type.Max

        raise GuardException  # pragma: no cover

    def _gen_init_reduce(self, reduce_var, reduce_op):
        """generate code to initialize reduction variables on non-root
        processors.
        """
        red_var_typ = self.state.typemap[reduce_var.name]
        el_typ = red_var_typ
        if is_np_array(self.state.typemap, reduce_var.name):
            el_typ = red_var_typ.dtype
        init_val = None
        pre_init_val = ""

        if reduce_op in [Reduce_Type.Sum, Reduce_Type.Or]:
            init_val = str(el_typ(0))
        if reduce_op == Reduce_Type.Prod:
            init_val = str(el_typ(1))
        if reduce_op == Reduce_Type.Min:
            init_val = "numba.targets.builtins.get_type_max_value(np.ones(1,dtype=np.{}).dtype)".format(el_typ)
        if reduce_op == Reduce_Type.Max:
            init_val = "numba.targets.builtins.get_type_min_value(np.ones(1,dtype=np.{}).dtype)".format(el_typ)
        if reduce_op in [Reduce_Type.Argmin, Reduce_Type.Argmax]:
            # don't generate initialization for argmin/argmax since they are not
            # initialized by user and correct initialization is already there
            return []

        assert init_val is not None

        if is_np_array(self.state.typemap, reduce_var.name):
            pre_init_val = "v = np.full_like(s, {}, s.dtype)".format(init_val)
            init_val = "v"

        f_text = "def f(s):\n  {}\n  s = hpat.distributed_lower._root_rank_select(s, {})".format(pre_init_val, init_val)
        loc_vars = {}
        exec(f_text, {'hpat': hpat}, loc_vars)
        f = loc_vars['f']

        f_block = compile_to_numba_ir(f,
                                      {'hpat': hpat, 'numba': numba, 'np': np},
                                      self.state.typingctx,
                                      (red_var_typ,),
                                      self.state.typemap,
                                      self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [reduce_var])
        nodes = f_block.body[:-3]
        nodes[-1].target = reduce_var
        return nodes

    def _get_tuple_varlist(self, tup_var, out):
        """ get the list of variables that hold values in the tuple variable.
        add node to out if code generation needed.
        """
        t_list = guard(find_build_tuple, self.state.func_ir, tup_var)
        if t_list is not None:
            return t_list
        assert isinstance(self.state.typemap[tup_var.name], types.UniTuple)
        ndims = self.state.typemap[tup_var.name].count
        f_text = "def f(tup_var):\n"
        for i in range(ndims):
            f_text += "  val{} = tup_var[{}]\n".format(i, i)
        loc_vars = {}
        exec(f_text, {}, loc_vars)
        f = loc_vars['f']
        f_block = compile_to_numba_ir(f,
                                      {},
                                      self.state.typingctx,
                                      (self.state.typemap[tup_var.name],
                                       ),
                                      self.state.typemap,
                                      self.state.calltypes).blocks.popitem()[1]
        replace_arg_nodes(f_block, [tup_var])
        nodes = f_block.body[:-3]
        vals_list = []
        for stmt in nodes:
            assert isinstance(stmt, ir.Assign)
            rhs = stmt.value
            assert isinstance(rhs, (ir.Var, ir.Const, ir.Expr))
            if isinstance(rhs, ir.Expr):
                assert rhs.op == 'static_getitem'
                vals_list.append(stmt.target)
        out += nodes
        return vals_list

    def _get_arg(self, f_name, args, kws, arg_no, arg_name, default=None, err_msg=None):
        arg = None
        if len(args) > arg_no:
            arg = args[arg_no]
        elif arg_name in kws:
            arg = kws[arg_name]

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = "{} requires '{}' argument".format(f_name, arg_name)
            raise ValueError(err_msg)
        return arg

    def _replace_func(self, func, args, const=False, pre_nodes=None, extra_globals=None):
        glbls = {'numba': numba, 'np': np, 'hpat': hpat}
        if extra_globals is not None:
            glbls.update(extra_globals)
        arg_typs = tuple(self.state.typemap[v.name] for v in args)
        if const:
            new_args = []
            for i, arg in enumerate(args):
                val = guard(find_const, self.state.func_ir, arg)
                if val:
                    new_args.append(types.literal(val))
                else:
                    new_args.append(arg_typs[i])
            arg_typs = tuple(new_args)
        return ReplaceFunc(func, arg_typs, args, glbls, pre_nodes)

    def _get_arr_ndim(self, arrname):
        if self.state.typemap[arrname] == string_array_type:
            return 1
        return self.state.typemap[arrname].ndim

    def _is_1D_arr(self, arr_name):
        # some arrays like stencil buffers are added after analysis so
        # they are not in dists list
        return ((arr_name in self._dist_analysis.array_dists
                 and self._dist_analysis.array_dists[arr_name] == Distribution.OneD))

    def _is_1D_Var_arr(self, arr_name):
        # some arrays like stencil buffers are added after analysis so
        # they are not in dists list
        return ((arr_name in self._dist_analysis.array_dists
                 and self._dist_analysis.array_dists[arr_name] == Distribution.OneD_Var))

    def _is_REP(self, arr_name):
        return ((arr_name not in self._dist_analysis.array_dists
                 or self._dist_analysis.array_dists[arr_name] == Distribution.REP))


def _find_first_print(body):
    """ This function finds the first print of something """
    for (i, inst) in enumerate(body):
        if isinstance(inst, ir.Print):
            return i
    return -1


def _set_getsetitem_index(node, new_ind):
    if ((isinstance(node, ir.Expr) and node.op == 'static_getitem') or isinstance(node, ir.StaticSetItem)):
        node.index_var = new_ind
        node.index = None
        return

    assert ((isinstance(node, ir.Expr) and node.op == 'getitem') or isinstance(node, ir.SetItem))
    node.index = new_ind


def dprint(*s):  # pragma: no cover
    if debug_prints():
        print(*s)

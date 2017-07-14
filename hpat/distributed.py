from __future__ import print_function, division, absolute_import

import types as pytypes  # avoid confusion with numba.types
import copy
import numba
from numba import (ir, types, typing, config, numpy_support,
                    ir_utils, postproc)
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc,
                            get_global_func_typ, find_op_typ, get_name_var_table,
                            get_call_table, get_tuple_table, remove_dels)
from numba.typing import signature
from numba.parfor import (get_parfor_reductions, get_parfor_params,
                            wrap_parfor_blocks, unwrap_parfor_blocks)
from numba.parfor import Parfor, lower_parfor_sequential
import numpy as np

import hpat
from hpat import (distributed_api,
                  distributed_lower)  # import lower for module initialization

from hpat.distributed_analysis import (Distribution,
                                       DistributedAnalysis,
                                       get_stencil_accesses)
import h5py
import time
# from mpi4py import MPI


distributed_run_extensions = {}

class DistributedPass(object):
    """analyze program and transfrom to distributed"""
    def __init__(self, func_ir, typemap, calltypes):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes

        self._call_table,_ = get_call_table(func_ir.blocks)
        self._tuple_table = get_tuple_table(func_ir.blocks)

        self._dist_analysis = None
        self._T_arrs = None  # set of transposed arrays (taken from analysis)

        self._rank_var = None # will be set in run
        self._size_var = None
        self._g_dist_var = None
        self._set1_var = None # variable set to 1
        self._set0_var = None # variable set to 0
        self._array_starts = {}
        self._array_counts = {}

        # keep shape attr calls on parallel arrays like X.shape
        self._shape_attrs = {}
        # keep array sizes of parallel arrays to handle shape attrs
        self._array_sizes = {}

    def run(self):
        remove_dels(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "starting distributed pass")
        dist_analysis_pass = DistributedAnalysis(self.func_ir, self.typemap,
                                                                self.calltypes)
        self._dist_analysis = dist_analysis_pass.run()
        self._T_arrs = dist_analysis_pass._T_arrs
        if config.DEBUG_ARRAY_OPT==1:
            print("distributions: ", self._dist_analysis)

        self._gen_dist_inits()
        self._run_dist_pass(self.func_ir.blocks)
        self.func_ir.blocks = self._dist_prints(self.func_ir.blocks)
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names, self.typemap)
        dprint_func_ir(self.func_ir, "after distributed pass")
        lower_parfor_sequential(self.func_ir, self.typemap, self.calltypes)
        post_proc = postproc.PostProcessor(self.func_ir)
        post_proc.run()

    def _run_dist_pass(self, blocks):
        topo_order = find_topo_order(blocks)
        namevar_table = get_name_var_table(blocks)
        #
        for label in topo_order:
            new_body = []
            for inst in blocks[label].body:
                if type(inst) in distributed_run_extensions:
                    f = distributed_run_extensions[type(inst)]
                    new_body += f(inst, self.typemap, self.calltypes)
                    continue
                if isinstance(inst, Parfor):
                    new_body += self._run_parfor(inst, namevar_table)
                    continue
                if isinstance(inst, ir.Assign):
                    lhs = inst.target.name
                    rhs = inst.value
                    if isinstance(rhs, ir.Expr):
                        if rhs.op=='call':
                            new_body += self._run_call(inst, blocks[label].body)
                            continue
                        if rhs.op=='getitem':
                            new_body += self._run_getsetitem(rhs.value.name,
                                rhs.index, rhs, inst)
                            continue
                        if (rhs.op=='getattr'
                                and self._is_1D_arr(rhs.value.name)
                                and rhs.attr=='shape'):
                            self._shape_attrs[lhs] = rhs.value.name
                        if (rhs.op=='getattr'
                                and self._is_1D_arr(rhs.value.name)
                                and rhs.attr=='T'):
                            assert lhs in self._T_arrs
                            orig_arr = rhs.value.name
                            self._array_starts[lhs] = copy.copy(
                                self._array_starts[orig_arr]).reverse()
                            self._array_counts[lhs] = copy.copy(
                                self._array_counts[orig_arr]).reverse()
                            self._array_sizes[lhs] = copy.copy(
                                self._array_sizes[orig_arr]).reverse()
                        if (rhs.op=='exhaust_iter'
                                and rhs.value.name in self._shape_attrs):
                            self._shape_attrs[lhs] = self._shape_attrs[rhs.value.name]
                        if (rhs.op=='static_getitem'
                                and rhs.value.name in self._shape_attrs):
                            arr = self._shape_attrs[rhs.value.name]
                            ndims = self.typemap[arr].ndim
                            sizes = self._array_sizes[arr]
                            if arr not in self._T_arrs and rhs.index==0:
                                inst.value = sizes[rhs.index]
                            # last dimension of transposed arrays is partitioned
                            if arr in self._T_arrs and rhs.index==ndims-1:
                                inst.value = sizes[rhs.index]
                    if isinstance(rhs, ir.Var) and self._is_1D_arr(rhs.name):
                        self._array_starts[lhs] = self._array_starts[rhs.name]
                        self._array_counts[lhs] = self._array_counts[rhs.name]
                        self._array_sizes[lhs] = self._array_sizes[rhs.name]
                if isinstance(inst, ir.SetItem):
                    new_body += self._run_getsetitem(inst.target.name,
                        inst.index, inst, inst)
                    continue
                new_body.append(inst)
            blocks[label].body = new_body

    def _gen_dist_inits(self):
        # add initializations
        topo_order = find_topo_order(self.func_ir.blocks)
        first_block = self.func_ir.blocks[topo_order[0]]
        # set scope and loc of generated code to the first variable in block
        scope = first_block.scope
        loc = first_block.loc
        out = []
        self._set1_var = ir.Var(scope, mk_unique_var("$const_parallel"), loc)
        self.typemap[self._set1_var.name] = types.int64
        set1_assign = ir.Assign(ir.Const(1, loc), self._set1_var, loc)
        out.append(set1_assign)
        self._set0_var = ir.Var(scope, mk_unique_var("$const_parallel"), loc)
        self.typemap[self._set0_var.name] = types.int64
        set0_assign = ir.Assign(ir.Const(0, loc), self._set0_var, loc)
        out.append(set0_assign)
        # g_dist_var = Global(hpat.distributed_api)
        g_dist_var = ir.Var(scope, mk_unique_var("$distributed_g_var"), loc)
        self._g_dist_var = g_dist_var
        self.typemap[g_dist_var.name] = types.misc.Module(hpat.distributed_api)
        g_dist = ir.Global('distributed_api', hpat.distributed_api, loc)
        g_dist_assign = ir.Assign(g_dist, g_dist_var, loc)
        # attr call: rank_attr = getattr(g_dist_var, get_rank)
        rank_attr_call = ir.Expr.getattr(g_dist_var, "get_rank", loc)
        rank_attr_var = ir.Var(scope, mk_unique_var("$get_rank_attr"), loc)
        self.typemap[rank_attr_var.name] = get_global_func_typ(
                                                    distributed_api.get_rank)
        rank_attr_assign = ir.Assign(rank_attr_call, rank_attr_var, loc)
        # rank_var = hpat.distributed_api.get_rank()
        rank_var = ir.Var(scope, mk_unique_var("$rank"), loc)
        self.typemap[rank_var.name] = types.int32
        rank_call = ir.Expr.call(rank_attr_var, [], (), loc)
        self.calltypes[rank_call] = self.typemap[rank_attr_var.name].get_call_type(
            typing.Context(), [], {})
        rank_assign = ir.Assign(rank_call, rank_var, loc)
        self._rank_var = rank_var
        out += [g_dist_assign, rank_attr_assign, rank_assign]

        # attr call: size_attr = getattr(g_dist_var, get_size)
        size_attr_call = ir.Expr.getattr(g_dist_var, "get_size", loc)
        size_attr_var = ir.Var(scope, mk_unique_var("$get_size_attr"), loc)
        self.typemap[size_attr_var.name] = get_global_func_typ(
                                                    distributed_api.get_size)
        size_attr_assign = ir.Assign(size_attr_call, size_attr_var, loc)
        # size_var = hpat.distributed_api.get_size()
        size_var = ir.Var(scope, mk_unique_var("$dist_size"), loc)
        self.typemap[size_var.name] = types.int32
        size_call = ir.Expr.call(size_attr_var, [], (), loc)
        self.calltypes[size_call] = self.typemap[size_attr_var.name].get_call_type(
            typing.Context(), [], {})
        size_assign = ir.Assign(size_call, size_var, loc)
        self._size_var = size_var
        out += [size_attr_assign, size_assign]
        first_block.body = out+first_block.body

    def _run_call(self, assign, block_body):
        lhs = assign.target.name
        rhs = assign.value
        func_var = rhs.func.name
        scope = assign.target.scope
        loc = assign.target.loc
        out = [assign]
        # shortcut if we don't know the call
        if func_var not in self._call_table or not self._call_table[func_var]:
            return out
        call_list = self._call_table[func_var]

        # divide 1D alloc
        if self._is_1D_arr(lhs) and self._is_alloc_call(func_var):
            size_var = rhs.args[0]
            if self.typemap[size_var.name]==types.intp:
                self._array_sizes[lhs] = [size_var]
                out, start_var, end_var = self._gen_1D_div(size_var, scope, loc,
                    "$alloc", "get_node_portion", distributed_api.get_node_portion)
                self._array_starts[lhs] = [start_var]
                self._array_counts[lhs] = [end_var]
                rhs.args[0] = end_var
            else:
                # size should be either int or tuple of ints
                assert size_var.name in self._tuple_table
                size_list = self._tuple_table[size_var.name]
                self._array_sizes[lhs] = size_list
                out, start_var, end_var = self._gen_1D_div(size_list[0], scope, loc,
                    "$alloc", "get_node_portion", distributed_api.get_node_portion)
                ndims = len(size_list)
                new_size_list = copy.copy(size_list)
                new_size_list[0] = end_var
                tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
                self.typemap[tuple_var.name] = self.typemap[size_var.name]
                tuple_call = ir.Expr.build_tuple(new_size_list, loc)
                tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
                out.append(tuple_assign)
                rhs.args[0] = tuple_var
                self._array_starts[lhs] = [self._set0_var]*ndims
                self._array_starts[lhs][0] = start_var
                self._array_counts[lhs] = new_size_list
            out.append(assign)

        if (self._is_h5_read_write_call(func_var)
                and self._is_1D_arr(rhs.args[6].name)):
            arr = rhs.args[6].name
            ndims = len(self._array_starts[arr])
            starts_var = ir.Var(scope, mk_unique_var("$h5_starts"), loc)
            self.typemap[starts_var.name] = types.containers.UniTuple(types.int64, ndims)
            start_tuple_call = ir.Expr.build_tuple(self._array_starts[arr], loc)
            starts_assign = ir.Assign(start_tuple_call, starts_var, loc)
            rhs.args[3] = starts_var
            counts_var = ir.Var(scope, mk_unique_var("$h5_counts"), loc)
            self.typemap[counts_var.name] = types.containers.UniTuple(types.int64, ndims)
            count_tuple_call = ir.Expr.build_tuple(self._array_counts[arr], loc)
            counts_assign = ir.Assign(count_tuple_call, counts_var, loc)
            out = [starts_assign, counts_assign, assign]
            rhs.args[4] = counts_var
            rhs.args[5] = self._set1_var
            # set parallel arg in file open
            # TODO: generalize to all blocks
            file_var = rhs.args[0].name
            for stmt in block_body:
                if isinstance(stmt, ir.Assign) and stmt.target.name==file_var:
                    rhs = stmt.value
                    assert isinstance(rhs, ir.Expr)
                    rhs.args[2] = self._set1_var

        # output array has same properties (starts etc.) as input array
        if (len(call_list)==2 and call_list[1]==np
                and call_list[0] in ['cumsum', 'cumprod', 'empty_like',
                    'zeros_like', 'ones_like', 'full_like', 'copy']):
            in_arr = rhs.args[0].name
            self._array_starts[lhs] = self._array_starts[in_arr]
            self._array_counts[lhs] = self._array_counts[in_arr]
            self._array_sizes[lhs] = self._array_sizes[in_arr]

        if (len(call_list)==2 and call_list[1]==np
                and call_list[0] in ['cumsum', 'cumprod']):
            in_arr = rhs.args[0].name
            in_arr_var = rhs.args[0]
            lhs_var = assign.target
            # allocate output array
            # TODO: compute inplace if input array is dead
            out = mk_alloc(self.typemap, self.calltypes, lhs_var,
                            tuple(self._array_sizes[in_arr]),
                            self.typemap[in_arr].dtype, scope, loc)
            # generate distributed call
            dist_attr_var = ir.Var(scope, mk_unique_var("$dist_attr"), loc)
            dist_func_name = "dist_"+call_list[0]
            dist_func = getattr(distributed_api, dist_func_name)
            dist_attr_call = ir.Expr.getattr(self._g_dist_var, dist_func_name, loc)
            self.typemap[dist_attr_var.name] = get_global_func_typ(dist_func)
            dist_func_assign = ir.Assign(dist_attr_call, dist_attr_var, loc)
            err_var = ir.Var(scope, mk_unique_var("$dist_err_var"), loc)
            self.typemap[err_var.name] = types.int32
            dist_call = ir.Expr.call(dist_attr_var, [in_arr_var, lhs_var], (), loc)
            self.calltypes[dist_call] = self.typemap[dist_attr_var.name].get_call_type(
                typing.Context(), [self.typemap[in_arr], self.typemap[lhs]], {})
            dist_assign = ir.Assign(dist_call, err_var, loc)
            return out+[dist_func_assign, dist_assign]

        if self._is_call(func_var, ['dot', np]):
            arg0 = rhs.args[0].name
            arg1 = rhs.args[1].name
            ndim0 = self.typemap[arg0].ndim
            ndim1 = self.typemap[arg1].ndim
            # Fortran layout is caused by X.T and means transpose
            t0 = arg0 in self._T_arrs
            t1 = arg1 in self._T_arrs

            # reduction across dataset
            if self._is_1D_arr(arg0) and self._is_1D_arr(arg1):
                dprint("run dot dist reduce:", arg0, arg1)
                reduce_attr_var = ir.Var(scope, mk_unique_var("$reduce_attr"), loc)
                reduce_func_name = "dist_arr_reduce"
                reduce_func = distributed_api.dist_arr_reduce
                # output of vector dot() is scalar
                if ndim0==1 and ndim1==1:
                    reduce_func_name = "dist_reduce"
                    reduce_func = distributed_api.dist_reduce
                reduce_attr_call = ir.Expr.getattr(self._g_dist_var, reduce_func_name, loc)
                self.typemap[reduce_attr_var.name] = get_global_func_typ(
                                                                    reduce_func)
                reduce_assign = ir.Assign(reduce_attr_call, reduce_attr_var, loc)
                out.append(reduce_assign)
                err_var = ir.Var(scope, mk_unique_var("$reduce_err_var"), loc)
                self.typemap[err_var.name] = types.int32
                # scalar reduce is not updated inplace
                if ndim0==1 and ndim1==1:
                    err_var = assign.target
                reduce_var = assign.target
                reduce_call = ir.Expr.call(reduce_attr_var, [reduce_var], (), loc)
                self.calltypes[reduce_call] = self.typemap[reduce_attr_var.name].get_call_type(
                    typing.Context(), [self.typemap[reduce_var.name]], {})
                reduce_assign = ir.Assign(reduce_call, err_var, loc)
                out.append(reduce_assign)

            # assign starts/counts/sizes data structures for output array
            if ndim0==2 and ndim1==1 and not t0 and self._is_1D_arr(arg0):
                # special case were arg1 vector is treated as column vector
                # samples dot weights: np.dot(X,w)
                # output is 1D array same size as dim 0 of X
                assert self.typemap[lhs].ndim==1
                assert self._is_1D_arr(lhs)
                self._array_starts[lhs] = [self._array_starts[arg0][0]]
                self._array_counts[lhs] = [self._array_counts[arg0][0]]
                self._array_sizes[lhs] = [self._array_sizes[rhs.name][0]]
                dprint("run dot case 1 Xw:", arg0, arg1)
            if ndim0==2 and ndim1==2 and not t0 and not t1:
                # samples dot weights: np.dot(X,w)
                assert self._is_1D_arr(lhs)
                # first dimension is same as X
                # second dimension not needed
                self._array_starts[lhs] = [self._array_starts[arg0][0], -1]
                self._array_counts[lhs] = [self._array_counts[arg0][0], -1]
                self._array_sizes[lhs] = [self._array_sizes[arg0][0], -1]
                dprint("run dot case 4 Xw:", arg0, arg1)

        return out

    def _run_getsetitem(self, arr, index_var, node, full_node):
        out = [full_node]
        if self._is_1D_arr(arr):
            scope = index_var.scope
            loc = index_var.loc
            ndims = self.typemap[arr].ndim
            if ndims==1:
                sub_assign = self._get_ind_sub(index_var, self._array_starts[arr][0])
                out = [sub_assign]
                node.index = sub_assign.target
            else:
                assert index_var.name in self._tuple_table
                index_list = self._tuple_table[index_var.name]
                sub_assign = self._get_ind_sub(index_list[0], self._array_starts[arr][0])
                out = [sub_assign]
                new_index_list = copy.copy(index_list)
                new_index_list[0] = sub_assign.target
                tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
                self.typemap[tuple_var.name] = self.typemap[index_var.name]
                tuple_call = ir.Expr.build_tuple(new_index_list, loc)
                tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
                out.append(tuple_assign)
                node.index = tuple_var

            out.append(full_node)

        return out

    def _run_parfor(self, parfor, namevar_table):
        stencil_accesses, arrays_accessed = get_stencil_accesses(
            parfor.loop_body, parfor.loop_nests[0].index_variable.name)
        # run dist pass recursively
        blocks = wrap_parfor_blocks(parfor)
        self._run_dist_pass(blocks)
        unwrap_parfor_blocks(parfor)

        if self._dist_analysis.parfor_dists[parfor.id]!=Distribution.OneD:
            # TODO: make sure loop index is not used for calculations in
            # OneD_Var parfors
            if config.DEBUG_ARRAY_OPT==1:
                print("parfor "+str(parfor.id)+" not parallelized.")
            return [parfor]
        #
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        range_size = parfor.loop_nests[0].stop

        out, start_var, end_var = self._gen_1D_div(range_size, scope, loc,
                                    "$loop", "get_end", distributed_api.get_end)
        # print_node = ir.Print([start_var, end_var], None, loc)
        # self.calltypes[print_node] = signature(types.none, types.int64, types.int64)
        # out.append(print_node)

        if stencil_accesses:
            # TODO assuming single array in stencil
            arr_set = set(arrays_accessed.values())
            arr = arr_set.pop()
            assert not arr_set  # only one array
            self._run_parfor_stencil(parfor, out, start_var, end_var,
                                        stencil_accesses, namevar_table[arr])
        else:
            parfor.loop_nests[0].start = start_var
            parfor.loop_nests[0].stop = end_var
            out.append(parfor)

        _, reductions = get_parfor_reductions(parfor, parfor.params)

        if len(reductions)!=0:
            reduce_attr_var = ir.Var(scope, mk_unique_var("$reduce_attr"), loc)
            reduce_attr_call = ir.Expr.getattr(self._g_dist_var, "dist_reduce", loc)
            self.typemap[reduce_attr_var.name] = get_global_func_typ(
                                                    distributed_api.dist_reduce)
            reduce_assign = ir.Assign(reduce_attr_call, reduce_attr_var, loc)
            out.append(reduce_assign)

        for reduce_varname, (_, reduce_func, _) in reductions.items():
            reduce_var = namevar_table[reduce_varname]
            reduce_call = ir.Expr.call(reduce_attr_var, [reduce_var], (), loc)
            self.calltypes[reduce_call] = self.typemap[reduce_attr_var.name].get_call_type(
                typing.Context(), [self.typemap[reduce_varname]], {})
            reduce_assign = ir.Assign(reduce_call, reduce_var, loc)
            out.append(reduce_assign)

        return out

    def _run_parfor_stencil(self, parfor, out, start_var, end_var,
                                                    stencil_accesses, arr_var):
        #
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        left_length = -min(stencil_accesses.values())
        right_length = max(stencil_accesses.values())
        dtype = self.typemap[arr_var.name].dtype

        # post left send/receive
        if left_length != 0:
            # allocate left tmp buffer for irecv
            left_recv_buff = ir.Var(scope, mk_unique_var("left_recv_buff"), loc)
            self.typemap[left_recv_buff.name] = self.typemap[arr_var.name]
            out += mk_alloc(self.typemap, self.calltypes, left_recv_buff,
                                (left_length,), dtype, scope, loc)

            # recv from left
            left_recv_req = self._gen_stencil_comm(left_recv_buff, left_length, out, is_left=True, is_send=False)

            # send to right to match recv
            right_send_buff = ir.Var(scope, mk_unique_var("right_send_buff"), loc)
            self.typemap[right_send_buff.name] = self.typemap[arr_var.name]
            # const = -size
            const_msize = ir.Var(scope, mk_unique_var("const_msize"), loc)
            self.typemap[const_msize.name] = types.intp
            out.append(ir.Assign(ir.Const(-left_length, loc), const_msize, loc))
            # const = none
            const_none = ir.Var(scope, mk_unique_var("const_none"), loc)
            self.typemap[const_none.name] = types.none
            out.append(ir.Assign(ir.Const(None, loc), const_none, loc))
            # g_slice = Global(slice)
            g_slice_var = ir.Var(scope, mk_unique_var("g_slice_var"), loc)
            self.typemap[g_slice_var.name] = get_global_func_typ(slice)
            out.append(ir.Assign(ir.Global('slice', slice, loc),
                                                        g_slice_var, loc))
            # slice_ind_out = slice(-size, none)
            slice_ind_out = ir.Var(scope, mk_unique_var("slice_ind_out"), loc)
            slice_call = ir.Expr.call(g_slice_var, [const_msize,
                            const_none], (), loc)
            self.calltypes[slice_call] = self.typemap[g_slice_var.name].get_call_type(typing.Context(),
                                                        [types.intp, types.none], {})
            self.typemap[slice_ind_out.name] = self.calltypes[slice_call].return_type
            out.append(ir.Assign(slice_call, slice_ind_out, loc))
            # right_send_buff = A[slice]
            getslice_call = ir.Expr.static_getitem(arr_var, slice(-3, None, None), slice_ind_out, loc)
            self.calltypes[getslice_call] = signature(
                                self.typemap[right_send_buff.name],
                                self.typemap[arr_var.name],
                                self.typemap[slice_ind_out.name])
            out.append(ir.Assign(getslice_call, right_send_buff, loc))

            right_send_req = self._gen_stencil_comm(right_send_buff, left_length, out, is_left=False, is_send=True)

        out.append(parfor)

        # wait on isend/irecv
        if left_length != 0:
            self._gen_stencil_wait(left_recv_req, out, is_left=True)
            self._gen_stencil_wait(right_send_req, out, is_left=False)

        return

    def _gen_stencil_wait(self, req, out, is_left):
        scope = req.scope
        loc = req.loc
        wait_cond = self._get_comm_cond(out, scope, loc, is_left)
        # wait_err = wait(req)
        wait_err = ir.Var(scope, mk_unique_var("wait_err"), loc)
        self.typemap[wait_err.name] = types.int32
        # attr call: wait_attr = getattr(g_dist_var, irecv)
        wait_attr_call = ir.Expr.getattr(self._g_dist_var, "wait", loc)
        wait_attr_var = ir.Var(scope, mk_unique_var("$get_wait_attr"), loc)
        self.typemap[wait_attr_var.name] = get_global_func_typ(distributed_api.wait)
        out.append(ir.Assign(wait_attr_call, wait_attr_var, loc))
        wait_call = ir.Expr.call(wait_attr_var, [req, wait_cond], (), loc)
        self.calltypes[wait_call] = self.typemap[wait_attr_var.name].get_call_type(
            typing.Context(), [types.int32, types.boolean], {})
        out.append(ir.Assign(wait_call, wait_err, loc))

    def _gen_stencil_comm(self, buff, size, out, is_left, is_send):
        scope = buff.scope
        loc = buff.loc
        rank_op = '+'
        if is_left:
            rank_op = '-'
        comm_name = 'irecv'
        comm_call = distributed_api.irecv
        if is_send:
            comm_name = 'isend'
            comm_call = distributed_api.isend
        comm_tag_const = 22

        # comm_size = size
        comm_size = ir.Var(scope, mk_unique_var("comm_size"), loc)
        self.typemap[comm_size.name] = types.int32
        out.append(ir.Assign(ir.Const(size, loc), comm_size, loc))

        # comm_pe = rank +/- 1
        comm_pe = ir.Var(scope, mk_unique_var("comm_pe"), loc)
        self.typemap[comm_pe.name] = types.int32
        comm_pe_call = ir.Expr.binop(rank_op, self._rank_var, self._set1_var, loc)
        if comm_pe_call not in self.calltypes:
            self.calltypes[comm_pe_call] = find_op_typ(rank_op, [types.int32, types.int64])
        out.append(ir.Assign(comm_pe_call, comm_pe, loc))

        # comm_tag = 22
        comm_tag = ir.Var(scope, mk_unique_var("comm_tag"), loc)
        self.typemap[comm_tag.name] = types.int32
        out.append(ir.Assign(ir.Const(comm_tag_const, loc), comm_tag, loc))

        comm_cond = self._get_comm_cond(out, scope, loc, is_left)

        # comm_req = irecv()
        comm_req = ir.Var(scope, mk_unique_var("comm_req"), loc)
        self.typemap[comm_req.name] = types.int32
        # attr call: icomm_attr = getattr(g_dist_var, irecv)
        icomm_attr_call = ir.Expr.getattr(self._g_dist_var, comm_name, loc)
        icomm_attr_var = ir.Var(scope, mk_unique_var("$get_"+comm_name+"_attr"), loc)
        self.typemap[icomm_attr_var.name] = get_global_func_typ(comm_call)
        out.append(ir.Assign(icomm_attr_call, icomm_attr_var, loc))
        icomm_call = ir.Expr.call(icomm_attr_var, [buff, comm_size,
            comm_pe, comm_tag, comm_cond], (), loc)
        self.calltypes[icomm_call] = self.typemap[icomm_attr_var.name].get_call_type(
            typing.Context(), [self.typemap[buff.name], types.int32,
            types.int32, types.int32, types.boolean], {})
        out.append(ir.Assign(icomm_call, comm_req, loc))
        return comm_req

    def _get_comm_cond(self, out, scope, loc, is_left):
        if is_left:
            last_pe = self._set0_var
        else:
            # last_pe = num_pes - 1
            last_pe = ir.Var(scope, mk_unique_var("last_pe"), loc)
            self.typemap[last_pe.name] = types.int32
            last_pe_call = ir.Expr.binop('-', self._size_var, self._set1_var, loc)
            if last_pe_call not in self.calltypes:
                self.calltypes[last_pe_call] = find_op_typ('-', [types.int32, types.int64])
            out.append(ir.Assign(last_pe_call, last_pe, loc))

        # comm_cond = rank != 0
        comm_cond = ir.Var(scope, mk_unique_var("comm_cond"), loc)
        self.typemap[comm_cond.name] = types.boolean
        comm_cond_call = ir.Expr.binop('!=', self._rank_var, last_pe, loc)
        if comm_cond_call not in self.calltypes:
            self.calltypes[comm_cond_call] = find_op_typ('!=', [types.int32, types.int64])
        out.append(ir.Assign(comm_cond_call, comm_cond, loc))

        return comm_cond

    def _gen_1D_div(self, size_var, scope, loc, prefix, end_call_name, end_call):
        div_nodes = []
        if isinstance(size_var, int):
            new_size_var = ir.Var(scope, mk_unique_var(prefix+"_size_var"), loc)
            self.typemap[new_size_var.name] = types.int64
            size_assign = ir.Assign(ir.Const(size_var, loc), new_size_var, loc)
            div_nodes.append(size_assign)
            size_var = new_size_var
        div_var = ir.Var(scope, mk_unique_var(prefix+"_div_var"), loc)
        self.typemap[div_var.name] = types.int64
        div_expr = ir.Expr.binop('//', size_var, self._size_var, loc)
        self.calltypes[div_expr] = find_op_typ('//', [types.int64, types.int32])
        div_assign = ir.Assign(div_expr, div_var, loc)

        start_var = ir.Var(scope, mk_unique_var(prefix+"_start_var"), loc)
        self.typemap[start_var.name] = types.int64
        start_expr = ir.Expr.binop('*', div_var, self._rank_var, loc)
        self.calltypes[start_expr] = find_op_typ('*', [types.int64, types.int32])
        start_assign = ir.Assign(start_expr, start_var, loc)
        # attr call: end_attr = getattr(g_dist_var, get_end)
        end_attr_call = ir.Expr.getattr(self._g_dist_var, end_call_name, loc)
        end_attr_var = ir.Var(scope, mk_unique_var("$get_end_attr"), loc)
        self.typemap[end_attr_var.name] = get_global_func_typ(end_call)
        end_attr_assign = ir.Assign(end_attr_call, end_attr_var, loc)

        end_var = ir.Var(scope, mk_unique_var(prefix+"_end_var"), loc)
        self.typemap[end_var.name] = types.int64
        end_expr = ir.Expr.call(end_attr_var, [size_var, div_var,
            self._size_var, self._rank_var], (), loc)
        self.calltypes[end_expr] = self.typemap[end_attr_var.name].get_call_type(
            typing.Context(), [types.int64, types.int64, types.int32, types.int32], {})
        end_assign = ir.Assign(end_expr, end_var, loc)
        div_nodes += [div_assign, start_assign, end_attr_assign, end_assign]
        return div_nodes, start_var, end_var

    def _get_ind_sub(self, ind_var, start_var):
        sub_var = ir.Var(ind_var.scope, mk_unique_var("$sub_var"), ind_var.loc)
        self.typemap[sub_var.name] = types.int64
        sub_expr = ir.Expr.binop('-', ind_var, start_var, ind_var.loc)
        self.calltypes[sub_expr] = find_op_typ('-', [types.int64, types.int64])
        sub_assign = ir.Assign(sub_expr, sub_var, ind_var.loc)
        return sub_assign

    def _dist_prints(self, blocks):
        new_blocks = {}
        for (block_label, block) in blocks.items():
            scope = block.scope
            i = _find_first_print(block.body)
            while i!=-1:
                inst = block.body[i]
                loc = inst.loc
                # split block across print
                prev_block = ir.Block(scope, loc)
                new_blocks[block_label] = prev_block
                block_label = ir_utils.next_label()
                print_label = ir_utils.next_label()

                prev_block.body = block.body[:i]
                rank_comp_var = ir.Var(scope, mk_unique_var("$rank_comp"), loc)
                self.typemap[rank_comp_var.name] = types.boolean
                comp_expr = ir.Expr.binop('==', self._rank_var, self._set0_var, loc)
                expr_typ = find_op_typ('==',[types.int32, types.int64])
                self.calltypes[comp_expr] = expr_typ
                comp_assign = ir.Assign(comp_expr, rank_comp_var, loc)
                prev_block.body.append(comp_assign)
                print_branch = ir.Branch(rank_comp_var, print_label, block_label, loc)
                prev_block.body.append(print_branch)

                print_block = ir.Block(scope, loc)
                print_block.body.append(inst)
                print_block.body.append(ir.Jump(block_label, loc))
                new_blocks[print_label] = print_block

                block.body = block.body[i+1:]
                i = _find_first_print(block.body)
            new_blocks[block_label] = block
        return new_blocks

    def _isarray(self, varname):
        return (varname in self.typemap
                and isinstance(self.typemap[varname], types.npytypes.Array))

    def _is_1D_arr(self, arr_name):
        return (self._isarray(arr_name) and
                self._dist_analysis.array_dists[arr_name]==Distribution.OneD)

    def _is_alloc_call(self, func_var):
        if func_var not in self._call_table:
            return False
        return self._call_table[func_var]==['empty', np]

    def _is_h5_read_write_call(self, func_var):
        if func_var not in self._call_table:
            return False
        return (self._call_table[func_var]==['h5read', hpat.pio_api]
                or self._call_table[func_var]==['h5write', hpat.pio_api])

    def _is_call(self, func_var, call_list):
        if func_var not in self._call_table:
            return False
        return self._call_table[func_var]==call_list


def _find_first_print(body):
    for (i, inst) in enumerate(body):
        if isinstance(inst, ir.Print):
            return i
    return -1

def dprint(*s):
    if config.DEBUG_ARRAY_OPT==1:
        print(*s)

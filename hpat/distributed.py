from __future__ import print_function, division, absolute_import
from collections import namedtuple

import types as pytypes # avoid confusion with numba.types
import copy
import numba
from numba import (ir, analysis, types, typing, config, numpy_support, cgutils,
                    ir_utils, postproc)
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc,
                            get_global_func_typ, find_op_typ, get_name_var_table,
                            get_call_table, get_tuple_table, remove_dels)

from numba.parfor import (get_parfor_reductions, get_parfor_params,
                            wrap_parfor_blocks, unwrap_parfor_blocks)
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
from numba.parfor import Parfor, lower_parfor_sequential
import numpy as np

import hpat
import h5py
import time
# from mpi4py import MPI

from enum import Enum
class Distribution(Enum):
    REP = 1
    OneD = 3
    TwoD = 2

_dist_analysis_result = namedtuple('dist_analysis_result', 'array_dists,parfor_dists')

class DistributedPass(object):
    """analyze program and transfrom to distributed"""
    def __init__(self, func_ir, typemap, calltypes):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self._call_table,_ = get_call_table(func_ir.blocks)
        self._tuple_table = get_tuple_table(func_ir.blocks)
        self._rank_var = None # will be set in run
        self._size_var = None
        self._dist_analysis = None
        self._g_dist_var = None
        self._set1_var = None # variable set to 1
        self._set0_var = None # variable set to 0
        self._array_starts = {}
        self._array_counts = {}
        self._parallel_accesses = set()
        # keep shape attr calls on parallel arrays like X.shape
        self._shape_attrs = {}
        # keep array sizes of parallel arrays to handle shape attrs
        self._array_sizes = {}

    def run(self):
        remove_dels(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "starting distributed pass")
        self._dist_analysis = self._analyze_dist(self.func_ir.blocks)
        if config.DEBUG_ARRAY_OPT==1:
            print("distributions: ", self._dist_analysis)
        self._gen_dist_inits()
        self._run_dist_pass(self.func_ir.blocks)
        self.func_ir.blocks = self._dist_prints(self.func_ir.blocks)
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        dprint_func_ir(self.func_ir, "after distributed pass")
        lower_parfor_sequential(self.func_ir, self.typemap, self.calltypes)
        post_proc = postproc.PostProcessor(self.func_ir)
        post_proc.run()

    def _analyze_dist(self, blocks, array_dists={}, parfor_dists={}):
        topo_order = find_topo_order(blocks)
        save_array_dists = {}
        save_parfor_dists = {1:1} # dummy value
        # fixed-point iteration
        while array_dists!=save_array_dists or parfor_dists!=save_parfor_dists:
            save_array_dists = copy.copy(array_dists)
            save_parfor_dists = copy.copy(parfor_dists)
            for label in topo_order:
                self._analyze_block(blocks[label], array_dists, parfor_dists)

        return _dist_analysis_result(array_dists=array_dists, parfor_dists=parfor_dists)

    def _analyze_block(self, block, array_dists, parfor_dists):
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                self._analyze_assign(inst, array_dists, parfor_dists)
            elif isinstance(inst, Parfor):
                self._analyze_parfor(inst, array_dists, parfor_dists)
            elif (isinstance(inst, ir.SetItem)
                    and (inst.target.name,inst.index.name)
                    in self._parallel_accesses):
                pass # parallel access, don't make REP
            else:
                self._set_REP(inst.list_vars(), array_dists)

    def _analyze_assign(self, inst, array_dists, parfor_dists):
        lhs = inst.target.name
        rhs = inst.value
        # treat return casts like assignments
        if isinstance(rhs, ir.Expr) and rhs.op=='cast':
            rhs = rhs.value

        if isinstance(rhs, ir.Var) and self._isarray(lhs):
            lhs_dist = Distribution.OneD
            if lhs in array_dists:
                lhs_dist = array_dists[lhs]
            new_dist = Distribution(min(lhs_dist.value, array_dists[rhs.name].value))
            array_dists[lhs] = new_dist
            array_dists[rhs.name] = new_dist
            return

        elif (isinstance(rhs, ir.Expr) and rhs.op=='getitem'
                and (rhs.value.name,rhs.index.name) in self._parallel_accesses):
            return
        elif isinstance(rhs, ir.Expr) and rhs.op=='getattr':# and rhs.attr=='shape':
            # TODO: handle X.T properly
            if self._isarray(lhs) and lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return
        elif isinstance(rhs, ir.Expr) and rhs.op=='call':
            self._analyze_call(lhs, rhs.func.name, rhs.args, array_dists)
        else:
            self._set_REP(inst.list_vars(), array_dists)
        return

    def _analyze_parfor(self, parfor, array_dists, parfor_dists):
        if parfor.id not in parfor_dists:
            parfor_dists[parfor.id] = Distribution.OneD

        # analyze init block first to see array definitions
        self._analyze_block(parfor.init_block, array_dists, parfor_dists)
        out_dist = Distribution.OneD

        parfor_arrs = set() # arrays this parfor accesses in parallel
        array_accesses = ir_utils.get_array_accesses(parfor.loop_body)
        par_index_var = parfor.loop_nests[0].index_variable.name
        for (arr,index) in array_accesses.items():
            if index==par_index_var:
                parfor_arrs.add(arr)
                self._parallel_accesses.add((arr,index))
            if index in self._tuple_table:
                index_tuple = [(var.name if isinstance(var, ir.Var) else var)
                    for var in self._tuple_table[index]]
                if index_tuple[0]==par_index_var:
                    parfor_arrs.add(arr)
                    self._parallel_accesses.add((arr,index))
                if par_index_var in index_tuple[1:]:
                    out_dist = Distribution.REP
            # TODO: check for index dependency

        for arr in parfor_arrs:
            out_dist = Distribution(min(out_dist.value, array_dists[arr].value))
        parfor_dists[parfor.id] = out_dist
        for arr in parfor_arrs:
            array_dists[arr] = out_dist

        # run analysis recursively on parfor body
        blocks = wrap_parfor_blocks(parfor)
        for b in blocks.values():
            self._analyze_block(b, array_dists, parfor_dists)
        unwrap_parfor_blocks(parfor)
        return

    def _analyze_call(self, lhs, func_var, args, array_dists):
        if self._is_call(func_var, ['empty',np]):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return
        if self._is_call(func_var, ['h5read', hpat.pio]):
            return
        if self._is_call(func_var, ['dot', np]):
            arg0 = args[0].name
            arg1 = args[1].name
            ndim0 = self.typemap[arg0].ndim
            ndim1 = self.typemap[arg1].ndim
            # Fortran layout is caused by X.T and means transpose
            t0 = self.typemap[arg0].layout=='F'
            t1 = self.typemap[arg1].layout=='F'
            if ndim0==1 and ndim1==1:
                # vector dot, both vectors should have same layout
                new_dist = Distribution(min(array_dists[arg0].value,
                                                    array_dists[arg1].value))
                array_dists[arg0] = new_dist
                array_dists[arg1] = new_dist
                return
            if (not t0 and not t1 and array_dists[arg0]==Distribution.OneD
                    and array_dists[arg1]==Distribution.OneD):
                # reduction across samples np.dot(Y,X)
                array_dists[lhs] = Distribution.REP
                # print("case 1 ", arg0, arg1)
                return
            if not t0 and not t1 and array_dists[arg0]==Distribution.OneD:
                # samples dot weights: np.dot(X,w)
                array_dists[lhs] = Distribution.OneD
                # print("case 2 ", arg0, arg1)
                return
        for v in args:
            if self._isarray(v.name):
                array_dists[v.name] = Distribution.REP
        if self._isarray(lhs):
            array_dists[lhs] = Distribution.REP

    def _set_REP(self, var_list, array_dists):
        for var in var_list:
            varname = var.name
            if self._isarray(varname):
                array_dists[varname] = Distribution.REP

    def _run_dist_pass(self, blocks):
        topo_order = find_topo_order(blocks)
        namevar_table = get_name_var_table(blocks)
        #
        for label in topo_order:
            new_body = []
            for inst in blocks[label].body:
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
                        if (rhs.op=='exhaust_iter'
                                and rhs.value.name in self._shape_attrs):
                            self._shape_attrs[lhs] = self._shape_attrs[rhs.value.name]
                        if (rhs.op=='static_getitem'
                                and rhs.value.name in self._shape_attrs):
                            arr = self._shape_attrs[rhs.value.name]
                            sizes = self._array_sizes[arr]
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
        # g_dist_var = Global(hpat.distributed)
        g_dist_var = ir.Var(scope, mk_unique_var("$distributed_g_var"), loc)
        self._g_dist_var = g_dist_var
        self.typemap[g_dist_var.name] = types.misc.Module(hpat.distributed)
        g_dist = ir.Global('distributed', hpat.distributed, loc)
        g_dist_assign = ir.Assign(g_dist, g_dist_var, loc)
        # attr call: rank_attr = getattr(g_dist_var, get_rank)
        rank_attr_call = ir.Expr.getattr(g_dist_var, "get_rank", loc)
        rank_attr_var = ir.Var(scope, mk_unique_var("$get_rank_attr"), loc)
        self.typemap[rank_attr_var.name] = get_global_func_typ(get_rank)
        rank_attr_assign = ir.Assign(rank_attr_call, rank_attr_var, loc)
        # rank_var = hpat.distributed.get_rank()
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
        self.typemap[size_attr_var.name] = get_global_func_typ(get_size)
        size_attr_assign = ir.Assign(size_attr_call, size_attr_var, loc)
        # size_var = hpat.distributed.get_size()
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
        # divide 1D alloc
        if self._is_1D_arr(lhs) and self._is_alloc_call(func_var):
            size_var = rhs.args[0]
            if self.typemap[size_var.name]==types.intp:
                self._array_sizes[lhs] = [size_var]
                out, start_var, end_var = self._gen_1D_div(size_var, scope, loc,
                    "$alloc", "get_node_portion", get_node_portion)
                self._array_starts[lhs] = [start_var]
                self._array_counts[lhs] = [end_var]
                rhs.args[0] = end_var
            else:
                # size should be either int or tuple of ints
                assert size_var.name in self._tuple_table
                size_list = self._tuple_table[size_var.name]
                self._array_sizes[lhs] = size_list
                out, start_var, end_var = self._gen_1D_div(size_list[0], scope, loc,
                    "$alloc", "get_node_portion", get_node_portion)
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

        if self._is_h5_read_call(func_var) and self._is_1D_arr(rhs.args[6].name):
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
            file_var = rhs.args[0].name
            for stmt in block_body:
                if isinstance(stmt, ir.Assign) and stmt.target.name==file_var:
                    rhs = stmt.value
                    assert isinstance(rhs, ir.Expr)
                    rhs.args[2] = self._set1_var
        if self._is_call(func_var, ['dot',np]):
            arg0 = rhs.args[0].name
            arg1 = rhs.args[1].name
            ndim0 = self.typemap[arg0].ndim
            ndim1 = self.typemap[arg1].ndim
            # Fortran layout is caused by X.T and means transpose
            t0 = self.typemap[arg0].layout=='F'
            t1 = self.typemap[arg1].layout=='F'
            # reduction across dataset
            if self._is_1D_arr(rhs.args[0].name) and self._is_1D_arr(rhs.args[1].name):
                reduce_attr_var = ir.Var(scope, mk_unique_var("$reduce_attr"), loc)
                reduce_func_name = "dist_arr_reduce"
                reduce_func = dist_arr_reduce
                # output of vector dot() is scalar
                if ndim0==1 and ndim1==1:
                    reduce_func_name = "dist_reduce"
                    reduce_func = dist_reduce
                reduce_attr_call = ir.Expr.getattr(self._g_dist_var, reduce_func_name, loc)
                self.typemap[reduce_attr_var.name] = get_global_func_typ(reduce_func)
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
        # run dist pass recursively
        blocks = wrap_parfor_blocks(parfor)
        self._run_dist_pass(blocks)
        unwrap_parfor_blocks(parfor)

        if self._dist_analysis.parfor_dists[parfor.id]!=Distribution.OneD:
            if config.DEBUG_ARRAY_OPT==1:
                print("parfor "+str(parfor.id)+" not parallelized.")
            return [parfor]
        #
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        range_size = parfor.loop_nests[0].stop

        out, start_var, end_var = self._gen_1D_div(range_size, scope, loc, "$loop", "get_end", get_end)
        parfor.loop_nests[0].start = start_var
        parfor.loop_nests[0].stop = end_var
        # print_node = ir.Print([div_var, start_var, end_var], None, loc)
        # self.calltypes[print_node] = signature(types.none, types.int64, types.int64, types.int64)
        # out.append(print_node)
        out.append(parfor)
        parfor_params = get_parfor_params(parfor, self.func_ir)
        _, reductions = get_parfor_reductions(parfor, parfor_params)

        if len(reductions)!=0:
            reduce_attr_var = ir.Var(scope, mk_unique_var("$reduce_attr"), loc)
            reduce_attr_call = ir.Expr.getattr(self._g_dist_var, "dist_reduce", loc)
            self.typemap[reduce_attr_var.name] = get_global_func_typ(dist_reduce)
            reduce_assign = ir.Assign(reduce_attr_call, reduce_attr_var, loc)
            out.append(reduce_assign)

        for reduce_varname, (_, reduce_func) in reductions.items():
            reduce_var = namevar_table[reduce_varname]
            reduce_call = ir.Expr.call(reduce_attr_var, [reduce_var], (), loc)
            self.calltypes[reduce_call] = self.typemap[reduce_attr_var.name].get_call_type(
                typing.Context(), [self.typemap[reduce_varname]], {})
            reduce_assign = ir.Assign(reduce_call, reduce_var, loc)
            out.append(reduce_assign)

        return out

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
        return isinstance(self.typemap[varname], types.npytypes.Array)

    def _is_1D_arr(self, arr_name):
        return (self._isarray(arr_name) and
                self._dist_analysis.array_dists[arr_name]==Distribution.OneD)

    def _is_alloc_call(self, func_var):
        if func_var not in self._call_table:
            return False
        return self._call_table[func_var]==['empty', np]

    def _is_h5_read_call(self, func_var):
        if func_var not in self._call_table:
            return False
        return self._call_table[func_var]==['h5read', hpat.pio]

    def _is_call(self, func_var, call_list):
        if func_var not in self._call_table:
            return False
        return self._call_table[func_var]==call_list


def _find_first_print(body):
    for (i, inst) in enumerate(body):
        if isinstance(inst, ir.Print):
            return i
    return -1


from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature

def get_rank():
    """dummy function for C mpi get_rank"""
    return 0

def get_size():
    """dummy function for C mpi get_size"""
    return 0

def get_end(total_size, div, pes, rank):
    """get end point of range for parfor division"""
    return total_size if rank==pes-1 else (rank+1)*div

def get_node_portion(total_size, div, pes, rank):
    """get portion of size for alloc division"""
    return total_size-div*rank if rank==pes-1 else div

def dist_reduce(value):
    """dummy to implement simple reductions"""
    return value

def dist_arr_reduce(arr):
    """dummy to implement array reductions"""
    return -1

@infer_global(get_rank)
class DistRank(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==0
        return signature(types.int32, *args)

@infer_global(get_size)
class DistSize(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==0
        return signature(types.int32, *args)

@infer_global(get_end)
class DistEnd(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==4
        return signature(types.int64, *args)

@infer_global(get_node_portion)
class DistPortion(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==4
        return signature(types.int64, *args)

@infer_global(dist_reduce)
class DistReduce(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==1
        return signature(args[0], *args)

@infer_global(dist_arr_reduce)
class DistArrReduce(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==1
        return signature(types.int32, *args)

@infer_global(time.time)
class DistTime(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==0
        return signature(types.float64, *args)

from llvmlite import ir as lir
import hdist
import llvmlite.binding as ll
ll.add_symbol('hpat_dist_get_rank', hdist.hpat_dist_get_rank)
ll.add_symbol('hpat_dist_get_size', hdist.hpat_dist_get_size)
ll.add_symbol('hpat_dist_get_end', hdist.hpat_dist_get_end)
ll.add_symbol('hpat_dist_get_node_portion', hdist.hpat_dist_get_node_portion)
ll.add_symbol('hpat_dist_get_time', hdist.hpat_dist_get_time)
ll.add_symbol('hpat_dist_reduce', hdist.hpat_dist_reduce)
ll.add_symbol('hpat_dist_arr_reduce', hdist.hpat_dist_arr_reduce)

@lower_builtin(get_rank)
def dist_get_rank(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_rank")
    return builder.call(fn, [])

@lower_builtin(get_size)
def dist_get_size(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_size")
    return builder.call(fn, [])

@lower_builtin(get_end, types.int64, types.int64, types.int32, types.int32)
def dist_get_end(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64), lir.IntType(64),
                                            lir.IntType(32), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_end")
    return builder.call(fn, [args[0], args[1], args[2], args[3]])

@lower_builtin(get_node_portion, types.int64, types.int64, types.int32, types.int32)
def dist_get_portion(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64), lir.IntType(64),
                                            lir.IntType(32), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_node_portion")
    return builder.call(fn, [args[0], args[1], args[2], args[3]])

@lower_builtin(dist_reduce, types.int64)
@lower_builtin(dist_reduce, types.int32)
@lower_builtin(dist_reduce, types.float32)
@lower_builtin(dist_reduce, types.float64)
def lower_dist_reduce(context, builder, sig, args):
    ltyp = args[0].type
    fnty = lir.FunctionType(ltyp, [ltyp])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_reduce")
    return builder.call(fn, [args[0]])

@lower_builtin(dist_arr_reduce, types.npytypes.Array)
def lower_dist_arr_reduce(context, builder, sig, args):
    # store an int to specify data type
    typ_enum = hpat.pio._h5_typ_table[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))
    ndims = sig.args[0].ndim
    out = make_array(sig.args[0])(context, builder, args[0])
    # TODO: handle multi-dimensional arrays properly
    #size_arg = builder.alloca(lir.IntType(64))
    size_arg = builder.load(out.shape.value.operands[0])
    for i in range(1,ndims):
        size_arg = builder.mul(size_arg, out.shape.value.operands[i])
    ndim_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), sig.args[0].ndim))
    call_args = [builder.bitcast(out.data, lir.IntType(8).as_pointer()),
                size_arg, builder.load(ndim_arg), builder.load(typ_arg)]

    # array, shape, ndim, extra last arg type for type enum
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64),
        lir.IntType(32), lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_arr_reduce")
    return builder.call(fn, call_args)

@lower_builtin(time.time)
def dist_get_time(context, builder, sig, args):
    fnty = lir.FunctionType(lir.DoubleType(), [])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_dist_get_time")
    return builder.call(fn, [])

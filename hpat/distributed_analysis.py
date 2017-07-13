from __future__ import print_function, division, absolute_import
from collections import namedtuple
import copy

import numba
from numba import ir, ir_utils, config
from numba.ir_utils import get_call_table, get_tuple_table, find_topo_order
from numba.parfor import Parfor
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks

import numpy as np
import hpat

from enum import Enum
class Distribution(Enum):
    REP = 1
    OneD = 4
    OneD_Var = 3
    TwoD = 2

_dist_analysis_result = namedtuple('dist_analysis_result', 'array_dists,parfor_dists')

distributed_analysis_extensions = {}


class DistributedAnalysis(object):
    """analyze program for to distributed transfromation"""
    def __init__(self, func_ir, typemap, calltypes):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self._call_table,_ = get_call_table(func_ir.blocks)
        self._tuple_table = get_tuple_table(func_ir.blocks)
        self._parallel_accesses = set()
        self._T_arrs = set()

    def run(self):
        blocks = self.func_ir.blocks
        array_dists = {}
        parfor_dists = {}
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
            elif type(inst) in distributed_analysis_extensions:
                # let external calls handle stmt if type matches
                f = distributed_analysis_extensions[type(inst)]
                f(inst, array_dists)
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
        elif (isinstance(rhs, ir.Expr) and rhs.op=='getattr' and rhs.attr=='T'
                    and self._isarray(lhs)):
            # array and its transpose have same distributions
            arr = rhs.value.name
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            new_dist = Distribution(min(array_dists[lhs].value, array_dists[arr].value))
            array_dists[lhs] = new_dist
            array_dists[arr] = new_dist
            # keep lhs in table for dot() handling
            self._T_arrs.add(lhs)
            return
        elif isinstance(rhs, ir.Expr) and rhs.op=='getattr' and rhs.attr=='shape':
            pass # X.shape doesn't affect X distribution
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
        stencil_accesses = get_stencil_accesses(parfor.loop_body, par_index_var)
        for (arr,index) in array_accesses:
            if index==par_index_var or index in stencil_accesses:
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
        if func_var not in self._call_table or not self._call_table[func_var]:
            self._analyze_call_set_REP(lhs, func_var, args, array_dists)
            return

        call_list = self._call_table[func_var]

        if self._is_call(func_var, ['empty', np]):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if (self._is_call(func_var, ['h5read', hpat.pio_api])
                or self._is_call(func_var, ['h5write', hpat.pio_api])):
            return

        if (len(call_list)==2 and call_list[1]==np
                and call_list[0] in ['cumsum', 'cumprod', 'empty_like',
                    'zeros_like', 'ones_like', 'full_like', 'copy']):
            in_arr = args[0].name
            lhs_dist = Distribution.OneD
            if lhs in array_dists:
                lhs_dist = array_dists[lhs]
            new_dist = Distribution(min(lhs_dist.value, array_dists[in_arr].value))
            array_dists[lhs] = new_dist
            array_dists[in_arr] = new_dist
            return

        if self._is_call(func_var, ['dot', np]):
            arg0 = args[0].name
            arg1 = args[1].name
            ndim0 = self.typemap[arg0].ndim
            ndim1 = self.typemap[arg1].ndim
            dist0 = array_dists[arg0]
            dist1 = array_dists[arg1]
            # Fortran layout is caused by X.T and means transpose
            t0 = arg0 in self._T_arrs
            t1 = arg1 in self._T_arrs
            if ndim0==1 and ndim1==1:
                # vector dot, both vectors should have same layout
                new_dist = Distribution(min(array_dists[arg0].value,
                                                    array_dists[arg1].value))
                array_dists[arg0] = new_dist
                array_dists[arg1] = new_dist
                return
            if ndim0==2 and ndim1==1 and not t0:
                # special case were arg1 vector is treated as column vector
                # samples dot weights: np.dot(X,w)
                # w is always REP
                array_dists[arg1] = Distribution.REP
                if lhs not in array_dists:
                    array_dists[lhs] = Distribution.OneD
                # lhs and X have same distribution
                new_dist = Distribution(min(array_dists[arg0].value,
                                                    array_dists[lhs].value))
                array_dists[arg0] = new_dist
                array_dists[lhs] = new_dist
                dprint("dot case 1 Xw:", arg0, arg1)
                return
            if ndim0==1 and ndim1==2 and not t1:
                # reduction across samples np.dot(Y,X)
                # lhs is always REP
                array_dists[lhs] = Distribution.REP
                # Y and X have same distribution
                new_dist = Distribution(min(array_dists[arg0].value,
                                                    array_dists[arg1].value))
                array_dists[arg0] = new_dist
                array_dists[arg1] = new_dist
                dprint("dot case 2 YX:", arg0, arg1)
                return
            if ndim0==2 and ndim1==2 and t0 and not t1:
                # reduction across samples np.dot(X.T,Y)
                # lhs is always REP
                array_dists[lhs] = Distribution.REP
                # Y and X have same distribution
                new_dist = Distribution(min(array_dists[arg0].value,
                                                    array_dists[arg1].value))
                array_dists[arg0] = new_dist
                array_dists[arg1] = new_dist
                dprint("dot case 3 XtY:", arg0, arg1)
                return
            if ndim0==2 and ndim1==2 and not t0 and not t1:
                # samples dot weights: np.dot(X,w)
                # w is always REP
                array_dists[arg1] = Distribution.REP
                if lhs not in array_dists:
                    array_dists[lhs] = Distribution.OneD
                new_dist = Distribution(min(array_dists[arg0].value,
                                                    array_dists[lhs].value))
                array_dists[arg0] = new_dist
                array_dists[lhs] = new_dist
                dprint("dot case 4 Xw:", arg0, arg1)
                return
        # set REP if not found
        self._analyze_call_set_REP(lhs, func_var, args, array_dists)

    def _analyze_call_set_REP(self, lhs, func_var, args, array_dists):
        for v in args:
            if self._isarray(v.name):
                dprint("dist setting call arg REP {}".format(v.name))
                array_dists[v.name] = Distribution.REP
        if self._isarray(lhs):
            dprint("dist setting call out REP {}".format(lhs))
            array_dists[lhs] = Distribution.REP

    def _set_REP(self, var_list, array_dists):
        for var in var_list:
            varname = var.name
            if self._isarray(varname):
                dprint("dist setting REP {}".format(varname))
                array_dists[varname] = Distribution.REP

    def _isarray(self, varname):
        return (varname in self.typemap
            and isinstance(self.typemap[varname], numba.types.npytypes.Array))

    def _is_call(self, func_var, call_list):
        if func_var not in self._call_table:
            return False
        return self._call_table[func_var]==call_list


def get_stencil_accesses(body, par_index_var):
    # TODO support recursive parfor, multi-D, mutiple body blocks
    const_table = {}
    stencil_accesses = {}

    for block in body.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Const):
                lhs = stmt.target.name
                const_table[lhs] = stmt.value.value
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                lhs = stmt.target.name
                rhs = stmt.value
                if (rhs.op == 'binop' and rhs.fn == '+' and
                        rhs.lhs.name == par_index_var and
                        rhs.rhs.name in const_table):
                    stencil_accesses[lhs] = const_table[rhs.rhs.name]

    return stencil_accesses

def dprint(*s):
    if config.DEBUG_ARRAY_OPT==1:
        print(*s)

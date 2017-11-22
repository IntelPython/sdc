from __future__ import print_function, division, absolute_import
from collections import namedtuple
import copy

import numba
from numba import ir, ir_utils, types
from numba.ir_utils import get_call_table, get_tuple_table, find_topo_order, guard, get_definition
from numba.parfor import Parfor
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks

import numpy as np
import hpat
from hpat.utils import get_definitions, is_alloc_call

from enum import Enum
class Distribution(Enum):
    REP = 1
    Thread = 2
    TwoD = 3
    OneD_Var = 4
    OneD = 5

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
        self.second_pass = False
        self.in_parallel_parfor = False

    def run(self):
        blocks = self.func_ir.blocks
        array_dists = {}
        parfor_dists = {}
        topo_order = find_topo_order(blocks)
        self._run_analysis(self.func_ir.blocks, topo_order, array_dists, parfor_dists)
        self.second_pass = True
        self._run_analysis(self.func_ir.blocks, topo_order, array_dists, parfor_dists)

        return _dist_analysis_result(array_dists=array_dists, parfor_dists=parfor_dists)

    def _run_analysis(self, blocks, topo_order, array_dists, parfor_dists):
        save_array_dists = {}
        save_parfor_dists = {1:1} # dummy value
        # fixed-point iteration
        while array_dists!=save_array_dists or parfor_dists!=save_parfor_dists:
            save_array_dists = copy.copy(array_dists)
            save_parfor_dists = copy.copy(parfor_dists)
            for label in topo_order:
                self._analyze_block(blocks[label], array_dists, parfor_dists)

    def _analyze_block(self, block, array_dists, parfor_dists):
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                self._analyze_assign(inst, array_dists, parfor_dists)
            elif isinstance(inst, Parfor):
                self._analyze_parfor(inst, array_dists, parfor_dists)
            elif isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                if isinstance(inst, ir.SetItem):
                    index = inst.index.name
                else:
                    index = inst.index_var.name
                if ((inst.target.name, index) not in self._parallel_accesses):
                    # no parallel to parallel array set (TODO)
                    self._set_REP([inst.value], array_dists)
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
            self._meet_array_dists(lhs, rhs.name, array_dists)
            return

        elif isinstance(rhs, ir.Expr) and rhs.op in ['getitem', 'static_getitem']:
            self._analyze_getitem(inst, lhs, rhs, array_dists)
            return
        elif isinstance(rhs, ir.Expr) and rhs.op == 'build_tuple':
            # parallel arrays can be packed and unpacked from tuples
            # e.g. boolean array index in test_getitem_multidim
            return
        elif (isinstance(rhs, ir.Expr) and rhs.op=='getattr' and rhs.attr=='T'
                    and self._isarray(lhs)):
            # array and its transpose have same distributions
            arr = rhs.value.name
            self._meet_array_dists(lhs, arr, array_dists)
            # keep lhs in table for dot() handling
            self._T_arrs.add(lhs)
            return
        elif (isinstance(rhs, ir.Expr) and rhs.op == 'getattr'
                and rhs.attr in ['shape', 'ndim', 'size', 'strides', 'dtype']):
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
        if self.in_parallel_parfor:
            out_dist = Distribution.REP

        parfor_arrs = set() # arrays this parfor accesses in parallel
        array_accesses = ir_utils.get_array_accesses(parfor.loop_body)
        par_index_var = parfor.loop_nests[0].index_variable.name
        stencil_accesses, _ = get_stencil_accesses(parfor, self.typemap)
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
            if arr in array_dists:
                out_dist = Distribution(min(out_dist.value, array_dists[arr].value))
        parfor_dists[parfor.id] = out_dist
        for arr in parfor_arrs:
            if arr in array_dists:
                array_dists[arr] = out_dist

        # TODO: find prange actually coming from user
        # for pattern in parfor.patterns:
        #     if pattern[0] == 'prange' and not self.in_parallel_parfor:
        #         parfor_dists[parfor.id] = Distribution.OneD

        # run analysis recursively on parfor body
        if self.second_pass and out_dist==Distribution.OneD:
            self.in_parallel_parfor = True
        blocks = wrap_parfor_blocks(parfor)
        for b in blocks.values():
            self._analyze_block(b, array_dists, parfor_dists)
        unwrap_parfor_blocks(parfor)
        self.in_parallel_parfor = False
        return

    def _analyze_call(self, lhs, func_var, args, array_dists):
        if func_var not in self._call_table or not self._call_table[func_var]:
            self._analyze_call_set_REP(lhs, func_var, args, array_dists)
            return

        call_list = self._call_table[func_var]

        if is_alloc_call(func_var, self._call_table):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if self._is_call(func_var, [len]):
            return

        if self._is_call(func_var, ['ravel', np]):
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        if hpat.config._has_h5py and (self._is_call(func_var, ['h5read', hpat.pio_api])
                or self._is_call(func_var, ['h5write', hpat.pio_api])):
            return

        if hpat.config._has_pyarrow and call_list==[hpat.parquet_pio.read_parquet]:
            return

        if hpat.config._has_pyarrow and call_list==[hpat.parquet_pio.read_parquet_str]:
            # string read creates array in output
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if (len(call_list)==2 and call_list[1]==np
                and call_list[0] in ['cumsum', 'cumprod', 'empty_like',
                    'zeros_like', 'ones_like', 'full_like', 'copy']):
            in_arr = args[0].name
            self._meet_array_dists(lhs, in_arr, array_dists)
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
                self._meet_array_dists(lhs, arg0, array_dists)
                dprint("dot case 1 Xw:", arg0, arg1)
                return
            if ndim0==1 and ndim1==2 and not t1:
                # reduction across samples np.dot(Y,X)
                # lhs is always REP
                array_dists[lhs] = Distribution.REP
                # Y and X have same distribution
                self._meet_array_dists(arg0, arg1, array_dists)
                dprint("dot case 2 YX:", arg0, arg1)
                return
            if ndim0==2 and ndim1==2 and t0 and not t1:
                # reduction across samples np.dot(X.T,Y)
                # lhs is always REP
                array_dists[lhs] = Distribution.REP
                # Y and X have same distribution
                self._meet_array_dists(arg0, arg1, array_dists)
                dprint("dot case 3 XtY:", arg0, arg1)
                return
            if ndim0==2 and ndim1==2 and not t0 and not t1:
                # samples dot weights: np.dot(X,w)
                # w is always REP
                array_dists[arg1] = Distribution.REP
                self._meet_array_dists(lhs, arg0, array_dists)
                dprint("dot case 4 Xw:", arg0, arg1)
                return

        if call_list == ['train']:
            getattr_call = guard(get_definition, self.func_ir, func_var)
            if getattr_call and self.typemap[getattr_call.value.name] == hpat.ml.svc_type:
                self._meet_array_dists(args[0].name, args[1].name, array_dists, Distribution.Thread)
                return

        if call_list == ['predict']:
            getattr_call = guard(get_definition, self.func_ir, func_var)
            if getattr_call and self.typemap[getattr_call.value.name] == hpat.ml.svc_type:
                self._meet_array_dists(lhs, args[0].name, array_dists, Distribution.Thread)
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

    def _analyze_getitem(self, inst, lhs, rhs, array_dists):
        if rhs.op == 'static_getitem':
            if rhs.index_var is None:
                return
            index_var = rhs.index_var.name
        else:
            assert rhs.op == 'getitem'
            index_var = rhs.index.name

        if (rhs.value.name, index_var) in self._parallel_accesses:
            #self._set_REP([inst.target], array_dists)
            return

        # in multi-dimensional case, we only consider first dimension
        # TODO: extend to 2D distribution
        if index_var in self._tuple_table:
            inds = self._tuple_table[index_var]
            index_var = inds[0].name
            # rest of indices should be replicated if array
            self._set_REP(inds[1:], array_dists)

        # array selection with boolean index
        if (is_array(index_var, self.typemap)
                    and self.typemap[index_var].dtype==types.boolean):
            # input array and bool index have the same distribution
            new_dist = self._meet_array_dists(index_var, rhs.value.name, array_dists)
            array_dists[lhs] = Distribution(min(Distribution.OneD_Var.value, new_dist.value))
            return
        self._set_REP(inst.list_vars(), array_dists)
        return

    def _meet_array_dists(self, arr1, arr2, array_dists, top_dist=None):
        if top_dist is None:
            top_dist = Distribution.OneD
        if arr1 not in array_dists:
            array_dists[arr1] = top_dist
        if arr2 not in array_dists:
            array_dists[arr2] = top_dist

        new_dist = Distribution(min(array_dists[arr1].value,
                                            array_dists[arr2].value))
        new_dist = Distribution(min(new_dist.value, top_dist.value))
        array_dists[arr1] = new_dist
        array_dists[arr2] = new_dist
        return new_dist

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


def is_array(varname, typemap):
    #return True
    return (varname in typemap
        and isinstance(typemap[varname], numba.types.npytypes.Array))

def get_stencil_accesses(parfor, typemap):
    # if a parfor has stencil pattern, see which accesses depend on loop index
    # XXX: assuming loop index is not used for non-stencil arrays
    # TODO support recursive parfor, multi-D, mutiple body blocks

    # no access if not stencil
    is_stencil = False
    for pattern in parfor.patterns:
        if pattern[0] == 'stencil':
            is_stencil = True
            neighborhood = pattern[1]
    if not is_stencil:
        return {}, None

    par_index_var = parfor.loop_nests[0].index_variable
    body = parfor.loop_body
    body_defs = get_definitions(body)

    stencil_accesses = {}

    for block in body.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                lhs = stmt.target.name
                rhs = stmt.value
                if (rhs.op == 'getitem' and is_array(rhs.value.name, typemap)
                        and vars_dependent(body_defs, rhs.index, par_index_var)):
                    stencil_accesses[rhs.index.name] = rhs.value.name

    return stencil_accesses, neighborhood

def vars_dependent(defs, var1, var2):
    # see if var1 depends on var2 based on definitions in defs
    if len(defs[var1.name]) != 1:
        return False

    vardef = defs[var1.name][0]
    if isinstance(vardef, ir.Var) and vardef.name == var2.name:
        return True
    if isinstance(vardef, ir.Expr):
        for invar in vardef.list_vars():
            if invar.name == var2.name or vars_dependent(defs, invar, var2):
                return True
    return False

def dprint(*s):
    if numba.config.DEBUG_ARRAY_OPT==1:
        print(*s)

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
from collections import namedtuple
import copy
import warnings

import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (find_topo_order, guard, get_definition, require,
                            find_callname, mk_unique_var, compile_to_numba_ir,
                            replace_arg_nodes, build_definitions,
                            find_build_sequence, find_const)
from numba.parfor import Parfor
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks

import numpy as np
import hpat
import hpat.io
import hpat.io.np_io
from hpat.hiframes.pd_series_ext import SeriesType
from hpat.utils import (get_constant, is_alloc_callname,
                        is_whole_slice, is_array, is_array_container,
                        is_np_array, find_build_tuple, debug_prints,
                        is_const_slice)
from hpat.hiframes.pd_dataframe_ext import DataFrameType
from enum import Enum


class Distribution(Enum):
    REP = 1
    Thread = 2
    TwoD = 3
    OneD_Var = 4
    OneD = 5


_dist_analysis_result = namedtuple(
    'dist_analysis_result', 'array_dists,parfor_dists')

distributed_analysis_extensions = {}
auto_rebalance = False


class DistributedAnalysis(object):
    """Analyze program for distributed transformation"""

    _extra_call = {}

    @classmethod
    def add_call_analysis(cls, typ, func, analysis_func):
        '''
        External modules/packages (like daal4py) can register their own call-analysis.
        Analysis funcs are stored in a dict with keys (typ, funcname)
        '''
        assert (typ, func) not in cls._extra_call
        cls._extra_call[(typ, func)] = analysis_func

    def __init__(self, func_ir, typemap, calltypes, typingctx, metadata):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.typingctx = typingctx
        self.metadata = metadata

    def _init_run(self):
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        self._parallel_accesses = set()
        self._T_arrs = set()
        self.second_pass = False
        self.in_parallel_parfor = -1

    def run(self):
        self._init_run()
        blocks = self.func_ir.blocks
        array_dists = {}
        parfor_dists = {}
        topo_order = find_topo_order(blocks)
        self._run_analysis(self.func_ir.blocks, topo_order,
                           array_dists, parfor_dists)
        self.second_pass = True
        self._run_analysis(self.func_ir.blocks, topo_order,
                           array_dists, parfor_dists)
        # rebalance arrays if necessary
        if auto_rebalance and Distribution.OneD_Var in array_dists.values():
            changed = self._rebalance_arrs(array_dists, parfor_dists)
            if changed:
                return self.run()

        return _dist_analysis_result(array_dists=array_dists, parfor_dists=parfor_dists)

    def _run_analysis(self, blocks, topo_order, array_dists, parfor_dists):
        save_array_dists = {}
        save_parfor_dists = {1: 1}  # dummy value
        # fixed-point iteration
        while array_dists != save_array_dists or parfor_dists != save_parfor_dists:
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
                self._analyze_setitem(inst, array_dists)
            # elif isinstance(inst, ir.Print):
            #     continue
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
        if isinstance(rhs, ir.Expr) and rhs.op == 'cast':
            rhs = rhs.value

        if isinstance(rhs, ir.Var) and (is_array(self.typemap, lhs)
                                        or isinstance(self.typemap[lhs], (SeriesType, DataFrameType))
                                        or is_array_container(self.typemap, lhs)):
            self._meet_array_dists(lhs, rhs.name, array_dists)
            return
        elif (is_array(self.typemap, lhs)
                and isinstance(rhs, ir.Expr)
                and rhs.op == 'inplace_binop'):
            # distributions of all 3 variables should meet (lhs, arg1, arg2)
            arg1 = rhs.lhs.name
            arg2 = rhs.rhs.name
            dist = self._meet_array_dists(arg1, arg2, array_dists)
            dist = self._meet_array_dists(arg1, lhs, array_dists, dist)
            self._meet_array_dists(arg1, arg2, array_dists, dist)
            return
        elif isinstance(rhs, ir.Expr) and rhs.op in ['getitem', 'static_getitem']:
            self._analyze_getitem(inst, lhs, rhs, array_dists)
            return
        elif isinstance(rhs, ir.Expr) and rhs.op == 'build_tuple':
            # parallel arrays can be packed and unpacked from tuples
            # e.g. boolean array index in test_getitem_multidim
            return
        elif (isinstance(rhs, ir.Expr) and rhs.op == 'getattr' and rhs.attr == 'T'
              and is_array(self.typemap, lhs)):
            # array and its transpose have same distributions
            arr = rhs.value.name
            self._meet_array_dists(lhs, arr, array_dists)
            # keep lhs in table for dot() handling
            self._T_arrs.add(lhs)
            return
        elif (isinstance(rhs, ir.Expr) and rhs.op == 'getattr'
                and isinstance(self.typemap[rhs.value.name], DataFrameType)
                and rhs.attr == 'to_csv'):
            return
        elif (isinstance(rhs, ir.Expr) and rhs.op == 'getattr'
                and rhs.attr in ['shape', 'ndim', 'size', 'strides', 'dtype',
                                 'itemsize', 'astype', 'reshape', 'ctypes',
                                 'transpose', 'tofile', 'copy']):
            pass  # X.shape doesn't affect X distribution
        elif isinstance(rhs, ir.Expr) and rhs.op == 'call':
            self._analyze_call(lhs, rhs, rhs.func.name, rhs.args, array_dists)
        # handle for A in arr_container: ...
        # A = pair_first(iternext(getiter(arr_container)))
        # TODO: support getitem of container
        elif isinstance(rhs, ir.Expr) and rhs.op == 'pair_first' and is_array(self.typemap, lhs):
            arr_container = guard(_get_pair_first_container, self.func_ir, rhs)
            if arr_container is not None:
                self._meet_array_dists(lhs, arr_container.name, array_dists)
                return
        elif isinstance(rhs, ir.Expr) and rhs.op in ('getiter', 'iternext'):
            # analyze array container access in pair_first
            return
        elif isinstance(rhs, ir.Arg):
            if rhs.name in self.metadata['distributed']:
                if lhs not in array_dists:
                    array_dists[lhs] = Distribution.OneD
            elif rhs.name in self.metadata['threaded']:
                if lhs not in array_dists:
                    array_dists[lhs] = Distribution.Thread
            else:
                dprint("replicated input ", rhs.name, lhs)
                self._set_REP([inst.target], array_dists)
        else:
            self._set_REP(inst.list_vars(), array_dists)
        return

    def _analyze_parfor(self, parfor, array_dists, parfor_dists):
        if parfor.id not in parfor_dists:
            parfor_dists[parfor.id] = Distribution.OneD

        # analyze init block first to see array definitions
        self._analyze_block(parfor.init_block, array_dists, parfor_dists)
        out_dist = Distribution.OneD
        if self.in_parallel_parfor != -1:
            out_dist = Distribution.REP

        parfor_arrs = set()  # arrays this parfor accesses in parallel
        array_accesses = _get_array_accesses(
            parfor.loop_body, self.func_ir, self.typemap)
        par_index_var = parfor.loop_nests[0].index_variable.name
        #stencil_accesses, _ = get_stencil_accesses(parfor, self.typemap)
        for (arr, index) in array_accesses:
            # XXX sometimes copy propagation doesn't work for parfor indices
            # so see if the index has a single variable definition and use it
            # e.g. test_to_numeric
            ind_def = self.func_ir._definitions[index]
            if len(ind_def) == 1 and isinstance(ind_def[0], ir.Var):
                index = ind_def[0].name
            if index == par_index_var:  # or index in stencil_accesses:
                parfor_arrs.add(arr)
                self._parallel_accesses.add((arr, index))

            # multi-dim case
            tup_list = guard(find_build_tuple, self.func_ir, index)
            if tup_list is not None:
                index_tuple = [var.name for var in tup_list]
                if index_tuple[0] == par_index_var:
                    parfor_arrs.add(arr)
                    self._parallel_accesses.add((arr, index))
                if par_index_var in index_tuple[1:]:
                    out_dist = Distribution.REP
            # TODO: check for index dependency

        for arr in parfor_arrs:
            if arr in array_dists:
                out_dist = Distribution(
                    min(out_dist.value, array_dists[arr].value))
        parfor_dists[parfor.id] = out_dist
        for arr in parfor_arrs:
            if arr in array_dists:
                array_dists[arr] = out_dist

        # TODO: find prange actually coming from user
        # for pattern in parfor.patterns:
        #     if pattern[0] == 'prange' and not self.in_parallel_parfor:
        #         parfor_dists[parfor.id] = Distribution.OneD

        # run analysis recursively on parfor body
        if self.second_pass and out_dist in [Distribution.OneD,
                                             Distribution.OneD_Var]:
            self.in_parallel_parfor = parfor.id
        blocks = wrap_parfor_blocks(parfor)
        for b in blocks.values():
            self._analyze_block(b, array_dists, parfor_dists)
        unwrap_parfor_blocks(parfor)
        if self.in_parallel_parfor == parfor.id:
            self.in_parallel_parfor = -1
        return

    def _analyze_call(self, lhs, rhs, func_var, args, array_dists):
        """analyze array distributions in function calls
        """
        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            # check ObjModeLiftedWith, we assume distribution doesn't change
            # blocks of data are passed in, TODO: document
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Const) and isinstance(func_def.value,
                                                             numba.dispatcher.ObjModeLiftedWith):
                return
            warnings.warn(
                "function call couldn't be found for distributed analysis")
            self._analyze_call_set_REP(lhs, args, array_dists, fdef)
            return
        else:
            func_name, func_mod = fdef

        if is_alloc_callname(func_name, func_mod):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        # numpy direct functions
        if isinstance(func_mod, str) and func_mod == 'numpy':
            self._analyze_call_np(lhs, func_name, args, array_dists)
            return

        # handle array.func calls
        if isinstance(func_mod, ir.Var) and is_array(self.typemap, func_mod.name):
            self._analyze_call_array(lhs, func_mod, func_name, args, array_dists)
            return

        # handle df.func calls
        if isinstance(func_mod, ir.Var) and isinstance(
                self.typemap[func_mod.name], DataFrameType):
            self._analyze_call_df(lhs, func_mod, func_name, args, array_dists)
            return

        # hpat.distributed_api functions
        if isinstance(func_mod, str) and func_mod == 'hpat.distributed_api':
            self._analyze_call_hpat_dist(lhs, func_name, args, array_dists)
            return

        # len()
        if func_name == 'len' and func_mod in ('__builtin__', 'builtins'):
            return

        if hpat.config._has_h5py and (func_mod == 'hpat.io.pio_api'
                                      and func_name in ('h5read', 'h5write', 'h5read_filter')):
            return

        if hpat.config._has_h5py and (func_mod == 'hpat.io.pio_api'
                                      and func_name == 'get_filter_read_indices'):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if fdef == ('quantile', 'hpat.hiframes.api'):
            # quantile doesn't affect input's distribution
            return

        if fdef == ('nunique', 'hpat.hiframes.api'):
            # nunique doesn't affect input's distribution
            return

        if fdef == ('unique', 'hpat.hiframes.api'):
            # doesn't affect distribution of input since input can stay 1D
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD_Var

            new_dist = Distribution(min(array_dists[lhs].value,
                                        array_dists[rhs.args[0].name].value))
            array_dists[lhs] = new_dist
            return

        if fdef == ('rolling_fixed', 'hpat.hiframes.rolling'):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ('rolling_variable', 'hpat.hiframes.rolling'):
            # lhs, in_arr, on_arr should have the same distribution
            new_dist = self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            new_dist = self._meet_array_dists(lhs, rhs.args[1].name, array_dists, new_dist)
            array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ('shift', 'hpat.hiframes.rolling'):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ('pct_change', 'hpat.hiframes.rolling'):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ('nlargest', 'hpat.hiframes.api'):
            # output of nlargest is REP
            array_dists[lhs] = Distribution.REP
            return

        if fdef == ('median', 'hpat.hiframes.api'):
            return

        if fdef == ('concat', 'hpat.hiframes.api'):
            # hiframes concat is similar to np.concatenate
            self._analyze_call_np_concatenate(lhs, args, array_dists)
            return

        if fdef == ('isna', 'hpat.hiframes.api'):
            return

        if fdef == ('get_series_name', 'hpat.hiframes.api'):
            return

        # dummy hiframes functions
        if func_mod == 'hpat.hiframes.api' and func_name in ('get_series_data',
                                                             'get_series_index',
                                                             'to_arr_from_series', 'ts_series_to_arr_typ',
                                                             'to_date_series_type', 'dummy_unbox_series',
                                                             'parallel_fix_df_array'):
            # TODO: support Series type similar to Array
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ('init_series', 'hpat.hiframes.api'):
            # lhs, in_arr, and index should have the same distribution
            new_dist = self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            if len(rhs.args) > 1 and self.typemap[rhs.args[1].name] != types.none:
                new_dist = self._meet_array_dists(lhs, rhs.args[1].name, array_dists, new_dist)
                array_dists[rhs.args[0].name] = new_dist
            return

        if fdef == ('init_dataframe', 'hpat.hiframes.pd_dataframe_ext'):
            # lhs, data arrays, and index should have the same distribution
            df_typ = self.typemap[lhs]
            n_cols = len(df_typ.columns)
            for i in range(n_cols):
                new_dist = self._meet_array_dists(lhs, rhs.args[i].name, array_dists)
            # handle index
            if len(rhs.args) > n_cols and self.typemap[rhs.args[n_cols].name] != types.none:
                new_dist = self._meet_array_dists(lhs, rhs.args[n_cols].name, array_dists, new_dist)
            for i in range(n_cols):
                array_dists[rhs.args[i].name] = new_dist
            return

        if fdef == ('get_dataframe_data', 'hpat.hiframes.pd_dataframe_ext'):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ('compute_split_view', 'hpat.hiframes.split_impl'):
            self._meet_array_dists(lhs, rhs.args[0].name, array_dists)
            return

        if fdef == ('get_split_view_index', 'hpat.hiframes.split_impl'):
            # just used in str.get() implementation for now so we know it is
            # parallel
            # TODO: handle index similar to getitem to support more cases
            return

        if fdef == ('get_split_view_data_ptr', 'hpat.hiframes.split_impl'):
            return

        if fdef == ('setitem_str_arr_ptr', 'hpat.str_arr_ext'):
            return

        if fdef == ('num_total_chars', 'hpat.str_arr_ext'):
            return

        if fdef == ('_series_dropna_str_alloc_impl_inner', 'hpat.hiframes.series_kernels'):
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD_Var
            in_dist = array_dists[rhs.args[0].name]
            out_dist = array_dists[lhs]
            out_dist = Distribution(min(out_dist.value, in_dist.value))
            array_dists[lhs] = out_dist
            # output can cause input REP
            if out_dist != Distribution.OneD_Var:
                array_dists[rhs.args[0].name] = out_dist
            return

        if (fdef == ('copy_non_null_offsets', 'hpat.str_arr_ext')
                or fdef == ('copy_data', 'hpat.str_arr_ext')):
            out_arrname = rhs.args[0].name
            in_arrname = rhs.args[1].name
            self._meet_array_dists(out_arrname, in_arrname, array_dists)
            return

        if fdef == ('str_arr_item_to_numeric', 'hpat.str_arr_ext'):
            out_arrname = rhs.args[0].name
            in_arrname = rhs.args[2].name
            self._meet_array_dists(out_arrname, in_arrname, array_dists)
            return

        # np.fromfile()
        if fdef == ('file_read', 'hpat.io.np_io'):
            return

        if hpat.config._has_ros and fdef == ('read_ros_images_inner', 'hpat.ros'):
            return

        if hpat.config._has_pyarrow and fdef == ('read_parquet', 'hpat.io.parquet_pio'):
            return

        if hpat.config._has_pyarrow and fdef == ('read_parquet_str', 'hpat.io.parquet_pio'):
            # string read creates array in output
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        # TODO: fix "numba.extending" in function def
        if hpat.config._has_xenon and fdef == ('read_xenon_col', 'numba.extending'):
            array_dists[args[4].name] = Distribution.REP
            return

        if hpat.config._has_xenon and fdef == ('read_xenon_str', 'numba.extending'):
            array_dists[args[4].name] = Distribution.REP
            # string read creates array in output
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            return

        if func_name == 'train' and isinstance(func_mod, ir.Var):
            if self.typemap[func_mod.name] == hpat.ml.svc.svc_type:
                self._meet_array_dists(
                    args[0].name, args[1].name, array_dists, Distribution.Thread)
                return
            if self.typemap[func_mod.name] == hpat.ml.naive_bayes.mnb_type:
                self._meet_array_dists(args[0].name, args[1].name, array_dists)
                return

        if func_name == 'predict' and isinstance(func_mod, ir.Var):
            if self.typemap[func_mod.name] == hpat.ml.svc.svc_type:
                self._meet_array_dists(
                    lhs, args[0].name, array_dists, Distribution.Thread)
                return
            if self.typemap[func_mod.name] == hpat.ml.naive_bayes.mnb_type:
                self._meet_array_dists(lhs, args[0].name, array_dists)
                return

        # TODO: make sure assert_equiv is not generated unnecessarily
        # TODO: fix assert_equiv for np.stack from df.value
        if fdef == ('assert_equiv', 'numba.array_analysis'):
            return

        # we perform call-analysis from external at the end
        if isinstance(func_mod, ir.Var):
            ky = (self.typemap[func_mod.name], func_name)
            if ky in DistributedAnalysis._extra_call:
                if DistributedAnalysis._extra_call[ky](lhs, func_mod, *ky, args, array_dists):
                    return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, fdef)

    def _analyze_call_np(self, lhs, func_name, args, array_dists):
        """analyze distributions of numpy functions (np.func_name)
        """

        if func_name == 'ascontiguousarray':
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        if func_name == 'ravel':
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        if func_name == 'concatenate':
            self._analyze_call_np_concatenate(lhs, args, array_dists)
            return

        if func_name == 'array' and is_array(self.typemap, args[0].name):
            self._meet_array_dists(lhs, args[0].name, array_dists)
            return

        # sum over the first axis is distributed, A.sum(0)
        if func_name == 'sum' and len(args) == 2:
            axis_def = guard(get_definition, self.func_ir, args[1])
            if isinstance(axis_def, ir.Const) and axis_def.value == 0:
                array_dists[lhs] = Distribution.REP
                return

        if func_name == 'dot':
            self._analyze_call_np_dot(lhs, args, array_dists)
            return

        # used in df.values
        if func_name == 'stack':
            seq_info = guard(find_build_sequence, self.func_ir, args[0])
            if seq_info is None:
                self._analyze_call_set_REP(lhs, args, array_dists, 'np.' + func_name)
                return
            in_arrs, _ = seq_info

            axis = 0
            # TODO: support kws
            # if 'axis' in kws:
            #     axis = find_const(self.func_ir, kws['axis'])
            if len(args) > 1:
                axis = find_const(self.func_ir, args[1])

            # parallel if args are 1D and output is 2D and axis == 1
            if axis is not None and axis == 1 and self.typemap[lhs].ndim == 2:
                for v in in_arrs:
                    self._meet_array_dists(lhs, v.name, array_dists)
                return

        if (func_name in ['cumsum', 'cumprod', 'empty_like',
                          'zeros_like', 'ones_like', 'full_like', 'copy']):
            in_arr = args[0].name
            self._meet_array_dists(lhs, in_arr, array_dists)
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, 'np.' + func_name)

    def _analyze_call_array(self, lhs, arr, func_name, args, array_dists):
        """analyze distributions of array functions (arr.func_name)
        """
        if func_name == 'transpose':
            if len(args) == 0:
                raise ValueError("Transpose with no arguments is not"
                                 " supported")
            in_arr_name = arr.name
            arg0 = guard(get_constant, self.func_ir, args[0])
            if isinstance(arg0, tuple):
                arg0 = arg0[0]
            if arg0 != 0:
                raise ValueError("Transpose with non-zero first argument"
                                 " is not supported")
            self._meet_array_dists(lhs, in_arr_name, array_dists)
            return

        if func_name in ('astype', 'reshape', 'copy'):
            in_arr_name = arr.name
            self._meet_array_dists(lhs, in_arr_name, array_dists)
            # TODO: support 1D_Var reshape
            if func_name == 'reshape' and array_dists[lhs] == Distribution.OneD_Var:
                # HACK support A.reshape(n, 1) for 1D_Var
                if len(args) == 2 and guard(find_const, self.func_ir, args[1]) == 1:
                    return
                self._analyze_call_set_REP(lhs, args, array_dists, 'array.' + func_name)
            return

        # Array.tofile() is supported for all distributions
        if func_name == 'tofile':
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, 'array.' + func_name)

    def _analyze_call_df(self, lhs, arr, func_name, args, array_dists):
        # to_csv() can be parallelized
        if func_name == 'to_csv':
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, 'df.' + func_name)

    def _analyze_call_hpat_dist(self, lhs, func_name, args, array_dists):
        """analyze distributions of hpat distributed functions
        (hpat.distributed_api.func_name)
        """
        if func_name == 'local_len':
            return
        if func_name == 'parallel_print':
            return

        if func_name == 'dist_return':
            arr_name = args[0].name
            assert arr_name in array_dists, "array distribution not found"
            if array_dists[arr_name] == Distribution.REP:
                raise ValueError("distributed return of array {} not valid"
                                 " since it is replicated".format(arr_name))
            return

        if func_name == 'threaded_return':
            arr_name = args[0].name
            assert arr_name in array_dists, "array distribution not found"
            if array_dists[arr_name] == Distribution.REP:
                raise ValueError("threaded return of array {} not valid"
                                 " since it is replicated")
            array_dists[arr_name] = Distribution.Thread
            return

        if func_name == 'rebalance_array':
            if lhs not in array_dists:
                array_dists[lhs] = Distribution.OneD
            in_arr = args[0].name
            if array_dists[in_arr] == Distribution.OneD_Var:
                array_dists[lhs] = Distribution.OneD
            else:
                self._meet_array_dists(lhs, in_arr, array_dists)
            return

        # set REP if not found
        self._analyze_call_set_REP(lhs, args, array_dists, 'hpat.distributed_api.' + func_name)

    def _analyze_call_np_concatenate(self, lhs, args, array_dists):
        assert len(args) == 1
        tup_def = guard(get_definition, self.func_ir, args[0])
        assert isinstance(tup_def, ir.Expr) and tup_def.op == 'build_tuple'
        in_arrs = tup_def.items
        # input arrays have same distribution
        in_dist = Distribution.OneD
        for v in in_arrs:
            in_dist = Distribution(
                min(in_dist.value, array_dists[v.name].value))
        # OneD_Var since sum of block sizes might not be exactly 1D
        out_dist = Distribution.OneD_Var
        out_dist = Distribution(min(out_dist.value, in_dist.value))
        array_dists[lhs] = out_dist
        # output can cause input REP
        if out_dist != Distribution.OneD_Var:
            in_dist = out_dist
        for v in in_arrs:
            array_dists[v.name] = in_dist
        return

    def _analyze_call_np_dot(self, lhs, args, array_dists):

        arg0 = args[0].name
        arg1 = args[1].name
        ndim0 = self.typemap[arg0].ndim
        ndim1 = self.typemap[arg1].ndim
        dist0 = array_dists[arg0]
        dist1 = array_dists[arg1]
        # Fortran layout is caused by X.T and means transpose
        t0 = arg0 in self._T_arrs
        t1 = arg1 in self._T_arrs
        if ndim0 == 1 and ndim1 == 1:
            # vector dot, both vectors should have same layout
            new_dist = Distribution(min(array_dists[arg0].value,
                                        array_dists[arg1].value))
            array_dists[arg0] = new_dist
            array_dists[arg1] = new_dist
            return
        if ndim0 == 2 and ndim1 == 1 and not t0:
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
        if ndim0 == 1 and ndim1 == 2 and not t1:
            # reduction across samples np.dot(Y,X)
            # lhs is always REP
            array_dists[lhs] = Distribution.REP
            # Y and X have same distribution
            self._meet_array_dists(arg0, arg1, array_dists)
            dprint("dot case 2 YX:", arg0, arg1)
            return
        if ndim0 == 2 and ndim1 == 2 and t0 and not t1:
            # reduction across samples np.dot(X.T,Y)
            # lhs is always REP
            array_dists[lhs] = Distribution.REP
            # Y and X have same distribution
            self._meet_array_dists(arg0, arg1, array_dists)
            dprint("dot case 3 XtY:", arg0, arg1)
            return
        if ndim0 == 2 and ndim1 == 2 and not t0 and not t1:
            # samples dot weights: np.dot(X,w)
            # w is always REP
            array_dists[arg1] = Distribution.REP
            self._meet_array_dists(lhs, arg0, array_dists)
            dprint("dot case 4 Xw:", arg0, arg1)
            return

        # set REP if no pattern matched
        self._analyze_call_set_REP(lhs, args, array_dists, 'np.dot')

    def _analyze_call_set_REP(self, lhs, args, array_dists, fdef=None):
        for v in args:
            if (is_array(self.typemap, v.name)
                    or is_array_container(self.typemap, v.name)
                    or isinstance(self.typemap[v.name], DataFrameType)):
                dprint("dist setting call arg REP {} in {}".format(v.name, fdef))
                array_dists[v.name] = Distribution.REP
        if (is_array(self.typemap, lhs)
                or is_array_container(self.typemap, lhs)
                or isinstance(self.typemap[lhs], DataFrameType)):
            dprint("dist setting call out REP {} in {}".format(lhs, fdef))
            array_dists[lhs] = Distribution.REP

    def _analyze_getitem(self, inst, lhs, rhs, array_dists):
        # selecting an array from a tuple
        if (rhs.op == 'static_getitem'
                and isinstance(self.typemap[rhs.value.name], types.BaseTuple)
                and isinstance(rhs.index, int)):
            seq_info = guard(find_build_sequence, self.func_ir, rhs.value)
            if seq_info is not None:
                in_arrs, _ = seq_info
                arr = in_arrs[rhs.index]
                self._meet_array_dists(lhs, arr.name, array_dists)
                return

        if rhs.op == 'static_getitem':
            if rhs.index_var is None:
                # TODO: things like A[0] need broadcast
                self._set_REP(inst.list_vars(), array_dists)
                return
            index_var = rhs.index_var
        else:
            assert rhs.op == 'getitem'
            index_var = rhs.index

        if (rhs.value.name, index_var.name) in self._parallel_accesses:
            # XXX: is this always valid? should be done second pass?
            self._set_REP([inst.target], array_dists)
            return

        # in multi-dimensional case, we only consider first dimension
        # TODO: extend to 2D distribution
        tup_list = guard(find_build_tuple, self.func_ir, index_var)
        if tup_list is not None:
            index_var = tup_list[0]
            # rest of indices should be replicated if array
            other_ind_vars = tup_list[1:]
            self._set_REP(other_ind_vars, array_dists)

        if isinstance(index_var, int):
            self._set_REP(inst.list_vars(), array_dists)
            return
        assert isinstance(index_var, ir.Var)

        # array selection with boolean index
        if (is_np_array(self.typemap, index_var.name)
                and self.typemap[index_var.name].dtype == types.boolean):
            # input array and bool index have the same distribution
            new_dist = self._meet_array_dists(index_var.name, rhs.value.name,
                                              array_dists)
            array_dists[lhs] = Distribution(min(Distribution.OneD_Var.value,
                                                new_dist.value))
            return

        # array selection with permutation array index
        if is_np_array(self.typemap, index_var.name):
            arr_def = guard(get_definition, self.func_ir, index_var)
            if isinstance(arr_def, ir.Expr) and arr_def.op == 'call':
                fdef = guard(find_callname, self.func_ir, arr_def, self.typemap)
                if fdef == ('permutation', 'numpy.random'):
                    self._meet_array_dists(lhs, rhs.value.name, array_dists)
                    return

        # whole slice or strided slice access
        # for example: A = X[:,5], A = X[::2,5]
        if guard(is_whole_slice, self.typemap, self.func_ir, index_var,
                 accept_stride=True):
            self._meet_array_dists(lhs, rhs.value.name, array_dists)
            return

        # output of operations like S.head() is REP since it's a "small" slice
        # input can remain 1D
        if guard(is_const_slice, self.typemap, self.func_ir, index_var):
            array_dists[lhs] = Distribution.REP
            return

        self._set_REP(inst.list_vars(), array_dists)
        return

    def _analyze_setitem(self, inst, array_dists):
        if isinstance(inst, ir.SetItem):
            index_var = inst.index
        else:
            index_var = inst.index_var

        if ((inst.target.name, index_var.name) in self._parallel_accesses):
            # no parallel to parallel array set (TODO)
            return

        tup_list = guard(find_build_tuple, self.func_ir, index_var)
        if tup_list is not None:
            index_var = tup_list[0]
            # rest of indices should be replicated if array
            self._set_REP(tup_list[1:], array_dists)

        if guard(is_whole_slice, self.typemap, self.func_ir, index_var):
            # for example: X[:,3] = A
            self._meet_array_dists(
                inst.target.name, inst.value.name, array_dists)
            return

        self._set_REP([inst.value], array_dists)

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
            # Handle SeriesType since it comes from Arg node and it could
            # have user-defined distribution
            if (is_array(self.typemap, varname)
                    or is_array_container(self.typemap, varname)
                    or isinstance(
                        self.typemap[varname], (SeriesType, DataFrameType))):
                dprint("dist setting REP {}".format(varname))
                array_dists[varname] = Distribution.REP
            # handle tuples of arrays
            var_def = guard(get_definition, self.func_ir, var)
            if (var_def is not None and isinstance(var_def, ir.Expr)
                    and var_def.op == 'build_tuple'):
                tuple_vars = var_def.items
                self._set_REP(tuple_vars, array_dists)

    def _rebalance_arrs(self, array_dists, parfor_dists):
        # rebalance an array if it is accessed in a parfor that has output
        # arrays or is in a loop

        # find sequential loop bodies
        cfg = numba.analysis.compute_cfg_from_blocks(self.func_ir.blocks)
        loop_bodies = set()
        for loop in cfg.loops().values():
            loop_bodies |= loop.body

        rebalance_arrs = set()

        for label, block in self.func_ir.blocks.items():
            for inst in block.body:
                # TODO: handle hiframes filter etc.
                if (isinstance(inst, Parfor)
                        and parfor_dists[inst.id] == Distribution.OneD_Var):
                    array_accesses = _get_array_accesses(
                        inst.loop_body, self.func_ir, self.typemap)
                    onedv_arrs = set(arr for (arr, ind) in array_accesses
                                     if arr in array_dists and array_dists[arr] == Distribution.OneD_Var)
                    if (label in loop_bodies
                            or _arrays_written(onedv_arrs, inst.loop_body)):
                        rebalance_arrs |= onedv_arrs

        if len(rebalance_arrs) != 0:
            self._gen_rebalances(rebalance_arrs, self.func_ir.blocks)
            return True

        return False

    def _gen_rebalances(self, rebalance_arrs, blocks):
        #
        for block in blocks.values():
            new_body = []
            for inst in block.body:
                # TODO: handle hiframes filter etc.
                if isinstance(inst, Parfor):
                    self._gen_rebalances(rebalance_arrs, {0: inst.init_block})
                    self._gen_rebalances(rebalance_arrs, inst.loop_body)
                if isinstance(inst, ir.Assign) and inst.target.name in rebalance_arrs:
                    out_arr = inst.target
                    self.func_ir._definitions[out_arr.name].remove(inst.value)
                    # hold inst results in tmp array
                    tmp_arr = ir.Var(out_arr.scope,
                                     mk_unique_var("rebalance_tmp"),
                                     out_arr.loc)
                    self.typemap[tmp_arr.name] = self.typemap[out_arr.name]
                    inst.target = tmp_arr
                    nodes = [inst]

                    def f(in_arr):  # pragma: no cover
                        out_a = hpat.distributed_api.rebalance_array(in_arr)
                    f_block = compile_to_numba_ir(f, {'hpat': hpat}, self.typingctx,
                                                  (self.typemap[tmp_arr.name],),
                                                  self.typemap, self.calltypes).blocks.popitem()[1]
                    replace_arg_nodes(f_block, [tmp_arr])
                    nodes += f_block.body[:-3]  # remove none return
                    nodes[-1].target = out_arr
                    # update definitions
                    dumm_block = ir.Block(out_arr.scope, out_arr.loc)
                    dumm_block.body = nodes
                    build_definitions({0: dumm_block}, self.func_ir._definitions)
                    new_body += nodes
                else:
                    new_body.append(inst)

            block.body = new_body


def _get_pair_first_container(func_ir, rhs):
    assert isinstance(rhs, ir.Expr) and rhs.op == 'pair_first'
    iternext = get_definition(func_ir, rhs.value)
    require(isinstance(iternext, ir.Expr) and iternext.op == 'iternext')
    getiter = get_definition(func_ir, iternext.value)
    require(isinstance(iternext, ir.Expr) and getiter.op == 'getiter')
    return getiter.value


def _arrays_written(arrs, blocks):
    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, Parfor) and _arrays_written(arrs, inst.loop_body):
                return True
            if (isinstance(inst, (ir.SetItem, ir.StaticSetItem))
                    and inst.target.name in arrs):
                return True
    return False

# def get_stencil_accesses(parfor, typemap):
#     # if a parfor has stencil pattern, see which accesses depend on loop index
#     # XXX: assuming loop index is not used for non-stencil arrays
#     # TODO support recursive parfor, multi-D, mutiple body blocks

#     # no access if not stencil
#     is_stencil = False
#     for pattern in parfor.patterns:
#         if pattern[0] == 'stencil':
#             is_stencil = True
#             neighborhood = pattern[1]
#     if not is_stencil:
#         return {}, None

#     par_index_var = parfor.loop_nests[0].index_variable
#     body = parfor.loop_body
#     body_defs = build_definitions(body)

#     stencil_accesses = {}

#     for block in body.values():
#         for stmt in block.body:
#             if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
#                 lhs = stmt.target.name
#                 rhs = stmt.value
#                 if (rhs.op == 'getitem' and is_array(typemap, rhs.value.name)
#                         and vars_dependent(body_defs, rhs.index, par_index_var)):
#                     stencil_accesses[rhs.index.name] = rhs.value.name

#     return stencil_accesses, neighborhood


# def vars_dependent(defs, var1, var2):
#     # see if var1 depends on var2 based on definitions in defs
#     if len(defs[var1.name]) != 1:
#         return False

#     vardef = defs[var1.name][0]
#     if isinstance(vardef, ir.Var) and vardef.name == var2.name:
#         return True
#     if isinstance(vardef, ir.Expr):
#         for invar in vardef.list_vars():
#             if invar.name == var2.name or vars_dependent(defs, invar, var2):
#                 return True
#     return False


# array access code is copied from ir_utils to be able to handle specialized
# array access calls such as get_split_view_index()
# TODO: implement extendable version in ir_utils
def get_parfor_array_accesses(parfor, func_ir, typemap, accesses=None):
    if accesses is None:
        accesses = set()
    blocks = wrap_parfor_blocks(parfor)
    accesses = _get_array_accesses(blocks, func_ir, typemap, accesses)
    unwrap_parfor_blocks(parfor)
    return accesses


array_accesses_extensions = {}
array_accesses_extensions[Parfor] = get_parfor_array_accesses


def _get_array_accesses(blocks, func_ir, typemap, accesses=None):
    """returns a set of arrays accessed and their indices.
    """
    if accesses is None:
        accesses = set()

    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.SetItem):
                accesses.add((inst.target.name, inst.index.name))
            if isinstance(inst, ir.StaticSetItem):
                accesses.add((inst.target.name, inst.index_var.name))
            if isinstance(inst, ir.Assign):
                lhs = inst.target.name
                rhs = inst.value
                if isinstance(rhs, ir.Expr) and rhs.op == 'getitem':
                    accesses.add((rhs.value.name, rhs.index.name))
                if isinstance(rhs, ir.Expr) and rhs.op == 'static_getitem':
                    index = rhs.index
                    # slice is unhashable, so just keep the variable
                    if index is None or ir_utils.is_slice_index(index):
                        index = rhs.index_var.name
                    accesses.add((rhs.value.name, index))
                if isinstance(rhs, ir.Expr) and rhs.op == 'call':
                    fdef = guard(find_callname, func_ir, rhs, typemap)
                    if fdef is not None:
                        if fdef == ('get_split_view_index', 'hpat.hiframes.split_impl'):
                            accesses.add((rhs.args[0].name, rhs.args[1].name))
                        if fdef == ('setitem_str_arr_ptr', 'hpat.str_arr_ext'):
                            accesses.add((rhs.args[0].name, rhs.args[1].name))
                        if fdef == ('str_arr_item_to_numeric', 'hpat.str_arr_ext'):
                            accesses.add((rhs.args[0].name, rhs.args[1].name))
                            accesses.add((rhs.args[2].name, rhs.args[3].name))
            for T, f in array_accesses_extensions.items():
                if isinstance(inst, T):
                    f(inst, func_ir, typemap, accesses)
    return accesses


def dprint(*s):
    if debug_prints():
        print(*s)

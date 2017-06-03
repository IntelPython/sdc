from __future__ import print_function, division, absolute_import
import types as pytypes # avoid confusion with numba.types

import numba
from numba import ir, analysis, types, config, numpy_support, cgutils
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc)

from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
import numpy as np

import hpat
import h5py

class PIO(object):
    """analyze and transform hdf5 calls"""
    def __init__(self, func_ir, local_vars):
        self.func_ir = func_ir
        self.local_vars = local_vars
        self.h5_globals = []
        self.h5_file_calls = []
        self.h5_files = {}
        self.h5_dsets = {}
        # varname -> 'str'
        self.str_const_table = {}
        self.reverse_copies = {}

    def run(self):
        dprint_func_ir(self.func_ir, "starting IO")
        topo_order = find_topo_order(self.func_ir.blocks)
        for label in topo_order:
            new_body = []
            # copies are collected before running the pass since
            # variables typed in locals are assigned late
            self._get_reverse_copies(self.func_ir.blocks[label].body)
            for inst in self.func_ir.blocks[label].body:
                if isinstance(inst, ir.Assign):
                    inst_list = self._run_assign(inst)
                    if inst_list is not None:
                        new_body.extend(inst_list)
                else:
                    new_body.append(inst)
            self.func_ir.blocks[label].body = new_body
        remove_dead(self.func_ir.blocks, self.func_ir.arg_names)
        dprint_func_ir(self.func_ir, "after IO")
        if config.DEBUG_ARRAY_OPT==1:
            print("h5 files: ", self.h5_files)
            print("h5 dsets: ", self.h5_dsets)

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value
        # lhs = h5py
        if (isinstance(rhs, ir.Global) and isinstance(rhs.value, pytypes.ModuleType)
                    and rhs.value==h5py):
            self.h5_globals.append(lhs)
        if isinstance(rhs, ir.Expr):
            # f_call = h5py.File
            if rhs.op=='getattr' and rhs.value.name in self.h5_globals and rhs.attr=='File':
                self.h5_file_calls.append(lhs)
            # f = h5py.File(file_name, mode)
            if rhs.op=='call' and rhs.func.name in self.h5_file_calls:
                file_name = self.str_const_table[rhs.args[0].name]
                self.h5_files[lhs] = file_name
                # parallel arg = False for this stage
                loc = assign.target.loc
                scope = assign.target.scope
                parallel_var = ir.Var(scope, mk_unique_var("$const_parallel"), loc)
                parallel_assign = ir.Assign(ir.Const(0, loc), parallel_var, loc)
                rhs.args.append(parallel_var)
                return [parallel_assign, assign]
            # d = f['dset']
            if rhs.op=='static_getitem' and rhs.value.name in self.h5_files:
                self.h5_dsets[lhs] = (rhs.value, rhs.index_var)
            # x = f['dset'][:]
            if rhs.op=='static_getitem' and rhs.value.name in self.h5_dsets:
                return self._gen_h5read(assign.target, rhs)
        # handle copies lhs = f
        if isinstance(rhs, ir.Var) and rhs.name in self.h5_files:
            self.h5_files[lhs] = self.h5_files[rhs.name]
        if isinstance(rhs, ir.Const) and isinstance(rhs.value, str):
            self.str_const_table[lhs] = rhs.value
        return [assign]

    def _gen_h5read(self, lhs_var, rhs):
        f_id, dset  = self.h5_dsets[rhs.value.name]
        file_name = self.h5_files[f_id.name]
        dset_str = self.str_const_table[dset.name]
        dset_type = self._get_dset_type(lhs_var.name, file_name, dset_str)
        loc = rhs.value.loc
        scope = rhs.value.scope
        # TODO: generate size, alloc calls
        out = []
        size_vars = self._gen_h5size(f_id, dset, dset_type.ndim, scope, loc, out)
        out.extend(mk_alloc(None, None, lhs_var, tuple(size_vars), dset_type.dtype, scope, loc))
        self._gen_h5read_call(f_id, dset, size_vars, lhs_var, scope, loc, out)
        return out

    def _gen_h5size(self, f_id, dset, ndims, scope, loc, out):
        # g_pio_var = Global(hpat.pio)
        g_pio_var = ir.Var(scope, mk_unique_var("$pio_g_var"), loc)
        g_pio = ir.Global('pio', hpat.pio, loc)
        g_pio_assign = ir.Assign(g_pio, g_pio_var, loc)
        # attr call: h5size_attr = getattr(g_pio_var, h5size)
        h5size_attr_call = ir.Expr.getattr(g_pio_var, "h5size", loc)
        attr_var = ir.Var(scope, mk_unique_var("$h5size_attr"), loc)
        attr_assign = ir.Assign(h5size_attr_call, attr_var, loc)
        out += [g_pio_assign, attr_assign]

        size_vars = []
        for i in range(ndims):
            dim_var = ir.Var(scope, mk_unique_var("$h5_dim_var"), loc)
            dim_assign = ir.Assign(ir.Const(np.int32(i), loc), dim_var, loc)
            out.append(dim_assign)
            size_var = ir.Var(scope, mk_unique_var("$h5_size_var"), loc)
            size_vars.append(size_var)
            size_call = ir.Expr.call(attr_var, [f_id, dset, dim_var], (), loc)
            size_assign = ir.Assign(size_call, size_var, loc)
            out.append(size_assign)
        return size_vars

    def _gen_h5read_call(self, f_id, dset, size_vars, lhs_var, scope, loc, out):
        # g_pio_var = Global(hpat.pio)
        g_pio_var = ir.Var(scope, mk_unique_var("$pio_g_var"), loc)
        g_pio = ir.Global('pio', hpat.pio, loc)
        g_pio_assign = ir.Assign(g_pio, g_pio_var, loc)
        # attr call: h5size_attr = getattr(g_pio_var, h5read)
        h5size_attr_call = ir.Expr.getattr(g_pio_var, "h5read", loc)
        attr_var = ir.Var(scope, mk_unique_var("$h5read_attr"), loc)
        attr_assign = ir.Assign(h5size_attr_call, attr_var, loc)
        out += [g_pio_assign, attr_assign]

        # ndims args
        ndims = len(size_vars)
        ndims_var = ir.Var(scope, mk_unique_var("$h5_ndims"), loc)
        ndims_assign = ir.Assign(ir.Const(np.int32(ndims), loc), ndims_var, loc)
        # sizes arg
        sizes_var = ir.Var(scope, mk_unique_var("$h5_sizes"), loc)
        tuple_call = ir.Expr.build_tuple(size_vars, loc)
        sizes_assign = ir.Assign(tuple_call, sizes_var, loc)

        zero_var = ir.Var(scope, mk_unique_var("$const_zero"), loc)
        zero_assign = ir.Assign(ir.Const(0, loc), zero_var, loc)
        # starts: assign to zeros
        starts_var = ir.Var(scope, mk_unique_var("$h5_starts"), loc)
        start_tuple_call = ir.Expr.build_tuple([zero_var]*ndims, loc)
        starts_assign = ir.Assign(start_tuple_call, starts_var, loc)
        out += [ndims_assign, zero_assign, starts_assign, sizes_assign]

        err_var = ir.Var(scope, mk_unique_var("$h5_err_var"), loc)
        read_call = ir.Expr.call(attr_var, [f_id, dset, ndims_var, starts_var, sizes_var, zero_var, lhs_var], (), loc)
        out.append(ir.Assign(read_call, err_var, loc))
        return

    def _get_dset_type(self, lhs, file_name, dset_str):
        """get data set type from user-specified locals types or actual file"""
        if lhs in self.local_vars:
            return self.local_vars[lhs]
        if self.reverse_copies[lhs] in self.local_vars:
            return self.local_vars[self.reverse_copies[lhs]]

        f = h5py.File(file_name, "r")
        ndims = len(f[dset_str].shape)
        numba_dtype = numpy_support.from_dtype(f[dset_str].dtype)
        return types.Array(numba_dtype, ndims, 'C')

    def _get_reverse_copies(self, body):
        for inst in body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var):
                self.reverse_copies[inst.value.name] = inst.target.name
        return


from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature

def h5size():
    """dummy function for C h5_size"""
    return

def h5read():
    """dummy function for C h5_read"""
    return

@infer_global(h5py.File)
class H5File(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==3
        return signature(types.int32, *args)

@infer_global(h5size)
class H5Size(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==3
        return signature(types.int64, *args)

@infer_global(h5read)
class H5Read(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==7
        return signature(types.int32, *args)

from llvmlite import ir as lir

#@lower_builtin(h5py.File, types.string, types.string)
#@lower_builtin(h5py.File, types.string, types.Const)
#@lower_builtin(h5py.File, types.Const, types.string)
@lower_builtin(h5py.File, types.Const, types.Const, types.int64)
def h5_open(context, builder, sig, args):
    # works for constant strings only
    # TODO: extend to string variables
    arg1, arg2, _ = sig.args
    val1 = context.insert_const_string(builder.module, arg1.value)
    val2 = context.insert_const_string(builder.module, arg2.value)
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_open")
    return builder.call(fn, [val1, val2, args[2]])

@lower_builtin(h5size, types.int32, types.Const, types.int32)
def h5_size(context, builder, sig, args):
    # works for constant string only
    # TODO: extend to string variables
    arg1, arg2, args3 = sig.args
    val2 = context.insert_const_string(builder.module, arg2.value)
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(32), lir.IntType(8).as_pointer(), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_size")
    return builder.call(fn, [args[0], val2, args[2]])

@lower_builtin(h5read, types.int32, types.Const, types.int32,
    types.containers.UniTuple, types.containers.UniTuple, types.int64,
    types.npytypes.Array)
def h5_read(context, builder, sig, args):
    # insert the dset_name string arg
    dset_name_arg = sig.args[1]
    val2 = context.insert_const_string(builder.module, dset_name_arg.value)
    # extra last arg type for type enum
    arg_typs = [lir.IntType(32), lir.IntType(8).as_pointer(), lir.IntType(32),
        lir.IntType(64).as_pointer(), lir.IntType(64).as_pointer(),
        lir.IntType(64), lir.IntType(8).as_pointer(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="hpat_h5_read")
    out = make_array(sig.args[6])(context, builder, args[6])
    # store size vars array struct to pointer
    count_ptr = cgutils.alloca_once(builder, args[3].type)
    builder.store(args[3], count_ptr)
    size_ptr = cgutils.alloca_once(builder, args[4].type)
    builder.store(args[4], size_ptr)
    # store an int to specify data type
    typ_enum = _h5_typ_table[sig.args[6].dtype]
    typ_arg = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(32), typ_enum))
    call_args = [args[0], val2, args[2],
        builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
        builder.bitcast(size_ptr, lir.IntType(64).as_pointer()), args[5],
        builder.bitcast(out.data, lir.IntType(8).as_pointer()),
        builder.load(typ_arg)]

    return builder.call(fn, call_args)

_h5_typ_table = {
    types.int8:0,
    types.uint8:1,
    types.int32:2,
    types.int64:3,
    types.float32:4,
    types.float64:5
    }

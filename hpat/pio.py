from __future__ import print_function, division, absolute_import
import types as pytypes  # avoid confusion with numba.types

import numba
from numba import ir, analysis, types, config, numpy_support
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc,
                            find_callname, guard, require, get_definition,
                            build_definitions, find_const, compile_to_numba_ir,
                            replace_arg_nodes)

import numpy as np

import hpat
from hpat import pio_api, pio_lower, utils
from hpat.utils import get_constant, NOT_CONSTANT, debug_prints
import h5py


def remove_h5(rhs, lives, call_list):
    # the call is dead if the read array is dead
    if call_list == ['h5read', pio_api] and rhs.args[6].name not in lives:
        return True
    if call_list == ['h5size', pio_api]:
        return True
    return False


numba.ir_utils.remove_call_handlers.append(remove_h5)


class PIO(object):
    """analyze and transform hdf5 calls"""

    def __init__(self, func_ir, _locals, reverse_copies):
        self.func_ir = func_ir
        self.locals = _locals

        self.h5_files = {}
        # dset_var -> (f_id, dset_name)
        self.h5_dsets = {}
        self.h5_dsets_sizes = {}
        self.h5_create_dset_calls = {}
        self.h5_create_group_calls = {}
        self.reverse_copies = reverse_copies
        self.tuple_table = {}

    def handle_possible_h5_read(self, assign, lhs, rhs):
        tp = self._get_h5_type(lhs, rhs)
        if tp is not None:
            dtype_str = str(tp.dtype)
            func_text = "def _h5_read_impl(dset, index):\n"
            # TODO: index arg?
            func_text += "  arr = hpat.pio_api.h5_read_dummy(dset, {}, '{}')\n".format(tp.ndim, dtype_str)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _h5_read_impl = loc_vars['_h5_read_impl']
            f_block = compile_to_numba_ir(
                    _h5_read_impl, {'hpat': hpat}).blocks.popitem()[1]
            replace_arg_nodes(f_block, [rhs.value, rhs.index_var])
            nodes = f_block.body[:-3]  # remove none return
            nodes[-1].target = assign.target
            return nodes
        return None

    def _get_h5_type(self, lhs, rhs):
        tp = self._get_h5_type_locals(lhs)
        if tp is not None:
            return tp
        return guard(self._infer_h5_typ, rhs)

    def _infer_h5_typ(self, rhs):
        # infer the type if it is of the from f['A']['B'][:] 
        # with constant filename
        # TODO: static_getitem has index_var for sure?
        # make sure it's slice, TODO: support non-slice like integer
        require(rhs.op in ('getitem', 'static_getitem'))
        index_var = rhs.index if rhs.op == 'getitem' else rhs.index_var
        index_def = get_definition(self.func_ir, index_var)
        require(isinstance(index_def, ir.Expr) and index_def.op == 'call')
        require(find_callname(self.func_ir, index_def) == ('slice', 'builtins'))
        # collect object names until the call
        val_def = rhs
        obj_name_list = []
        while True:
            val_def = get_definition(self.func_ir, val_def.value)
            require(isinstance(val_def, ir.Expr))
            if val_def.op == 'call':
                return self._get_h5_type_file(val_def, obj_name_list)

            # object_name should be constant str
            require(val_def.op in ('getitem', 'static_getitem'))
            val_index_var = val_def.index if val_def.op == 'getitem' else val_def.index_var
            obj_name = find_const(self.func_ir, val_index_var)
            require(isinstance(obj_name, str))
            obj_name_list.append(obj_name)

    def _get_h5_type_file(self, val_def, obj_name_list):
        require(len(obj_name_list) > 0)
        require(find_callname(self.func_ir, val_def) == ('File', 'h5py'))
        require(len(val_def.args) > 0)
        f_name = find_const(self.func_ir, val_def.args[0])
        require(isinstance(f_name, str))
        obj_name_list.reverse()

        import h5py
        f = h5py.File(f_name, 'r')
        obj = f
        for obj_name in obj_name_list:
            obj = obj[obj_name]
        ndims = len(obj.shape)
        numba_dtype = numba.numpy_support.from_dtype(obj.dtype)
        f.close()
        return types.Array(numba_dtype, ndims, 'C')

    def _get_h5_type_locals(self, varname):
        if varname not in self.reverse_copies or (self.reverse_copies[varname] + ':h5_types') not in self.locals:
            return None
        new_name = self.reverse_copies[varname]
        typ = self.locals.pop(new_name + ":h5_types")
        return typ

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
                    new_body.extend(inst_list)
                elif isinstance(inst, ir.StaticSetItem):
                    inst_list = self._run_static_setitem(inst)
                    new_body.extend(inst_list)
                else:
                    new_body.append(inst)
            self.func_ir.blocks[label].body = new_body
        # iterative remove dead to make sure all extra code (e.g. df vars) is removed
        while remove_dead(self.func_ir.blocks, self.func_ir.arg_names, self.func_ir):
            pass
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "after IO")
        if debug_prints():
            print("h5 files: ", self.h5_files)
            print("h5 dsets: ", self.h5_dsets)

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value

        if isinstance(rhs, ir.Expr):

            if rhs.op == 'call':
                # f = h5py.File(file_name, mode)
                res = self._handle_h5_File_call(assign, assign.target, rhs)
                if res is not None:
                    return res
                # f.close()
                res = self._handle_f_close_call(assign, assign.target, rhs)
                if res is not None:
                    return res

            # f.create_dataset("points", (N,), dtype='f8')
            if rhs.op == 'call' and rhs.func.name in self.h5_create_dset_calls:
                return self._gen_h5create_dset(assign,
                                               self.h5_create_dset_calls[rhs.func.name])

            # f.create_group("subgroup")
            if rhs.op == 'call' and rhs.func.name in self.h5_create_group_calls:
                return self._gen_h5create_group(assign,
                                                self.h5_create_group_calls[rhs.func.name])

            # d = f['dset']
            if rhs.op == 'static_getitem' and rhs.value.name in self.h5_files:
                self.h5_dsets[lhs] = (rhs.value, rhs.index_var)
            if rhs.op == 'getitem' and rhs.value.name in self.h5_files:
                self.h5_dsets[lhs] = (rhs.value, rhs.index)
            # x = f['dset'][:]
            # x = f['dset'][a:b]
            if (rhs.op == 'static_getitem' or rhs.op == 'getitem') and rhs.value.name in self.h5_dsets:
                return self._gen_h5read(assign.target, rhs)

            # f.close or f.create_dataset
            if rhs.op == 'getattr' and rhs.value.name in self.h5_files:
                if rhs.attr == 'create_dataset':
                    self.h5_create_dset_calls[lhs] = rhs.value
                elif rhs.attr == 'create_group':
                    self.h5_create_group_calls[lhs] = rhs.value
                elif rhs.attr in ['keys', 'close']:
                    pass
                else:
                    raise NotImplementedError("file operation not supported")
            if rhs.op == 'build_tuple':
                self.tuple_table[lhs] = rhs.items
        # handle copies lhs = f
        if isinstance(rhs, ir.Var):
            if rhs.name in self.h5_files:
                self.h5_files[lhs] = self.h5_files[rhs.name]
            if rhs.name in self.h5_dsets:
                self.h5_dsets[lhs] = self.h5_dsets[rhs.name]
            if rhs.name in self.h5_dsets_sizes:
                self.h5_dsets_sizes[lhs] = self.h5_dsets_sizes[rhs.name]
        return [assign]

    def _handle_h5_File_call(self, assign, lhs, rhs):
        """
        Handle h5py.File calls like:
          f = h5py.File(file_name, mode)
        """
        if guard(find_callname, self.func_ir, rhs) == ('File', 'h5py'):
            self.h5_files[lhs.name] = rhs.args[0]
            # parallel arg = False for this stage
            loc = lhs.loc
            scope = lhs.scope
            parallel_var = ir.Var(scope, mk_unique_var("$const_parallel"), loc)
            parallel_assign = ir.Assign(ir.Const(0, loc), parallel_var, loc)
            rhs.args.append(parallel_var)
            return [parallel_assign, assign]
        return None

    def _run_static_setitem(self, stmt):
        # generate h5 write code for dset[:] = arr
        if stmt.target.name in self.h5_dsets:
            assert stmt.index == slice(None, None, None)
            f_id, dset_name = self.h5_dsets[stmt.target.name]
            return self._gen_h5write(f_id, stmt.target, stmt.value)
        return [stmt]

    def _gen_h5write(self, f_id, dset_var, arr_var):
        scope = dset_var.scope
        loc = dset_var.loc

        # g_pio_var = Global(hpat.pio_api)
        g_pio_var = ir.Var(scope, mk_unique_var("$pio_g_var"), loc)
        g_pio = ir.Global('pio_api', hpat.pio_api, loc)
        g_pio_assign = ir.Assign(g_pio, g_pio_var, loc)
        # attr call: h5write_attr = getattr(g_pio_var, h5write)
        h5write_attr_call = ir.Expr.getattr(g_pio_var, "h5write", loc)
        attr_var = ir.Var(scope, mk_unique_var("$h5write_attr"), loc)
        attr_assign = ir.Assign(h5write_attr_call, attr_var, loc)
        out = [g_pio_assign, attr_assign]

        # ndims args
        ndims = len(self.h5_dsets_sizes[dset_var.name])
        ndims_var = ir.Var(scope, mk_unique_var("$h5_ndims"), loc)
        ndims_assign = ir.Assign(
            ir.Const(np.int32(ndims), loc), ndims_var, loc)
        # sizes arg
        sizes_var = ir.Var(scope, mk_unique_var("$h5_sizes"), loc)
        tuple_call = ir.Expr.getattr(arr_var, 'shape', loc)
        sizes_assign = ir.Assign(tuple_call, sizes_var, loc)

        zero_var = ir.Var(scope, mk_unique_var("$const_zero"), loc)
        zero_assign = ir.Assign(ir.Const(0, loc), zero_var, loc)
        # starts: assign to zeros
        starts_var = ir.Var(scope, mk_unique_var("$h5_starts"), loc)
        start_tuple_call = ir.Expr.build_tuple([zero_var] * ndims, loc)
        starts_assign = ir.Assign(start_tuple_call, starts_var, loc)
        out += [ndims_assign, zero_assign, starts_assign, sizes_assign]

        # err = h5write(f_id)
        err_var = ir.Var(scope, mk_unique_var("$pio_ret_var"), loc)
        write_call = ir.Expr.call(attr_var, [f_id, dset_var, ndims_var,
                                             starts_var, sizes_var, zero_var, arr_var], (), loc)
        write_assign = ir.Assign(write_call, err_var, loc)
        out.append(write_assign)
        return out

    def _gen_h5read(self, lhs_var, rhs):
        f_id, dset = self.h5_dsets[rhs.value.name]
        dset_type = self._get_dset_type(
            lhs_var.name, self.h5_files[f_id.name], dset.name)
        loc = rhs.value.loc
        scope = rhs.value.scope
        # TODO: generate size, alloc calls
        out = []
        if rhs.op == 'static_getitem':
            start_vars = None
            size_vars = self._gen_h5size(
                f_id, dset, dset_type.ndim, scope, loc, out)
        else:
            start_vars, size_vars = self._get_slice_range(rhs.index, out)
        out.extend(mk_alloc(None, None, lhs_var, tuple(
            size_vars), dset_type.dtype, scope, loc))
        self._gen_h5read_call(f_id, dset, start_vars,
                              size_vars, lhs_var, scope, loc, out)
        return out

    def _gen_h5size(self, f_id, dset, ndims, scope, loc, out):
        # g_pio_var = Global(hpat.pio_api)
        g_pio_var = ir.Var(scope, mk_unique_var("$pio_g_var"), loc)
        g_pio = ir.Global('pio_api', hpat.pio_api, loc)
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

    def _gen_h5read_call(self, f_id, dset, start_vars, size_vars, lhs_var, scope, loc, out):
        # g_pio_var = Global(hpat.pio_api)
        g_pio_var = ir.Var(scope, mk_unique_var("$pio_g_var"), loc)
        g_pio = ir.Global('pio_api', hpat.pio_api, loc)
        g_pio_assign = ir.Assign(g_pio, g_pio_var, loc)
        # attr call: h5size_attr = getattr(g_pio_var, h5read)
        h5size_attr_call = ir.Expr.getattr(g_pio_var, "h5read", loc)
        attr_var = ir.Var(scope, mk_unique_var("$h5read_attr"), loc)
        attr_assign = ir.Assign(h5size_attr_call, attr_var, loc)
        out += [g_pio_assign, attr_assign]

        # ndims args
        ndims = len(size_vars)
        ndims_var = ir.Var(scope, mk_unique_var("$h5_ndims"), loc)
        ndims_assign = ir.Assign(
            ir.Const(np.int32(ndims), loc), ndims_var, loc)
        # sizes arg
        sizes_var = ir.Var(scope, mk_unique_var("$h5_sizes"), loc)
        tuple_call = ir.Expr.build_tuple(size_vars, loc)
        sizes_assign = ir.Assign(tuple_call, sizes_var, loc)

        zero_var = ir.Var(scope, mk_unique_var("$const_zero"), loc)
        zero_assign = ir.Assign(ir.Const(0, loc), zero_var, loc)
        # starts: assign to zeros
        if not start_vars:
            start_vars = [zero_var] * ndims
        starts_var = ir.Var(scope, mk_unique_var("$h5_starts"), loc)
        start_tuple_call = ir.Expr.build_tuple(start_vars, loc)
        starts_assign = ir.Assign(start_tuple_call, starts_var, loc)
        out += [ndims_assign, zero_assign, starts_assign, sizes_assign]

        err_var = ir.Var(scope, mk_unique_var("$h5_err_var"), loc)
        read_call = ir.Expr.call(attr_var, [f_id, dset, ndims_var, starts_var,
                                            sizes_var, zero_var, lhs_var], (), loc)
        out.append(ir.Assign(read_call, err_var, loc))
        return

    def _get_dset_type(self, lhs, file_var, dset_var):
        """get data set type from user-specified locals types or actual file"""
        if lhs in self.local_vars:
            return self.local_vars[lhs]
        if self.reverse_copies[lhs] in self.local_vars:
            return self.local_vars[self.reverse_copies[lhs]]

        # read type from file if file name and dset name are constant values
        # TODO: check for file availability
        file_name = get_constant(self.func_ir, file_var)
        dset_str = get_constant(self.func_ir, dset_var)
        if file_name is not NOT_CONSTANT and dset_str is not NOT_CONSTANT:
            f = h5py.File(file_name, "r")
            ndims = len(f[dset_str].shape)
            numba_dtype = numpy_support.from_dtype(f[dset_str].dtype)
            return types.Array(numba_dtype, ndims, 'C')

        raise RuntimeError("data set type not found")

    def _get_reverse_copies(self, body):
        for inst in body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var):
                self.reverse_copies[inst.value.name] = inst.target.name
        return

    def _handle_f_close_call(self, stmt, lhs_var, rhs):
        func_def = guard(get_definition, self.func_ir, rhs.func)
        assert func_def is not None
        # rare case where function variable is assigned to a new variable
        if isinstance(func_def, ir.Var):
            rhs.func = func_def
            return self._handle_f_close_call(stmt, lhs_var, rhs)
        if (isinstance(func_def, ir.Expr) and func_def.op == 'getattr'
                and func_def.value.name in self.h5_files
                and func_def.attr == 'close'):
            f_id = func_def.value
            scope = lhs_var.scope
            loc = lhs_var.loc
            # g_pio_var = Global(hpat.pio_api)
            g_pio_var = ir.Var(scope, mk_unique_var("$pio_g_var"), loc)
            g_pio = ir.Global('pio_api', hpat.pio_api, loc)
            g_pio_assign = ir.Assign(g_pio, g_pio_var, loc)
            # attr call: h5close_attr = getattr(g_pio_var, h5close)
            h5close_attr_call = ir.Expr.getattr(g_pio_var, "h5close", loc)
            attr_var = ir.Var(scope, mk_unique_var("$h5close_attr"), loc)
            attr_assign = ir.Assign(h5close_attr_call, attr_var, loc)
            # h5close(f_id)
            close_call = ir.Expr.call(attr_var, [f_id], (), loc)
            close_assign = ir.Assign(close_call, lhs_var, loc)
            return [g_pio_assign, attr_assign, close_assign]
        return None

    def _gen_h5create_dset(self, stmt, f_id):
        lhs_var = stmt.target
        scope = lhs_var.scope
        loc = lhs_var.loc
        args = [f_id] + stmt.value.args
        # append the dtype arg (e.g. dtype='f8')
        assert stmt.value.kws and stmt.value.kws[0][0] == 'dtype'
        args.append(stmt.value.kws[0][1])
        # g_pio_var = Global(hpat.pio_api)
        g_pio_var = ir.Var(scope, mk_unique_var("$pio_g_var"), loc)
        g_pio = ir.Global('pio_api', hpat.pio_api, loc)
        g_pio_assign = ir.Assign(g_pio, g_pio_var, loc)
        # attr call: h5create_dset_attr = getattr(g_pio_var, h5create_dset)
        h5create_dset_attr_call = ir.Expr.getattr(
            g_pio_var, "h5create_dset", loc)
        attr_var = ir.Var(scope, mk_unique_var("$h5create_dset_attr"), loc)
        attr_assign = ir.Assign(h5create_dset_attr_call, attr_var, loc)
        # dset_id = h5create_dset(f_id)
        create_dset_call = ir.Expr.call(attr_var, args, (), loc)
        create_dset_assign = ir.Assign(create_dset_call, lhs_var, loc)
        self.h5_dsets[lhs_var.name] = (f_id, args[1])
        self.h5_dsets_sizes[lhs_var.name] = self.tuple_table[args[2].name]
        return [g_pio_assign, attr_assign, create_dset_assign]

    def _gen_h5create_group(self, stmt, f_id):
        lhs_var = stmt.target
        scope = lhs_var.scope
        loc = lhs_var.loc
        args = [f_id] + stmt.value.args
        # g_pio_var = Global(hpat.pio_api)
        g_pio_var = ir.Var(scope, mk_unique_var("$pio_g_var"), loc)
        g_pio = ir.Global('pio_api', hpat.pio_api, loc)
        g_pio_assign = ir.Assign(g_pio, g_pio_var, loc)
        # attr call: h5create_group_attr = getattr(g_pio_var, h5create_group)
        h5create_group_attr_call = ir.Expr.getattr(
            g_pio_var, "h5create_group", loc)
        attr_var = ir.Var(scope, mk_unique_var("$h5create_group_attr"), loc)
        attr_assign = ir.Assign(h5create_group_attr_call, attr_var, loc)
        # group_id = h5create_group(f_id)
        create_group_call = ir.Expr.call(attr_var, args, (), loc)
        create_group_assign = ir.Assign(create_group_call, lhs_var, loc)
        # add to files since group behavior is same as files for many calls
        # FIXME:
        self.h5_files[lhs_var.name] = ir.Var(
            scope, mk_unique_var("$group"), loc)
        return [g_pio_assign, attr_assign, create_group_assign]

    def _get_slice_range(self, index_slice, out):
        scope = index_slice.scope
        loc = index_slice.loc
        # start = s.start
        start_var = ir.Var(scope, mk_unique_var("$pio_range_start"), loc)
        start_attr_call = ir.Expr.getattr(index_slice, "start", loc)
        start_assign = ir.Assign(start_attr_call, start_var, loc)
        # stop = s.stop
        stop_var = ir.Var(scope, mk_unique_var("$pio_range_stop"), loc)
        stop_attr_call = ir.Expr.getattr(index_slice, "stop", loc)
        stop_assign = ir.Assign(stop_attr_call, stop_var, loc)
        # size = stop-start
        size_var = ir.Var(scope, mk_unique_var("$pio_range_size"), loc)
        size_call = ir.Expr.binop('-', stop_var, start_var, loc)
        size_assign = ir.Assign(size_call, size_var, loc)
        out += [start_assign, stop_assign, size_assign]
        return [start_var], [size_var]

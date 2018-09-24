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
        self.reverse_copies = reverse_copies

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

    def _handle_h5_File_call(self, assign, lhs, rhs):
        """
        Handle h5py.File calls like:
          f = h5py.File(file_name, mode)
        """
        # parallel arg = False for this stage
        loc = lhs.loc
        scope = lhs.scope
        parallel_var = ir.Var(scope, mk_unique_var("$const_parallel"), loc)
        parallel_assign = ir.Assign(ir.Const(0, loc), parallel_var, loc)
        rhs.args.append(parallel_var)
        return [parallel_assign, assign]

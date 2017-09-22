from numba import ir, config, ir_utils, types
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc, remove_dels,
                            get_name_var_table, replace_var_names,
                            add_offset_to_labels, get_ir_of_code,
                            compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard, require, get_definition)

from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature
import numpy as np

_pq_type_to_numba = {'DOUBLE': types.Array(types.float64, 1, 'C'),
                    'INT64': types.Array(types.int64, 1, 'C')}

def read_parquet():
    return 0

class ParquetHandler(object):
    """analyze and transform parquet IO calls"""
    def __init__(self, func_ir, typingctx, args, _locals):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals

    def gen_parquet_read(self, file_name):
        import pyarrow.parquet as pq
        fname_def = guard(get_definition, self.func_ir, file_name)
        if isinstance(fname_def, ir.Const):
            assert isinstance(fname_def.value, str)
            file_name_str = fname_def.value
            f = pq.ParquetFile(file_name_str)
            scope = file_name.scope
            loc = file_name.loc
            col_names = f.schema.names
            out_nodes = []
            col_items = []
            for i, cname in enumerate(col_names):
                # get column type from schema
                c_type = _pq_type_to_numba[f.schema.column(i).physical_type]
                # create a variable for column and assign type
                varname = mk_unique_var(cname)
                self.locals[varname] = c_type
                cvar = ir.Var(scope, varname, loc)
                col_items.append((cname, cvar))
                # TODO: handle string constant
                dummy_const = "np.{}(0)".format(c_type.dtype)
                # generate column read call
                read_func_text = ('def f():\n  a = {}\n  {} = read_parquet("{}", {}, a)\n'.
                        format(dummy_const, cname, file_name_str, i))
                loc_vars = {}
                exec(read_func_text, {}, loc_vars)
                read_func = loc_vars['f']
                _, f_block = compile_to_numba_ir(read_func,
                            {'read_parquet': read_parquet, 'np': np}).blocks.popitem()

                # dummy_var = ir.Var(scope, mk_unique_var('dummy_const'), loc)
                # self.locals[dummy_var.name] = c_type.dtype
                # dummy_const_assign = ir.Assign(ir.Const(c_type.dtype(0), loc), dummy_var, loc)
                # out_nodes.append(dummy_const_assign)
                # replace_arg_nodes(f_block, [dummy_var])
                out_nodes += f_block.body[:-3]
                assign = ir.Assign(out_nodes[-1].target, cvar, loc)
                out_nodes.append(assign)

            return col_items, out_nodes
        raise ValueError("Parquet schema not available")

@infer_global(read_parquet)
class ReadParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==3
        array_ty = types.Array(ndim=1, layout='C', dtype=args[2])
        return signature(array_ty, *args)


from numba import cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
from llvmlite import ir as lir
import parquet_cpp
import llvmlite.binding as ll
ll.add_symbol('pq_read', parquet_cpp.read)
@lower_builtin(read_parquet, types.Type, types.Type, types.Type)
def pq_read_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="pq_read")
    return builder.call(fn, [args[0]])

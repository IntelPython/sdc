from numba import ir, config, ir_utils, types
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc, remove_dels,
                            get_name_var_table, replace_var_names,
                            add_offset_to_labels, get_ir_of_code,
                            compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard, require, get_definition)

_pq_type_to_numba = {'DOUBLE': types.float64[:], 'INT64': types.int64[:]}

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
            col_items = []
            for i, cname in enumerate(col_names):
                c_type = _pq_type_to_numba[f.schema.column(i).physical_type]
                varname = mk_unique_var(cname)
                self.locals[varname] = c_type
                cvar = ir.Var(scope, varname, loc)
                col_items.append((cname, cvar))
            # columns names as string constants
            read_args = ', '.join(list(map(lambda a: '"{}"'.format(a), col_names)))
            out_args = ', '.join(col_names)
            read_func_text = 'def f():\n  {} = read_parquet({})\n'.format(out_args, read_args)
            loc_vars = {}
            exec(read_func_text, {}, loc_vars)
            read_func = loc_vars['f']
            _, f_block = compile_to_numba_ir(read_func,
                        {'read_parquet': read_parquet}).blocks.popitem()
            call_nodes = f_block.body[:-3]
            # TODO: add f's signature to locals
            new_assigns = []
            for i, (_, cvar) in enumerate(reversed(col_items)):
                assign = ir.Assign(call_nodes[-i-1].target, cvar, loc)
                new_assigns.append(assign)
            call_nodes += new_assigns
            return col_items, call_nodes
        raise ValueError("Parquet schema not available")

import numba
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
from hpat.str_ext import StringType

_pq_type_to_numba = {'DOUBLE': types.Array(types.float64, 1, 'C'),
                    'INT64': types.Array(types.int64, 1, 'C')}

def read_parquet():
    return 0

def read_parquet_parallel():
    return 0

def get_column_size_parquet():
    return 0

def remove_parquet(rhs, lives, call_list):
    # the call is dead if the read array is dead
    if call_list == [read_parquet] and rhs.args[2].name not in lives:
        return True
    if call_list == [get_column_size_parquet]:
        return True
    return False

numba.ir_utils.remove_call_handlers.append(remove_parquet)

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
            col_names, col_types = parquet_file_schema(file_name_str)
            scope = file_name.scope
            loc = file_name.loc
            out_nodes = []
            col_items = []
            for i, cname in enumerate(col_names):
                # get column type from schema
                c_type = col_types[i]
                # create a variable for column and assign type
                varname = mk_unique_var(cname)
                self.locals[varname] = c_type
                cvar = ir.Var(scope, varname, loc)
                col_items.append((cname, cvar))

                size_func_text = ('def f():\n  col_size = get_column_size_parquet("{}", {})\n'.
                        format(file_name_str, i))
                size_func_text += '  column = np.empty(col_size, dtype=np.{})\n'.format(c_type.dtype)
                size_func_text += '  status = read_parquet("{}", {}, column)\n'.format(file_name_str, i)
                loc_vars = {}
                exec(size_func_text, {}, loc_vars)
                size_func = loc_vars['f']
                _, f_block = compile_to_numba_ir(size_func,
                            {'get_column_size_parquet': get_column_size_parquet,
                            'read_parquet': read_parquet, 'np': np}).blocks.popitem()

                out_nodes += f_block.body[:-3]
                for stmt in out_nodes:
                    if stmt.target.name.startswith("column"):
                        assign = ir.Assign(stmt.target, cvar, loc)
                        break

                out_nodes.append(assign)

            return col_items, out_nodes
        raise ValueError("Parquet schema not available")

def parquet_file_schema(file_name):
    import pyarrow.parquet as pq
    import pyarrow as pa
    col_names = []
    col_types = []

    if file_name.startswith("hdfs://"):
        fs = pa.hdfs.connect()
    else:
        fs = pa.LocalFileSystem()
    with fs.open(file_name) as _file:
        f = pq.ParquetFile(_file)
        col_names = f.schema.names
        num_cols = len(col_names)
        col_types = [_pq_type_to_numba[f.schema.column(i).physical_type]
                                                    for i in range(num_cols)]
    return col_names, col_types

@infer_global(get_column_size_parquet)
class SizeParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==2
        return signature(types.intp, *args)

@infer_global(read_parquet)
class ReadParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==3
        # array_ty = types.Array(ndim=1, layout='C', dtype=args[2])
        return signature(types.int32, *args)

@infer_global(read_parquet_parallel)
class ReadParallelParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args)==5
        # array_ty = types.Array(ndim=1, layout='C', dtype=args[2])
        return signature(types.int32, *args)

from numba import cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
from llvmlite import ir as lir
import llvmlite.binding as ll

from hpat.config import _has_pyarrow
if _has_pyarrow:
    import parquet_cpp
    ll.add_symbol('pq_read', parquet_cpp.read)
    ll.add_symbol('pq_read_parallel', parquet_cpp.read_parallel)
    ll.add_symbol('pq_get_size', parquet_cpp.get_size)

@lower_builtin(get_column_size_parquet, StringType, types.intp)
def pq_size_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64),
                            [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="pq_get_size")
    return builder.call(fn, args)

@lower_builtin(read_parquet, StringType, types.intp, types.Array)
def pq_read_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(64),
                             lir.IntType(8).as_pointer()])
    out_array = make_array(sig.args[2])(context, builder, args[2])

    fn = builder.module.get_or_insert_function(fnty, name="pq_read")
    return builder.call(fn, [args[0], args[1],
            builder.bitcast(out_array.data, lir.IntType(8).as_pointer())])

@lower_builtin(read_parquet_parallel, StringType, types.intp, types.Array, types.intp, types.intp)
def pq_read_parallel_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(64),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(64), lir.IntType(64)])
    out_array = make_array(sig.args[2])(context, builder, args[2])

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_parallel")
    return builder.call(fn, [args[0], args[1],
            builder.bitcast(out_array.data, lir.IntType(8).as_pointer()),
            args[3], args[4]])

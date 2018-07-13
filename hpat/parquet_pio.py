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
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed
import numpy as np
import hpat
from hpat.str_ext import StringType
from hpat.str_arr_ext import StringArray, StringArrayPayloadType, construct_string_array
from hpat.str_arr_ext import string_array_type

_pq_type_to_numba = {'BOOLEAN': types.Array(types.boolean, 1, 'C'),
                     'INT32': types.Array(types.int32, 1, 'C'),
                     'INT64': types.Array(types.int64, 1, 'C'),
                     'FLOAT': types.Array(types.float32, 1, 'C'),
                     'DOUBLE': types.Array(types.float64, 1, 'C'),
                     'BYTE_ARRAY': string_array_type,
                     'INT96': types.Array(types.NPDatetime('ns'), 1, 'C'),
                     }

# boolean, int32, int64, int96, float, double
# XXX arrow converts int96 timestamp to int64
_type_to_pq_dtype_number = {'bool_': 0, 'int32': 1, 'int64': 2,
                            'int96': 3, 'float32': 4, 'float64': 5,
                            'datetime64(ns)': 3}



def read_parquet():
    return 0


def read_parquet_str():
    return 0


def read_parquet_str_parallel():
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
    if call_list == [read_parquet_str]:
        return True
    return False


numba.ir_utils.remove_call_handlers.append(remove_parquet)


class ParquetHandler(object):
    """analyze and transform parquet IO calls"""

    def __init__(self, func_ir, typingctx, args, _locals, _reverse_copies):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals
        self.reverse_copies = _reverse_copies

    def gen_parquet_read(self, file_name, lhs):
        scope = file_name.scope
        loc = file_name.loc

        table_types = None
        # lhs is temporary and will possibly be assigned to user variable
        assert lhs.name.startswith('$')
        if lhs.name in self.reverse_copies and self.reverse_copies[lhs.name] in self.locals:
            table_types = self.locals[self.reverse_copies[lhs.name]]
            self.locals.pop(self.reverse_copies[lhs.name])

        convert_types = {}
        # user-specified type conversion
        if lhs.name in self.reverse_copies and (self.reverse_copies[lhs.name] + ':convert') in self.locals:
            convert_types = self.locals[self.reverse_copies[lhs.name] + ':convert']
            self.locals.pop(self.reverse_copies[lhs.name] + ':convert')

        if table_types is None:
            fname_def = guard(get_definition, self.func_ir, file_name)
            if not isinstance(fname_def, ir.Const) or not isinstance(fname_def.value, str):
                raise ValueError("Parquet schema not available")
            file_name_str = fname_def.value
            col_names, col_types = parquet_file_schema(file_name_str)
        else:
            col_names = list(table_types.keys())
            col_types = list(table_types.values())

        out_nodes = []
        col_items = []
        for i, cname in enumerate(col_names):
            # get column type from schema
            c_type = col_types[i]
            if cname in convert_types:
                c_type = convert_types[cname]

            # create a variable for column and assign type
            varname = mk_unique_var(cname)
            #self.locals[varname] = c_type
            cvar = ir.Var(scope, varname, loc)
            col_items.append((cname, cvar))

            out_nodes += get_column_read_nodes(c_type, cvar, file_name, i)

        return col_items, col_types, out_nodes


def get_column_read_nodes(c_type, cvar, file_name, i):

    loc = cvar.loc

    func_text = ('def f(fname):\n  col_size = get_column_size_parquet(fname, {})\n'.
                 format(i))
    # generate strings differently
    if c_type == string_array_type:
        # pass size for easier allocation and distributed analysis
        func_text += '  column = read_parquet_str(fname, {}, col_size)\n'.format(
            i)
    else:
        el_type = get_element_type(c_type.dtype)
        if el_type == 'datetime64(ns)':
            func_text += '  column_tmp = np.empty(col_size, dtype=np.int64)\n'
            func_text += '  column = hpat.hiframes_api.ts_series_to_arr_typ(column_tmp)\n'
        else:
            func_text += '  column = np.empty(col_size, dtype=np.{})\n'.format(
                el_type)
        func_text += '  status = read_parquet(fname, {}, column, np.int32({}))\n'.format(
            i, _type_to_pq_dtype_number[el_type])
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    size_func = loc_vars['f']
    _, f_block = compile_to_numba_ir(size_func,
                                     {'get_column_size_parquet': get_column_size_parquet,
                                      'read_parquet': read_parquet,
                                      'read_parquet_str': read_parquet_str,
                                      'np': np,
                                      'hpat': hpat,
                                      'StringArray': StringArray}).blocks.popitem()

    replace_arg_nodes(f_block, [file_name])
    out_nodes = f_block.body[:-3]
    for stmt in reversed(out_nodes):
        if stmt.target.name.startswith("column"):
            assign = ir.Assign(stmt.target, cvar, loc)
            break

    out_nodes.append(assign)
    return out_nodes


def get_element_type(dtype):
    out = repr(dtype)
    if out == 'bool':
        out = 'bool_'
    return out

def _get_numba_typ_from_pq_typ(pq_typ):
    if pq_typ not in _pq_type_to_numba:
        raise ValueError("parquet data type {} not supported yet".format(
                                                                       pq_typ))
    return _pq_type_to_numba[pq_typ]

def parquet_file_schema(file_name):
    import pyarrow.parquet as pq
    col_names = []
    col_types = []

    pq_dataset = pq.ParquetDataset(file_name)
    col_names = pq_dataset.schema.names
    num_cols = len(col_names)
    col_types = [_get_numba_typ_from_pq_typ(
                 pq_dataset.schema.column(i).physical_type)
                 for i in range(num_cols)]
    # TODO: close file?
    return col_names, col_types


@infer_global(get_column_size_parquet)
class SizeParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.intp, *args)


@infer_global(read_parquet)
class ReadParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        if args[2] == types.intp:  # string read call, returns string array
            return signature(string_array_type, *args)
        # array_ty = types.Array(ndim=1, layout='C', dtype=args[2])
        return signature(types.int64, *args)


@infer_global(read_parquet_str)
class ReadParquetStrInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        return signature(string_array_type, *args)


@infer_global(read_parquet_str_parallel)
class ReadParquetStrParallelInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        return signature(string_array_type, *args)


@infer_global(read_parquet_parallel)
class ReadParallelParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 6
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
    ll.add_symbol('pq_read_string', parquet_cpp.read_string)
    ll.add_symbol('pq_read_string_parallel', parquet_cpp.read_string_parallel)


@lower_builtin(get_column_size_parquet, StringType, types.intp)
def pq_size_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64),
                            [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="pq_get_size")
    return builder.call(fn, args)


@lower_builtin(read_parquet, StringType, types.intp, types.Array, types.int32)
def pq_read_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64),
                            [lir.IntType(8).as_pointer(), lir.IntType(64),
                             lir.IntType(8).as_pointer()], lir.IntType(32))
    out_array = make_array(sig.args[2])(context, builder, args[2])

    fn = builder.module.get_or_insert_function(fnty, name="pq_read")
    return builder.call(fn, [args[0], args[1],
                             builder.bitcast(
                                 out_array.data, lir.IntType(8).as_pointer()),
                             args[3]])


@lower_builtin(read_parquet_parallel, StringType, types.intp, types.Array, types.int32, types.intp, types.intp)
def pq_read_parallel_lower(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(64),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(32), lir.IntType(64), lir.IntType(64)])
    out_array = make_array(sig.args[2])(context, builder, args[2])

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_parallel")
    return builder.call(fn, [args[0], args[1],
                             builder.bitcast(
                                 out_array.data, lir.IntType(8).as_pointer()),
                             args[3], args[4], args[5]])

# read strings


@lower_builtin(read_parquet_str, StringType, types.intp, types.intp)
def pq_read_string_lower(context, builder, sig, args):
    typ = sig.return_type
    dtype = StringArrayPayloadType()
    meminfo, data_pointer = construct_string_array(context, builder)
    string_array = cgutils.create_struct_proxy(dtype)(context, builder)
    string_array.size = args[2]
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(64),
                             lir.IntType(8).as_pointer().as_pointer(),
                             lir.IntType(8).as_pointer().as_pointer()])

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_string")
    res = builder.call(fn, [args[0], args[1],
                            string_array._get_ptr_by_name('offsets'),
                            string_array._get_ptr_by_name('data')])
    builder.store(string_array._getvalue(),
                  data_pointer)
    inst_struct = context.make_helper(builder, typ)
    inst_struct.meminfo = meminfo
    ret = inst_struct._getvalue()
    return impl_ret_new_ref(context, builder, typ, ret)


@lower_builtin(read_parquet_str_parallel, StringType, types.intp, types.intp, types.intp)
def pq_read_string_parallel_lower(context, builder, sig, args):
    typ = sig.return_type
    dtype = StringArrayPayloadType()
    meminfo, data_pointer = construct_string_array(context, builder)
    string_array = cgutils.create_struct_proxy(dtype)(context, builder)
    string_array.size = args[3]
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(64),
                             lir.IntType(8).as_pointer().as_pointer(),
                             lir.IntType(8).as_pointer().as_pointer(), lir.IntType(64), lir.IntType(64)])

    fn = builder.module.get_or_insert_function(
        fnty, name="pq_read_string_parallel")
    res = builder.call(fn, [args[0], args[1],
                            string_array._get_ptr_by_name('offsets'),
                            string_array._get_ptr_by_name('data'), args[2],
                            args[3]])

    builder.store(string_array._getvalue(),
                  data_pointer)
    inst_struct = context.make_helper(builder, typ)
    inst_struct.meminfo = meminfo
    ret = inst_struct._getvalue()
    return impl_ret_new_ref(context, builder, typ, ret)

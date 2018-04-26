import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (mk_unique_var, replace_vars_inner, find_topo_order,
                            dprint_func_ir, remove_dead, mk_alloc, remove_dels,
                            get_name_var_table, replace_var_names,
                            add_offset_to_labels, get_ir_of_code,
                            compile_to_numba_ir, replace_arg_nodes,
                            find_callname, guard, require, get_definition,
                            build_definitions, replace_vars_stmt, replace_vars_inner)

from llvmlite import ir as lir
import llvmlite.binding as ll

import numpy as np
import hpat
from hpat.utils import get_constant, NOT_CONSTANT
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type

# TODO: implement in regular python
def read_xenon():
    return

def _handle_read(assign, lhs, rhs, func_ir):
    if not hpat.config._has_xenon:
        raise ValueError("Xenon support not available")

    # TODO: init only once
    import hxe_ext
    ll.add_symbol('get_column_size_xenon', hxe_ext.get_column_size_xenon)
    ll.add_symbol('c_read_xenon', hxe_ext.read_xenon_col)

    if len(rhs.args) != 1:
        raise ValueError("read_xenon expects one argument but received {}".format(len(rhs.args)))

    dset_name_var = rhs.args[0]
    dset_name = get_constant(func_ir, dset_name_var)
    if dset_name is NOT_CONSTANT:
        raise ValueError("Xenon dataset should be a constant string")

    col_names, col_types = get_dset_schema(dset_name)

    xe_typs = [str(get_xe_typ_enum(c_type)) for c_type in col_types]
    xe_typs_str = "np.array([" + ",".join(xe_typs) + "])"

    scope = dset_name_var.scope
    loc = dset_name_var.loc

    out_nodes = []
    col_items = []
    for i, cname in enumerate(col_names):
        # get column type from schema
        c_type = col_types[i]

        # create a variable for column and assign type
        varname = mk_unique_var(cname)
        cvar = ir.Var(scope, varname, loc)
        col_items.append((cname, cvar))

        out_nodes += get_column_read_nodes(c_type, cvar, dset_name_var, i, xe_typs_str)

    return col_items, out_nodes

_xe_type_to_numba = {'BOOL': types.Array(types.boolean, 1, 'C'),
                     'I8': types.Array(types.char, 1, 'C'),
                     'I16': types.Array(types.int16, 1, 'C'),
                     'I32': types.Array(types.int32, 1, 'C'),
                     'I64': types.Array(types.int64, 1, 'C'),
                     'FLOAT': types.Array(types.float32, 1, 'C'),
                     'DOUBLE': types.Array(types.float64, 1, 'C'),
                     'CHAR': string_array_type,
                     # TODO: handle decimal and blob types
                     }

_type_to_xe_dtype_number = {'int8': 0, 'int16': 1, 'int32': 2, 'int64': 3,
                            'float32': 4, 'float64': 5, 'DECIMAL': 6,
                             'bool_': 7, 'string': 8, 'BLOB': 9}

def get_xe_typ_enum(c_type):
    if c_type == string_array_type:
        return _type_to_xe_dtype_number['string']
    assert isinstance(c_type, types.Array)
    return _type_to_xe_dtype_number[get_element_type(c_type.dtype)]

def get_dset_schema(dset_name):
    import hxe_ext
    schema = hxe_ext.get_schema(dset_name)
    print("schema", schema)
    # example: {first:CHAR,last:CHAR,age:I32,street:CHAR,state:CHAR,zip:I32}
    # remove braces
    assert schema[0] == '{' and schema[-1] == '}'
    schema = schema[1:-1]
    col_names = []
    col_types = []
    for col_name_typ in schema.split(','):
        assert col_name_typ.count(":") == 1, "invalid Xenon schema"
        cname, xe_typ = col_name_typ.split(':')
        assert xe_typ in _xe_type_to_numba
        np_typ = _xe_type_to_numba[xe_typ]
        col_names.append(cname)
        col_types.append(np_typ)

    return col_names, col_types


def get_column_read_nodes(c_type, cvar, dset_name, i, xe_typs_str):

    loc = cvar.loc

    func_text = ('def f(dset_name):\n  col_size = get_column_size_xenon(dset_name, {})\n'.
                 format(i))
    # func_text += '  print(col_size)\n'
    func_text += '  schema_arr = {}\n'.format(xe_typs_str)
    # generate strings differently since upfront allocation is not possible
    if c_type == string_array_type:
        # pass size for easier allocation and distributed analysis
        func_text += '  column = 3#read_xenon_str(dset_name, {}, col_size)\n'.format(
            i)
    else:
        el_type = get_element_type(c_type.dtype)
        func_text += '  column = np.empty(col_size, dtype=np.{})\n'.format(
            el_type)
        func_text += '  status = read_xenon_col(dset_name, {}, column.ctypes, schema_arr.ctypes)\n'.format(i)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    size_func = loc_vars['f']
    _, f_block = compile_to_numba_ir(size_func,
                                     {'get_column_size_xenon': get_column_size_xenon,
                                      'read_xenon_col': read_xenon_col,
                                      #'read_xenon_str': read_xenon_str,
                                      'np': np,
                                      }).blocks.popitem()

    replace_arg_nodes(f_block, [dset_name])
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

get_column_size_xenon = types.ExternalFunction("get_column_size_xenon", types.int64(string_type, types.intp))
read_xenon_col = types.ExternalFunction("c_read_xenon", types.void(string_type, types.intp, types.voidptr, types.CPointer(types.int64)))

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
from numba.targets.imputils import impl_ret_new_ref
from numba.extending import lower_builtin, overload, intrinsic, register_model, models
from numba.typing import signature
from numba import cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array

import numpy as np
import hpat
from hpat.utils import get_constant, NOT_CONSTANT
from hpat.str_ext import string_type
from hpat.str_arr_ext import StringArray, StringArrayPayloadType, construct_string_array
from hpat.str_arr_ext import string_array_type

def remove_xenon(rhs, lives, call_list):
    # the call is dead if the read array is dead
    if call_list == [read_xenon_col] and rhs.args[3].name not in lives:
        return True
    if call_list == [get_column_size_xenon]:
        return True
    if call_list == [read_xenon_str]:
        return True
    return False


numba.ir_utils.remove_call_handlers.append(remove_xenon)

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
    ll.add_symbol('c_read_xenon_col_str', hxe_ext.read_xenon_col_str)
    ll.add_symbol('c_xe_connect', hxe_ext.c_xe_connect)
    ll.add_symbol('c_xe_open', hxe_ext.c_xe_open)
    ll.add_symbol('c_xe_close', hxe_ext.c_xe_close)

    if len(rhs.args) != 1:
        raise ValueError("read_xenon expects one argument but received {}".format(len(rhs.args)))

    dset_name_var = rhs.args[0]
    dset_name = get_constant(func_ir, dset_name_var)
    if dset_name is NOT_CONSTANT:
        raise ValueError("Xenon dataset should be a constant string")

    if dset_name.count("/") != 1:
        raise ValueError("invalid Xenon address {}".format(dset_name))
    address, dset_name = dset_name.split("/")
    col_names, col_types = get_dset_schema(address, dset_name)

    out_nodes, xe_connect_var, xe_dset_var = gen_init_xenon(address, dset_name)

    # generate array of schema types
    xe_typs = [str(get_xe_typ_enum(c_type)) for c_type in col_types]
    xe_typs_str = "np.array([" + ",".join(xe_typs) + "])"
    func_text = 'def f():\n  schema_arr = {}\n'.format(xe_typs_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    schm_func = loc_vars['f']
    f_block = compile_to_numba_ir(schm_func, {'np': np,}).blocks.popitem()[1]
    out_nodes += f_block.body[:-3]
    schema_arr_var = out_nodes[-1].target

    scope = dset_name_var.scope
    loc = dset_name_var.loc

    col_items = []
    for i, cname in enumerate(col_names):
        # get column type from schema
        c_type = col_types[i]

        # create a variable for column and assign type
        varname = mk_unique_var(cname)
        cvar = ir.Var(scope, varname, loc)
        col_items.append((cname, cvar))

        out_nodes += get_column_read_nodes(c_type, cvar, xe_connect_var, xe_dset_var, i, schema_arr_var)

    out_nodes += gen_close_xenon(xe_connect_var, xe_dset_var);

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

def get_dset_schema(address, dset_name):
    import hxe_ext
    schema = hxe_ext.get_schema(address, dset_name)
    # print("schema", schema)
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


def get_column_read_nodes(c_type, cvar, xe_connect_var, xe_dset_var, i, schema_arr_var):

    loc = cvar.loc

    func_text = ('def f(xe_connect_var, xe_dset_var, schema_arr):\n  col_size = get_column_size_xenon(xe_connect_var, xe_dset_var, {})\n'.
                 format(i))
    # func_text += '  print(col_size)\n'
    # generate strings differently since upfront allocation is not possible
    if c_type == string_array_type:
        # pass size for easier allocation and distributed analysis
        func_text += '  column = read_xenon_str(xe_connect_var, xe_dset_var, {}, col_size, schema_arr.ctypes)\n'.format(
            i)
    else:
        el_type = get_element_type(c_type.dtype)
        func_text += '  column = np.empty(col_size, dtype=np.{})\n'.format(
            el_type)
        func_text += '  status = read_xenon_col(xe_connect_var, xe_dset_var, {}, column, schema_arr.ctypes)\n'.format(i)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    size_func = loc_vars['f']
    _, f_block = compile_to_numba_ir(size_func,
                                     {'get_column_size_xenon': get_column_size_xenon,
                                      'read_xenon_col': read_xenon_col,
                                      'read_xenon_str': read_xenon_str,
                                      'np': np,
                                      }).blocks.popitem()

    replace_arg_nodes(f_block, [xe_connect_var, xe_dset_var, schema_arr_var])
    out_nodes = f_block.body[:-3]
    for stmt in reversed(out_nodes):
        if stmt.target.name.startswith("column"):
            assign = ir.Assign(stmt.target, cvar, loc)
            break

    out_nodes.append(assign)
    return out_nodes

def gen_init_xenon(address, dset_name):
    # TODO: support non-constant address/dset_name
    func_text = ('def f():\n  connect_t = c_xe_connect("{}")\n'.format(address))
    func_text += '  dset_t = c_xe_open(connect_t, "{}")\n'.format(dset_name)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    init_func = loc_vars['f']
    f_block = compile_to_numba_ir(init_func,
                                         {'c_xe_connect': c_xe_connect,
                                         'c_xe_open': c_xe_open}).blocks.popitem()[1]

    connect_var = None
    dset_t_var = None

    out_nodes = f_block.body[:-3]
    for stmt in reversed(out_nodes):
        if stmt.target.name.startswith("connect_t"):
             connect_var = stmt.target
        if stmt.target.name.startswith("dset_t"):
             dset_t_var = stmt.target

    assert connect_var is not None and dset_t_var is not None
    return out_nodes, connect_var, dset_t_var

def gen_close_xenon(connect_var, dset_t_var):
    #
    def close_func(connect_var, dset_t_var):
        s = c_xe_close(connect_var, dset_t_var)

    f_block = compile_to_numba_ir(close_func,
                                         {'c_xe_close': c_xe_close}).blocks.popitem()[1]

    replace_arg_nodes(f_block, [connect_var, dset_t_var])
    out_nodes = f_block.body[:-3]
    return out_nodes


def get_element_type(dtype):
    out = repr(dtype)
    if out == 'bool':
        out = 'bool_'
    return out

class XeConnectType(types.Opaque):
    def __init__(self):
        super(XeConnectType, self).__init__(name='XeConnectType')

xe_connect_type = XeConnectType()

register_model(XeConnectType)(models.OpaqueModel)

class XeDSetType(types.Opaque):
    def __init__(self):
        super(XeDSetType, self).__init__(name='XeDSetType')

xe_dset_type = XeDSetType()

register_model(XeDSetType)(models.OpaqueModel)

get_column_size_xenon = types.ExternalFunction("get_column_size_xenon", types.int64(xe_connect_type, xe_dset_type, types.intp))
# read_xenon_col = types.ExternalFunction("c_read_xenon", types.void(string_type, types.intp, types.voidptr, types.CPointer(types.int64)))
c_xe_connect = types.ExternalFunction("c_xe_connect", xe_connect_type(string_type))
c_xe_open = types.ExternalFunction("c_xe_open", xe_dset_type(xe_connect_type, string_type))
c_xe_close = types.ExternalFunction("c_xe_close", types.void(xe_connect_type, xe_dset_type))


# TODO: fix liveness/alias in Numba to be able to use arr.ctypes directly
@intrinsic
def read_xenon_col(typingctx, connect_tp, dset_tp, col_id_tp, column_tp, schema_arr_tp):
    def codegen(context, builder, sig, args):
        arr_info = context.make_array(column_tp)(context, builder, value=args[3])
        ctinfo = context.make_helper(builder, schema_arr_tp, value=args[4])
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(64),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(64).as_pointer()])

        fn = builder.module.get_or_insert_function(fnty, name="c_read_xenon")
        res = builder.call(fn, [args[0], args[1], args[2],
                                builder.bitcast(arr_info.data, lir.IntType(8).as_pointer()), ctinfo.data])
        return context.get_dummy_value()
    return signature(types.none, connect_tp, dset_tp, col_id_tp, column_tp, schema_arr_tp), codegen

@intrinsic
def read_xenon_str(typingctx, connect_tp, dset_tp, col_id_tp, size_tp, schema_arr_tp):
    def codegen(context, builder, sig, args):
        typ = sig.return_type
        dtype = StringArrayPayloadType()
        meminfo, data_pointer = construct_string_array(context, builder)
        string_array = cgutils.create_struct_proxy(dtype)(context, builder)
        string_array.size = args[3]

        ctinfo = context.make_helper(builder, schema_arr_tp, value=args[4])
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(64),
                                 lir.IntType(8).as_pointer().as_pointer(),
                                 lir.IntType(8).as_pointer().as_pointer(),
                                 lir.IntType(64).as_pointer()])

        fn = builder.module.get_or_insert_function(fnty, name="c_read_xenon_col_str")
        res = builder.call(fn, [args[0], args[1], args[2],
                                string_array._get_ptr_by_name('offsets'),
                                string_array._get_ptr_by_name('data'), ctinfo.data])
        builder.store(string_array._getvalue(),
                      data_pointer)
        inst_struct = context.make_helper(builder, typ)
        inst_struct.meminfo = meminfo
        ret = inst_struct._getvalue()
        return impl_ret_new_ref(context, builder, typ, ret)
    return signature(string_array_type, connect_tp, dset_tp, col_id_tp, size_tp, schema_arr_tp), codegen

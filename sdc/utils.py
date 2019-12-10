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


from . import hstr_ext
import llvmlite.binding as ll
from llvmlite import ir as lir
from collections import namedtuple
import operator
import numba
from numba import ir_utils, ir, types, cgutils
from numba.ir_utils import (guard, get_definition, find_callname, require,
                            add_offset_to_labels, find_topo_order, find_const)
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import lower_builtin
from numba.extending import overload, intrinsic, lower_cast
import numpy as np
import sdc
from sdc.str_ext import string_type, list_string_array_type
from sdc.str_arr_ext import string_array_type, num_total_chars, pre_alloc_string_array
from enum import Enum
import types as pytypes
from numba.extending import overload, overload_method, overload_attribute


# int values for types to pass to C code
# XXX: _hpat_common.h
class CTypeEnum(Enum):
    Int8 = 0
    UInt8 = 1
    Int32 = 2
    UInt32 = 3
    Int64 = 4
    UInt64 = 7
    Float32 = 5
    Float64 = 6
    Int16 = 8
    UInt16 = 9


_numba_to_c_type_map = {
    types.int8: CTypeEnum.Int8.value,
    types.uint8: CTypeEnum.UInt8.value,
    types.int32: CTypeEnum.Int32.value,
    types.uint32: CTypeEnum.UInt32.value,
    types.int64: CTypeEnum.Int64.value,
    types.uint64: CTypeEnum.UInt64.value,
    types.float32: CTypeEnum.Float32.value,
    types.float64: CTypeEnum.Float64.value,
    types.NPDatetime('ns'): CTypeEnum.UInt64.value,
    # XXX: Numpy's bool array uses a byte for each value but regular booleans
    # are not bytes
    # TODO: handle boolean scalars properly
    types.bool_: CTypeEnum.UInt8.value,
    types.int16: CTypeEnum.Int16.value,
    types.uint16: CTypeEnum.UInt16.value,
}


# silence Numba error messages for now
# TODO: customize through @sdc.jit
numba.errors.error_extras = {
    'unsupported_error': '',
    'typing': '',
    'reportable': '',
    'interpreter': '',
    'constant_inference': ''}

# sentinel value representing non-constant values


class NotConstant:
    pass


NOT_CONSTANT = NotConstant()

ReplaceFunc = namedtuple("ReplaceFunc",
                         ["func", "arg_types", "args", "glbls", "pre_nodes"])

np_alloc_callnames = ('empty', 'zeros', 'ones', 'full')


def unliteral_all(args):
    return tuple(types.unliteral(a) for a in args)

# TODO: move to Numba


class BooleanLiteral(types.Literal, types.Boolean):

    def can_convert_to(self, typingctx, other):
        # similar to IntegerLiteral
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, types.Conversion.promote)


types.Literal.ctor_map[bool] = BooleanLiteral

numba.datamodel.register_default(
    BooleanLiteral)(numba.extending.models.BooleanModel)


@lower_cast(BooleanLiteral, types.Boolean)
def literal_bool_cast(context, builder, fromty, toty, val):
    lit = context.get_constant_generic(
        builder,
        fromty.literal_type,
        fromty.literal_value,
    )
    return context.cast(builder, lit, fromty.literal_type, toty)


def get_constant(func_ir, var, default=NOT_CONSTANT):
    def_node = guard(get_definition, func_ir, var)
    if def_node is None:
        return default
    if isinstance(def_node, ir.Const):
        return def_node.value
    # call recursively if variable assignment
    if isinstance(def_node, ir.Var):
        return get_constant(func_ir, def_node, default)
    return default


def inline_new_blocks(func_ir, block, i, callee_blocks, work_list=None):
    # adopted from inline_closure_call
    scope = block.scope
    instr = block.body[i]

    # 1. relabel callee_ir by adding an offset
    callee_blocks = add_offset_to_labels(callee_blocks, ir_utils._max_label + 1)
    callee_blocks = ir_utils.simplify_CFG(callee_blocks)
    max_label = max(callee_blocks.keys())
    #    reset globals in ir_utils before we use it
    ir_utils._max_label = max_label
    topo_order = find_topo_order(callee_blocks)

    # 5. split caller blocks into two
    new_blocks = []
    new_block = ir.Block(scope, block.loc)
    new_block.body = block.body[i + 1:]
    new_label = ir_utils.next_label()
    func_ir.blocks[new_label] = new_block
    new_blocks.append((new_label, new_block))
    block.body = block.body[:i]
    min_label = topo_order[0]
    block.body.append(ir.Jump(min_label, instr.loc))

    # 6. replace Return with assignment to LHS
    numba.inline_closurecall._replace_returns(callee_blocks, instr.target, new_label)
    #    remove the old definition of instr.target too
    if (instr.target.name in func_ir._definitions):
        func_ir._definitions[instr.target.name] = []

    # 7. insert all new blocks, and add back definitions
    for label in topo_order:
        # block scope must point to parent's
        block = callee_blocks[label]
        block.scope = scope
        numba.inline_closurecall._add_definitions(func_ir, block)
        func_ir.blocks[label] = block
        new_blocks.append((label, block))

    if work_list is not None:
        for block in new_blocks:
            work_list.append(block)
    return callee_blocks


def is_alloc_call(func_var, call_table):
    """
    return true if func_var represents an array creation call
    """
    assert func_var in call_table
    call_list = call_table[func_var]
    return ((len(call_list) == 2 and call_list[1] == np
             and call_list[0] in ['empty', 'zeros', 'ones', 'full'])
            or call_list == [numba.unsafe.ndarray.empty_inferred])


def is_alloc_callname(func_name, mod_name):
    """
    return true if function represents an array creation call
    """
    return isinstance(mod_name, str) and ((mod_name == 'numpy'
                                           and func_name in np_alloc_callnames)
                                          or (func_name == 'empty_inferred'
                                              and mod_name in ('numba.extending', 'numba.unsafe.ndarray'))
                                          or (func_name == 'pre_alloc_string_array'
                                              and mod_name == 'sdc.str_arr_ext')
                                          or (func_name in ('alloc_str_list', 'alloc_list_list_str')
                                              and mod_name == 'sdc.str_ext'))


def find_build_tuple(func_ir, var):
    """Check if a variable is constructed via build_tuple
    and return the sequence or raise GuardException otherwise.
    """
    # variable or variable name
    require(isinstance(var, (ir.Var, str)))
    var_def = get_definition(func_ir, var)
    require(isinstance(var_def, ir.Expr))
    require(var_def.op == 'build_tuple')
    return var_def.items


def cprint(*s):
    print(*s)


@infer_global(cprint)
class CprintInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *unliteral_all(args))


typ_to_format = {
    types.int32: 'd',
    types.uint32: 'u',
    types.int64: 'lld',
    types.uint64: 'llu',
    types.float32: 'f',
    types.float64: 'lf',
}

ll.add_symbol('print_str', hstr_ext.print_str)
ll.add_symbol('print_char', hstr_ext.print_char)


@lower_builtin(cprint, types.VarArg(types.Any))
def cprint_lower(context, builder, sig, args):
    from sdc.str_ext import string_type, char_type

    for i, val in enumerate(args):
        typ = sig.args[i]
        if typ == string_type:
            fnty = lir.FunctionType(
                lir.VoidType(), [lir.IntType(8).as_pointer()])
            fn = builder.module.get_or_insert_function(fnty, name="print_str")
            builder.call(fn, [val])
            cgutils.printf(builder, " ")
            continue
        if typ == char_type:
            fnty = lir.FunctionType(
                lir.VoidType(), [lir.IntType(8)])
            fn = builder.module.get_or_insert_function(fnty, name="print_char")
            builder.call(fn, [val])
            cgutils.printf(builder, " ")
            continue
        if isinstance(typ, types.ArrayCTypes):
            cgutils.printf(builder, "%p ", val)
            continue
        format_str = typ_to_format[typ]
        cgutils.printf(builder, "%{} ".format(format_str), val)
    cgutils.printf(builder, "\n")
    return context.get_dummy_value()


def print_dist(d):
    from sdc.distributed_analysis import Distribution
    if d == Distribution.REP:
        return "REP"
    if d == Distribution.OneD:
        return "1D_Block"
    if d == Distribution.OneD_Var:
        return "1D_Block_Var"
    if d == Distribution.Thread:
        return "Multi-thread"
    if d == Distribution.TwoD:
        return "2D_Block"


def distribution_report():
    import sdc.distributed
    if sdc.distributed.dist_analysis is None:
        return
    print("Array distributions:")
    for arr, dist in sdc.distributed.dist_analysis.array_dists.items():
        print("   {0:20} {1}".format(arr, print_dist(dist)))
    print("\nParfor distributions:")
    for p, dist in sdc.distributed.dist_analysis.parfor_dists.items():
        print("   {0:<20} {1}".format(p, print_dist(dist)))


def is_whole_slice(typemap, func_ir, var, accept_stride=False):
    """ return True if var can be determined to be a whole slice """
    require(typemap[var.name] == types.slice2_type
            or (accept_stride and typemap[var.name] == types.slice3_type))
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == 'call')
    assert (len(call_expr.args) == 2
            or (accept_stride and len(call_expr.args) == 3))
    assert find_callname(func_ir, call_expr) == ('slice', 'builtins')
    arg0_def = get_definition(func_ir, call_expr.args[0])
    arg1_def = get_definition(func_ir, call_expr.args[1])
    require(isinstance(arg0_def, ir.Const) and arg0_def.value is None)
    require(isinstance(arg1_def, ir.Const) and arg1_def.value is None)
    return True


def is_const_slice(typemap, func_ir, var, accept_stride=False):
    """ return True if var can be determined to be a constant size slice """
    require(typemap[var.name] == types.slice2_type
            or (accept_stride and typemap[var.name] == types.slice3_type))
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == 'call')
    assert (len(call_expr.args) == 2
            or (accept_stride and len(call_expr.args) == 3))
    assert find_callname(func_ir, call_expr) == ('slice', 'builtins')
    arg0_def = get_definition(func_ir, call_expr.args[0])
    require(isinstance(arg0_def, ir.Const) and arg0_def.value is None)
    size_const = find_const(func_ir, call_expr.args[1])
    require(isinstance(size_const, int))
    return True


def get_slice_step(typemap, func_ir, var):
    require(typemap[var.name] == types.slice3_type)
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == 'call')
    assert len(call_expr.args) == 3
    return call_expr.args[2]


def is_array(typemap, varname):
    return (varname in typemap
            and (is_np_array(typemap, varname)
                 or typemap[varname] in (string_array_type, list_string_array_type,
                                         sdc.hiframes.split_impl.string_array_split_view_type)
                 or isinstance(typemap[varname], sdc.hiframes.pd_series_ext.SeriesType)))


def is_np_array(typemap, varname):
    return (varname in typemap
            and isinstance(typemap[varname], types.Array))


def is_array_container(typemap, varname):
    return (varname in typemap
            and isinstance(typemap[varname], (types.List, types.Set))
            and (isinstance(typemap[varname].dtype, types.Array)
                 or typemap[varname].dtype == string_array_type
                 or isinstance(typemap[varname].dtype,
                               sdc.hiframes.pd_series_ext.SeriesType)))


# converts an iterable to array, similar to np.array, but can support
# other things like StringArray
# TODO: other types like datetime?
def to_array(A):
    return np.array(A)


@overload(to_array)
def to_array_overload(A):
    # try regular np.array and return it if it works
    def to_array_impl(A):
        return np.array(A)
    try:
        numba.njit(to_array_impl).get_call_template((A,), {})
        return to_array_impl
    except BaseException:
        pass  # should be handled elsewhere (e.g. Set)


def empty_like_type(n, arr):
    return np.empty(n, arr.dtype)


@overload(empty_like_type)
def empty_like_type_overload(n, arr):
    if isinstance(arr, sdc.hiframes.pd_categorical_ext.CategoricalArray):
        from sdc.hiframes.pd_categorical_ext import fix_cat_array_type
        return lambda n, arr: fix_cat_array_type(np.empty(n, arr.dtype))
    if isinstance(arr, types.Array):
        return lambda n, arr: np.empty(n, arr.dtype)
    if isinstance(arr, types.List) and arr.dtype == string_type:
        def empty_like_type_str_list(n, arr):
            return [''] * n
        return empty_like_type_str_list

    # string array buffer for join
    assert arr == string_array_type

    def empty_like_type_str_arr(n, arr):
        # average character heuristic
        avg_chars = 20  # heuristic
        if len(arr) != 0:
            avg_chars = num_total_chars(arr) // len(arr)
        return pre_alloc_string_array(n, n * avg_chars)

    return empty_like_type_str_arr


def alloc_arr_tup(n, arr_tup, init_vals=()):
    arrs = []
    for in_arr in arr_tup:
        arrs.append(np.empty(n, in_arr.dtype))
    return tuple(arrs)


@overload(alloc_arr_tup)
def alloc_arr_tup_overload(n, data, init_vals=()):
    count = data.count

    allocs = ','.join(["empty_like_type(n, data[{}])".format(i)
                       for i in range(count)])

    if init_vals is not ():
        # TODO check for numeric value
        allocs = ','.join(["np.full(n, init_vals[{}], data[{}].dtype)".format(i, i)
                           for i in range(count)])

    func_text = "def f(n, data, init_vals=()):\n"
    func_text += "  return ({}{})\n".format(allocs,
                                            "," if count == 1 else "")  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {'empty_like_type': empty_like_type, 'np': np}, loc_vars)
    alloc_impl = loc_vars['f']
    return alloc_impl


@intrinsic
def get_ctypes_ptr(typingctx, ctypes_typ=None):
    assert isinstance(ctypes_typ, types.ArrayCTypes)

    def codegen(context, builder, sig, args):
        in_carr, = args
        ctinfo = context.make_helper(builder, sig.args[0], in_carr)
        return ctinfo.data

    return types.voidptr(ctypes_typ), codegen


def remove_return_from_block(last_block):
    # remove const none, cast, return nodes
    assert isinstance(last_block.body[-1], ir.Return)
    last_block.body.pop()
    assert (isinstance(last_block.body[-1], ir.Assign)
            and isinstance(last_block.body[-1].value, ir.Expr)
            and last_block.body[-1].value.op == 'cast')
    last_block.body.pop()
    if (isinstance(last_block.body[-1], ir.Assign)
            and isinstance(last_block.body[-1].value, ir.Const)
            and last_block.body[-1].value.value is None):
        last_block.body.pop()


def include_new_blocks(blocks, new_blocks, label, new_body, remove_non_return=True, work_list=None, func_ir=None):
    inner_blocks = add_offset_to_labels(new_blocks, ir_utils._max_label + 1)
    blocks.update(inner_blocks)
    ir_utils._max_label = max(blocks.keys())
    scope = blocks[label].scope
    loc = blocks[label].loc
    inner_topo_order = find_topo_order(inner_blocks)
    inner_first_label = inner_topo_order[0]
    inner_last_label = inner_topo_order[-1]
    if remove_non_return:
        remove_return_from_block(inner_blocks[inner_last_label])
    new_body.append(ir.Jump(inner_first_label, loc))
    blocks[label].body = new_body
    label = ir_utils.next_label()
    blocks[label] = ir.Block(scope, loc)
    if remove_non_return:
        inner_blocks[inner_last_label].body.append(ir.Jump(label, loc))
    # new_body.clear()
    if work_list is not None:
        topo_order = find_topo_order(inner_blocks)
        for _label in topo_order:
            block = inner_blocks[_label]
            block.scope = scope
            numba.inline_closurecall._add_definitions(func_ir, block)
            work_list.append((_label, block))
    return label


def find_str_const(func_ir, var):
    """Check if a variable can be inferred as a string constant, and return
    the constant value, or raise GuardException otherwise.
    """
    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)
    if isinstance(var_def, ir.Const):
        val = var_def.value
        require(isinstance(val, str))
        return val

    # only add supported (s1+s2), TODO: extend to other expressions
    require(isinstance(var_def, ir.Expr) and var_def.op == 'binop'
            and var_def.fn == operator.add)
    arg1 = find_str_const(func_ir, var_def.lhs)
    arg2 = find_str_const(func_ir, var_def.rhs)
    return arg1 + arg2


def gen_getitem(out_var, in_var, ind, calltypes, nodes):
    loc = out_var.loc
    getitem = ir.Expr.static_getitem(in_var, ind, None, loc)
    calltypes[getitem] = None
    nodes.append(ir.Assign(getitem, out_var, loc))


def sanitize_varname(varname):
    return varname.replace('$', '_').replace('.', '_')


def is_call_assign(stmt):
    return (isinstance(stmt, ir.Assign)
            and isinstance(stmt.value, ir.Expr)
            and stmt.value.op == 'call')


def is_call(expr):
    return (isinstance(expr, ir.Expr)
            and expr.op == 'call')


def is_var_assign(inst):
    return isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var)


def is_assign(inst):
    return isinstance(inst, ir.Assign)


def dump_node_list(node_list):
    for n in node_list:
        print("   ", n)


def debug_prints():
    return numba.config.DEBUG_ARRAY_OPT == 1


def update_globals(func, glbls):
    if isinstance(func, pytypes.FunctionType):
        func.__globals__.update(glbls)


def sdc_overload(func):
    return overload(func, inline='always')


def sdc_overload_method(typ, name):
    return overload_method(typ, name, inline='always')


def sdc_overload_attribute(typ, name):
    return overload_attribute(typ, name, inline='always')

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

from sys import modules

from types import FunctionType

from numba.ir import (Const, Global, Var, FreeVar,
                      Expr, Assign, Del,
                      unknown_loc)
from numba.ir_utils import (guard, find_const, mk_unique_var)
from numba.extending import _Intrinsic


def filter_block_statements(block, stmt_types):
    """
    Filters given block returning statments of specific type
    """

    for stmt in block.body:
        if isinstance(stmt, stmt_types):
            yield stmt


def find_operations(block, op_name):
    """
    Filter given block returning statements with specific expressions
    """

    for assign in filter_block_statements(block, Assign):
        rhs = assign.value
        if isinstance(rhs, Expr) and rhs.op == op_name:
            yield assign


def find_declarations(block, var_types=None):
    """
    Filter given block returning statements with variables assignments of specific type
    """

    all_var_types = (Const, Global, Var, FreeVar)
    if var_types is None:
        var_types = all_var_types

    if isinstance(var_types, tuple):
        assert all(typ in all_var_types for typ in var_types)
    else:
        assert var_types in all_var_types

    for assign in filter_block_statements(block, (Assign)):
        rhs = assign.value
        if isinstance(rhs, var_types):
            yield assign


def make_var_name(prefix=None, name=None):
    """
    Creates variable name.
    If prefix is given it would be extended with postfix to create unique name.
    If name is given returning this name.
    Either prefix or name could be not None at the same time
    """

    assert prefix is None or name is None
    var_name = None
    if name is not None:
        var_name = name
    else:
        if prefix is None:
            prefix = '$_var_'

        var_name = mk_unique_var(prefix)

    return var_name


def make_assign(expr, scope, func_ir, loc=unknown_loc, prefix=None, name=None):
    """
    Creates variable, assign statement and add variable to the function definition section
    """

    var_name = make_var_name(prefix=prefix, name=name)
    var = Var(scope, var_name, loc)
    stmt = Assign(expr, var, loc)

    func_ir._definitions[var.name] = [expr]

    return stmt


def declare_constant(value, block, func_ir, loc=unknown_loc, prefix=None, name=None):
    """
    Creates variable and constant object with given value. Assign created variable to the constant
    """

    if prefix is None and name is None:
        prefix = '$_const_'

    stmt = make_assign(Const(value, loc, True), block.scope, func_ir, loc, prefix=prefix, name=name)

    block.prepend(stmt)

    return stmt


def declare_global(global_name, value, block, func_ir, loc=unknown_loc, prefix=None, name=None):
    """
    Creates variable and global object. Assign created variable to the global
    """

    if prefix is None and name is None:
        prefix = '$_global_' + global_name

    stmt = make_assign(Global(global_name, value, loc), block.scope, func_ir, loc, prefix=prefix, name=name)

    block.prepend(stmt)

    return stmt


def make_getattr(attr, var, block, func_ir, loc=unknown_loc, prefix=None, name=None):
    """
    Creates variable and assign to it provided attribute.
    """

    if prefix is None and name is None:
        prefix = '$_attr_' + attr
    stmt_ = make_assign(Expr.getattr(var, attr, loc), block.scope, func_ir, loc, prefix=prefix, name=name)

    assign = block.find_variable_assignment(var.name)
    if assign is not None:
        block.insert_after(stmt_, assign)
    else:
        block.prepend(assign)

    return stmt_


def is_dict(var, func_ir):
    """
    Checks if variable is a dictionary
    """

    try:
        rhs = func_ir.get_definition(var)
        if isinstance(rhs, Expr):
            return rhs.op == 'build_map'
    except Exception:
        pass

    return False


def get_dict_items(var, func_ir):
    """
    Returns dictionary items
    """

    assert is_dict(var, func_ir)
    rhs = func_ir.get_definition(var)

    return rhs.items


def is_list(var, func_ir):
    """
    Checks if variable is a list
    """

    try:
        rhs = func_ir.get_definition(var)
        if isinstance(rhs, Expr):
            return rhs.op == 'build_list'
    except Exception:
        pass

    return False


def is_tuple(var, func_ir):
    """
    Checks if variable is either constant or non-constant tuple
    """

    val = guard(find_const, func_ir, var)
    if val is not None:
        return isinstance(val, tuple)

    try:
        rhs = func_ir.get_definition(var)
        if isinstance(rhs, Expr):
            return rhs.op == 'build_tuple'
    except Exception:
        pass

    return False


def get_tuple_items(var, block, func_ir):
    """
    Returns tuple items. If tuple is constant creates and returns constants
    """

    def wrap_into_var(value, block, func_ir, loc):
        stmt = declare_constant(value, block, func_ir, loc)

        return stmt.target

    val = guard(find_const, func_ir, var)
    if val is not None:
        if isinstance(val, tuple):
            return [wrap_into_var(v, block, func_ir, var.loc) for v in val]

        return None

    try:
        rhs = func_ir.get_definition(var)
        if isinstance(rhs, Expr):
            if rhs.op == 'build_tuple':
                return list(rhs.items)
    except Exception:
        pass

    return None


def find_usage(var, block):
    """
    Filters given block by statements with given variable
    """
    assert isinstance(var, Var)

    # TODO handle usage outside of given block
    for stmt in block.body:
        if var in stmt.list_vars() or (isinstance(stmt, Del) and stmt.value == var.name):
            yield stmt


def _remove_unused_internal(var, block, func_ir):
    """
    Search given block for variable usages. If it is used only in one assignment and one Del - remove this variable
    """

    usage_list = []
    use_count = 0

    for stmt in find_usage(var, block):
        usage_list.append(stmt)
        if isinstance(stmt, Del):
            use_count -= 1
        else:
            use_count += 1

    if use_count == 0:
        for stmt in usage_list:
            block.remove(stmt)

        del func_ir._definitions[var.name]

        return True, usage_list

    return False, None


def remove_unused(var, block, func_ir):
    """
    Search given block for variable usages. If it is used only in one assignment and one Del - remove this variable
    """

    result, _ = _remove_unused_internal(var, block, func_ir)

    return result


def remove_var(var, block, func_ir):
    """
    Remove all statements with given variable from the block
    """

    for stmt in find_usage(var, block):
        block.remove(stmt)

    del func_ir._definitions[var.name]


def remove_unused_recursively(var, block, func_ir):
    """
    Search given block for variable usages.
    If it is used only in one assignment and one Del - remove it and all it's aliases
    """

    result, usage = _remove_unused_internal(var, block, func_ir)

    while usage:
        for stmt in usage:
            if isinstance(stmt, Del):
                continue

            if isinstance(stmt.value, Var):
                result, usage = _remove_unused_internal(stmt.value, block, func_ir)
            else:
                usage = None
                break

    return result


def get_call_parameters(call, arg_names):
    """
    Extracts call parameters into dict
    """

    params = dict(zip(arg_names, call.args))
    params.update(call.kws)

    return params


def import_object(object_name, module_name, block, func_ir, prefix=None, name=None):
    """
    Imports specific object from a specific module. Do nothing if object is already imported in the block
    """

    def get_module_stmt(module_name, block, func_ir):
        module = modules[module_name]
        module_stmt = None

        for stmt in find_declarations(block, Global):
            rhs = stmt.value
            if rhs.value == module:
                module_stmt = stmt

        if module_stmt is None:
            stmt = declare_global(module_name, module, block, func_ir)
            module_stmt = stmt

        assert module_stmt is not None

        return module_stmt

    def find_object_stmt(object_name, block):
        object_stmt = None

        for stmt in find_operations(block, 'getattr'):
            expr = stmt.value
            if expr.attr == object_name:
                object_stmt = stmt

        return object_stmt

    object_stmt = find_object_stmt(object_name, block)

    if object_stmt is None:
        module_stmt = get_module_stmt(module_name, block, func_ir)
        object_stmt = make_getattr(object_name, module_stmt.target, block, func_ir, prefix=prefix, name=name)

    assert object_stmt is not None

    return object_stmt


def import_function(func, block, func_ir, prefix=None, name=None):
    """
    Creates variable and imports function. Assign created variable to the function
    """

    assert isinstance(func, (FunctionType, _Intrinsic))

    if isinstance(func, _Intrinsic):
        func = func._defn

    func_name = func.__name__
    module_name = func.__module__

    return import_object(func_name, module_name, block, func_ir, prefix=prefix, name=name)


def make_call(func, args, kwargs, block, func_ir, loc=unknown_loc, prefix=None, name=None):
    """
    Creates variable and call expression with given functions object, intrinsic or function variable.
    Returns assignment statement of created variable to call expression
    """
    assert isinstance(func, (FunctionType, _Intrinsic, Var))
    func_var = func

    if isinstance(func, (FunctionType, _Intrinsic)):
        func_var = import_function(func, block, func_ir).target

    return make_assign(Expr.call(func_var, args, kwargs, loc), block.scope, func_ir, loc, prefix, name)


def insert_before(block, stmt, other):
    """
    Inserts given statement before another statement in the given block
    """

    index = block.body.index(other)
    block.body.insert(index, stmt)

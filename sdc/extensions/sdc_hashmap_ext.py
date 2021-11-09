# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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

import llvmlite.binding as ll
import llvmlite.llvmpy.core as lc
import numba
import numpy as np
import operator
import sdc

from glob import glob
from llvmlite import ir as lir
from numba import types, cfunc
from numba.core import cgutils
from numba.extending import (typeof_impl, type_callable, models, register_model, NativeValue,
                             lower_builtin, box, unbox, lower_getattr, intrinsic,
                             overload_method, overload, overload_attribute)
from numba.cpython.hashing import _Py_hash_t
from numba.core.imputils import (impl_ret_new_ref, impl_ret_borrowed, iternext_impl, RefType)
from numba.cpython.listobj import ListInstance
from numba.core.typing.templates import (infer_global, AbstractTemplate, infer,
                                         signature, AttributeTemplate, infer_getattr, bound_function)
from numba.typed.typedobjectutils import _as_bytes

from sdc.str_arr_type import StringArrayType
from sdc import hconc_dict
from sdc.extensions.sdc_hashmap_type import (ConcurrentDict, ConcurrentDictType,
                                             ConcDictKeysIterableType, ConcDictIteratorType,
                                             ConcDictItemsIterableType, ConcDictValuesIterableType)
from numba.extending import register_jitable

from sdc.datatypes.sdc_typeref import ConcurrentDictTypeRef
from sdc.utilities.sdc_typing_utils import TypingError, TypeChecker, check_types_comparable
from itertools import product

from numba.typed.dictobject import _cast


def gen_func_suffixes():
    key_suffixes = ['int32_t', 'int64_t', 'voidptr']
    val_suffixes = ['int32_t', 'int64_t', 'float', 'double', 'voidptr']
    return map(lambda x: f'{x[0]}_to_{x[1]}',
               product(key_suffixes, val_suffixes))


def load_native_func(fname, module, suffixes=None, skip_check=None):
    suffixes = suffixes or ['', ]
    for s in suffixes:
        if skip_check and skip_check(s):
            continue
        fsuffix = f'_{s}' if s else ''
        full_func_name = f'{fname}{fsuffix}'
        ll.add_symbol(full_func_name,
                      getattr(module, full_func_name))


hashmap_func_suffixes = list(gen_func_suffixes())
load_native_func('hashmap_create', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_size', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_set', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_contains', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_lookup', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_clear', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_pop', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_update', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_create_from_data', hconc_dict, hashmap_func_suffixes, lambda x: 'voidptr' in x)
load_native_func('hashmap_getiter', hconc_dict, hashmap_func_suffixes)
load_native_func('hashmap_iternext', hconc_dict, hashmap_func_suffixes)


supported_numeric_key_types = [
    types.int32,
    types.uint32,
    types.int64,
    types.uint64
]


supported_numeric_value_types = [
    types.int32,
    types.uint32,
    types.int64,
    types.uint64,
    types.float32,
    types.float64,
]


# to avoid over-specialization native hashmap structs always use signed integers
# and function arguments and return values are converted when needed
reduced_type_map = {
    types.int32: types.int32,
    types.uint32: types.int32,
    types.int64: types.int64,
    types.uint64: types.int64,
}


map_numba_type_to_prefix = {
    types.int32: 'int32_t',
    types.int64: 'int64_t',
    types.float32: 'float',
    types.float64: 'double',
}


def _get_types_postfixes(key_type, value_type):
    _key_type = reduced_type_map[key_type] if isinstance(key_type, types.Integer) else key_type
    _value_type = reduced_type_map[value_type] if isinstance(value_type, types.Integer) else value_type

    key_postfix = map_numba_type_to_prefix.get(_key_type, 'voidptr')
    value_postfix = map_numba_type_to_prefix.get(_value_type, 'voidptr')

    return (key_postfix, value_postfix)


def gen_deref_voidptr(key_type):
    @intrinsic
    def deref_voidptr(typingctx, data_ptr_type):
        if data_ptr_type is not types.voidptr:
            return None

        ret_type = key_type

        def codegen(context, builder, sig, args):
            str_val, = args

            ty_ret_type_pointer = lir.PointerType(context.get_data_type(ret_type))
            casted_ptr = builder.bitcast(str_val, ty_ret_type_pointer)

            return impl_ret_borrowed(context, builder, ret_type, builder.load(casted_ptr))

        return ret_type(types.voidptr), codegen

    return deref_voidptr


def gen_hash_compare_ops(key_type):

    deref_voidptr = gen_deref_voidptr(key_type)
    c_sig_hash = types.uintp(types.voidptr)
    c_sig_eq = types.boolean(types.voidptr, types.voidptr)

    @cfunc(c_sig_hash)
    def hash_func_adaptor(voidptr_to_data):
        obj = deref_voidptr(voidptr_to_data)
        return hash(obj)

    @cfunc(c_sig_eq)
    def eq_func_adaptor(lhs_ptr, rhs_ptr):
        lhs_str = deref_voidptr(lhs_ptr)
        rhs_str = deref_voidptr(rhs_ptr)
        return lhs_str == rhs_str

    hasher_ptr = hash_func_adaptor.address
    eq_ptr = eq_func_adaptor.address

    return hasher_ptr, eq_ptr


@intrinsic
def call_incref(typingctx, val_type):
    ret_type = types.void

    def codegen(context, builder, sig, args):
        [arg_val] = args
        [arg_type] = sig.args

        if context.enable_nrt:
            context.nrt.incref(builder, arg_type, arg_val)

    return ret_type(val_type), codegen


@intrinsic
def call_decref(typingctx, val_type):
    ret_type = types.void

    def codegen(context, builder, sig, args):
        [arg_val] = args
        [arg_type] = sig.args

        if context.enable_nrt:
            context.nrt.decref(builder, arg_type, arg_val)

    return ret_type(val_type), codegen


def gen_incref_decref_ops(key_type):

    deref_voidptr = gen_deref_voidptr(key_type)
    c_sig_incref = types.void(types.voidptr)
    c_sig_decref = types.void(types.voidptr)

    @cfunc(c_sig_incref)
    def incref_func_adaptor(voidptr_to_data):
        obj = deref_voidptr(voidptr_to_data)
        return call_incref(obj)

    @cfunc(c_sig_decref)
    def decref_func_adaptor(voidptr_to_data):
        obj = deref_voidptr(voidptr_to_data)
        return call_decref(obj)

    incref_ptr = incref_func_adaptor.address
    decref_ptr = decref_func_adaptor.address

    return incref_ptr, decref_ptr


def codegen_get_voidptr(context, builder, ty_var, var_val):
    dm_key = context.data_model_manager[ty_var]
    data_val = dm_key.as_data(builder, var_val)
    ptr_var = cgutils.alloca_once_value(builder, data_val)
    val_as_voidptr = _as_bytes(builder, ptr_var)

    return val_as_voidptr


def transform_input_arg(context, builder, ty_arg, val):
    """ This function should adjust key to satisfy argument type of native function to
    which it will be passed later """

    if isinstance(ty_arg, types.Number):
        arg_native_type = reduced_type_map.get(ty_arg, ty_arg)
        key_val = val
        if ty_arg is not arg_native_type:
            key_val = context.cast(builder, key_val, ty_arg, arg_native_type)
        lir_key_type = context.get_value_type(arg_native_type)
    else:
        key_val = codegen_get_voidptr(context, builder, ty_arg, val)
        lir_key_type = context.get_value_type(types.voidptr)

    return (key_val, lir_key_type)


def alloc_native_value(context, builder, ty_arg):
    """ This function allocates argument to be used as return value of a native function """

    if isinstance(ty_arg, types.Number):
        native_arg_type = reduced_type_map.get(ty_arg, ty_arg)
    else:
        native_arg_type = types.voidptr

    lir_val_type = context.get_value_type(native_arg_type)
    ret_val_ptr = cgutils.alloca_once(builder, lir_val_type)

    return (ret_val_ptr, lir_val_type)


def transform_native_val(context, builder, ty_arg, val):
    """ This function should cast value returned from native func back to dicts value_type """

    if isinstance(ty_arg, types.Number):
        reduced_value_type = reduced_type_map.get(ty_arg, ty_arg)
        result_value = context.cast(builder, val, reduced_value_type, ty_arg)
    else:
        # for values stored as void* in native dict we also need to dereference
        lir_typed_value_ptr = context.get_value_type(ty_arg).as_pointer()
        casted_ptr = builder.bitcast(val, lir_typed_value_ptr)
        result_value = builder.load(casted_ptr)

    return result_value


@intrinsic
def hashmap_create(typingctx, key, value):

    key_numeric = isinstance(key, types.NumberClass)
    val_numeric = isinstance(value, types.NumberClass)
    dict_key_type = key.dtype if key_numeric else key.instance_type
    dict_val_type = value.dtype if val_numeric else value.instance_type
    dict_type = ConcurrentDictType(dict_key_type, dict_val_type)

    hash_func_addr, eq_func_addr = gen_hash_compare_ops(dict_key_type)
    key_incref_func_addr, key_decref_func_addr = gen_incref_decref_ops(dict_key_type)
    val_incref_func_addr, val_decref_func_addr = gen_incref_decref_ops(dict_val_type)

    key_type_postfix, value_type_postfix = _get_types_postfixes(dict_key_type, dict_val_type)

    def codegen(context, builder, sig, args):
        nrt_table = context.nrt.get_nrt_api(builder)

        llptrtype = context.get_value_type(types.intp)
        cdict = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        fnty = lir.FunctionType(lir.VoidType(),
                                [cdict.meminfo.type.as_pointer(),     # meminfo to fill
                                 lir.IntType(8).as_pointer(),         # NRT API func table
                                 lir.IntType(8), lir.IntType(8),      # gen_key, gen_value flags
                                 llptrtype, llptrtype,                # hash_func, equality func
                                 llptrtype, llptrtype,                # key incref, decref
                                 llptrtype, llptrtype,                # val incref, decref
                                 lir.IntType(64), lir.IntType(64)])   # key size, val size
        func_name = f"hashmap_create_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_create = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)

        gen_key = context.get_constant(types.int8, types.int8(not key_numeric))
        gen_val = context.get_constant(types.int8, types.int8(not val_numeric))

        lir_key_type = context.get_value_type(dict_key_type)
        hash_func_addr_const = context.get_constant(types.intp, hash_func_addr)
        eq_func_addr_const = context.get_constant(types.intp, eq_func_addr)
        key_incref = context.get_constant(types.intp, key_incref_func_addr)
        key_decref = context.get_constant(types.intp, key_decref_func_addr)
        key_type_size = context.get_constant(types.int64, context.get_abi_sizeof(lir_key_type))

        lir_val_type = context.get_value_type(dict_val_type)
        val_incref = context.get_constant(types.intp, val_incref_func_addr)
        val_decref = context.get_constant(types.intp, val_decref_func_addr)
        val_type_size = context.get_constant(types.int64, context.get_abi_sizeof(lir_val_type))

        builder.call(fn_hashmap_create,
                     [cdict._get_ptr_by_name('meminfo'),
                      nrt_table,
                      gen_key,
                      gen_val,
                      hash_func_addr_const,
                      eq_func_addr_const,
                      key_incref,
                      key_decref,
                      val_incref,
                      val_decref,
                      key_type_size,
                      val_type_size])

        cdict.data_ptr = context.nrt.meminfo_data(builder, cdict.meminfo)
        return cdict._getvalue()

    return dict_type(key, value), codegen


@overload_method(ConcurrentDictTypeRef, 'empty')
def concurrent_dict_empty(cls, key_type, value_type):

    if cls.instance_type is not ConcurrentDictType:
        return

    _func_name = 'Method ConcurrentDictTypeRef::empty().'
    ty_checker = TypeChecker(_func_name)

    supported_key_types = (types.NumberClass, types.TypeRef)
    supported_value_types = (types.NumberClass, types.TypeRef)

    if not isinstance(key_type, supported_key_types):
        ty_checker.raise_exc(key_type, f'Numba type of dict keys (e.g. types.int32)', 'key_type')
    if not isinstance(value_type, supported_value_types):
        ty_checker.raise_exc(value_type, f'Numba type of dict values (e.g. types.int32)', 'value_type')

    if (isinstance(key_type, types.NumberClass)
            and key_type.dtype not in supported_numeric_key_types or
        isinstance(key_type, types.TypeRef)
            and not isinstance(key_type.instance_type, (types.UnicodeType, types.Hashable) or
        isinstance(value_type, types.NumberClass)
            and value_type.dtype not in supported_numeric_value_types)):
        error_msg = '{} SDC ConcurrentDict({}, {}) is not supported. '
        raise TypingError(error_msg.format(_func_name, key_type, value_type))

    def concurrent_dict_empty_impl(cls, key_type, value_type):
        return hashmap_create(key_type, value_type)

    return concurrent_dict_empty_impl


@intrinsic
def hashmap_size(typingctx, dict_type):

    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    def codegen(context, builder, sig, args):
        dict_val, = args

        cdict = cgutils.create_struct_proxy(dict_type)(
                    context, builder, value=dict_val)
        fnty = lir.FunctionType(lir.IntType(64),
                                [lir.IntType(8).as_pointer()])
        func_name = f"hashmap_size_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_size = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)
        ret = builder.call(fn_hashmap_size, [cdict.data_ptr])
        return ret

    return types.uint64(dict_type), codegen


@overload(len)
def concurrent_dict_len_ovld(cdict):
    if not isinstance(cdict, ConcurrentDictType):
        return None

    def concurrent_dict_len_impl(cdict):
        return hashmap_size(cdict)

    return concurrent_dict_len_impl


@intrinsic
def hashmap_set(typingctx, dict_type, key_type, value_type):

    key_type_postfix, value_type_postfix = _get_types_postfixes(key_type, value_type)

    def codegen(context, builder, sig, args):
        dict_val, key_val, value_val = args

        key_val, lir_key_type = transform_input_arg(context, builder, key_type, key_val)
        val_val, lir_val_type = transform_input_arg(context, builder, value_type, value_val)

        cdict = cgutils.create_struct_proxy(dict_type)(
                    context, builder, value=dict_val)
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer(),
                                 lir_key_type,
                                 lir_val_type])

        func_name = f"hashmap_set_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_insert = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)

        builder.call(fn_hashmap_insert, [cdict.data_ptr, key_val, val_val])
        return

    return types.void(dict_type, key_type, value_type), codegen


@overload(operator.setitem, prefer_literal=False)
def concurrent_dict_set_ovld(self, key, value):
    if not isinstance(self, ConcurrentDictType):
        return None

    dict_key_type, dict_value_type = self.key_type, self.value_type
    cast_key = key is not dict_key_type
    cast_value = value is not dict_value_type

    def concurrent_dict_set_impl(self, key, value):
        _key = key if cast_key == False else _cast(key, dict_key_type)  # noqa
        _value = value if cast_value == False else _cast(value, dict_value_type)  # noqa
        return hashmap_set(self, _key, _value)

    return concurrent_dict_set_impl


@intrinsic
def hashmap_contains(typingctx, dict_type, key_type):

    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    def codegen(context, builder, sig, args):
        dict_val, key_val = args

        key_val, lir_key_type = transform_input_arg(context, builder, key_type, key_val)
        cdict = cgutils.create_struct_proxy(dict_type)(
                    context, builder, value=dict_val)
        fnty = lir.FunctionType(lir.IntType(8),
                                [lir.IntType(8).as_pointer(),
                                 lir_key_type])
        func_name = f"hashmap_contains_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_contains = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)

        res = builder.call(fn_hashmap_contains, [cdict.data_ptr, key_val])
        return context.cast(builder, res, types.uint8, types.bool_)

    return types.bool_(dict_type, key_type), codegen


@overload(operator.contains, prefer_literal=False)
def concurrent_dict_contains_ovld(self, key):
    if not isinstance(self, ConcurrentDictType):
        return None

    dict_key_type = self.key_type
    cast_key = key is not dict_key_type

    def concurrent_dict_contains_impl(self, key):
        _key = key if cast_key == False else _cast(key, dict_key_type)  # noqa
        return hashmap_contains(self, _key)

    return concurrent_dict_contains_impl


@intrinsic
def hashmap_lookup(typingctx, dict_type, key_type):

    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    return_type = types.Tuple([types.bool_, types.Optional(ty_val)])
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    def codegen(context, builder, sig, args):
        dict_val, key_val = args

        key_val, lir_key_type = transform_input_arg(context, builder, key_type, key_val)
        native_value_ptr, lir_value_type = alloc_native_value(context, builder, ty_val)

        cdict = cgutils.create_struct_proxy(dict_type)(context, builder, value=dict_val)
        fnty = lir.FunctionType(lir.IntType(8),
                                [lir.IntType(8).as_pointer(),
                                 lir_key_type,
                                 lir_value_type.as_pointer()
                                 ])
        func_name = f"hashmap_lookup_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_lookup = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)

        status = builder.call(fn_hashmap_lookup, [cdict.data_ptr, key_val, native_value_ptr])
        status_as_bool = context.cast(builder, status, types.uint8, types.bool_)

        # if key was not found nothing would be stored to native_value_ptr, so depending on status
        # we either deref it or not, wrapping final result into types.Optional value
        result_ptr = cgutils.alloca_once(builder,
                                         context.get_value_type(types.Optional(ty_val)))
        with builder.if_else(status_as_bool, likely=True) as (if_ok, if_not_ok):
            with if_ok:
                native_value = builder.load(native_value_ptr)
                result_value = transform_native_val(context, builder, ty_val, native_value)

                if context.enable_nrt:
                    context.nrt.incref(builder, ty_val, result_value)

                builder.store(context.make_optional_value(builder, ty_val, result_value),
                              result_ptr)

            with if_not_ok:
                builder.store(context.make_optional_none(builder, ty_val),
                              result_ptr)

        opt_result = builder.load(result_ptr)
        return context.make_tuple(builder, return_type, [status_as_bool, opt_result])

    func_sig = return_type(dict_type, key_type)
    return func_sig, codegen


@overload(operator.getitem, prefer_literal=False)
def concurrent_dict_lookup_ovld(self, key):
    if not isinstance(self, ConcurrentDictType):
        return None

    dict_key_type = self.key_type
    cast_key = key is not dict_key_type

    def concurrent_dict_lookup_impl(self, key):
        _key = key if cast_key == False else _cast(key, dict_key_type)  # noqa
        found, res = hashmap_lookup(self, _key)

        # Note: this function raises exception so expect no scaling if you use it in prange
        if not found:
            raise KeyError("ConcurrentDict key not found")
        return res

    return concurrent_dict_lookup_impl


@intrinsic
def hashmap_clear(typingctx, dict_type):

    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    def codegen(context, builder, sig, args):
        dict_val, = args

        cdict = cgutils.create_struct_proxy(dict_type)(
                    context, builder, value=dict_val)
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer()])
        func_name = f"hashmap_clear_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_clear = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)
        builder.call(fn_hashmap_clear, [cdict.data_ptr])
        return

    return types.void(dict_type), codegen


@overload_method(ConcurrentDictType, 'clear')
def concurrent_dict_clear_ovld(self):
    if not isinstance(self, ConcurrentDictType):
        return None

    def concurrent_dict_clear_impl(self):
        hashmap_clear(self)

    return concurrent_dict_clear_impl


@overload_method(ConcurrentDictType, 'get')
def concurrent_dict_get_ovld(self, key, default=None):
    if not isinstance(self, ConcurrentDictType):
        return None

    _func_name = f'Method {self}::get()'
    ty_checker = TypeChecker(_func_name)

    # default value is expected to be of the same (or safely casted) type as dict's value_type
    no_default = isinstance(default, (types.NoneType, types.Omitted)) or default is None
    default_is_optional = isinstance(default, types.Optional)
    if not (no_default or check_types_comparable(default, self.value_type)
            or default_is_optional and check_types_comparable(default.type, self.value_type)):
        ty_checker.raise_exc(default, f'{self.value_type} or convertible or None', 'default')

    dict_key_type, dict_value_type = self.key_type, self.value_type
    cast_key = key is not dict_key_type

    def concurrent_dict_get_impl(self, key, default=None):
        _key = key if cast_key == False else _cast(key, dict_key_type)  # noqa
        found, res = hashmap_lookup(self, _key)

        if not found:
            # just to make obvious that return type is types.Optional(dict.value_type)
            if no_default == False:  # noqa
                return _cast(default, dict_value_type)
            else:
                return None
        return res

    return concurrent_dict_get_impl


@intrinsic
def hashmap_pop(typingctx, dict_type, key_type):

    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    return_type = types.Tuple([types.bool_, types.Optional(ty_val)])
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    def codegen(context, builder, sig, args):
        dict_val, key_val = args

        key_val, lir_key_type = transform_input_arg(context, builder, key_type, key_val)

        # unlike in lookup operation we allocate value here and pass into native function
        # voidptr to allocated data, which copies and frees it's copy
        if isinstance(ty_val, types.Number):
            ret_val_ptr, lir_val_type = alloc_native_value(context, builder, ty_val)
        else:
            lir_val_type = context.get_value_type(ty_val)
            ret_val_ptr = cgutils.alloca_once(builder, lir_val_type)

        llvoidptr = context.get_value_type(types.voidptr)
        ret_val_ptr = builder.bitcast(ret_val_ptr, llvoidptr)

        cdict = cgutils.create_struct_proxy(dict_type)(context, builder, value=dict_val)
        fnty = lir.FunctionType(lir.IntType(8),
                                [lir.IntType(8).as_pointer(),
                                 lir_key_type,
                                 llvoidptr,
                                 ])
        func_name = f"hashmap_pop_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_pop = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)

        status = builder.call(fn_hashmap_pop, [cdict.data_ptr, key_val, ret_val_ptr])
        status_as_bool = context.cast(builder, status, types.uint8, types.bool_)

        # same logic to handle non-existing key as in hashmap_lookup
        result_ptr = cgutils.alloca_once(builder,
                                         context.get_value_type(types.Optional(ty_val)))
        with builder.if_else(status_as_bool, likely=True) as (if_ok, if_not_ok):
            with if_ok:

                ret_val_ptr = builder.bitcast(ret_val_ptr, lir_val_type.as_pointer())
                native_value = builder.load(ret_val_ptr)
                if isinstance(ty_val, types.Number):
                    reduced_value_type = reduced_type_map.get(ty_val, ty_val)
                    native_value = context.cast(builder, native_value, reduced_value_type, ty_val)

                # no incref of the value here, since it was removed from the dict
                # w/o decref to consider the case when value in the dict had refcnt == 1

                builder.store(context.make_optional_value(builder, ty_val, native_value),
                              result_ptr)

            with if_not_ok:
                builder.store(context.make_optional_none(builder, ty_val),
                              result_ptr)

        opt_result = builder.load(result_ptr)
        return context.make_tuple(builder, return_type, [status_as_bool, opt_result])

    func_sig = return_type(dict_type, key_type)
    return func_sig, codegen


@overload_method(ConcurrentDictType, 'pop', prefer_literal=False)
def concurrent_dict_pop_ovld(self, key, default=None):
    if not isinstance(self, ConcurrentDictType):
        return None

    _func_name = f'Method {self}::pop()'
    ty_checker = TypeChecker(_func_name)

    # default value is expected to be of the same (or safely casted) type as dict's value_type
    no_default = isinstance(default, (types.NoneType, types.Omitted)) or default is None
    default_is_optional = isinstance(default, types.Optional)
    if not (no_default or check_types_comparable(default, self.value_type)
            or default_is_optional and check_types_comparable(default.type, self.value_type)):
        ty_checker.raise_exc(default, f'{self.value_type} or convertible or None', 'default')

    dict_key_type, dict_value_type = self.key_type, self.value_type
    cast_key = key is not dict_key_type

    def concurrent_dict_pop_impl(self, key, default=None):
        _key = key if cast_key == False else _cast(key, dict_key_type)  # noqa
        found, res = hashmap_pop(self, _key)

        if not found:
            if no_default == False:  # noqa
                return _cast(default, dict_value_type)
            else:
                return None
        return res

    return concurrent_dict_pop_impl


@intrinsic
def hashmap_update(typingctx, dict_type, other_dict_type):

    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    return_type = types.void
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    def codegen(context, builder, sig, args):
        dict_val, other_dict_val = args

        self_cdict = cgutils.create_struct_proxy(dict_type)(context, builder, value=dict_val)
        other_cdict = cgutils.create_struct_proxy(other_dict_type)(context, builder, value=other_dict_val)
        fnty = lir.FunctionType(lir.IntType(8),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(8).as_pointer()
                                 ])
        func_name = f"hashmap_update_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_update = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)

        builder.call(fn_hashmap_update, [self_cdict.data_ptr, other_cdict.data_ptr])
        return

    func_sig = return_type(dict_type, other_dict_type)
    return func_sig, codegen


@overload_method(ConcurrentDictType, 'update', prefer_literal=False)
def concurrent_dict_update_ovld(self, other):
    if not ((self, ConcurrentDictType) and isinstance(other, ConcurrentDictType)):
        return None

    _func_name = f'Method {self}::update()'
    ty_checker = TypeChecker(_func_name)

    if self is not other:
        ty_checker.raise_exc(other, f'{self}', 'other')

    def concurrent_dict_update_impl(self, other):
        return hashmap_update(self, other)

    return concurrent_dict_update_impl


@overload_method(ConcurrentDictType, 'fromkeys', prefer_literal=False)
def concurrent_dict_fromkeys_ovld(self, keys, value):
    if not isinstance(self, ConcurrentDictType):
        return None

    def wrapper_impl(self, keys, value):
        return ConcurrentDict.fromkeys(keys, value)

    return wrapper_impl


@register_jitable
def get_min_size(A, B):
    return min(len(A), len(B))


@intrinsic
def create_from_arrays(typingctx, keys, values):

    ty_key, ty_val = keys.dtype, values.dtype
    dict_type = ConcurrentDictType(ty_key, ty_val)
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)
    get_min_size_sig = signature(types.int64, keys, values)

    def codegen(context, builder, sig, args):
        keys_val, values_val = args
        nrt_table = context.nrt.get_nrt_api(builder)

        keys_ctinfo = context.make_helper(builder, keys, keys_val)
        values_ctinfo = context.make_helper(builder, values, values_val)
        size_val = context.compile_internal(
            builder,
            lambda k, v: get_min_size(k, v),
            get_min_size_sig,
            [keys_val, values_val]
        )

        # create concurrent dict struct and call native ctor filling meminfo
        lir_key_type = context.get_value_type(reduced_type_map.get(ty_key, ty_key))
        lir_value_type = context.get_value_type(reduced_type_map.get(ty_val, ty_val))
        cdict = cgutils.create_struct_proxy(dict_type)(context, builder)
        fnty = lir.FunctionType(lir.VoidType(),
                                [cdict.meminfo.type.as_pointer(),    # meminfo to fill
                                 lir.IntType(8).as_pointer(),        # NRT API func table
                                 lir_key_type.as_pointer(),          # array of keys
                                 lir_value_type.as_pointer(),        # array of values
                                 lir.IntType(64),                    # size
                                 ])
        func_name = f"hashmap_create_from_data_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_create = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)
        builder.call(fn_hashmap_create,
                     [cdict._get_ptr_by_name('meminfo'),
                      nrt_table,
                      keys_ctinfo.data,
                      values_ctinfo.data,
                      size_val
                      ])
        cdict.data_ptr = context.nrt.meminfo_data(builder, cdict.meminfo)
        return cdict._getvalue()

    return dict_type(keys, values), codegen


@overload_method(ConcurrentDictTypeRef, 'from_arrays')
def concurrent_dict_from_arrays_ovld(cls, keys, values):
    if cls.instance_type is not ConcurrentDictType:
        return

    _func_name = f'Method ConcurrentDict::from_arrays()'
    if not (isinstance(keys, types.Array) and keys.ndim == 1
            and isinstance(values, types.Array) and values.ndim == 1):
        raise TypingError('{} Supported only with 1D arrays of keys and values'
                          'Given: keys={}, values={}'.format(_func_name, keys, values))

    def concurrent_dict_from_arrays_impl(cls, keys, values):
        return create_from_arrays(keys, values)

    return concurrent_dict_from_arrays_impl


@overload_method(ConcurrentDictTypeRef, 'fromkeys', prefer_literal=False)
def concurrent_dict_type_fromkeys_ovld(cls, keys, value):
    if cls.instance_type is not ConcurrentDictType:
        return

    _func_name = f'Method ConcurrentDict::fromkeys()'
    ty_checker = TypeChecker(_func_name)

    valid_keys_types = (types.Sequence, types.Array, StringArrayType)
    if not isinstance(keys, valid_keys_types):
        ty_checker.raise_exc(keys, f'array or sequence', 'keys')

    dict_key_type, dict_value_type = keys.dtype, value
    if isinstance(keys, (types.Array, StringArrayType)):
        def concurrent_dict_fromkeys_impl(cls, keys, value):
            res = ConcurrentDict.empty(dict_key_type, dict_value_type)
            for i in numba.prange(len(keys)):
                res[keys[i]] = value
            return res
    else:  # generic for all other iterables
        def concurrent_dict_fromkeys_impl(cls, keys, value):
            res = ConcurrentDict.empty(dict_key_type, dict_value_type)
            for k in keys:
                res[k] = value
            return res

    return concurrent_dict_fromkeys_impl


@intrinsic
def _hashmap_dump(typingctx, dict_type):

    # load hashmap_dump here as otherwise module import will fail
    # since it's included in debug build only
    load_native_func('hashmap_dump', hconc_dict, hashmap_func_suffixes)
    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    def codegen(context, builder, sig, args):
        dict_val, = args

        cdict = cgutils.create_struct_proxy(dict_type)(
                    context, builder, value=dict_val)
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer()])
        func_name = f"hashmap_dump_{key_type_postfix}_to_{value_type_postfix}"
        fn_hashmap_dump = cgutils.get_or_insert_function(builder.module,
            fnty, name=func_name)
        builder.call(fn_hashmap_dump, [cdict.data_ptr])
        return

    return types.void(dict_type), codegen


def _iterator_codegen(resty):
    """The common codegen for iterator intrinsics.

    Populates the iterator struct and increfs.
    """

    def codegen(context, builder, sig, args):
        [d] = args
        [td] = sig.args
        iterhelper = context.make_helper(builder, resty)
        iterhelper.parent = d
        iterhelper.state = iterhelper.state.type(None)
        return impl_ret_borrowed(
            context,
            builder,
            resty,
            iterhelper._getvalue(),
        )

    return codegen


@intrinsic
def _conc_dict_items(typingctx, d):
    """Get dictionary iterator for .items()"""
    resty = ConcDictItemsIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
    return sig, codegen


@intrinsic
def _conc_dict_keys(typingctx, d):
    """Get dictionary iterator for .keys()"""
    resty = ConcDictKeysIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
    return sig, codegen


@intrinsic
def _conc_dict_values(typingctx, d):
    """Get dictionary iterator for .values()"""
    resty = ConcDictValuesIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
    return sig, codegen


@overload_method(ConcurrentDictType, 'items')
def impl_items(d):
    if not isinstance(d, ConcurrentDictType):
        return

    def impl(d):
        it = _conc_dict_items(d)
        return it

    return impl


@overload_method(ConcurrentDictType, 'keys')
def impl_keys(d):
    if not isinstance(d, ConcurrentDictType):
        return

    def impl(d):
        return _conc_dict_keys(d)

    return impl


@overload_method(ConcurrentDictType, 'values')
def impl_values(d):
    if not isinstance(d, ConcurrentDictType):
        return

    def impl(d):
        return _conc_dict_values(d)

    return impl


def call_native_getiter(context, builder, dict_type, dict_val, it):
    """ This function should produce LLVM code for calling native
    hashmap_getiter and fill iterator data accordingly """

    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    nrt_table = context.nrt.get_nrt_api(builder)
    llvoidptr = context.get_value_type(types.voidptr)
    fnty = lir.FunctionType(llvoidptr,
                            [it.meminfo.type.as_pointer(),
                             llvoidptr,
                             llvoidptr])
    func_name = f"hashmap_getiter_{key_type_postfix}_to_{value_type_postfix}"
    fn_hashmap_getiter = cgutils.get_or_insert_function(builder.module,
        fnty, name=func_name)

    cdict = cgutils.create_struct_proxy(dict_type)(context, builder, value=dict_val)
    it.state = builder.call(fn_hashmap_getiter,
                            [it._get_ptr_by_name('meminfo'),
                             nrt_table,
                             cdict.data_ptr])

    # store the reference to parent and incref
    it.parent = dict_val
    if context.enable_nrt:
        context.nrt.incref(builder, dict_type, dict_val)


@lower_builtin('getiter', ConcDictItemsIterableType)
@lower_builtin('getiter', ConcDictKeysIterableType)
@lower_builtin('getiter', ConcDictValuesIterableType)
def impl_iterable_getiter(context, builder, sig, args):
    """Implement iter() for .keys(), .values(), .items()
    """
    iterablety, = sig.args
    iter_val, = args

    # iter_val is an empty dict iterator created with call to _iterator_codegen()
    # this iterator has no state or meminfo filled yet, only parent (i.e. dict),
    # which we use to make actual call
    dict_type = iterablety.parent
    it = context.make_helper(builder, iterablety.iterator_type, iter_val)
    call_native_getiter(context, builder, dict_type, it.parent, it)

    # this is a new NRT managed iterator, so no need to use impl_ret_borrowed
    return it._getvalue()


@lower_builtin('getiter', ConcurrentDictType)
def impl_conc_dict_getiter(context, builder, sig, args):
    dict_type, = sig.args
    dict_val, = args

    iterablety = ConcDictKeysIterableType(dict_type)
    it = context.make_helper(builder, iterablety.iterator_type)
    call_native_getiter(context, builder, dict_type, dict_val, it)

    # this is a new NRT managed iterator, so no need to use impl_ret_borrowed
    return it._getvalue()


@lower_builtin('iternext', ConcDictIteratorType)
@iternext_impl(RefType.BORROWED)
def impl_iterator_iternext(context, builder, sig, args, result):
    iter_type, = sig.args
    iter_val, = args

    dict_type = iter_type.parent
    ty_key, ty_val = dict_type.key_type, dict_type.value_type
    key_type_postfix, value_type_postfix = _get_types_postfixes(ty_key, ty_val)

    native_key_ptr, lir_key_type = alloc_native_value(context, builder, ty_key)
    native_value_ptr, lir_value_type = alloc_native_value(context, builder, ty_val)

    llvoidptr = context.get_value_type(types.voidptr)
    fnty = lir.FunctionType(lir.IntType(8),
                            [llvoidptr,
                             lir_key_type.as_pointer(),
                             lir_value_type.as_pointer()])
    func_name = f"hashmap_iternext_{key_type_postfix}_to_{value_type_postfix}"
    fn_hashmap_iternext = cgutils.get_or_insert_function(builder.module,
        fnty, name=func_name)

    iter_ctinfo = context.make_helper(builder, iter_type, iter_val)
    status = builder.call(fn_hashmap_iternext,
                          [iter_ctinfo.state,
                           native_key_ptr,
                           native_value_ptr])

    # TODO: no handling of error state i.e. mutated dictionary
    #       all errors are treated as exhausted iterator
    is_valid = builder.icmp_unsigned('==', status, status.type(0))
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        yield_type = iter_type.yield_type

        native_key = builder.load(native_key_ptr)
        result_key = transform_native_val(context, builder, ty_key, native_key)

        native_val = builder.load(native_value_ptr)
        result_val = transform_native_val(context, builder, ty_val, native_val)

        # All dict iterators use this common implementation.
        # Their differences are resolved here.
        if isinstance(iter_type.iterable, ConcDictItemsIterableType):
            # .items()
            tup = context.make_tuple(builder, yield_type, [result_key, result_val])
            result.yield_(tup)
        elif isinstance(iter_type.iterable, ConcDictKeysIterableType):
            # .keys()
            result.yield_(result_key)
        elif isinstance(iter_type.iterable, ConcDictValuesIterableType):
            # .values()
            result.yield_(result_val)
        else:
            # unreachable
            raise AssertionError('unknown type: {}'.format(iter_type.iterable))

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


import operator
import numba
from numba import types, typing
from numba.typing.templates import (signature, AbstractTemplate, infer,
                                    ConcreteTemplate, AttributeTemplate, bound_function, infer_global)
from numba.extending import typeof_impl, lower_cast
from numba.extending import type_callable, box, unbox, NativeValue
from numba.extending import models, register_model, infer_getattr
from numba.extending import lower_builtin, overload_method, overload
from numba.targets.imputils import (impl_ret_new_ref, impl_ret_borrowed,
                                    iternext_impl, RefType)
from hpat.str_ext import string_type, gen_unicode_to_std_str, gen_std_str_to_unicode
from numba import cgutils
from llvmlite import ir as lir
import llvmlite.binding as ll
from . import hdict_ext
from hpat.utils import unliteral_all

ll_voidp = lir.IntType(8).as_pointer()


class ByteVecType(types.Opaque):
    def __init__(self):
        super(ByteVecType, self).__init__(
            name='byte_vec')


byte_vec_type = ByteVecType()
register_model(ByteVecType)(models.OpaqueModel)


class DictType(types.Opaque):
    def __init__(self, key_typ, val_typ):
        self.key_typ = key_typ
        self.val_typ = val_typ
        super(DictType, self).__init__(
            name='DictType{}{}'.format(key_typ, val_typ))

    @property
    def key(self):
        return self.key_typ, self.val_typ

    @property
    def iterator_type(self):
        return DictKeyIteratorType(self.key_typ, self.val_typ)

    def is_precise(self):
        return self.key_typ.is_precise() and self.val_typ.is_precise()


elem_types = [
    types.int8,
    types.int16,
    types.int32,
    types.int64,
    types.uint8,
    types.uint16,
    types.uint32,
    types.uint64,
    types.boolean,
    types.float32,
    types.float64,
    string_type
]


def typ_str_to_obj(typ_str):
    if typ_str == types.boolean:
        return "types.boolean"
    if typ_str == string_type:
        return "string_type"
    return "types.{}".format(typ_str)


def _add_dict_symbols(key_str, val_str):
    # init dict object
    exec("ll.add_symbol('dict_{0}_{1}_init', hdict_ext.dict_{0}_{1}_init)".format(key_str, val_str))
    # setitem
    exec("ll.add_symbol('dict_{0}_{1}_setitem', hdict_ext.dict_{0}_{1}_setitem)".format(key_str, val_str))
    # getitem
    exec("ll.add_symbol('dict_{0}_{1}_getitem', hdict_ext.dict_{0}_{1}_getitem)".format(key_str, val_str))
    # in
    exec("ll.add_symbol('dict_{0}_{1}_in', hdict_ext.dict_{0}_{1}_in)".format(key_str, val_str))
    # print
    exec("ll.add_symbol('dict_{0}_{1}_print', hdict_ext.dict_{0}_{1}_print)".format(key_str, val_str))
    # get
    exec("ll.add_symbol('dict_{0}_{1}_get', hdict_ext.dict_{0}_{1}_get)".format(key_str, val_str))
    # pop
    exec("ll.add_symbol('dict_{0}_{1}_pop', hdict_ext.dict_{0}_{1}_pop)".format(key_str, val_str))
    # keys
    exec("ll.add_symbol('dict_{0}_{1}_keys', hdict_ext.dict_{0}_{1}_keys)".format(key_str, val_str))
    # min
    exec("ll.add_symbol('dict_{0}_{1}_min', hdict_ext.dict_{0}_{1}_min)".format(key_str, val_str))
    # max
    exec("ll.add_symbol('dict_{0}_{1}_max', hdict_ext.dict_{0}_{1}_max)".format(key_str, val_str))
    # not_empty
    exec("ll.add_symbol('dict_{0}_{1}_not_empty', hdict_ext.dict_{0}_{1}_not_empty)".format(key_str, val_str))


for key_typ in elem_types:
    for val_typ in elem_types:
        k_obj = typ_str_to_obj(key_typ)
        v_obj = typ_str_to_obj(val_typ)
        key_str = str(key_typ)
        val_str = str(val_typ)
        _add_dict_symbols(key_str, val_str)
        # create types
        exec("dict_{}_{}_type = DictType({}, {})".format(key_str, val_str, k_obj, v_obj))
        exec_format_line = "dict_{0}_{1}_init = types.ExternalFunction('dict_{0}_{1}_init', dict_{0}_{1}_type())"
        exec(exec_format_line.format(key_str, val_str))

dict_byte_vec_int64_type = DictType(byte_vec_type, types.int64)
dict_byte_vec_int64_init = types.ExternalFunction('dict_byte_vec_int64_init', dict_byte_vec_int64_type())
_add_dict_symbols('byte_vec', 'int64')

ll.add_symbol('byte_vec_init', hdict_ext.byte_vec_init)
ll.add_symbol('byte_vec_set', hdict_ext.byte_vec_set)
ll.add_symbol('byte_vec_free', hdict_ext.byte_vec_free)
ll.add_symbol('byte_vec_resize', hdict_ext.byte_vec_resize)

byte_vec_init = types.ExternalFunction('byte_vec_init', byte_vec_type(types.int64, types.voidptr))
byte_vec_set = types.ExternalFunction(
    'byte_vec_set',
    types.void(
        byte_vec_type,
        types.int64,
        types.voidptr,
        types.int64))
byte_vec_resize = types.ExternalFunction('byte_vec_resize', types.void(byte_vec_type, types.int64))
byte_vec_free = types.ExternalFunction('byte_vec_free', types.void(byte_vec_type))


class MultiMapType(types.Opaque):
    def __init__(self, key_typ, val_typ):
        self.key_typ = key_typ
        self.val_typ = val_typ
        super(MultiMapType, self).__init__(
            name='MultiMapType{}{}'.format(key_typ, val_typ))

    @property
    def key(self):
        return self.key_typ, self.val_typ

    def is_precise(self):
        return self.key_typ.is_precise() and self.val_typ.is_precise()


register_model(MultiMapType)(models.OpaqueModel)


class MultiMapRangeIteratorType(types.SimpleIteratorType):
    def __init__(self, key_typ, val_typ):
        self.key_typ = key_typ
        self.val_typ = val_typ
        yield_type = val_typ
        super(MultiMapRangeIteratorType, self).__init__(
            'MultiMapRangeIteratorType{}{}'.format(key_typ, val_typ), yield_type)

        @property
        def iterator_type(self):
            return self

        @property
        def key(self):
            return self.key_typ, self.val_typ

        def is_precise(self):
            return self.key_typ.is_precise() and self.val_typ.is_precise()


multimap_int64_range_iterator_type = MultiMapRangeIteratorType(types.intp, types.intp)

register_model(MultiMapRangeIteratorType)(models.OpaqueModel)

multimap_int64_type = MultiMapType(types.int64, types.int64)
multimap_int64_init = types.ExternalFunction(
    'multimap_int64_init', multimap_int64_type())
multimap_int64_insert = types.ExternalFunction(
    'multimap_int64_insert',
    types.void(multimap_int64_type, types.int64, types.int64))
multimap_int64_equal_range = types.ExternalFunction(
    'multimap_int64_equal_range',
    multimap_int64_range_iterator_type(multimap_int64_type, types.int64))


# store the iterator pair type in same storage and avoid repeated alloc
multimap_int64_equal_range_alloc = types.ExternalFunction(
    'multimap_int64_equal_range_alloc', multimap_int64_range_iterator_type())

multimap_int64_equal_range_dealloc = types.ExternalFunction(
    'multimap_int64_equal_range_dealloc',
    types.void(multimap_int64_range_iterator_type))

multimap_int64_equal_range_inplace = types.ExternalFunction(
    'multimap_int64_equal_range_inplace',
    multimap_int64_range_iterator_type(multimap_int64_type, types.int64,
                                       multimap_int64_range_iterator_type))

ll.add_symbol('multimap_int64_init', hdict_ext.multimap_int64_init)
ll.add_symbol('multimap_int64_insert', hdict_ext.multimap_int64_insert)
ll.add_symbol('multimap_int64_equal_range', hdict_ext.multimap_int64_equal_range)
ll.add_symbol('multimap_int64_equal_range_alloc', hdict_ext.multimap_int64_equal_range_alloc)
ll.add_symbol('multimap_int64_equal_range_dealloc', hdict_ext.multimap_int64_equal_range_dealloc)
ll.add_symbol('multimap_int64_equal_range_inplace', hdict_ext.multimap_int64_equal_range_inplace)
ll.add_symbol('multimap_int64_it_is_valid', hdict_ext.multimap_int64_it_is_valid)
ll.add_symbol('multimap_int64_it_get_value', hdict_ext.multimap_int64_it_get_value)
ll.add_symbol('multimap_int64_it_inc', hdict_ext.multimap_int64_it_inc)


@lower_builtin('getiter', MultiMapRangeIteratorType)
def iterator_getiter(context, builder, sig, args):
    it, = args
    # return impl_ret_borrowed(context, builder, sig.return_type, it)
    return it


@lower_builtin('iternext', MultiMapRangeIteratorType)
@iternext_impl(RefType.UNTRACKED)
def iternext_listiter(context, builder, sig, args, result):
    ll_bool = context.get_value_type(types.bool_)  # lir.IntType(1)?

    # is valid
    fnty = lir.FunctionType(ll_bool, [ll_voidp])
    it_is_valid = builder.module.get_or_insert_function(fnty, name="multimap_int64_it_is_valid")

    # get value
    val_typ = context.get_value_type(sig.args[0].val_typ)
    fnty = lir.FunctionType(val_typ, [ll_voidp])
    get_value = builder.module.get_or_insert_function(fnty, name="multimap_int64_it_get_value")

    # increment
    fnty = lir.FunctionType(lir.VoidType(), [ll_voidp])
    inc_it = builder.module.get_or_insert_function(fnty, name="multimap_int64_it_inc")

    range_it, = args

    # it != range.second
    is_valid = builder.call(it_is_valid, [range_it])
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        # it->second
        val = builder.call(get_value, [range_it])
        result.yield_(val)
        builder.call(inc_it, [range_it])


# XXX: needs Numba #3014 resolved
# @overload("in")
# def in_dict(key_typ, dict_typ):
#     def f(k, dict_int):
#         return dict_int_int_in(dict_int, k)
#     return f

# XXX possible overload bug
# @overload(operator.setitem)
# def setitem_dict(dict_typ, key_typ, val_typ):
#     def f(k, dict_int):
#         return dict_int_int_in(dict_int, k)
#     return f

@infer
class InDict(AbstractTemplate):
    key = "in"

    def generic(self, args, kws):
        _, cont = args
        if isinstance(cont, DictType):
            return signature(types.boolean, cont.key_typ, cont)


@infer_global(operator.contains)
class InDictOp(AbstractTemplate):
    def generic(self, args, kws):
        # contains operator reverses the args
        cont, _ = args
        if isinstance(cont, DictType):
            return signature(types.boolean, cont, cont.key_typ)


dict_int_int_type = DictType(types.intc, types.intc)
dict_int32_int32_type = DictType(types.int32, types.int32)


class DictIntInt(object):
    def __new__(cls, *args):
        return {}


class DictInt32Int32(object):
    def __new__(cls, *args):
        return {}


@typeof_impl.register(DictIntInt)
def typeof_dict_int(val, c):
    return dict_int_int_type


@typeof_impl.register(DictInt32Int32)
def typeof_dict_int32(val, c):
    return dict_int32_int32_type


@type_callable(DictIntInt)
def type_dict_int(context):
    def typer():
        return dict_int_int_type
    return typer


@type_callable(DictInt32Int32)
def type_dict_int32(context):
    def typer():
        return dict_int32_int32_type
    return typer


@infer_global(operator.setitem)
class SetItemDict(AbstractTemplate):
    def generic(self, args, kws):
        dict_t, _, _ = args
        if isinstance(dict_t, DictType):
            return signature(types.none, dict_t, dict_t.key_typ, dict_t.val_typ)


@infer_global(operator.getitem)
class GetItemDict(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        dict_t, _ = args
        if isinstance(dict_t, DictType):
            return signature(dict_t.val_typ, dict_t, dict_t.key_typ)


@infer
class PrintDictIntInt(ConcreteTemplate):
    key = "print_item"
    cases = [signature(types.none, dict_int_int_type),
             signature(types.none, dict_int32_int32_type)]


@infer_getattr
class DictAttribute(AttributeTemplate):
    key = DictType

    @bound_function("dict.get")
    def resolve_get(self, dict, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(args[1], *unliteral_all(args))

    @bound_function("dict.pop")
    def resolve_pop(self, dict, args, kws):
        assert not kws
        return signature(dict.val_typ, *unliteral_all(args))

    @bound_function("dict.keys")
    def resolve_keys(self, dict, args, kws):
        assert not kws
        return signature(DictKeyIteratorType(dict.key_typ, dict.val_typ))


register_model(DictType)(models.OpaqueModel)


@box(DictType)
def box_dict(typ, val, c):
    """
    """
    # interval = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    # lo_obj = c.pyapi.float_from_double(interval.lo)
    # hi_obj = c.pyapi.float_from_double(interval.hi)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(DictIntInt))
    res = c.pyapi.call_function_objargs(class_obj, (val,))
    # c.pyapi.decref(lo_obj)
    # c.pyapi.decref(hi_obj)
    c.pyapi.decref(class_obj)
    return res


class DictKeyIteratorType(types.Opaque):
    def __init__(self, key_typ, val_typ):
        self.key_typ = key_typ
        self.val_typ = val_typ
        super(DictKeyIteratorType, self).__init__(
            'DictKeyIteratorType{}{}'.format(key_typ, val_typ))


dict_key_iterator_int_int_type = DictKeyIteratorType(types.intp, types.intp)
dict_key_iterator_int32_int32_type = DictKeyIteratorType(
    types.int32, types.int32)

register_model(DictKeyIteratorType)(models.OpaqueModel)


@infer_global(min)
@infer_global(max)
class MinMaxDict(AbstractTemplate):
    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], DictKeyIteratorType):
            return signature(args[0].key_typ, *unliteral_all(args))


# dict_int_int_in = types.ExternalFunction("dict_int_int_in", types.boolean(dict_int_int_type, types.intp))

@lower_builtin(DictIntInt)
def impl_dict_int_int(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [])
    fn = builder.module.get_or_insert_function(fnty, name="dict_int_int_init")
    return builder.call(fn, [])


@lower_builtin(operator.setitem, DictType, types.Any, types.Any)
def setitem_dict(context, builder, sig, args):
    _, key_typ, val_typ = sig.args
    dct, key, val = args
    fname = "dict_{}_{}_setitem".format(key_typ, val_typ)

    if key_typ == string_type:
        key_typ = types.voidptr
        key = gen_unicode_to_std_str(context, builder, key)

    if val_typ == string_type:
        val_typ = types.voidptr
        val = gen_unicode_to_std_str(context, builder, val)

    fnty = lir.FunctionType(lir.VoidType(),
                            [lir.IntType(8).as_pointer(),
                             context.get_value_type(key_typ),
                             context.get_value_type(val_typ)])
    fn = builder.module.get_or_insert_function(fnty, name=fname)
    return builder.call(fn, [dct, key, val])


@lower_builtin("print_item", dict_int_int_type)
def print_dict(context, builder, sig, args):
    # pyapi = context.get_python_api(builder)
    # strobj = pyapi.unserialize(pyapi.serialize_object("hello!"))
    # pyapi.print_object(strobj)
    # pyapi.decref(strobj)
    # return context.get_dummy_value()
    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="dict_int_int_print")
    return builder.call(fn, args)


@lower_builtin("dict.get", DictType, types.intp, types.intp)
def lower_dict_get(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(
        8).as_pointer(), lir.IntType(64), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="dict_int_int_get")
    return builder.call(fn, args)


@lower_builtin(operator.getitem, DictType, types.Any)
def lower_dict_getitem(context, builder, sig, args):
    dict_typ, key_typ = sig.args
    dct, key = args
    val_typ = dict_typ.val_typ

    fname = "dict_{}_{}_getitem".format(key_typ, val_typ)

    if key_typ == string_type:
        key_typ = types.voidptr
        key = gen_unicode_to_std_str(context, builder, key)

    ll_val_typ = context.get_value_type(val_typ)
    if val_typ == string_type:
        ll_val_typ = context.get_value_type(types.voidptr)

    fnty = lir.FunctionType(ll_val_typ,
                            [lir.IntType(8).as_pointer(), context.get_value_type(key_typ)])

    fn = builder.module.get_or_insert_function(fnty, name=fname)
    val = builder.call(fn, [dct, key])
    if val_typ == string_type:
        val = gen_std_str_to_unicode(context, builder, val)
    return val


@lower_builtin("dict.pop", DictType, types.intp)
def lower_dict_pop(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(
        64), [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="dict_int_int_pop")
    return builder.call(fn, args)


@lower_builtin("dict.keys", dict_int_int_type)
def lower_dict_keys(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [
                            lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="dict_int_int_keys")
    return builder.call(fn, args)


@lower_builtin(min, dict_key_iterator_int_int_type)
def lower_dict_min(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="dict_int_int_min")
    return builder.call(fn, args)


@lower_builtin(max, dict_key_iterator_int_int_type)
def lower_dict_max(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="dict_int_int_max")
    return builder.call(fn, args)


@lower_builtin("in", types.Any, DictType)
def lower_dict_in(context, builder, sig, args):
    key_typ, dict_typ = sig.args
    key, dct = args

    fname = "dict_{}_{}_in".format(key_typ, dict_typ.val_typ)

    if key_typ == string_type:
        key_typ = types.voidptr
        key = gen_unicode_to_std_str(context, builder, key)

    fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer(),
                                             context.get_value_type(key_typ), ])
    fn = builder.module.get_or_insert_function(fnty, name=fname)
    val = builder.call(fn, [dct, key])
    if dict_typ.val_typ == string_type:
        val = gen_std_str_to_unicode(context, builder, val)
    return val


@lower_builtin(operator.contains, DictType, types.Any)
def lower_dict_in_op(context, builder, sig, args):
    dict_typ, key_typ = sig.args
    dct, key = args

    fname = "dict_{}_{}_in".format(key_typ, dict_typ.val_typ)

    if key_typ == string_type:
        key_typ = types.voidptr
        key = gen_unicode_to_std_str(context, builder, key)

    fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer(),
                                             context.get_value_type(key_typ), ])
    fn = builder.module.get_or_insert_function(fnty, name=fname)
    return builder.call(fn, [dct, key])


@lower_cast(dict_int_int_type, types.boolean)
def dict_empty(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int_int_not_empty")
    return builder.call(fn, (val,))


# ------ int32 versions ------
@lower_builtin(DictInt32Int32)
def impl_dict_int32_int32(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_init")
    return builder.call(fn, [])


# @lower_builtin(operator.setitem, DictType, types.int32, types.int32)
# def setitem_dict_int32(context, builder, sig, args):
#     fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(
#         8).as_pointer(), lir.IntType(32), lir.IntType(32)])
#     fn = builder.module.get_or_insert_function(
#         fnty, name="dict_int32_int32_setitem")
#     return builder.call(fn, args)


@lower_builtin("print_item", dict_int32_int32_type)
def print_dict_int32(context, builder, sig, args):
    # pyapi = context.get_python_api(builder)
    # strobj = pyapi.unserialize(pyapi.serialize_object("hello!"))
    # pyapi.print_object(strobj)
    # pyapi.decref(strobj)
    # return context.get_dummy_value()
    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_print")
    return builder.call(fn, args)


@lower_builtin("dict.get", DictType, types.int32, types.int32)
def lower_dict_get_int32(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(
        8).as_pointer(), lir.IntType(32), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_get")
    return builder.call(fn, args)


# @lower_builtin(operator.getitem, DictType, types.int32)
# def lower_dict_getitem_int32(context, builder, sig, args):
#     fnty = lir.FunctionType(lir.IntType(
#         32), [lir.IntType(8).as_pointer(), lir.IntType(32)])
#     fn = builder.module.get_or_insert_function(
#         fnty, name="dict_int32_int32_getitem")
#     return builder.call(fn, args)


@lower_builtin("dict.pop", DictType, types.int32)
def lower_dict_pop_int32(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(
        32), [lir.IntType(8).as_pointer(), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_pop")
    return builder.call(fn, args)


@lower_builtin("dict.keys", dict_int32_int32_type)
def lower_dict_keys_int32(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [
                            lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_keys")
    return builder.call(fn, args)


@lower_builtin(min, dict_key_iterator_int32_int32_type)
def lower_dict_min_int32(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_min")
    return builder.call(fn, args)


@lower_builtin(max, dict_key_iterator_int32_int32_type)
def lower_dict_max_int32(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_max")
    return builder.call(fn, args)


@lower_cast(dict_int32_int32_type, types.boolean)
def dict_empty_int32(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_not_empty")
    return builder.call(fn, (val,))

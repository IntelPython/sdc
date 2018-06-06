import numba
from numba import types, typing
from numba.typing.templates import (signature, AbstractTemplate, infer,
                                    ConcreteTemplate, AttributeTemplate, bound_function, infer_global)
from numba.extending import typeof_impl, lower_cast
from numba.extending import type_callable, box, unbox, NativeValue
from numba.extending import models, register_model, infer_getattr
from numba.extending import lower_builtin, overload_method, overload
from hpat.str_ext import string_type
from numba import cgutils
from llvmlite import ir as lir
import llvmlite.binding as ll
import hdict_ext


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

for key_typ in []:#elem_types:
    for val_typ in []:#elem_types:
        k_obj = typ_str_to_obj(key_typ)
        v_obj = typ_str_to_obj(val_typ)
        # create types
        exec("dict_{}_{}_type = DictType({}, {})".format(key_typ, val_typ,
                                                                k_obj, v_obj))
        # init dict object
        exec("ll.add_symbol('init_dict_{}_{}', hdict_ext.init_dict_{}_{})".format(
            key_typ, val_typ, key_typ, val_typ))
        exec("init_dict_{}_{} = types.ExternalFunction('init_dict_{}_{}', dict_{}_{}_type())".format(
            key_typ, val_typ, key_typ, val_typ, key_typ, val_typ))
        # setitem
        exec("ll.add_symbol('dict_setitem_{}_{}', hdict_ext.dict_setitem_{}_{})".format(
            key_typ, val_typ, key_typ, val_typ))



# XXX: needs Numba #3014 resolved
# @overload("in")
# def in_dict(key_typ, dict_typ):
#     def f(k, dict_int):
#         return dict_int_int_in(dict_int, k)
#     return f

# XXX possible overload bug
# @overload("setitem")
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

dict_int_int_type = DictType(types.intp, types.intp)
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


@infer
class SetItemDict(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        dict_t, idx, value = args
        if isinstance(dict_t, DictType):
            return signature(types.none, dict_t, dict_t.key_typ, dict_t.val_typ)


@infer
class GetItemDict(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        dict_t, idx = args
        if isinstance(dict_t, DictType):
            if isinstance(idx, types.Integer):
                return signature(dict_t.val_typ, dict_t, idx)


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
        return signature(args[1], *args)

    @bound_function("dict.pop")
    def resolve_pop(self, dict, args, kws):
        assert not kws
        return signature(dict.val_typ, *args)

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
            return signature(args[0].key_typ, *args)


ll.add_symbol('init_dict_int_int', hdict_ext.init_dict_int_int)
ll.add_symbol('dict_int_int_setitem', hdict_ext.dict_int_int_setitem)
ll.add_symbol('dict_int_int_print', hdict_ext.dict_int_int_print)
ll.add_symbol('dict_int_int_get', hdict_ext.dict_int_int_get)
ll.add_symbol('dict_int_int_getitem', hdict_ext.dict_int_int_getitem)
ll.add_symbol('dict_int_int_pop', hdict_ext.dict_int_int_pop)
ll.add_symbol('dict_int_int_keys', hdict_ext.dict_int_int_keys)
ll.add_symbol('dict_int_int_min', hdict_ext.dict_int_int_min)
ll.add_symbol('dict_int_int_max', hdict_ext.dict_int_int_max)
ll.add_symbol('dict_int_int_in', hdict_ext.dict_int_int_in)
ll.add_symbol('dict_int_int_not_empty', hdict_ext.dict_int_int_not_empty)

# int32 versions
ll.add_symbol('init_dict_int32_int32', hdict_ext.init_dict_int32_int32)
ll.add_symbol('dict_int32_int32_setitem', hdict_ext.dict_int32_int32_setitem)
ll.add_symbol('dict_int32_int32_print', hdict_ext.dict_int32_int32_print)
ll.add_symbol('dict_int32_int32_get', hdict_ext.dict_int32_int32_get)
ll.add_symbol('dict_int32_int32_getitem', hdict_ext.dict_int32_int32_getitem)
ll.add_symbol('dict_int32_int32_pop', hdict_ext.dict_int32_int32_pop)
ll.add_symbol('dict_int32_int32_keys', hdict_ext.dict_int32_int32_keys)
ll.add_symbol('dict_int32_int32_min', hdict_ext.dict_int32_int32_min)
ll.add_symbol('dict_int32_int32_max', hdict_ext.dict_int32_int32_max)
ll.add_symbol('dict_int32_int32_not_empty',
              hdict_ext.dict_int32_int32_not_empty)


# dict_int_int_in = types.ExternalFunction("dict_int_int_in", types.boolean(dict_int_int_type, types.intp))

@lower_builtin(DictIntInt)
def impl_dict_int_int(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [])
    fn = builder.module.get_or_insert_function(fnty, name="init_dict_int_int")
    return builder.call(fn, [])


@lower_builtin('setitem', DictType, types.Any, types.Any)
def setitem_dict(context, builder, sig, args):
    dict_typ, key_typ, val_typ = sig.args
    fname = "dict_setitem_{}_{}".format(key_typ, val_typ)
    fnty = lir.FunctionType(lir.VoidType(),
        [lir.IntType(8).as_pointer(),
        context.get_value_type(key_typ),
        context.get_value_type(val_typ)])
    fn = builder.module.get_or_insert_function(fnty, name=fname)
    return builder.call(fn, args)


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


@lower_builtin("getitem", DictType, types.intp)
def lower_dict_getitem(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(
        64), [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int_int_getitem")
    return builder.call(fn, args)


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

@lower_builtin("in", types.int64, dict_int_int_type)
def lower_dict_in(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer(),
                                                lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="dict_int_int_in")
    return builder.call(fn, [args[1], args[0]])


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
        fnty, name="init_dict_int32_int32")
    return builder.call(fn, [])


# @lower_builtin('setitem', DictType, types.int32, types.int32)
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


@lower_builtin("getitem", DictType, types.int32)
def lower_dict_getitem_int32(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(
        32), [lir.IntType(8).as_pointer(), lir.IntType(32)])
    fn = builder.module.get_or_insert_function(
        fnty, name="dict_int32_int32_getitem")
    return builder.call(fn, args)


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

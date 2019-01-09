import operator
import numba
from numba import types, typing, generated_jit
from numba.extending import box, unbox, NativeValue
from numba.extending import models, register_model
from numba.extending import lower_builtin, overload_method, overload, intrinsic
from numba.targets.imputils import (impl_ret_new_ref, impl_ret_borrowed,
                                    iternext_impl, impl_ret_untracked)
from numba import cgutils
from numba.typing.templates import signature, AbstractTemplate, infer, infer_global

from llvmlite import ir as lir
import llvmlite.binding as ll
import hset_ext
ll.add_symbol('init_set_string', hset_ext.init_set_string)
ll.add_symbol('insert_set_string', hset_ext.insert_set_string)
ll.add_symbol('len_set_string', hset_ext.len_set_string)
ll.add_symbol('set_in_string', hset_ext.set_in_string)
ll.add_symbol('set_iterator_string', hset_ext.set_iterator_string)
ll.add_symbol('set_itervalid_string', hset_ext.set_itervalid_string)
ll.add_symbol('set_nextval_string', hset_ext.set_nextval_string)
ll.add_symbol('num_total_chars_set_string', hset_ext.num_total_chars_set_string)
ll.add_symbol('populate_str_arr_from_set', hset_ext.populate_str_arr_from_set)

import hpat
from hpat.utils import to_array
from hpat.str_ext import string_type
from hpat.str_arr_ext import (StringArray, StringArrayType, string_array_type,
                              pre_alloc_string_array, StringArrayPayloadType,
                              is_str_arr_typ)


# similar to types.Container.Set
class SetType(types.Container):
    def __init__(self, dtype):
        self.dtype = dtype
        super(SetType, self).__init__(
            name='SetType({})'.format(dtype))

    @property
    def key(self):
        return self.dtype

    @property
    def iterator_type(self):
        return SetIterType(self)

    def is_precise(self):
        return self.dtype.is_precise()

set_string_type = SetType(string_type)


class SetIterType(types.BaseContainerIterator):
    container_class = SetType


register_model(SetType)(models.OpaqueModel)


_init_set_string = types.ExternalFunction("init_set_string",
                                         set_string_type())

def init_set_string():
    return set()

@overload(init_set_string)
def init_set_overload():
    return lambda: _init_set_string()

add_set_string = types.ExternalFunction("insert_set_string",
                                    types.void(set_string_type, string_type))

len_set_string = types.ExternalFunction("len_set_string",
                                    types.intp(set_string_type))

num_total_chars_set_string = types.ExternalFunction("num_total_chars_set_string",
                                    types.intp(set_string_type))

# TODO: box set(string)


@generated_jit(nopython=True, cache=True)
def build_set(A):
    if is_str_arr_typ(A):
        return _build_str_set_impl
    else:
        return lambda A: set(A)


def _build_str_set_impl(A):
    str_arr = hpat.hiframes_api.dummy_unbox_series(A)
    str_set = init_set_string()
    n = len(str_arr)
    for i in range(n):
        str = str_arr[i]
        str_set.add(str)
    return str_set

# TODO: remove since probably unused
@overload(set)
def init_set_string_array(in_typ):
    if is_str_arr_typ(in_typ):
        return _build_str_set_impl


@overload_method(SetType, 'add')
def set_add_overload(set_obj_typ, item_typ):
    # TODO: expand to other set types
    assert set_obj_typ == set_string_type and item_typ == string_type
    def add_impl(set_obj, item):
        return add_set_string(set_obj, item)
    return add_impl

@overload(len)
def len_set_str_overload(in_typ):
    if in_typ == set_string_type:
        def len_impl(str_set):
            return len_set_string(str_set)
        return len_impl

# FIXME: overload fails in lowering sometimes!
@lower_builtin(len, set_string_type)
def lower_len_set_impl(context, builder, sig, args):

    def len_impl(str_set):
        return len_set_string(str_set)

    res = context.compile_internal(builder, len_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@infer
class InSet(AbstractTemplate):
    key = "in"

    def generic(self, args, kws):
        _, cont_typ = args
        if cont_typ == set_string_type:
            return signature(types.boolean, cont_typ.dtype, cont_typ)


@infer_global(operator.contains)
class InSetOp(AbstractTemplate):
    def generic(self, args, kws):
        cont_typ, _ = args
        if cont_typ == set_string_type:
            return signature(types.boolean, cont_typ, cont_typ.dtype)


@lower_builtin("in", string_type, set_string_type)
def lower_dict_in(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer(),
                                                lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="set_in_string")
    return builder.call(fn, args)

@lower_builtin(operator.contains, set_string_type, string_type)
def lower_dict_in_op(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1), [lir.IntType(8).as_pointer(),
                                                lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="set_in_string")
    return builder.call(fn, [args[1], args[0]])


@overload(to_array)
def to_array_overload(in_typ):
    if in_typ == set_string_type:
        #
        def set_string_to_array(str_set):
            num_total_chars = num_total_chars_set_string(str_set)
            num_strs = len(str_set)
            str_arr = pre_alloc_string_array(num_strs, num_total_chars)
            populate_str_arr_from_set(str_set, str_arr)
            return str_arr

        return set_string_to_array

@intrinsic
def populate_str_arr_from_set(typingctx, in_set_typ, in_str_arr_typ=None):
    assert in_set_typ == set_string_type
    assert is_str_arr_typ(in_str_arr_typ)
    def codegen(context, builder, sig, args):
        in_set, in_str_arr = args

        string_array = context.make_helper(builder, string_array_type, in_str_arr)

        fnty = lir.FunctionType( lir.VoidType(),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(32).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                ])
        fn_getitem = builder.module.get_or_insert_function(fnty,
                                                           name="populate_str_arr_from_set")
        builder.call(fn_getitem, [in_set, string_array.offsets,
                                         string_array.data])
        return context.get_dummy_value()

    return types.void(set_string_type, string_array_type), codegen

# TODO: delete iterator

@register_model(SetIterType)
class StrSetIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [('itp', types.Opaque('SetIterPtr')),
                   ('set', set_string_type)]
        super(StrSetIteratorModel, self).__init__(dmm, fe_type, members)


@lower_builtin('getiter', SetType)
def getiter_set(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                                                [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="set_iterator_string")
    itp = builder.call(fn, args)

    iterobj = context.make_helper(builder, sig.return_type)

    iterobj.itp = itp
    iterobj.set = args[0]

    return iterobj._getvalue()


@lower_builtin('iternext', SetIterType)
@iternext_impl
def iternext_setiter(context, builder, sig, args, result):
    iterty, = sig.args
    it, = args
    iterobj = context.make_helper(builder, iterty, value=it)

    fnty = lir.FunctionType(lir.IntType(1),
                    [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="set_itervalid_string")
    is_valid = builder.call(fn, [iterobj.itp, iterobj.set])
    result.set_valid(is_valid)

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                    [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="set_nextval_string")

    with builder.if_then(is_valid):
        val = builder.call(fn, [iterobj.itp])
        result.yield_(val)

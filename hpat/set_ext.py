import numba
from numba import types, typing
from numba.extending import box, unbox, NativeValue
from numba.extending import models, register_model
from numba.extending import lower_builtin, overload_method, overload, intrinsic
from numba import cgutils

from llvmlite import ir as lir
import llvmlite.binding as ll
import hset_ext
ll.add_symbol('init_set_string', hset_ext.init_set_string)
ll.add_symbol('insert_set_string', hset_ext.insert_set_string)
ll.add_symbol('len_set_string', hset_ext.len_set_string)
ll.add_symbol('num_total_chars_set_string', hset_ext.num_total_chars_set_string)
ll.add_symbol('populate_str_arr_from_set', hset_ext.populate_str_arr_from_set)

import hpat
from hpat.utils import to_array
from hpat.str_ext import StringType, string_type
from hpat.str_arr_ext import (StringArray, StringArrayType, string_array_type,
                              pre_alloc_string_array, StringArrayPayloadType)

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

init_set_string = types.ExternalFunction("init_set_string",
                                         set_string_type())

add_set_string = types.ExternalFunction("insert_set_string",
                                    types.void(set_string_type, string_type))

len_set_string = types.ExternalFunction("len_set_string",
                                    types.intp(set_string_type))

num_total_chars_set_string = types.ExternalFunction("num_total_chars_set_string",
                                    types.intp(set_string_type))

# TODO: box set(string)

@overload(set)
def init_set_string_array(in_typ):
    if in_typ == string_array_type:
        def f(str_arr):
            str_set = init_set_string()
            n = len(str_arr)
            for i in range(n):
                str = str_arr[i]
                str_set.add(str)
                hpat.str_ext.del_str(str)
            return str_set
        return f


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
def populate_str_arr_from_set(typingctx, in_set_typ, in_str_arr_typ):
    assert in_set_typ == set_string_type
    assert in_str_arr_typ == string_array_type
    def codegen(context, builder, sig, args):
        in_set, in_str_arr = args
        dtype = StringArrayPayloadType()

        inst_struct = context.make_helper(builder, string_array_type, in_str_arr)
        data_pointer = context.nrt.meminfo_data(builder, inst_struct.meminfo)
        data_pointer = builder.bitcast(data_pointer,
                                       context.get_data_type(dtype).as_pointer())

        string_array = cgutils.create_struct_proxy(dtype)(context, builder, builder.load(data_pointer))
        fnty = lir.FunctionType( lir.VoidType(),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                ])
        fn_getitem = builder.module.get_or_insert_function(fnty,
                                                           name="populate_str_arr_from_set")
        builder.call(fn_getitem, [in_set, string_array.offsets,
                                         string_array.data])
        return context.get_dummy_value()

    return types.void(set_string_type, string_array_type), codegen

#
# TODO: implement iterator
# @register_model(SetIterType)
# class SetIterTypeModel(models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [
#             ('set', SetType),
#             ('index', types.int64),
#         ]
#         models.StructModel.__init__(self, dmm, fe_type, members)

#
# class SetTypeIterInstance(_SetPayloadMixin):
#
#     def __init__(self, context, builder, iter_type, iter_val):
#         self._context = context
#         self._builder = builder
#         self._ty = iter_type
#         self._iter = context.make_helper(builder, iter_type, iter_val)
#         self._datamodel = context.data_model_manager[iter_type.yield_type]
#
#     @classmethod
#     def from_set(cls, context, builder, iter_type, set_val):
#         set_inst = SetInstance(context, builder, iter_type.container, set_val)
#         self = cls(context, builder, iter_type, None)
#         index = context.get_constant(types.intp, 0)
#         self._iter.index = cgutils.alloca_once_value(builder, index)
#         self._iter.meminfo = set_inst.meminfo
#         return self
#
#     @property
#     def _payload(self):
#         # This cannot be cached as it can be reallocated
#         return get_set_payload(self._context, self._builder,
#                                 self._ty.container, self._iter)
#
#     @property
#     def value(self):
#         return self._iter._getvalue()
#
#     @property
#     def index(self):
#         return self._builder.load(self._iter.index)
#
#     @index.setter
#     def index(self, value):
#         self._builder.store(value, self._iter.index)
#
#
# @lower_builtin('getiter', SetType)
# def getiter_set(context, builder, sig, args):
#     inst = SetTypeIterInstance.from_set(context, builder, sig.return_type, args[0])
#     return impl_ret_borrowed(context, builder, sig.return_type, inst.value)
#
# @lower_builtin('iternext', types.SetIter)
# @iternext_impl
# def iternext_setiter(context, builder, sig, args, result):
#     inst = SetTypeIterInstance(context, builder, sig.args[0], args[0])
#
#     index = inst.index
#     nitems = inst.size
#     is_valid = builder.icmp_signed('<', index, nitems)
#     result.set_valid(is_valid)
#
#     with builder.if_then(is_valid):
#         result.yield_(inst.getitem(index))
#         inst.index = builder.add(index, context.get_constant(types.intp, 1))

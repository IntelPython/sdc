import operator
import numpy as np
import numba
import hpat
from numba import types
from numba.typing.templates import (infer_global, AbstractTemplate, infer,
    signature, AttributeTemplate, infer_getattr, bound_function)
import numba.typing.typeof
from numba.extending import (typeof_impl, type_callable, models, register_model, NativeValue,
                             make_attribute_wrapper, lower_builtin, box, unbox,
                             lower_getattr, intrinsic, overload_method, overload, overload_attribute)
from numba import cgutils
from hpat.str_ext import string_type
from numba.targets.imputils import (impl_ret_new_ref, impl_ret_borrowed,
    iternext_impl, RefType)
from hpat.str_arr_ext import (string_array_type, get_data_ptr,
    is_str_arr_typ, pre_alloc_string_array)

import llvmlite.llvmpy.core as lc
from llvmlite import ir as lir
import llvmlite.binding as ll

from .. import hstr_ext
ll.add_symbol('dtor_str_arr_split_view', hstr_ext.dtor_str_arr_split_view)
ll.add_symbol('str_arr_split_view_impl', hstr_ext.str_arr_split_view_impl)

char_typ = types.uint8
offset_typ = types.uint32

data_ctypes_type = types.ArrayCTypes(types.Array(char_typ, 1, 'C'))
offset_ctypes_type = types.ArrayCTypes(types.Array(offset_typ, 1, 'C'))


# nested offset structure to represent S.str.split()
# data_offsets array includes offsets to character data array
# index_offsets array includes offsets to data_offsets array to identify lists
class StringArraySplitViewType(types.IterableType):
    def __init__(self):
        super(StringArraySplitViewType, self).__init__(
            name='StringArraySplitViewType()')

    @property
    def dtype(self):
        # TODO: optimized list type
        return types.List(string_type)

    # TODO
    @property
    def iterator_type(self):
        return #StringArrayIterator()

    def copy(self):
        return StringArraySplitViewType()


string_array_split_view_type = StringArraySplitViewType()



class StringArraySplitViewPayloadType(types.Type):
    def __init__(self):
        super(StringArraySplitViewPayloadType, self).__init__(
            name='StringArraySplitViewPayloadType()')


str_arr_split_view_payload_type = StringArraySplitViewPayloadType()


# XXX: C equivalent in _str_ext.cpp
@register_model(StringArraySplitViewPayloadType)
class StringArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('index_offsets', types.CPointer(offset_typ)),
            ('data_offsets', types.CPointer(offset_typ)),
            #('null_bitmap', types.CPointer(char_typ)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)

str_arr_model_members = [
    ('num_items', types.uint64),  # number of lists
    #('num_total_strs', types.uint64),  # number of strings total
    #('num_total_chars', types.uint64),
    ('index_offsets', types.CPointer(offset_typ)),
    ('data_offsets', types.CPointer(offset_typ)),
    ('data', data_ctypes_type),
    #('null_bitmap', types.CPointer(char_typ)),
    ('meminfo', types.MemInfoPointer(str_arr_split_view_payload_type)),
]

@register_model(StringArraySplitViewType)
class StringArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        models.StructModel.__init__(self, dmm, fe_type, str_arr_model_members)


def construct_str_arr_split_view(context, builder):
    """Creates meminfo and sets dtor.
    """
    alloc_type = context.get_data_type(str_arr_split_view_payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = lir.FunctionType(lir.VoidType(),
                                  [llvoidptr, llsize, llvoidptr])
    dtor_fn = builder.module.get_or_insert_function(
        dtor_ftype, name="dtor_str_arr_split_view")

    meminfo = context.nrt.meminfo_alloc_dtor(
        builder,
        context.get_constant(types.uintp, alloc_size),
        dtor_fn,
    )
    meminfo_data_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_data_ptr,
                                   alloc_type.as_pointer())

    # Nullify all data
    # builder.store( cgutils.get_null_value(alloc_type),
    #             meminfo_data_ptr)
    return meminfo, meminfo_data_ptr

@intrinsic
def compute_split_view(typingctx, str_arr_typ, sep_typ=None):
    assert str_arr_typ == string_array_type and isinstance(sep_typ, types.StringLiteral)
    def codegen(context, builder, sig, args):
        str_arr, _ = args
        meminfo, meminfo_data_ptr = construct_str_arr_split_view(
            context, builder)

        in_str_arr = context.make_helper(
            builder, string_array_type, str_arr)

        # (str_arr_split_view_payload* out_view, int64_t n_strs,
        #  uint32_t* offsets, char* data, char sep)
        fnty = lir.FunctionType(lir.VoidType(),
                                [meminfo_data_ptr.type,
                                 lir.IntType(64),
                                 lir.IntType(32).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(8)])
        fn_impl = builder.module.get_or_insert_function(
            fnty, name="str_arr_split_view_impl")

        sep_val = context.get_constant(types.int8, ord(sep_typ.literal_value))
        builder.call(fn_impl,
            [meminfo_data_ptr, in_str_arr.num_items,
            in_str_arr.offsets, in_str_arr.data, sep_val])

        view_payload = cgutils.create_struct_proxy(
            str_arr_split_view_payload_type)(
            context, builder, value=builder.load(meminfo_data_ptr))

        out_view = context.make_helper(builder, string_array_split_view_type)
        out_view.num_items = in_str_arr.num_items
        out_view.index_offsets = view_payload.index_offsets
        out_view.data_offsets = view_payload.data_offsets
        out_view.data = context.compile_internal(
            builder, lambda S: get_data_ptr(S),
            data_ctypes_type(string_array_type), [str_arr])
        # out_view.null_bitmap = view_payload.null_bitmap
        out_view.meminfo = meminfo
        ret = out_view._getvalue()
        #context.nrt.decref(builder, ty, ret)

        return impl_ret_new_ref(
            context, builder, string_array_split_view_type, ret)

    return string_array_split_view_type(
        string_array_type, sep_typ), codegen


@box(StringArraySplitViewType)
def box_str_arr_split_view(typ, val, c):
    context = c.context
    builder = c.builder
    sp_view = context.make_helper(builder, string_array_split_view_type, val)

    # create array of objects with num_items shape
    mod_name = c.context.insert_const_string(c.builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype = c.pyapi.object_getattr_string(np_class_obj, 'object_')
    l_num_items = builder.sext(sp_view.num_items, c.pyapi.longlong)
    num_items_obj = c.pyapi.long_from_longlong(l_num_items)
    out_arr = c.pyapi.call_method(
        np_class_obj, "ndarray", (num_items_obj, dtype))


    # for each string
    with cgutils.for_range(builder, sp_view.num_items) as loop:
        str_ind = loop.index
        # start and end offset of string's list in index_offsets
        # sp_view.index_offsets[str_ind]
        list_start_offset = builder.sext(builder.load(builder.gep(sp_view.index_offsets, [str_ind])), lir.IntType(64))
        # sp_view.index_offsets[str_ind+1]
        list_end_offset = builder.sext(builder.load(builder.gep(sp_view.index_offsets, [builder.add(str_ind, str_ind.type(1))])), lir.IntType(64))
        # cgutils.printf(builder, "%d %d\n", list_start, list_end)

        # Build a new Python list
        nitems = builder.sub(list_end_offset, list_start_offset)
        cgutils.printf(builder, "str %lld n %lld\n", str_ind, nitems)
        list_obj = c.pyapi.list_new(nitems)
        with c.builder.if_then(cgutils.is_not_null(c.builder, list_obj),
                               likely=True):
            with cgutils.for_range(c.builder, nitems) as loop:
                # data_offsets of current list
                start_index = builder.add(list_start_offset, loop.index)
                data_start = builder.load(builder.gep(sp_view.data_offsets, [start_index]))
                # add 1 since starts from -1
                data_start = builder.add(data_start, data_start.type(1))
                data_end = builder.load(builder.gep(sp_view.data_offsets, [builder.add(start_index, start_index.type(1))]))
                cgutils.printf(builder, "ind %lld %lld\n", data_start, data_end)
                #itemobj =
                #c.pyapi.list_setitem(obj, loop.index, itemobj)



    c.pyapi.decref(np_class_obj)
    return out_arr


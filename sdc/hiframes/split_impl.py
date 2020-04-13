# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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


from numba import types

from numba.datamodel import StructModel
from numba.extending import (typeof_impl, type_callable, models, register_model, NativeValue,
                             make_attribute_wrapper, lower_builtin, box, unbox,
                             lower_getattr, intrinsic, overload_method, overload, overload_attribute)
from numba import cgutils
from sdc.str_ext import string_type

from llvmlite import ir as lir
import llvmlite.binding as ll
from llvmlite.llvmpy.core import Type as LLType
from .. import hstr_ext
ll.add_symbol('array_setitem', hstr_ext.array_setitem)
ll.add_symbol('array_getptr1', hstr_ext.array_getptr1)

ll.add_symbol('dtor_str_arr_split_view', hstr_ext.dtor_str_arr_split_view)
ll.add_symbol('str_arr_split_view_impl', hstr_ext.str_arr_split_view_impl)
ll.add_symbol('str_arr_split_view_alloc', hstr_ext.str_arr_split_view_alloc)

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
        return  # StringArrayIterator()

    def copy(self):
        return StringArraySplitViewType()


string_array_split_view_type = StringArraySplitViewType()


class SplitViewStringMethodsType(types.IterableType):
    """
    Type definition for pandas.core.strings.StringMethods functions handling.

    Members
    ----------
    _data: :class:`SeriesType`
        input arg
    """

    def __init__(self, data):
        self.data = data
        name = 'SplitViewStringMethodsType({})'.format(self.data)
        super(SplitViewStringMethodsType, self).__init__(name)

    @property
    def iterator_type(self):
        return None


@register_model(SplitViewStringMethodsType)
class SplitViewStringMethodsTypeModel(StructModel):
    """
    Model for SplitViewStringMethodsType type
    All members must be the same as main type for this model
    """

    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.data)
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


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

    # Array setitem call
    arr_get_fnty = LLType.function(
        lir.IntType(8).as_pointer(), [c.pyapi.pyobj, c.pyapi.py_ssize_t])
    arr_get_fn = c.pyapi._get_function(arr_get_fnty, name="array_getptr1")
    arr_setitem_fnty = LLType.function(
        lir.VoidType(),
        [c.pyapi.pyobj, lir.IntType(8).as_pointer(), c.pyapi.pyobj])
    arr_setitem_fn = c.pyapi._get_function(
        arr_setitem_fnty, name="array_setitem")

    # for each string
    with cgutils.for_range(builder, sp_view.num_items) as loop:
        str_ind = loop.index
        # start and end offset of string's list in index_offsets
        # sp_view.index_offsets[str_ind]
        list_start_offset = builder.sext(builder.load(builder.gep(sp_view.index_offsets, [str_ind])), lir.IntType(64))
        # sp_view.index_offsets[str_ind+1]
        list_end_offset = builder.sext(
            builder.load(
                builder.gep(
                    sp_view.index_offsets, [
                        builder.add(
                            str_ind, str_ind.type(1))])), lir.IntType(64))
        # cgutils.printf(builder, "%d %d\n", list_start, list_end)

        # Build a new Python list
        nitems = builder.sub(list_end_offset, list_start_offset)
        nitems = builder.sub(nitems, nitems.type(1))
        # cgutils.printf(builder, "str %lld n %lld\n", str_ind, nitems)
        list_obj = c.pyapi.list_new(nitems)
        with c.builder.if_then(cgutils.is_not_null(c.builder, list_obj),
                               likely=True):
            with cgutils.for_range(c.builder, nitems) as loop:
                # data_offsets of current list
                start_index = builder.add(list_start_offset, loop.index)
                data_start = builder.load(builder.gep(sp_view.data_offsets, [start_index]))
                # add 1 since starts from -1
                data_start = builder.add(data_start, data_start.type(1))
                data_end = builder.load(
                    builder.gep(
                        sp_view.data_offsets, [
                            builder.add(
                                start_index, start_index.type(1))]))
                # cgutils.printf(builder, "ind %lld %lld\n", data_start, data_end)
                data_ptr = builder.gep(builder.extract_value(sp_view.data, 0), [data_start])
                str_size = builder.sext(builder.sub(data_end, data_start), lir.IntType(64))
                str_obj = c.pyapi.string_from_string_and_size(data_ptr, str_size)
                c.pyapi.list_setitem(list_obj, loop.index, str_obj)

        arr_ptr = builder.call(arr_get_fn, [out_arr, str_ind])
        builder.call(arr_setitem_fn, [out_arr, arr_ptr, list_obj])

    c.pyapi.decref(np_class_obj)
    return out_arr

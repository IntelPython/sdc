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
import numpy
import pandas
import numba
import sdc
from numba import types
from numba.typing.templates import (infer_global, AbstractTemplate, infer,
                                    signature, AttributeTemplate, infer_getattr, bound_function)
import numba.typing.typeof
from numba.datamodel import StructModel
from numba.errors import TypingError
from numba.extending import (typeof_impl, type_callable, models, register_model, NativeValue,
                             make_attribute_wrapper, lower_builtin, box, unbox,
                             lower_getattr, intrinsic, overload_method, overload, overload_attribute)
from numba import cgutils
from sdc.str_ext import string_type
from numba.targets.imputils import (impl_ret_new_ref, impl_ret_borrowed,
                                    iternext_impl, RefType)
from sdc.str_arr_ext import (string_array_type, get_data_ptr,
                              is_str_arr_typ, pre_alloc_string_array, _memcpy)

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
    # ('num_total_strs', types.uint64),  # number of strings total
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


make_attribute_wrapper(StringArraySplitViewType, 'num_items', '_num_items')
make_attribute_wrapper(StringArraySplitViewType, 'index_offsets', '_index_offsets')
make_attribute_wrapper(StringArraySplitViewType, 'data_offsets', '_data_offsets')
make_attribute_wrapper(StringArraySplitViewType, 'data', '_data')


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
        super(SplitViewStringMethodsType, self).__init__('SplitViewStringMethodsType')

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


make_attribute_wrapper(SplitViewStringMethodsType, 'data', '_data')


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
        # TODO: incref?
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


@intrinsic
def pre_alloc_str_arr_view(typingctx, num_items_t, num_offsets_t, data_t=None):
    assert num_items_t == types.intp and num_offsets_t == types.intp

    def codegen(context, builder, sig, args):
        num_items, num_offsets, data_ptr = args
        meminfo, meminfo_data_ptr = construct_str_arr_split_view(
            context, builder)

        # (str_arr_split_view_payload* out_view, int64_t num_items,
        #  int64_t num_offsets)
        fnty = lir.FunctionType(
            lir.VoidType(),
            [meminfo_data_ptr.type, lir.IntType(64), lir.IntType(64)])

        fn_impl = builder.module.get_or_insert_function(
            fnty, name="str_arr_split_view_alloc")

        builder.call(fn_impl,
                     [meminfo_data_ptr, num_items, num_offsets])

        view_payload = cgutils.create_struct_proxy(
            str_arr_split_view_payload_type)(
            context, builder, value=builder.load(meminfo_data_ptr))

        out_view = context.make_helper(builder, string_array_split_view_type)
        out_view.num_items = num_items
        out_view.index_offsets = view_payload.index_offsets
        out_view.data_offsets = view_payload.data_offsets
        # TODO: incref?
        out_view.data = data_ptr
        if context.enable_nrt:
            context.nrt.incref(builder, data_t, data_ptr)
        # out_view.null_bitmap = view_payload.null_bitmap
        out_view.meminfo = meminfo
        ret = out_view._getvalue()

        return impl_ret_new_ref(
            context, builder, string_array_split_view_type, ret)

    return string_array_split_view_type(
        types.intp, types.intp, data_t), codegen


@intrinsic
def get_c_arr_ptr(typingctx, c_arr, ind_t=None):
    assert isinstance(c_arr, (types.CPointer, types.ArrayCTypes))

    def codegen(context, builder, sig, args):
        in_arr, ind = args
        if isinstance(sig.args[0], types.ArrayCTypes):
            in_arr = builder.extract_value(in_arr, 0)

        return builder.bitcast(
            builder.gep(in_arr, [ind]), lir.IntType(8).as_pointer())

    return types.voidptr(c_arr, ind_t), codegen


@intrinsic
def getitem_c_arr(typingctx, c_arr, ind_t=None):
    def codegen(context, builder, sig, args):
        in_arr, ind = args
        return builder.load(builder.gep(in_arr, [ind]))

    return c_arr.dtype(c_arr, ind_t), codegen


@intrinsic
def setitem_c_arr(typingctx, c_arr, ind_t, item_t=None):
    def codegen(context, builder, sig, args):
        in_arr, ind, item = args
        ptr = builder.gep(in_arr, [ind])
        builder.store(item, ptr)

    return types.void(c_arr, ind_t, c_arr.dtype), codegen


@intrinsic
def get_array_ctypes_ptr(typingctx, arr_ctypes_t, ind_t=None):
    def codegen(context, builder, sig, args):
        in_arr_ctypes, ind = args

        arr_ctypes = context.make_helper(
            builder, arr_ctypes_t, in_arr_ctypes)

        out = context.make_helper(builder, arr_ctypes_t)
        out.data = builder.gep(arr_ctypes.data, [ind])
        out.meminfo = arr_ctypes.meminfo
        res = out._getvalue()
        return impl_ret_borrowed(context, builder, arr_ctypes_t, res)

    return arr_ctypes_t(arr_ctypes_t, ind_t), codegen


@numba.njit(no_cpython_wrapper=True)
def get_split_view_index(arr, item_ind, str_ind):
    start_index = getitem_c_arr(arr._index_offsets, item_ind)
    # TODO: check num strings and support NAN
    # end_index = getitem_c_arr(arr._index_offsets, item_ind+1)
    data_start = getitem_c_arr(
        arr._data_offsets, start_index + str_ind)
    data_start += 1
    # get around -1 storage in uint32 problem
    if start_index + str_ind == 0:
        data_start = 0
    data_end = getitem_c_arr(
        arr._data_offsets, start_index + str_ind + 1)
    return data_start, (data_end - data_start)


@numba.njit(no_cpython_wrapper=True)
def get_split_view_data_ptr(arr, data_start):
    return get_array_ctypes_ptr(arr._data, data_start)


@overload(len)
def str_arr_split_view_len_overload(arr):
    if arr == string_array_split_view_type:
        return lambda arr: arr._num_items


@overload_method(SplitViewStringMethodsType, 'len')
def hpat_pandas_spliview_stringmethods_len(self):
    """
    Pandas Series method :meth:`pandas.core.strings.StringMethods.len()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests sdc.tests.test_hiframes.TestHiFrames.test_str_split_filter

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    if not isinstance(self, SplitViewStringMethodsType):
        raise TypingError(
            'Method len(). The object must be a pandas.core.strings. Given: {}'.format(
                self))

    def hpat_pandas_spliview_stringmethods_len_impl(self):
        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.int64)
        local_data = self._data._data
        for i in range(len(local_data)):
            result[i] = len(local_data[i])

        return pandas.Series(result, name=self._data._name)

    return hpat_pandas_spliview_stringmethods_len_impl


# @infer_global(operator.getitem)
class GetItemStringArraySplitView(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        if ary == string_array_split_view_type:
            if isinstance(idx, types.SliceType):
                return signature(string_array_split_view_type, *args)
            elif isinstance(idx, types.Integer):
                return signature(types.List(string_type), *args)
            elif idx == types.Array(types.bool_, 1, 'C'):
                return signature(string_array_split_view_type, *args)
            elif idx == types.Array(types.intp, 1, 'C'):
                return signature(string_array_split_view_type, *args)


@overload(operator.getitem)
def str_arr_split_view_getitem_overload(A, ind):
    if A == string_array_split_view_type and isinstance(ind, types.Integer):
        kind = numba.unicode.PY_UNICODE_1BYTE_KIND

        def _impl(A, ind):
            start_index = getitem_c_arr(A._index_offsets, ind)
            end_index = getitem_c_arr(A._index_offsets, ind + 1)
            n = end_index - start_index - 1

            str_list = sdc.str_ext.alloc_str_list(n)
            for i in range(n):
                data_start = getitem_c_arr(
                    A._data_offsets, start_index + i)
                data_start += 1
                # get around -1 storage in uint32 problem
                if start_index + i == 0:
                    data_start = 0
                data_end = getitem_c_arr(
                    A._data_offsets, start_index + i + 1)
                length = data_end - data_start
                _str = numba.unicode._empty_string(kind, length)
                ptr = get_array_ctypes_ptr(A._data, data_start)
                _memcpy(_str._data, ptr, length, 1)
                str_list[i] = _str

            return str_list

        return _impl

    if A == string_array_split_view_type and ind == types.Array(types.bool_, 1, 'C'):
        def _impl(A, ind):
            n = len(A)
            if n != len(ind):
                raise IndexError("boolean index did not match indexed array"
                                 " along dimension 0")

            num_items = 0
            num_offsets = 0
            for i in range(n):
                if ind[i]:
                    num_items += 1
                    start_index = getitem_c_arr(A._index_offsets, i)
                    end_index = getitem_c_arr(A._index_offsets, i + 1)
                    num_offsets += end_index - start_index

            out_arr = pre_alloc_str_arr_view(num_items, num_offsets, A._data)
            item_ind = 0
            offset_ind = 0
            for i in range(n):
                if ind[i]:
                    start_index = getitem_c_arr(A._index_offsets, i)
                    end_index = getitem_c_arr(A._index_offsets, i + 1)
                    n_offsets = end_index - start_index

                    setitem_c_arr(out_arr._index_offsets, item_ind, offset_ind)
                    ptr = get_c_arr_ptr(A._data_offsets, start_index)
                    out_ptr = get_c_arr_ptr(out_arr._data_offsets, offset_ind)
                    _memcpy(out_ptr, ptr, n_offsets, 4)
                    item_ind += 1
                    offset_ind += n_offsets

            # last item
            setitem_c_arr(out_arr._index_offsets, item_ind, offset_ind)
            return out_arr

        return _impl

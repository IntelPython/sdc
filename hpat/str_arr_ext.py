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
from numba.targets.hashing import _Py_hash_t
import llvmlite.llvmpy.core as lc
from glob import glob

char_typ = types.uint8
offset_typ = types.uint32

data_ctypes_type = types.ArrayCTypes(types.Array(char_typ, 1, 'C'))
offset_ctypes_type = types.ArrayCTypes(types.Array(offset_typ, 1, 'C'))

class StringArray(object):
    def __init__(self, str_list):
        # dummy constructor
        self.num_strings = len(str_list)
        self.offsets = str_list
        self.data = str_list

    def __repr__(self):
        return 'StringArray({})'.format(self.data)


class StringArrayType(types.IterableType):
    def __init__(self):
        super(StringArrayType, self).__init__(
            name='StringArrayType()')

    @property
    def dtype(self):
        return string_type

    @property
    def iterator_type(self):
        return StringArrayIterator()

    def copy(self):
        return StringArrayType()

string_array_type = StringArrayType()


@typeof_impl.register(StringArray)
def typeof_string_array(val, c):
    return string_array_type

# @type_callable(StringArray)
# def type_string_array_call(context):
#     def typer(offset, data):
#         return string_array_type
#     return typer


@type_callable(StringArray)
def type_string_array_call2(context):
    def typer(string_list=None):
        return string_array_type
    return typer


class StringArrayPayloadType(types.Type):
    def __init__(self):
        super(StringArrayPayloadType, self).__init__(
            name='StringArrayPayloadType()')

str_arr_payload_type = StringArrayPayloadType()

# XXX: C equivalent in _str_ext.cpp
@register_model(StringArrayPayloadType)
class StringArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('offsets', types.CPointer(offset_typ)),
            ('data', types.CPointer(char_typ)),
            ('null_bitmap', types.CPointer(char_typ)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)

str_arr_model_members = [
    ('num_items', types.uint64),
    ('num_total_chars', types.uint64),
    ('offsets', types.CPointer(offset_typ)),
    ('data', types.CPointer(char_typ)),
    ('null_bitmap', types.CPointer(char_typ)),
    ('meminfo', types.MemInfoPointer(str_arr_payload_type)),
]

@register_model(StringArrayType)
class StringArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        models.StructModel.__init__(self, dmm, fe_type, str_arr_model_members)

# TODO: fix overload for things like 'getitem'
# @overload(operator.getitem)
# def str_arr_getitem_bool_overload(str_arr_tp, bool_arr_tp):
#     import pdb; pdb.set_trace()
#     if str_arr_tp == string_array_type and bool_arr_tp == types.Array(types.bool_, 1, 'C'):
#         def str_arr_bool_impl(str_arr, bool_arr):
#             n = len(str_arr)
#             if n!=len(bool_arr):
#                 raise IndexError("boolean index did not match indexed array along dimension 0")
#             return str_arr
#         return str_arr_bool_impl

class StringArrayIterator(types.SimpleIteratorType):
    """
    Type class for iterators of string arrays.
    """

    def __init__(self):
        name = "iter(String)"
        yield_type = string_type
        super(StringArrayIterator, self).__init__(name, yield_type)

@register_model(StringArrayIterator)
class StrArrayIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('array', string_array_type)]
        super(StrArrayIteratorModel, self).__init__(dmm, fe_type, members)

lower_builtin('getiter', string_array_type)(numba.targets.arrayobj.getiter_array)

@lower_builtin('iternext', StringArrayIterator)
@iternext_impl(RefType.NEW)
def iternext_str_array(context, builder, sig, args, result):
    [iterty] = sig.args
    [iter_arg] = args

    iterobj = context.make_helper(builder, iterty, value=iter_arg)
    len_sig = signature(types.intp, string_array_type)
    nitems = context.compile_internal(builder, lambda a: len(a), len_sig, [iterobj.array])

    index = builder.load(iterobj.index)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        getitem_sig = signature(string_type, string_array_type, types.intp)
        value = context.compile_internal(builder, lambda a,i: a[i], getitem_sig, [iterobj.array, index])
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)

@intrinsic
def num_total_chars(typingctx, str_arr_typ=None):
    # None default to make IntelliSense happy
    assert is_str_arr_typ(str_arr_typ)
    def codegen(context, builder, sig, args):
        in_str_arr, = args
        string_array = context.make_helper(builder, string_array_type, in_str_arr)
        return string_array.num_total_chars

    return types.uint64(string_array_type), codegen


@intrinsic
def get_offset_ptr(typingctx, str_arr_typ=None):
    assert is_str_arr_typ(str_arr_typ)
    def codegen(context, builder, sig, args):
        in_str_arr, = args

        string_array = context.make_helper(builder, string_array_type, in_str_arr)
        #return string_array.offsets
        # # Create new ArrayCType structure
        ctinfo = context.make_helper(builder, offset_ctypes_type)
        ctinfo.data = builder.bitcast(string_array.offsets, lir.IntType(32).as_pointer())
        ctinfo.meminfo = string_array.meminfo
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, offset_ctypes_type, res)

    return offset_ctypes_type(string_array_type), codegen

@intrinsic
def get_data_ptr(typingctx, str_arr_typ=None):
    assert is_str_arr_typ(str_arr_typ)
    def codegen(context, builder, sig, args):
        in_str_arr, = args

        string_array = context.make_helper(builder, string_array_type, in_str_arr)
        #return string_array.data
        # Create new ArrayCType structure
        # TODO: put offset/data in main structure since immutable
        ctinfo = context.make_helper(builder, data_ctypes_type)
        ctinfo.data = string_array.data
        ctinfo.meminfo = string_array.meminfo
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, data_ctypes_type, res)

    return data_ctypes_type(string_array_type), codegen

@intrinsic
def get_data_ptr_ind(typingctx, str_arr_typ, int_t=None):
    assert is_str_arr_typ(str_arr_typ)
    def codegen(context, builder, sig, args):
        in_str_arr, ind = args

        string_array = context.make_helper(
            builder, string_array_type, in_str_arr)
        # Create new ArrayCType structure
        # TODO: put offset/data in main structure since immutable
        ctinfo = context.make_helper(builder, data_ctypes_type)
        ctinfo.data = builder.gep(string_array.data, [ind])
        ctinfo.meminfo = string_array.meminfo
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, data_ctypes_type, res)

    return data_ctypes_type(string_array_type, types.intp), codegen

@intrinsic
def getitem_str_offset(typingctx, str_arr_typ, ind_t=None):
    def codegen(context, builder, sig, args):
        in_str_arr, ind = args

        string_array = context.make_helper(builder, string_array_type, in_str_arr)
        offsets = builder.bitcast(string_array.offsets, lir.IntType(32).as_pointer())
        return builder.load(builder.gep(offsets, [ind]))

    return types.uint32(string_array_type, ind_t), codegen

# TODO: fix this for join
@intrinsic
def setitem_str_offset(typingctx, str_arr_typ, ind_t, val_t=None):
    def codegen(context, builder, sig, args):
        in_str_arr, ind, val = args

        string_array = context.make_helper(builder, string_array_type, in_str_arr)
        offsets = builder.bitcast(string_array.offsets, lir.IntType(32).as_pointer())
        builder.store(val, builder.gep(offsets, [ind]))
        return context.get_dummy_value()

    return types.void(string_array_type, ind_t, types.uint32), codegen

@intrinsic
def copy_str_arr_slice(typingctx, str_arr_typ, out_str_arr_typ, ind_t=None):
    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr, ind = args

        in_string_array = context.make_helper(builder, string_array_type, in_str_arr)

        out_string_array = context.make_helper(builder, string_array_type, out_str_arr)

        in_offsets = builder.bitcast(in_string_array.offsets, lir.IntType(32).as_pointer())
        out_offsets = builder.bitcast(out_string_array.offsets, lir.IntType(32).as_pointer())

        ind_p1 = builder.add(ind, context.get_constant(types.intp, 1))
        cgutils.memcpy(builder, out_offsets, in_offsets, ind_p1)
        cgutils.memcpy(builder, out_string_array.data, in_string_array.data, builder.load(builder.gep(in_offsets, [ind])))
        # n_bytes = (num_strings+sizeof(uint8_t)-1)/sizeof(uint8_t)
        ind_p7 = builder.add(ind, lir.Constant(lir.IntType(64), 7))
        n_bytes = builder.lshr(ind_p7, lir.Constant(lir.IntType(64), 3))
        # assuming rest of last byte is set to all ones (e.g. from prealloc)
        cgutils.memcpy(builder, out_string_array.null_bitmap, in_string_array.null_bitmap, n_bytes)
        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type, ind_t), codegen

@intrinsic
def copy_data(typingctx, str_arr_typ, out_str_arr_typ=None):
    # precondition: output is allocated with data the same size as input's data
    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr = args

        in_string_array = context.make_helper(builder, string_array_type, in_str_arr)
        out_string_array = context.make_helper(builder, string_array_type, out_str_arr)

        cgutils.memcpy(builder, out_string_array.data, in_string_array.data,
                        in_string_array.num_total_chars)
        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type), codegen

@intrinsic
def copy_non_null_offsets(typingctx, str_arr_typ, out_str_arr_typ=None):
    # precondition: output is allocated with offset the size non-nulls in input
    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr = args

        in_string_array = context.make_helper(builder, string_array_type, in_str_arr)
        out_string_array = context.make_helper(builder, string_array_type, out_str_arr)
        n = in_string_array.num_items
        zero = context.get_constant(offset_typ, 0)
        curr_offset_ptr = cgutils.alloca_once_value(builder, zero)
        # XXX: assuming last offset is already set by allocate_string_array

        # for i in range(n)
        #   if not isna():
        #     out_offset[curr] = offset[i]
        with cgutils.for_range(builder, n) as loop:
            isna = lower_is_na(context, builder, in_string_array.null_bitmap, loop.index)
            with cgutils.if_likely(builder, builder.not_(isna)):
                in_val = builder.load(builder.gep(in_string_array.offsets, [loop.index]))
                curr_offset = builder.load(curr_offset_ptr)
                builder.store(in_val, builder.gep(out_string_array.offsets, [curr_offset]))
                builder.store(builder.add(curr_offset, lir.Constant(context.get_data_type(offset_typ), 1)), curr_offset_ptr)

        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type), codegen

@intrinsic
def str_copy(typingctx, buff_arr_typ, ind_typ, str_typ, len_typ=None):
    def codegen(context, builder, sig, args):
        buff_arr, ind, str, len_str = args
        buff_arr = context.make_array(sig.args[0])(context, builder, buff_arr)
        ptr = builder.gep(buff_arr.data, [ind])
        cgutils.raw_memcpy(builder, ptr, str, len_str, 1)
        return context.get_dummy_value()

    return types.void(types.Array(types.uint8, 1, 'C'), types.intp, types.voidptr, types.intp), codegen


@intrinsic
def str_copy_ptr(typingctx, ptr_typ, ind_typ, str_typ, len_typ=None):
    def codegen(context, builder, sig, args):
        ptr, ind, _str, len_str = args
        ptr = builder.gep(ptr, [ind])
        cgutils.raw_memcpy(builder, ptr, _str, len_str, 1)
        return context.get_dummy_value()

    return types.void(types.voidptr, types.intp, types.voidptr, types.intp), codegen


# convert array to list of strings if it is StringArray
# just return it otherwise
def to_string_list(arr):
    return arr

@overload(to_string_list)
def to_string_list_overload(data):
    if is_str_arr_typ(data):
        def to_string_impl(data):
            n = len(data)
            l_str = []
            for i in range(n):
                l_str.append(data[i])
            return l_str
        return to_string_impl

    if isinstance(data, (types.Tuple, types.UniTuple)):
        count = data.count

        func_text = "def f(data):\n"
        func_text += "  return ({}{})\n".format(','.join(["to_string_list(data[{}])".format(
            i) for i in range(count)]),
            "," if count == 1 else "")  # single value needs comma to become tuple

        loc_vars = {}
        exec(func_text, {'to_string_list': to_string_list}, loc_vars)
        to_str_impl = loc_vars['f']
        return to_str_impl

    return lambda data: data

def cp_str_list_to_array(str_arr, str_list):
    return

@overload(cp_str_list_to_array)
def cp_str_list_to_array_overload(str_arr, list_data):
    if is_str_arr_typ(str_arr):
        def cp_str_list_impl(str_arr, list_data):
            n = len(list_data)
            for i in range(n):
                _str = list_data[i]
                str_arr[i] = _str

        return cp_str_list_impl

    if isinstance(str_arr, (types.Tuple, types.UniTuple)):
        count = str_arr.count

        func_text = "def f(str_arr, list_data):\n"
        for i in range(count):
            func_text += "  cp_str_list_to_array(str_arr[{}], list_data[{}])\n".format(i, i)
        func_text += "  return\n"

        loc_vars = {}
        exec(func_text, {'cp_str_list_to_array': cp_str_list_to_array}, loc_vars)
        cp_str_impl = loc_vars['f']
        return cp_str_impl

    return lambda str_arr, list_data: None


def str_list_to_array(str_list):
    return str_list

@overload(str_list_to_array)
def str_list_to_array_overload(str_list):
    if str_list == types.List(string_type):
        def str_list_impl(str_list):
            n = len(str_list)
            n_char = 0
            for i in range(n):
                _str = str_list[i]
                n_char += get_utf8_size(_str)
            str_arr = pre_alloc_string_array(n, n_char)
            for i in range(n):
                _str = str_list[i]
                str_arr[i] = _str
            return str_arr

        return str_list_impl

    return lambda str_list: str_list

@infer_global(operator.getitem)
class GetItemStringArray(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        if isinstance(ary, StringArrayType):
            if isinstance(idx, types.SliceType):
                return signature(string_array_type, *args)
            # elif isinstance(idx, types.Integer):
            #     return signature(string_type, *args)
            elif idx == types.Array(types.bool_, 1, 'C'):
                return signature(string_array_type, *args)
            elif idx == types.Array(types.intp, 1, 'C'):
                return signature(string_array_type, *args)

@infer_global(operator.setitem)
class SetItemStringArray(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        if (ary == string_array_type and isinstance(idx, types.Integer)
                and val == string_type):
            return signature(types.none, *args)


@infer
@infer_global(operator.eq)
@infer_global(operator.ne)
@infer_global(operator.ge)
@infer_global(operator.gt)
@infer_global(operator.le)
@infer_global(operator.lt)
class CmpOpEqStringArray(AbstractTemplate):
    key = '=='

    def generic(self, args, kws):
        assert not kws
        [va, vb] = args
        # if one of the inputs is string array
        if va == string_array_type or vb == string_array_type:
            # inputs should be either string array or string
            assert is_str_arr_typ(va) or va == string_type
            assert is_str_arr_typ(vb) or vb == string_type
            return signature(types.Array(types.boolean, 1, 'C'), va, vb)


@infer
class CmpOpNEqStringArray(CmpOpEqStringArray):
    key = '!='

@infer
class CmpOpGEStringArray(CmpOpEqStringArray):
    key = '>='

@infer
class CmpOpGTStringArray(CmpOpEqStringArray):
    key = '>'

@infer
class CmpOpLEStringArray(CmpOpEqStringArray):
    key = '<='

@infer
class CmpOpLTStringArray(CmpOpEqStringArray):
    key = '<'

def is_str_arr_typ(typ):
    from hpat.hiframes.pd_series_ext import is_str_series_typ
    return typ == string_array_type or is_str_series_typ(typ)

# @infer_global(len)
# class LenStringArray(AbstractTemplate):
#     def generic(self, args, kws):
#         if not kws and len(args)==1 and args[0]==string_array_type:
#             return signature(types.intp, *args)

# XXX: should these be exposed?
# make_attribute_wrapper(StringArrayType, 'num_items', 'num_items')
# make_attribute_wrapper(StringArrayType, 'num_total_chars', 'num_total_chars')
# make_attribute_wrapper(StringArrayType, 'offsets', 'offsets')
# make_attribute_wrapper(StringArrayType, 'data', 'data')

# make_attribute_wrapper(StringArrayPayloadType, 'offsets', 'offsets')
# make_attribute_wrapper(StringArrayPayloadType, 'data', 'data')

# XXX can't use this with overload_method
@infer_getattr
class StrArrayAttribute(AttributeTemplate):
    key = string_array_type

    def resolve_size(self, ctflags):
        return types.intp

    @bound_function("str_arr.copy")
    def resolve_copy(self, ary, args, kws):
        return signature(string_array_type, *args)

@lower_builtin("str_arr.copy", string_array_type)
def str_arr_copy_impl(context, builder, sig, args):
    return context.compile_internal(builder, copy_impl, sig, args)


def copy_impl(arr):
    n = len(arr)
    n_chars = num_total_chars(arr)
    new_arr = pre_alloc_string_array(n, np.int64(n_chars))
    copy_str_arr_slice(new_arr, arr, n)
    return new_arr

# @overload_method(StringArrayType, 'copy')
# def string_array_copy(arr_t):
#     return copy_impl


# @overload_attribute(string_array_type, 'size')
# def string_array_attr_size(arr_t):
#     return get_str_arr_size

# def get_str_arr_size(arr):  # pragma: no cover
#     return len(arr)

# @infer_global(get_str_arr_size)
# class StrArrSizeInfer(AbstractTemplate):
#     def generic(self, args, kws):
#         assert not kws
#         assert len(args) == 1 and args[0] == string_array_type
#         return signature(types.intp, *args)

# @lower_builtin(get_str_arr_size, string_array_type)
# def str_arr_size_impl(context, builder, sig, args):

@lower_getattr(string_array_type, 'size')
def str_arr_size_impl(context, builder, typ, val):
    string_array = context.make_helper(builder, string_array_type, val)

    attrval = string_array.num_items
    attrty = types.intp
    return impl_ret_borrowed(context, builder, attrty, attrval)

# @lower_builtin(StringArray, types.Type, types.Type)
# def impl_string_array(context, builder, sig, args):
#     typ = sig.return_type
#     offsets, data = args
#     string_array = cgutils.create_struct_proxy(typ)(context, builder)
#     string_array.offsets = offsets
#     string_array.data = data
#     return string_array._getvalue()



@overload(len)
def str_arr_len_overload(str_arr):
    if is_str_arr_typ(str_arr):
        def str_arr_len(str_arr):
            return str_arr.size
        return str_arr_len


from numba.targets.listobj import ListInstance
from llvmlite import ir as lir
import llvmlite.binding as ll
from . import hstr_ext
ll.add_symbol('get_str_len', hstr_ext.get_str_len)
ll.add_symbol('allocate_string_array', hstr_ext.allocate_string_array)
ll.add_symbol('setitem_string_array', hstr_ext.setitem_string_array)
ll.add_symbol('getitem_string_array', hstr_ext.getitem_string_array)
ll.add_symbol('getitem_string_array_std', hstr_ext.getitem_string_array_std)
ll.add_symbol('is_na', hstr_ext.is_na)
ll.add_symbol('string_array_from_sequence', hstr_ext.string_array_from_sequence)
ll.add_symbol('np_array_from_string_array', hstr_ext.np_array_from_string_array)
ll.add_symbol('print_int', hstr_ext.print_int)
ll.add_symbol('convert_len_arr_to_offset', hstr_ext.convert_len_arr_to_offset)
ll.add_symbol('set_string_array_range', hstr_ext.set_string_array_range)
ll.add_symbol('str_arr_to_int64', hstr_ext.str_arr_to_int64)
ll.add_symbol('str_arr_to_float64', hstr_ext.str_arr_to_float64)
ll.add_symbol('dtor_string_array', hstr_ext.dtor_string_array)
ll.add_symbol('c_glob', hstr_ext.c_glob)
ll.add_symbol('init_memsys', hstr_ext.init_memsys)
ll.add_symbol('decode_utf8', hstr_ext.decode_utf8)

convert_len_arr_to_offset = types.ExternalFunction("convert_len_arr_to_offset", types.void(types.voidptr, types.intp))


setitem_string_array = types.ExternalFunction("setitem_string_array",
            types.void(types.voidptr, types.voidptr, types.intp, string_type,
            types.intp))
_get_utf8_size = types.ExternalFunction("get_utf8_size",
            types.intp(types.voidptr, types.intp, types.int32))


def construct_string_array(context, builder):
    """Creates meminfo and sets dtor.
    """
    alloc_type = context.get_data_type(str_arr_payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = lir.FunctionType(lir.VoidType(),
                                  [llvoidptr, llsize, llvoidptr])
    dtor_fn = builder.module.get_or_insert_function(
        dtor_ftype, name="dtor_string_array")

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


# TODO: overload of constructor doesn't work
# @overload(StringArray)
# def string_array_const(in_list=None):
#     if in_list is None:
#         return lambda: pre_alloc_string_array(0, 0)

#     def str_arr_from_list(in_list):
#         n_strs = len(in_list)
#         total_chars = 0
#         # TODO: use vector to avoid two passes?
#         # get total number of chars
#         for s in in_list:
#             total_chars += len(s)

#         A = pre_alloc_string_array(n_strs, total_chars)
#         for i in range(n_strs):
#             A[i] = in_list[i]

#         return A

#     return str_arr_from_list


# used in pd.DataFrame() and pd.Series() to convert list of strings
@lower_builtin(StringArray)
@lower_builtin(StringArray, types.List)
@lower_builtin(StringArray, types.UniTuple)
def impl_string_array_single(context, builder, sig, args):
    if isinstance(args[0], types.UniTuple):
        assert args[0].dtype == string_type

    if not sig.args:  # return empty string array if no args
        res = context.compile_internal(
            builder, lambda: pre_alloc_string_array(0, 0), sig, args)
        return res

    def str_arr_from_list(in_list):
        n_strs = len(in_list)
        total_chars = 0
        # TODO: use vector to avoid two passes?
        # get total number of chars
        for s in in_list:
            total_chars += get_utf8_size(s)

        A = pre_alloc_string_array(n_strs, total_chars)
        for i in range(n_strs):
            A[i] = in_list[i]

        return A

    res = context.compile_internal(builder, str_arr_from_list, sig, args)
    return res

# @lower_builtin(StringArray)
# @lower_builtin(StringArray, types.List)
# def impl_string_array_single(context, builder, sig, args):
#     typ = sig.return_type
#     zero = context.get_constant(types.intp, 0)
#     meminfo, meminfo_data_ptr = construct_string_array(context, builder)

#     str_arr_payload = cgutils.create_struct_proxy(str_arr_payload_type)(context, builder)
#     if not sig.args:  # return empty string array if no args
#         # XXX alloc empty arrays for dtor to safely delete?
#         builder.store(str_arr_payload._getvalue(), meminfo_data_ptr)
#         string_array = context.make_helper(builder, typ)
#         string_array.meminfo = meminfo
#         string_array.num_items = zero
#         string_array.num_total_chars = zero
#         ret = string_array._getvalue()
#         #context.nrt.decref(builder, ty, ret)

#         return impl_ret_new_ref(context, builder, typ, ret)

#     string_list = ListInstance(context, builder, sig.args[0], args[0])

#     # get total size of string buffer
#     fnty = lir.FunctionType(lir.IntType(64),
#                             [lir.IntType(8).as_pointer()])
#     fn_len = builder.module.get_or_insert_function(fnty, name="get_str_len")
#     total_size = cgutils.alloca_once_value(builder, zero)

#     # loop through all strings and get length
#     with cgutils.for_range(builder, string_list.size) as loop:
#         str_value = string_list.getitem(loop.index)
#         str_len = builder.call(fn_len, [str_value])
#         builder.store(builder.add(builder.load(total_size), str_len), total_size)

#     # allocate string array
#     fnty = lir.FunctionType(lir.VoidType(),
#                             [lir.IntType(32).as_pointer().as_pointer(),
#                              lir.IntType(8).as_pointer().as_pointer(),
#                              lir.IntType(8).as_pointer().as_pointer(),
#                              lir.IntType(64),
#                              lir.IntType(64)])
#     fn_alloc = builder.module.get_or_insert_function(fnty,
#                                                      name="allocate_string_array")
#     builder.call(fn_alloc, [str_arr_payload._get_ptr_by_name('offsets'),
#                             str_arr_payload._get_ptr_by_name('data'),
#                             str_arr_payload._get_ptr_by_name('null_bitmap'),
#                             string_list.size, builder.load(total_size)])

#     # set string array values
#     fnty = lir.FunctionType(lir.VoidType(),
#                             [lir.IntType(32).as_pointer(),
#                              lir.IntType(8).as_pointer(),
#                              lir.IntType(8).as_pointer(),
#                              lir.IntType(64)])
#     fn_setitem = builder.module.get_or_insert_function(fnty,
#                                                        name="setitem_string_array")

#     with cgutils.for_range(builder, string_list.size) as loop:
#         str_value = string_list.getitem(loop.index)
#         builder.call(fn_setitem, [str_arr_payload.offsets, str_arr_payload.data,
#                                   str_value, loop.index])

#     builder.store(str_arr_payload._getvalue(), meminfo_data_ptr)

#     string_array = context.make_helper(builder, typ)
#     string_array.num_items = string_list.size
#     string_array.num_total_chars = builder.load(total_size)
#     #cgutils.printf(builder, "str %d %d\n", string_array.num_items, string_array.num_total_chars)
#     string_array.offsets = str_arr_payload.offsets
#     string_array.data = str_arr_payload.data
#     string_array.null_bitmap = str_arr_payload.null_bitmap
#     string_array.meminfo = meminfo
#     ret = string_array._getvalue()
#     #context.nrt.decref(builder, ty, ret)

#     return impl_ret_new_ref(context, builder, typ, ret)

@intrinsic
def pre_alloc_string_array(typingctx, num_strs_typ, num_total_chars_typ=None):
    assert isinstance(num_strs_typ, types.Integer) and isinstance(num_total_chars_typ, types.Integer)
    def codegen(context, builder, sig, args):
        num_strs, num_total_chars = args
        meminfo, meminfo_data_ptr = construct_string_array(context, builder)

        str_arr_payload = cgutils.create_struct_proxy(str_arr_payload_type)(context, builder)

        # allocate string array
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(32).as_pointer().as_pointer(),
                                 lir.IntType(8).as_pointer().as_pointer(),
                                 lir.IntType(8).as_pointer().as_pointer(),
                                 lir.IntType(64),
                                 lir.IntType(64)])
        fn_alloc = builder.module.get_or_insert_function(fnty,
                                                         name="allocate_string_array")
        builder.call(fn_alloc, [str_arr_payload._get_ptr_by_name('offsets'),
                                str_arr_payload._get_ptr_by_name('data'),
                                str_arr_payload._get_ptr_by_name('null_bitmap'),
                                num_strs,
                                num_total_chars])

        builder.store(str_arr_payload._getvalue(), meminfo_data_ptr)
        string_array = context.make_helper(builder, string_array_type)
        string_array.num_items = num_strs
        string_array.num_total_chars = num_total_chars
        string_array.offsets = str_arr_payload.offsets
        string_array.data = str_arr_payload.data
        string_array.null_bitmap = str_arr_payload.null_bitmap
        string_array.meminfo = meminfo
        ret = string_array._getvalue()
        #context.nrt.decref(builder, ty, ret)

        return impl_ret_new_ref(context, builder, string_array_type, ret)

    return string_array_type(types.intp, types.intp), codegen


@intrinsic
def set_string_array_range(typingctx, out_typ, in_typ, curr_str_typ, curr_chars_typ=None):
    assert is_str_arr_typ(out_typ) and is_str_arr_typ(in_typ)
    assert curr_str_typ == types.intp and curr_chars_typ == types.intp
    def codegen(context, builder, sig, args):
        out_arr, in_arr, curr_str_ind, curr_chars_ind = args

        # get input/output struct
        out_string_array = context.make_helper(builder, string_array_type, out_arr)
        in_string_array = context.make_helper(builder, string_array_type, in_arr)

        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(32).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(32).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(64),
                                 lir.IntType(64),
                                 lir.IntType(64),
                                 lir.IntType(64),])
        fn_alloc = builder.module.get_or_insert_function(fnty,
                                                         name="set_string_array_range")
        builder.call(fn_alloc, [out_string_array.offsets,
                                out_string_array.data,
                                in_string_array.offsets,
                                in_string_array.data,
                                curr_str_ind,
                                curr_chars_ind,
                                in_string_array.num_items,
                                in_string_array.num_total_chars,
                                ])

        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type, types.intp, types.intp), codegen

# box series calls this too
@box(StringArrayType)
def box_str_arr(typ, val, c):
    """
    """

    string_array = c.context.make_helper(c.builder, string_array_type, val)

    fnty = lir.FunctionType(c.context.get_argument_type(types.pyobject), #lir.IntType(8).as_pointer(),
                            [lir.IntType(64),
                             lir.IntType(32).as_pointer(),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(8).as_pointer(),
                            ])
    fn_get = c.builder.module.get_or_insert_function(fnty, name="np_array_from_string_array")
    arr = c.builder.call(fn_get, [string_array.num_items, string_array.offsets, string_array.data, string_array.null_bitmap])

    # TODO: double check refcounting here
    # c.context.nrt.decref(c.builder, typ, val)
    return arr #c.builder.load(arr)

@intrinsic
def str_arr_is_na(typingctx, str_arr_typ, ind_typ=None):
    # None default to make IntelliSense happy
    assert is_str_arr_typ(str_arr_typ)
    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        string_array = context.make_helper(builder, string_array_type, in_str_arr)

        # (null_bitmap[i / 8] & kBitmask[i % 8]) == 0;
        byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
        bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
        byte = builder.load(builder.gep(string_array.null_bitmap, [byte_ind], inbounds=True))
        ll_typ_mask = lir.ArrayType(lir.IntType(8), 8)
        mask_tup = cgutils.alloca_once_value(builder, lir.Constant(ll_typ_mask, (1, 2, 4, 8, 16, 32, 64, 128)))
        mask = builder.load(builder.gep(mask_tup, [lir.Constant(lir.IntType(64), 0), bit_ind], inbounds=True))
        return builder.icmp_unsigned('==', builder.and_(byte, mask), lir.Constant(lir.IntType(8), 0))

    return types.bool_(string_array_type, types.intp), codegen


@intrinsic
def str_arr_set_na(typingctx, str_arr_typ, ind_typ=None):
    # None default to make IntelliSense happy
    assert is_str_arr_typ(str_arr_typ)
    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        string_array = context.make_helper(builder, string_array_type, in_str_arr)

        # bits[i / 8] |= kBitmask[i % 8];
        byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
        bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
        byte_ptr = builder.gep(string_array.null_bitmap, [byte_ind], inbounds=True)
        byte = builder.load(byte_ptr)
        ll_typ_mask = lir.ArrayType(lir.IntType(8), 8)
        mask_tup = cgutils.alloca_once_value(builder, lir.Constant(ll_typ_mask, (1, 2, 4, 8, 16, 32, 64, 128)))
        mask = builder.load(builder.gep(mask_tup, [lir.Constant(lir.IntType(64), 0), bit_ind], inbounds=True))
        # flip all bits of mask e.g. 11111101
        mask = builder.xor(mask, lir.Constant(lir.IntType(8), -1))
        # unset masked bit
        builder.store(builder.and_(byte, mask), byte_ptr)
        return context.get_dummy_value()

    return types.void(string_array_type, types.intp), codegen


@intrinsic
def set_null_bits(typingctx, str_arr_typ=None):
    assert is_str_arr_typ(str_arr_typ)
    def codegen(context, builder, sig, args):
        in_str_arr, = args
        string_array = context.make_helper(builder, string_array_type, in_str_arr)
        # n_bytes = (num_strings+sizeof(uint8_t)-1)/sizeof(uint8_t);
        n_bytes = builder.udiv(builder.add(string_array.num_items, lir.Constant(lir.IntType(64), 7)), lir.Constant(lir.IntType(64), 8))
        cgutils.memset(builder, string_array.null_bitmap, n_bytes, -1)
        return context.get_dummy_value()

    return types.none(string_array_type), codegen

# XXX: setitem works only if value is same size as the previous value
@lower_builtin(operator.setitem, StringArrayType, types.Integer, string_type)
def setitem_str_arr(context, builder, sig, args):
    arr, ind, val = args
    uni_str = cgutils.create_struct_proxy(string_type)(
        context, builder, value=val)
    string_array = context.make_helper(builder, string_array_type, arr)
    fnty = lir.FunctionType(lir.VoidType(),
                            [lir.IntType(32).as_pointer(),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(64),
                             lir.IntType(8).as_pointer(),
                             lir.IntType(64),
                             lir.IntType(32),
                             lir.IntType(64)])
    fn_setitem = builder.module.get_or_insert_function(
        fnty, name="setitem_string_array")
    builder.call(fn_setitem, [string_array.offsets, string_array.data,
                              string_array.num_total_chars,
                              uni_str.data, uni_str.length, uni_str.kind, ind])
    return context.get_dummy_value()


@numba.njit(no_cpython_wrapper=True)
def get_utf8_size(s):
    return _get_utf8_size(s._data, s._length, s._kind)


@intrinsic
def setitem_str_arr_ptr(typingctx, str_arr_t, ind_t, ptr_t, len_t=None):
    def codegen(context, builder, sig, args):
        arr, ind, ptr, length = args
        string_array = context.make_helper(builder, string_array_type, arr)
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(32).as_pointer(),
                                lir.IntType(8).as_pointer(),
                                lir.IntType(64),
                                lir.IntType(8).as_pointer(),
                                lir.IntType(64),
                                lir.IntType(32),
                                lir.IntType(64)])
        fn_setitem = builder.module.get_or_insert_function(
            fnty, name="setitem_string_array")
        builder.call(fn_setitem, [string_array.offsets, string_array.data,
                    string_array.num_total_chars,
                    builder.extract_value(ptr, 0), length, uni_str.kind, ind])
        return context.get_dummy_value()

    return types.void(str_arr_t, ind_t, ptr_t, len_t), codegen

def lower_is_na(context, builder, bull_bitmap, ind):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(),
                             lir.IntType(64)])
    fn_getitem = builder.module.get_or_insert_function(fnty,
                                                       name="is_na")
    return builder.call(fn_getitem, [bull_bitmap,
                                     ind])


@intrinsic
def _memcpy(typingctx, dest_t, src_t, count_t, item_size_t=None):
    def codegen(context, builder, sig, args):
        dst, src, count, itemsize = args
        # buff_arr = context.make_array(sig.args[0])(context, builder, buff_arr)
        # ptr = builder.gep(buff_arr.data, [ind])
        cgutils.raw_memcpy(builder, dst, src, count, itemsize)
        return context.get_dummy_value()

    return types.void(types.voidptr, types.voidptr, types.intp, types.intp), codegen


# TODO: use overload
@overload(operator.getitem)
def str_arr_getitem_int(A, i):
    if A == string_array_type and isinstance(i, types.Integer):
        kind = numba.unicode.PY_UNICODE_1BYTE_KIND
        def str_arr_getitem_impl(A, i):
            start_offset = getitem_str_offset(A, i)
            end_offset = getitem_str_offset(A, i + 1)
            length = end_offset - start_offset
            ptr = get_data_ptr_ind(A, start_offset)
            ret = decode_utf8(ptr, length)
            # ret = numba.unicode._empty_string(kind, length)
            # _memcpy(ret._data, ptr, length, 1)
            return ret

        return str_arr_getitem_impl


@intrinsic
def decode_utf8(typingctx, ptr_t, len_t=None):
    def codegen(context, builder, sig, args):
        ptr, length = args

        # initialize alloc/dealloc from Numba's runtime pointers
        allocator = context.get_constant(types.intp,
            numba.runtime._nrt_python.c_helpers['MemInfo_alloc_safe'])
        deallocator = context.get_constant(types.intp,
            numba.runtime._nrt_python.c_helpers['MemInfo_call_dtor'])
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(64),
            lir.IntType(64)])
        fn_init_memsys = builder.module.get_or_insert_function(
            fnty, name="init_memsys")
        builder.call(fn_init_memsys, [allocator, deallocator])

        # create str and call decode with internal pointers
        uni_str = cgutils.create_struct_proxy(string_type)(context, builder)
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(32).as_pointer(),
            lir.IntType(64).as_pointer(),
            uni_str.meminfo.type.as_pointer()])
        fn_decode = builder.module.get_or_insert_function(
            fnty, name="decode_utf8")
        builder.call(fn_decode, [ptr, length,
                                uni_str._get_ptr_by_name('kind'),
                                uni_str._get_ptr_by_name('length'),
                                uni_str._get_ptr_by_name('meminfo')])
        uni_str.hash = context.get_constant(_Py_hash_t, -1)
        uni_str.data = context.nrt.meminfo_data(builder, uni_str.meminfo)
        # Set parent to NULL
        uni_str.parent = cgutils.get_null_value(uni_str.parent.type)
        return uni_str._getvalue()

    return string_type(types.voidptr, types.intp), codegen


# @lower_builtin(operator.getitem, StringArrayType, types.Integer)
# @lower_builtin(operator.getitem, StringArrayType, types.IntegerLiteral)
# def lower_string_arr_getitem(context, builder, sig, args):
#     # TODO: support multibyte unicode
#     # TODO: support Null
#     kind = numba.unicode.PY_UNICODE_1BYTE_KIND
#     def str_arr_getitem_impl(A, i):
#         start_offset = getitem_str_offset(A, i)
#         end_offset = getitem_str_offset(A, i + 1)
#         length = end_offset - start_offset
#         ret = numba.unicode._empty_string(kind, length)
#         ptr = get_data_ptr_ind(A, start_offset)
#         _memcpy(ret._data, ptr, length, 1)
#         return ret

#     res = context.compile_internal(builder, str_arr_getitem_impl, sig, args)
#     return res

    # typ = sig.args[0]
    # ind = args[1]

    # string_array = context.make_helper(builder, typ, args[0])

    # # check for NA
    # # i/8, XXX: lshr since always positive
    # #byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
    # #bit_ind = builder.srem

    # # cgutils.printf(builder, "calling bitmap\n")
    # # with cgutils.if_unlikely(builder, lower_is_na(context, builder, string_array.null_bitmap, ind)):
    # #     cgutils.printf(builder, "is_na %d \n", ind)
    # # cgutils.printf(builder, "calling bitmap done\n")

    # fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
    #                         [lir.IntType(32).as_pointer(),
    #                          lir.IntType(8).as_pointer(),
    #                          lir.IntType(64)])
    # fn_getitem = builder.module.get_or_insert_function(fnty,
    #                                                    name="getitem_string_array_std")
    # return builder.call(fn_getitem, [string_array.offsets,
    #                                  string_array.data, args[1]])


@lower_builtin(operator.getitem, StringArrayType, types.Array(types.bool_, 1, 'C'))
def lower_string_arr_getitem_bool(context, builder, sig, args):
    def str_arr_bool_impl(str_arr, bool_arr):
        n = len(str_arr)
        if n != len(bool_arr):
            raise IndexError("boolean index did not match indexed array along dimension 0")
        n_strs = 0
        n_chars = 0
        for i in range(n):
            if bool_arr[i] == True:
                # TODO: use get_cstr_and_len instead of getitem
                _str = str_arr[i]
                n_strs += 1
                n_chars += get_utf8_size(_str)
        out_arr = pre_alloc_string_array(n_strs, n_chars)
        str_ind = 0
        for i in range(n):
            if bool_arr[i] == True:
                _str = str_arr[i]
                out_arr[str_ind] = _str
                str_ind += 1
        return out_arr
    res = context.compile_internal(builder, str_arr_bool_impl, sig, args)
    return res


@lower_builtin(operator.getitem, StringArrayType, types.Array(types.intp, 1, 'C'))
def lower_string_arr_getitem_arr(context, builder, sig, args):
    def str_arr_arr_impl(str_arr, ind_arr):
        n = len(ind_arr)
        # get lengths
        n_strs = 0
        n_chars = 0
        for i in range(n):
            # TODO: use get_cstr_and_len instead of getitem
            _str = str_arr[ind_arr[i]]
            n_strs += 1
            n_chars += get_utf8_size(_str)

        out_arr = pre_alloc_string_array(n_strs, n_chars)
        str_ind = 0
        for i in range(n):
            _str = str_arr[ind_arr[i]]
            out_arr[str_ind] = _str
            str_ind += 1
        return out_arr
    res = context.compile_internal(builder, str_arr_arr_impl, sig, args)
    return res


@lower_builtin(operator.getitem, StringArrayType, types.SliceType)
def lower_string_arr_getitem_slice(context, builder, sig, args):
    def str_arr_slice_impl(str_arr, idx):
        n = len(str_arr)
        slice_idx = numba.unicode._normalize_slice(idx, n)
        span = numba.unicode._slice_span(slice_idx)

        if slice_idx.step == 1:
            start_offset = getitem_str_offset(str_arr, slice_idx.start)
            end_offset = getitem_str_offset(str_arr, slice_idx.stop)
            n_chars = end_offset - start_offset
            new_arr = pre_alloc_string_array(span, np.int64(n_chars))
            # TODO: more efficient copy
            for i in range(span):
                new_arr[i] = str_arr[slice_idx.start+i]
            return new_arr
        else:  # TODO: test
            # get number of chars
            n_chars = 0
            for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                _str = str_arr[i]
                n_chars += get_utf8_size(_str)
            new_arr = pre_alloc_string_array(span, np.int64(n_chars))
            # TODO: more efficient copy
            for i in range(span):
                new_arr[i] = str_arr[slice_idx.start+i*slice_idx.step]
            return new_arr

    res = context.compile_internal(builder, str_arr_slice_impl, sig, args)
    return res


@numba.njit(no_cpython_wrapper=True)
def str_arr_item_to_numeric(out_arr, out_ind, str_arr, ind):
    return _str_arr_item_to_numeric(hpat.hiframes.split_impl.get_c_arr_ptr(
        out_arr.ctypes, out_ind), str_arr, ind, out_arr.dtype)


@intrinsic
def _str_arr_item_to_numeric(typingctx, out_ptr_t, str_arr_t, ind_t,
                                                             out_dtype_t=None):
    assert str_arr_t == string_array_type
    assert ind_t == types.int64

    def codegen(context, builder, sig, args):
        # TODO: return tuple with value and error and avoid array arg?
        out_ptr, arr, ind, _dtype = args
        string_array = context.make_helper(builder, string_array_type, arr)
        fnty = lir.FunctionType(
            lir.IntType(32),
            [out_ptr.type,
             lir.IntType(32).as_pointer(),
             lir.IntType(8).as_pointer(),
             lir.IntType(64)])
        fname = 'str_arr_to_int64'
        if sig.args[3].dtype == types.float64:
            fname = 'str_arr_to_float64'
        else:
            assert sig.args[3].dtype == types.int64
        fn_to_numeric = builder.module.get_or_insert_function(fnty, fname)
        return builder.call(
            fn_to_numeric,
            [out_ptr, string_array.offsets, string_array.data, ind])

    return types.int32(
        out_ptr_t,  string_array_type, types.int64, out_dtype_t), codegen


# TODO: support array of strings
# @typeof_impl.register(np.ndarray)
# def typeof_np_string(val, c):
#     arr_typ = numba.typing.typeof._typeof_ndarray(val, c)
#     # match string dtype
#     if isinstance(arr_typ.dtype, (types.UnicodeCharSeq, types.CharSeq)):
#         return string_array_type
#     return arr_typ


@unbox(StringArrayType)
def unbox_str_series(typ, val, c):
    """
    Unbox a Pandas String Series. We just redirect to StringArray implementation.
    """
    dtype = StringArrayPayloadType()
    payload = cgutils.create_struct_proxy(dtype)(c.context, c.builder)
    string_array = c.context.make_helper(c.builder, typ)

    # function signature of string_array_from_sequence
    # we use void* instead of PyObject*
    fnty = lir.FunctionType(lir.VoidType(),
                            [lir.IntType(8).as_pointer(),
                             lir.IntType(64).as_pointer(),
                             lir.IntType(32).as_pointer().as_pointer(),
                             lir.IntType(8).as_pointer().as_pointer(),
                             lir.IntType(8).as_pointer().as_pointer(),
                             ])
    fn = c.builder.module.get_or_insert_function(fnty, name="string_array_from_sequence")
    c.builder.call(fn, [val,
                        string_array._get_ptr_by_name('num_items'),
                        payload._get_ptr_by_name('offsets'),
                        payload._get_ptr_by_name('data'),
                        payload._get_ptr_by_name('null_bitmap'),
                    ])

    # the raw data is now copied to payload
    # The native representation is a proxy to the payload, we need to
    # get a proxy and attach the payload and meminfo
    meminfo, meminfo_data_ptr = construct_string_array(c.context, c.builder)
    c.builder.store(payload._getvalue(), meminfo_data_ptr)

    string_array.meminfo = meminfo
    string_array.offsets = payload.offsets
    string_array.data = payload.data
    string_array.null_bitmap = payload.null_bitmap
    string_array.num_total_chars = c.builder.zext(c.builder.load(
        c.builder.gep(string_array.offsets, [string_array.num_items])), lir.IntType(64))

    # FIXME how to check that the returned size is > 0?
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(string_array._getvalue(), is_error=is_error)

# zero = context.get_constant(types.intp, 0)
# cond = builder.icmp_signed('>=', size, zero)
# with cgutils.if_unlikely(builder, cond):
# http://llvmlite.readthedocs.io/en/latest/user-guide/ir/ir-builder.html#comparisons


#### glob support #####

@infer_global(glob)
class GlobInfer(AbstractTemplate):
    def generic(self, args, kws):
        if not kws and len(args)==1 and args[0]==string_type:
            return signature(string_array_type, *args)


@lower_builtin(glob, string_type)
def lower_glob(context, builder, sig, args):
    path = args[0]
    uni_str = cgutils.create_struct_proxy(string_type)(
        context, builder, value=path)
    path = uni_str.data
    typ = sig.return_type
    dtype = StringArrayPayloadType()
    meminfo, meminfo_data_ptr = construct_string_array(context, builder)
    string_array = context.make_helper(builder, typ)
    str_arr_payload = cgutils.create_struct_proxy(dtype)(context, builder)

    # call glob in C
    fnty = lir.FunctionType(lir.VoidType(),
                            [lir.IntType(32).as_pointer().as_pointer(),
                             lir.IntType(8).as_pointer().as_pointer(),
                             lir.IntType(8).as_pointer().as_pointer(),
                             lir.IntType(64).as_pointer(),
                             lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="c_glob")
    builder.call(fn, [str_arr_payload._get_ptr_by_name('offsets'),
                            str_arr_payload._get_ptr_by_name('data'),
                            str_arr_payload._get_ptr_by_name('null_bitmap'),
                            string_array._get_ptr_by_name('num_items'),
                            path])

    builder.store(str_arr_payload._getvalue(), meminfo_data_ptr)

    string_array.meminfo = meminfo
    string_array.offsets = str_arr_payload.offsets
    string_array.data = str_arr_payload.data
    string_array.null_bitmap = str_arr_payload.null_bitmap
    string_array.num_total_chars = builder.zext(builder.load(
        builder.gep(string_array.offsets, [string_array.num_items])), lir.IntType(64))

    # cgutils.printf(builder, "n %d\n", string_array.num_items)
    ret = string_array._getvalue()
    #context.nrt.decref(builder, ty, ret)

    return impl_ret_new_ref(context, builder, typ, ret)

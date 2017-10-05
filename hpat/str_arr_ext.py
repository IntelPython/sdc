import numba
import hpat
from numba import types
from numba.typing.templates import infer_global, AbstractTemplate, infer, signature
from numba.extending import (typeof_impl, type_callable, models, register_model,
                                make_attribute_wrapper, lower_builtin, box)
from numba import cgutils
from hpat.str_ext import string_type

class StringArray(object):
    def __init__(self, offsets, data, size):
        self.size = size
        self.offsets = offsets
        self.data = data

    def __repr__(self):
        return 'StringArray({}, {}, {})'.format(self.offsets, self.data, self.size)

class StringArrayType(types.Type):
    def __init__(self):
        super(StringArrayType, self).__init__(
                                    name='StringArrayType()')

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

@register_model(StringArrayType)
class StringArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('size', types.intp),
            ('offsets', types.Opaque('offsets')),
            ('data', hpat.str_ext.string_type),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

@infer
class GetItemStringArray(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        if isinstance(ary, StringArrayType):
            if isinstance(idx, types.SliceType):
                return signature(string_array_type, *args)
            else:
                assert isinstance(idx, types.Integer)
                return signature(string_type, *args)


@infer
class CmpOpEqStringArray(AbstractTemplate):
    key = '=='

    def generic(self, args, kws):
        assert not kws
        [va, vb] = args
        # if one of the inputs is string array
        if va==string_array_type or vb==string_array_type:
            # inputs should be either string array or string
            assert va == string_array_type or va == string_type
            assert vb == string_array_type or vb == string_type
            return signature(types.Array(types.boolean, 1, 'C'), va, vb)

@infer
class CmpOpNEqStringArray(CmpOpEqStringArray):
    key = '!='

make_attribute_wrapper(StringArrayType, 'size', 'size')
make_attribute_wrapper(StringArrayType, 'offsets', 'offsets')
make_attribute_wrapper(StringArrayType, 'data', 'data')

# @lower_builtin(StringArray, types.Type, types.Type)
# def impl_string_array(context, builder, sig, args):
#     typ = sig.return_type
#     offsets, data = args
#     string_array = cgutils.create_struct_proxy(typ)(context, builder)
#     string_array.offsets = offsets
#     string_array.data = data
#     return string_array._getvalue()

from numba.targets.listobj import ListInstance
from llvmlite import ir as lir
import llvmlite.binding as ll
import hstr_ext
ll.add_symbol('get_str_len', hstr_ext.get_str_len)
ll.add_symbol('allocate_string_array', hstr_ext.allocate_string_array)
ll.add_symbol('setitem_string_array', hstr_ext.setitem_string_array)
ll.add_symbol('getitem_string_array', hstr_ext.getitem_string_array)
ll.add_symbol('getitem_string_array_std', hstr_ext.getitem_string_array_std)
ll.add_symbol('print_int', hstr_ext.print_int)

@lower_builtin(StringArray)
@lower_builtin(StringArray, types.List)
def impl_string_array_single(context, builder, sig, args):
    typ = sig.return_type
    string_array = cgutils.create_struct_proxy(typ)(context, builder)
    if not sig.args:  # return empty string array if no args
        return string_array._getvalue()

    string_list = ListInstance(context, builder, sig.args[0], args[0])

    # get total size of string buffer
    fnty = lir.FunctionType( lir.IntType(64),
                            [lir.IntType(8).as_pointer()])
    fn_len = builder.module.get_or_insert_function(fnty, name="get_str_len")
    zero = context.get_constant(types.intp, 0)
    total_size = cgutils.alloca_once_value(builder, zero)
    string_array.size = string_list.size

    # loop through all strings and get length
    with cgutils.for_range(builder, string_list.size) as loop:
        str_value = string_list.getitem(loop.index)
        str_len = builder.call(fn_len, [str_value])
        builder.store(builder.add(builder.load(total_size), str_len), total_size)

    # allocate string array
    fnty = lir.FunctionType( lir.VoidType(),
                            [lir.IntType(8).as_pointer().as_pointer(),
                            lir.IntType(8).as_pointer().as_pointer(),
                            lir.IntType(64),
                            lir.IntType(64)])
    fn_alloc = builder.module.get_or_insert_function(fnty,
                                                name="allocate_string_array")
    builder.call(fn_alloc, [string_array._get_ptr_by_name('offsets'),
                            string_array._get_ptr_by_name('data'),
                            string_list.size, builder.load(total_size)])

    # set string array values
    fnty = lir.FunctionType( lir.VoidType(),
                            [lir.IntType(8).as_pointer(),
                            lir.IntType(8).as_pointer(),
                            lir.IntType(8).as_pointer(),
                            lir.IntType(64)])
    fn_setitem = builder.module.get_or_insert_function(fnty,
                                                name="setitem_string_array")

    with cgutils.for_range(builder, string_list.size) as loop:
        str_value = string_list.getitem(loop.index)
        builder.call(fn_setitem, [string_array.offsets, string_array.data,
                        str_value, loop.index])

    return string_array._getvalue()

@box(StringArrayType)
def box_str(typ, val, c):
    """
    """
    string_array = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    # fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(64)])
    # fn_print_int = c.builder.module.get_or_insert_function(fnty,
    #                                             name="print_int")
    # c.builder.call(fn_print_int, [string_array.size])

    string_list = c.pyapi.list_new(string_array.size)
    res = cgutils.alloca_once(c.builder, lir.IntType(8).as_pointer())
    c.builder.store(string_list, res)

    fnty = lir.FunctionType( lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(),
                            lir.IntType(8).as_pointer(),
                            lir.IntType(64)])
    fn_getitem = c.builder.module.get_or_insert_function(fnty,
                                                name="getitem_string_array")

    with cgutils.for_range(c.builder, string_array.size) as loop:
        c_str = c.builder.call(fn_getitem, [string_array.offsets,
                                    string_array.data, loop.index])
        pystr = c.pyapi.string_from_string(c_str)
        c.pyapi.list_setitem(string_list, loop.index, pystr)

    c.context.nrt.decref(c.builder, typ, val)
    return c.builder.load(res)


@lower_builtin('getitem', StringArrayType, types.Integer)
def lower_string_arr_getitem(context, builder, sig, args):
    typ = sig.args[0]
    string_array = cgutils.create_struct_proxy(typ)(context, builder, args[0])
    fnty = lir.FunctionType( lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(),
                            lir.IntType(8).as_pointer(),
                            lir.IntType(64)])
    fn_getitem = builder.module.get_or_insert_function(fnty,
                                                name="getitem_string_array_std")
    return builder.call(fn_getitem, [string_array.offsets,
                                string_array.data, args[1]])

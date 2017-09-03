import numba
from numba.extending import (box, unbox, typeof_impl, register_model, models,
                            NativeValue, lower_builtin, lower_cast)
from numba.targets.imputils import lower_constant, impl_ret_new_ref
from numba import types, typing
from numba.typing.templates import (signature, AbstractTemplate, infer, infer_getattr,
        ConcreteTemplate, AttributeTemplate, bound_function, infer_global)
from numba import cgutils
from llvmlite import ir as lir
import llvmlite.binding as ll

class StringType(types.Opaque):
    def __init__(self):
        super(StringType, self).__init__(name='StringType')

string_type = StringType()

@typeof_impl.register(str)
def _typeof_str(val, c):
    return string_type

register_model(StringType)(models.OpaqueModel)

@infer
class StringAdd(ConcreteTemplate):
    key = "+"
    cases = [signature(string_type, string_type, string_type)]

@infer
class StringOpEq(AbstractTemplate):
    key = '=='
    def generic(self, args, kws):
        assert not kws
        (arg1, arg2) = args
        if isinstance(arg1, StringType) and isinstance(arg2, StringType):
            return signature(types.boolean, arg1, arg2)

@infer
class StringOpNotEq(StringOpEq):
    key = '!='

@infer_getattr
class StringAttribute(AttributeTemplate):
    key = StringType

    @bound_function("str.split")
    def resolve_split(self, dict, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(types.List(string_type), *args)

@infer
class GetItemString(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        if (len(args) == 2 and isinstance(args[0], StringType)
                and isinstance(args[1], types.Integer)):
            return signature(args[0], *args)

@infer_global(int)
class StrToInt(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, StringType):
            return signature(types.intp, arg)

import hstr_ext
ll.add_symbol('init_string', hstr_ext.init_string)
ll.add_symbol('init_string_const', hstr_ext.init_string_const)
ll.add_symbol('get_c_str', hstr_ext.get_c_str)
ll.add_symbol('str_concat', hstr_ext.str_concat)
ll.add_symbol('str_equal', hstr_ext.str_equal)
ll.add_symbol('str_split', hstr_ext.str_split)
ll.add_symbol('str_substr_int', hstr_ext.str_substr_int)
ll.add_symbol('str_to_int64', hstr_ext.str_to_int64)

@unbox(StringType)
def unbox_string(typ, obj, c):
    """
    """
    ok, buffer, size = c.pyapi.string_as_string_and_size(obj)

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = c.builder.module.get_or_insert_function(fnty, name="init_string")
    ret = c.builder.call(fn, [buffer, size])

    return NativeValue(ret, is_error=c.builder.not_(ok))

@box(StringType)
def box_str(typ, val, c):
    """
    """
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = c.builder.module.get_or_insert_function(fnty, name="get_c_str")
    c_str = c.builder.call(fn, [val])
    pystr = c.pyapi.string_from_string(c_str)
    return pystr

@lower_constant(StringType)
def const_string(context, builder, ty, pyval):
    cstr = context.insert_const_string(builder.module, pyval)

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="init_string_const")
    ret = builder.call(fn, [cstr])
    return ret

@lower_builtin("+", string_type, string_type)
def impl_string_concat(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_concat")
    return builder.call(fn, args)

@lower_builtin('==', string_type, string_type)
def string_eq_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                    [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_equal")
    return builder.call(fn, args)

@lower_builtin('!=', string_type, string_type)
def string_neq_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                    [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_equal")
    return builder.not_(builder.call(fn, args))

@lower_builtin("str.split", string_type, string_type)
def string_split_impl(context, builder, sig, args):
    nitems = cgutils.alloca_once(builder, lir.IntType(64))
    # input str, sep, size pointer
    fnty = lir.FunctionType(lir.IntType(8).as_pointer().as_pointer(),
                [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer(),
                lir.IntType(64).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_split")
    ptr = builder.call(fn, args+[nitems])
    size = builder.load(nitems)
    # TODO: use ptr instead of allocating and copying, use NRT_MemInfo_new
    # TODO: deallocate ptr
    _list = numba.targets.listobj.ListInstance.allocate(context, builder,
                                    sig.return_type, size)
    _list.size = size
    with cgutils.for_range(builder, size) as loop:
        value = builder.load(cgutils.gep_inbounds(builder, ptr, loop.index))
        _list.setitem(loop.index, value)
    return impl_ret_new_ref(context, builder, sig.return_type, _list.value)

@lower_builtin('getitem', StringType, types.Integer)
def getitem_string(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                    [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="str_substr_int")
    # TODO: handle reference counting
    #return impl_ret_new_ref(builder.call(fn, args))
    return (builder.call(fn, args))

@lower_cast(StringType, types.int64)
def dict_empty(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_to_int64")
    return builder.call(fn, (val,))

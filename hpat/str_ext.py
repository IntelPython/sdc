import operator
import numba
from numba.extending import (box, unbox, typeof_impl, register_model, models,
                             NativeValue, lower_builtin, lower_cast, overload,
                             type_callable, overload_method)
from numba.targets.imputils import lower_constant, impl_ret_new_ref, impl_ret_untracked
from numba import types, typing
from numba.typing.templates import (signature, AbstractTemplate, infer, infer_getattr,
                                    ConcreteTemplate, AttributeTemplate, bound_function, infer_global)
from numba import cgutils
import llvmlite.llvmpy.core as lc
from llvmlite import ir as lir
import llvmlite.binding as ll
import hpat
#from hpat.utils import unliteral_all
# TODO: resolve import conflict
def unliteral_all(args):
    return tuple(types.unliteral(a) for a in args)

import hstr_ext
ll.add_symbol('get_char_from_string', hstr_ext.get_char_from_string)
ll.add_symbol('get_char_ptr', hstr_ext.get_char_ptr)
ll.add_symbol('del_str', hstr_ext.del_str)
ll.add_symbol('_hash_str', hstr_ext.hash_str)

class StringType(types.Opaque):
    def __init__(self):
        super(StringType, self).__init__(name='StringType')

string_type = StringType()


@typeof_impl.register(str)
def _typeof_str(val, c):
    return string_type


register_model(StringType)(models.OpaqueModel)

# XXX: should be subtype of StringType?
class CharType(types.Type):
    def __init__(self):
        super(CharType, self).__init__(name='CharType')
        self.bitwidth = 8

char_type = CharType()
register_model(CharType)(models.IntegerModel)

@overload(operator.getitem)
def char_getitem_overload(_str, ind):
    if _str == string_type and isinstance(ind, types.Integer):
        sig = char_type(
                    string_type,   # string
                    types.intp,    # index
                    )
        get_char_from_string = types.ExternalFunction("get_char_from_string", sig)
        def impl(s, i):
            return get_char_from_string(s, i)

        return impl

# XXX: fix overload for getitem and use it
@lower_builtin(operator.getitem, StringType, types.Integer)
def getitem_string(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8),
                            [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="get_char_from_string")
    return builder.call(fn, args)


@box(CharType)
def box_char(typ, val, c):
    """
    """
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8)])
    fn = c.builder.module.get_or_insert_function(fnty, name="get_char_ptr")
    c_str = c.builder.call(fn, [val])
    pystr = c.pyapi.string_from_string_and_size(c_str, c.context.get_constant(types.intp, 1))
    # TODO: delete ptr
    return pystr

del_str = types.ExternalFunction("del_str", types.void(string_type))
_hash_str = types.ExternalFunction("_hash_str", types.int64(string_type))
get_c_str = types.ExternalFunction("get_c_str", types.voidptr(string_type))

@overload_method(StringType, 'c_str')
def str_c_str(str_typ):
    return lambda s: get_c_str(s)

@overload_method(StringType, 'join')
def str_join(str_typ, iterable_typ):
    # TODO: more efficient implementation (e.g. C++ string buffer)
    def str_join_impl(sep_str, str_container):
        res = ""
        counter = 0
        for s in str_container:
            if counter != 0:
                res += sep_str
            counter += 1
            res += s
        return res
    return str_join_impl

@overload(hash)
def hash_overload(str_typ):
    if str_typ == string_type:
        return lambda s: _hash_str(s)

@infer
@infer_global(operator.add)
@infer_global(operator.iadd)
class StringAdd(ConcreteTemplate):
    key = "+"
    cases = [signature(string_type, string_type, string_type)]


@infer
@infer_global(operator.eq)
@infer_global(operator.ne)
@infer_global(operator.ge)
@infer_global(operator.gt)
@infer_global(operator.le)
@infer_global(operator.lt)
class StringOpEq(AbstractTemplate):
    key = '=='

    def generic(self, args, kws):
        assert not kws
        (arg1, arg2) = args
        if isinstance(arg1, StringType) and isinstance(arg2, StringType):
            return signature(types.boolean, arg1, arg2)
        if arg1 == char_type and arg2 == char_type:
            return signature(types.boolean, arg1, arg2)


@infer
class StringOpNotEq(StringOpEq):
    key = '!='

@infer
class StringOpGE(StringOpEq):
    key = '>='

@infer
class StringOpGT(StringOpEq):
    key = '>'

@infer
class StringOpLE(StringOpEq):
    key = '<='

@infer
class StringOpLT(StringOpEq):
    key = '<'


@infer_getattr
class StringAttribute(AttributeTemplate):
    key = StringType

    @bound_function("str.split")
    def resolve_split(self, dict, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(types.List(string_type), types.unliteral(args[0]))


# @infer_global(operator.getitem)
# class GetItemString(AbstractTemplate):
#     key = operator.getitem
#
#     def generic(self, args, kws):
#         assert not kws
#         if (len(args) == 2 and isinstance(args[0], StringType)
#                 and isinstance(args[1], types.Integer)):
#             return signature(args[0], *args)


@infer_global(len)
class LenStringArray(AbstractTemplate):
    def generic(self, args, kws):
        if not kws and len(args) == 1 and args[0] == string_type:
            return signature(types.intp, *args)


@infer_global(int)
class StrToInt(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, StringType):
            return signature(types.intp, arg)


@infer_global(float)
class StrToFloat(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, StringType):
            return signature(types.float64, arg)


@infer_global(str)
class StrConstInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        assert args[0] in [types.int32, types.int64, types.float32, types.float64, string_type]
        return signature(string_type, *args)


class RegexType(types.Opaque):
    def __init__(self):
        super(RegexType, self).__init__(name='RegexType')


regex_type = RegexType()

register_model(RegexType)(models.OpaqueModel)


def compile_regex(pat):
    return 0


@infer_global(compile_regex)
class CompileRegexInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(regex_type, *unliteral_all(args))


def contains_regex(str, pat):
    return False


def contains_noregex(str, pat):
    return False


@infer_global(contains_regex)
@infer_global(contains_noregex)
class ContainsInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.boolean, *unliteral_all(args))

ll.add_symbol('init_string', hstr_ext.init_string)
ll.add_symbol('init_string_const', hstr_ext.init_string_const)
ll.add_symbol('get_c_str', hstr_ext.get_c_str)
ll.add_symbol('str_concat', hstr_ext.str_concat)
ll.add_symbol('str_compare', hstr_ext.str_compare)
ll.add_symbol('str_equal', hstr_ext.str_equal)
ll.add_symbol('str_equal_cstr', hstr_ext.str_equal_cstr)
ll.add_symbol('str_split', hstr_ext.str_split)
ll.add_symbol('str_substr_int', hstr_ext.str_substr_int)
ll.add_symbol('str_to_int64', hstr_ext.str_to_int64)
ll.add_symbol('str_to_float64', hstr_ext.str_to_float64)
ll.add_symbol('get_str_len', hstr_ext.get_str_len)
ll.add_symbol('compile_regex', hstr_ext.compile_regex)
ll.add_symbol('str_contains_regex', hstr_ext.str_contains_regex)
ll.add_symbol('str_contains_noregex', hstr_ext.str_contains_noregex)
ll.add_symbol('str_from_int32', hstr_ext.str_from_int32)
ll.add_symbol('str_from_int64', hstr_ext.str_from_int64)
ll.add_symbol('str_from_float32', hstr_ext.str_from_float32)
ll.add_symbol('str_from_float64', hstr_ext.str_from_float64)


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

def getpointer(str):
    pass

@type_callable(getpointer)
def type_string_getpointer(context):
    def typer(val):
        return types.voidptr
    return typer

@lower_builtin(getpointer, StringType)
def getpointer_from_string(context, builder, sig, args):
    val = args[0]
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="get_c_str")
    c_str = builder.call(fn, [val])
    return c_str

@lower_builtin(getpointer, types.StringLiteral)
def getpointer_from_string_literal(context, builder, sig, args):
    cstr = context.insert_const_string(builder.module, sig.args[0].literal_value)
    return cstr

@lower_cast(StringType, types.StringLiteral)
def string_type_to_const(context, builder, fromty, toty, val):
    # calling str() since the const value can be non-str like tuple const (CSV)
    cstr = context.insert_const_string(builder.module, str(toty.literal_value))
    # check to make sure Const value matches stored string
    # call str == cstr
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_equal_cstr")
    match = builder.call(fn, [val, cstr])
    with cgutils.if_unlikely(builder, builder.not_(match)):
        # Raise RuntimeError about the assumption violation
        usermsg = "constant string assumption violated"
        errmsg = "{}: expecting {}".format(usermsg, toty.literal_value)
        context.call_conv.return_user_exc(builder, RuntimeError, (errmsg,))

    return impl_ret_untracked(context, builder, toty, cstr)


@lower_constant(StringType)
def const_string(context, builder, ty, pyval):
    cstr = context.insert_const_string(builder.module, pyval)

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="init_string_const")
    ret = builder.call(fn, [cstr])
    return ret

@lower_cast(types.StringLiteral, StringType)
def const_to_string_type(context, builder, fromty, toty, val):
    cstr = context.insert_const_string(builder.module, fromty.literal_value)

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="init_string_const")
    ret = builder.call(fn, [cstr])
    return ret

@lower_builtin(str, types.Any)
def string_from_impl(context, builder, sig, args):
    in_typ = sig.args[0]
    if in_typ == string_type:
        return args[0]
    ll_in_typ = context.get_value_type(sig.args[0])
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [ll_in_typ])
    fn = builder.module.get_or_insert_function(
        fnty, name="str_from_" + str(in_typ))
    return builder.call(fn, args)


@lower_builtin(operator.add, string_type, string_type)
@lower_builtin("+", string_type, string_type)
def impl_string_concat(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_concat")
    return builder.call(fn, args)


@lower_builtin(operator.eq, string_type, string_type)
@lower_builtin('==', string_type, string_type)
def string_eq_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_equal")
    return builder.call(fn, args)

@lower_builtin(operator.eq, char_type, char_type)
@lower_builtin('==', char_type, char_type)
def char_eq_impl(context, builder, sig, args):
    def char_eq_impl(c1, c2):
        return c1==c2
    new_sig = signature(sig.return_type, types.uint8, types.uint8)
    res = context.compile_internal(builder, char_eq_impl, new_sig, args)
    return res


@lower_builtin(operator.ne, string_type, string_type)
@lower_builtin('!=', string_type, string_type)
def string_neq_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_equal")
    return builder.not_(builder.call(fn, args))

@lower_builtin(operator.ge, string_type, string_type)
@lower_builtin('>=', string_type, string_type)
def string_ge_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_compare")
    comp_val = builder.call(fn, args)
    zero = context.get_constant(types.int32, 0)
    res = builder.icmp(lc.ICMP_SGE, comp_val, zero)
    return res

@lower_builtin(operator.gt, string_type, string_type)
@lower_builtin('>', string_type, string_type)
def string_gt_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_compare")
    comp_val = builder.call(fn, args)
    zero = context.get_constant(types.int32, 0)
    res = builder.icmp(lc.ICMP_SGT, comp_val, zero)
    return res

@lower_builtin(operator.le, string_type, string_type)
@lower_builtin('<=', string_type, string_type)
def string_le_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_compare")
    comp_val = builder.call(fn, args)
    zero = context.get_constant(types.int32, 0)
    res = builder.icmp(lc.ICMP_SLE, comp_val, zero)
    return res

@lower_builtin(operator.lt, string_type, string_type)
@lower_builtin('<', string_type, string_type)
def string_lt_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_compare")
    comp_val = builder.call(fn, args)
    zero = context.get_constant(types.int32, 0)
    res = builder.icmp(lc.ICMP_SLT, comp_val, zero)
    return res


@lower_builtin("str.split", string_type, string_type)
def string_split_impl(context, builder, sig, args):
    nitems = cgutils.alloca_once(builder, lir.IntType(64))
    # input str, sep, size pointer
    fnty = lir.FunctionType(lir.IntType(8).as_pointer().as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer(),
                             lir.IntType(64).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_split")
    ptr = builder.call(fn, args + [nitems])
    size = builder.load(nitems)
    # TODO: use ptr instead of allocating and copying, use NRT_MemInfo_new
    # TODO: deallocate ptr
    _list = numba.targets.listobj.ListInstance.allocate(context, builder,
                                                        sig.return_type, size)
    _list.size = size
    with cgutils.for_range(builder, size) as loop:
        value = builder.load(cgutils.gep_inbounds(builder, ptr, loop.index))
        # TODO: refcounted str
        _list.setitem(loop.index, value, incref=False)
    return impl_ret_new_ref(context, builder, sig.return_type, _list.value)


# @lower_builtin(operator.getitem, StringType, types.Integer)
# def getitem_string(context, builder, sig, args):
#     fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
#                             [lir.IntType(8).as_pointer(), lir.IntType(64)])
#     fn = builder.module.get_or_insert_function(fnty, name="str_substr_int")
#     # TODO: handle reference counting
#     # return impl_ret_new_ref(builder.call(fn, args))
#     return (builder.call(fn, args))


@lower_cast(StringType, types.int64)
def cast_str_to_int64(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_to_int64")
    return builder.call(fn, (val,))


@lower_cast(StringType, types.float64)
def cast_str_to_float64(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.DoubleType(), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_to_float64")
    return builder.call(fn, (val,))


@lower_builtin(len, StringType)
def len_string(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(64),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="get_str_len")
    return (builder.call(fn, args))


@lower_builtin(compile_regex, string_type)
def lower_compile_regex(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="compile_regex")
    return builder.call(fn, args)


@lower_builtin(contains_regex, string_type, regex_type)
def impl_string_contains_regex(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_contains_regex")
    return builder.call(fn, args)


@lower_builtin(contains_noregex, string_type, string_type)
def impl_string_contains_noregex(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(
        fnty, name="str_contains_noregex")
    return builder.call(fn, args)

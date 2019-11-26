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
import re
import llvmlite.llvmpy.core as lc
from llvmlite import ir as lir
import llvmlite.binding as ll

import numba
from numba import cgutils, types
from numba.extending import (box, unbox, typeof_impl, register_model, models,
                             NativeValue, lower_builtin, lower_cast, overload,
                             type_callable, overload_method, intrinsic)
import numba.targets.hashing
from numba.targets.imputils import lower_constant, impl_ret_new_ref, impl_ret_untracked
from numba.typing.templates import (signature, AbstractTemplate, infer, infer_getattr,
                                    ConcreteTemplate, AttributeTemplate, bound_function, infer_global)

import sdc
from . import hstr_ext
# from sdc.utils import unliteral_all
# TODO: resolve import conflict


def unliteral_all(args):
    return tuple(types.unliteral(a) for a in args)


# relative import seems required for C extensions
ll.add_symbol('get_char_from_string', hstr_ext.get_char_from_string)
ll.add_symbol('get_char_ptr', hstr_ext.get_char_ptr)
ll.add_symbol('del_str', hstr_ext.del_str)
ll.add_symbol('_hash_str', hstr_ext.hash_str)


string_type = types.unicode_type


# XXX setting hash secret for hash(unicode_type) to be consistent across
# processes. Other wise, shuffle operators like unique_str_parallel will fail.
# TODO: use a seperate implementation?
# TODO: make sure hash(str) is not already instantiated in overloads
# def _rm_hash_str_overload():
#     try:
#         fn = numba.targets.registry.cpu_target.typing_context.resolve_value_type(hash)
#         sig = signature(types.int64, types.unicode_type)
#         key = fn.get_impl_key(sig)
#         numba.targets.registry.cpu_target.target_context._defns[key]._cache.pop(sig.args)
#     except:
#         pass

# _rm_hash_str_overload()

numba.targets.hashing._Py_HashSecret_djbx33a_suffix = 0
numba.targets.hashing._Py_HashSecret_siphash_k0 = 0
numba.targets.hashing._Py_HashSecret_siphash_k1 = 0


# use objmode for string methods for now

# string methods that take no arguments and return a string
str2str_noargs = ('capitalize', 'casefold', 'lower', 'swapcase', 'title', 'upper')


def str_overload_noargs(method):
    @overload_method(types.UnicodeType, method)
    def str_overload(in_str):
        def _str_impl(in_str):
            with numba.objmode(out='unicode_type'):
                out = getattr(in_str, method)()
            return out

        return _str_impl


for method in str2str_noargs:
    str_overload_noargs(method)

# strip string methods that take one argument and return a string
# Numba bug https://github.com/numba/numba/issues/4731
str2str_1arg = ('lstrip', 'rstrip', 'strip')


def str_overload_1arg(method):
    @overload_method(types.UnicodeType, method)
    def str_overload(in_str, arg1):
        def _str_impl(in_str, arg1):
            with numba.objmode(out='unicode_type'):
                out = getattr(in_str, method)(arg1)
            return out

        return _str_impl


for method in str2str_1arg:
    str_overload_1arg(method)


@overload_method(types.UnicodeType, 'replace')
def str_replace_overload(in_str, old, new, count=-1):

    def _str_replace_impl(in_str, old, new, count=-1):
        with numba.objmode(out='unicode_type'):
            out = in_str.replace(old, new, count)
        return out

    return _str_replace_impl


# ********************  re support  *******************

class RePatternType(types.Opaque):
    def __init__(self):
        super(RePatternType, self).__init__(name='RePatternType')


re_pattern_type = RePatternType()
types.re_pattern_type = re_pattern_type

register_model(RePatternType)(models.OpaqueModel)


@box(RePatternType)
def box_re_pattern(typ, val, c):
    # TODO: fix
    c.pyapi.incref(val)
    return val


@unbox(RePatternType)
def unbox_re_pattern(typ, obj, c):
    # TODO: fix
    c.pyapi.incref(obj)
    return NativeValue(obj)


# jitoptions until numba #4020 is resolved
@overload(re.compile, jit_options={'no_cpython_wrapper': False})
def re_compile_overload(pattern, flags=0):
    def _re_compile_impl(pattern, flags=0):
        with numba.objmode(pat='re_pattern_type'):
            pat = re.compile(pattern, flags)
        return pat
    return _re_compile_impl


@overload_method(RePatternType, 'sub')
def re_sub_overload(p, repl, string, count=0):
    def _re_sub_impl(p, repl, string, count=0):
        with numba.objmode(out='unicode_type'):
            out = p.sub(repl, string, count)
        return out
    return _re_sub_impl


# **********************  type for std string pointer  ************************


class StringType(types.Opaque, types.Hashable):
    def __init__(self):
        super(StringType, self).__init__(name='StringType')


std_str_type = StringType()

# XXX enabling this turns on old std::string implementation
# string_type = StringType()

# @typeof_impl.register(str)
# def _typeof_str(val, c):
#     return string_type


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
    if _str == std_str_type and isinstance(ind, types.Integer):
        sig = char_type(
            std_str_type,   # string
            types.intp,    # index
        )
        get_char_from_string = types.ExternalFunction("get_char_from_string", sig)

        def impl(_str, ind):
            return get_char_from_string(_str, ind)

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


del_str = types.ExternalFunction("del_str", types.void(std_str_type))
_hash_str = types.ExternalFunction("_hash_str", types.int64(std_str_type))
get_c_str = types.ExternalFunction("get_c_str", types.voidptr(std_str_type))


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


# TODO: using lower_builtin since overload fails for str tuple
# TODO: constant hash like hash("ss",) fails
# @overload(hash)
# def hash_overload(str_typ):
#     if str_typ == string_type:
#         return lambda s: _hash_str(s)

@lower_builtin(hash, std_str_type)
def hash_str_lower(context, builder, sig, args):
    return context.compile_internal(
        builder, lambda s: _hash_str(s), sig, args)

# XXX: use Numba's hash(str) when available
@lower_builtin(hash, string_type)
def hash_unicode_lower(context, builder, sig, args):
    std_str = gen_unicode_to_std_str(context, builder, args[0])
    return hash_str_lower(
        context, builder, signature(sig.return_type, std_str_type), [std_str])


@infer
@infer_global(operator.add)
@infer_global(operator.iadd)
class StringAdd(ConcreteTemplate):
    key = "+"
    cases = [signature(std_str_type, std_str_type, std_str_type)]


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
        return signature(types.List(std_str_type), types.unliteral(args[0]))


# @infer_global(operator.getitem)
# class GetItemString(AbstractTemplate):
#     key = operator.getitem
#
#     def generic(self, args, kws):
#         assert not kws
#         if (len(args) == 2 and isinstance(args[0], StringType)
#                 and isinstance(args[1], types.Integer)):
#             return signature(args[0], *args)


# @infer_global(len)
# class LenStringArray(AbstractTemplate):
#     def generic(self, args, kws):
#         if not kws and len(args) == 1 and args[0] == std_str_type:
#             return signature(types.intp, *args)


@overload(int)
def int_str_overload(in_str):
    if in_str == string_type:
        def _str_to_int_impl(in_str):
            return _str_to_int64(in_str._data, in_str._length)

        return _str_to_int_impl


# @infer_global(int)
# class StrToInt(AbstractTemplate):
#     def generic(self, args, kws):
#         assert not kws
#         [arg] = args
#         if isinstance(arg, StringType):
#             return signature(types.intp, arg)
#         # TODO: implement int(str) in Numba
#         if arg == string_type:
#             return signature(types.intp, arg)


@infer_global(float)
class StrToFloat(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, StringType):
            return signature(types.float64, arg)
        # TODO: implement int(str) in Numba
        if arg == string_type:
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
ll.add_symbol('std_str_to_int64', hstr_ext.std_str_to_int64)
ll.add_symbol('str_to_float64', hstr_ext.str_to_float64)
ll.add_symbol('get_str_len', hstr_ext.get_str_len)
ll.add_symbol('compile_regex', hstr_ext.compile_regex)
ll.add_symbol('str_contains_regex', hstr_ext.str_contains_regex)
ll.add_symbol('str_contains_noregex', hstr_ext.str_contains_noregex)
ll.add_symbol('str_replace_regex', hstr_ext.str_replace_regex)
ll.add_symbol('str_replace_noregex', hstr_ext.str_replace_noregex)
ll.add_symbol('str_from_int32', hstr_ext.str_from_int32)
ll.add_symbol('str_from_int64', hstr_ext.str_from_int64)
ll.add_symbol('str_from_float32', hstr_ext.str_from_float32)
ll.add_symbol('str_from_float64', hstr_ext.str_from_float64)

get_std_str_len = types.ExternalFunction(
    "get_str_len", signature(types.intp, std_str_type))
init_string_from_chars = types.ExternalFunction(
    "init_string_const", std_str_type(types.voidptr, types.intp))

_str_to_int64 = types.ExternalFunction(
    "str_to_int64", signature(types.intp, types.voidptr, types.intp))

str_replace_regex = types.ExternalFunction(
    "str_replace_regex", std_str_type(std_str_type, regex_type, std_str_type))

str_replace_noregex = types.ExternalFunction(
    "str_replace_noregex", std_str_type(std_str_type, std_str_type, std_str_type))


def gen_unicode_to_std_str(context, builder, unicode_val):
    #
    uni_str = cgutils.create_struct_proxy(string_type)(
        context, builder, value=unicode_val)
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="init_string_const")
    return builder.call(fn, [uni_str.data, uni_str.length])


def gen_std_str_to_unicode(context, builder, std_str_val, del_str=False):
    kind = numba.unicode.PY_UNICODE_1BYTE_KIND

    def _std_str_to_unicode(std_str):
        length = sdc.str_ext.get_std_str_len(std_str)
        ret = numba.unicode._empty_string(kind, length)
        sdc.str_arr_ext._memcpy(
            ret._data, sdc.str_ext.get_c_str(std_str), length, 1)
        if del_str:
            sdc.str_ext.del_str(std_str)
        return ret
    val = context.compile_internal(
        builder,
        _std_str_to_unicode,
        string_type(sdc.str_ext.std_str_type),
        [std_str_val])
    return val


def gen_get_unicode_chars(context, builder, unicode_val):
    uni_str = cgutils.create_struct_proxy(string_type)(
        context, builder, value=unicode_val)
    return uni_str.data


def unicode_to_char_ptr(in_str):
    return in_str


@overload(unicode_to_char_ptr)
def unicode_to_char_ptr_overload(a):
    # str._data is not safe since str might be literal
    # overload resolves str literal to unicode_type
    if a == string_type:
        return lambda a: a._data


@intrinsic
def unicode_to_std_str(typingctx, unicode_t=None):
    def codegen(context, builder, sig, args):
        return gen_unicode_to_std_str(context, builder, args[0])
    return std_str_type(string_type), codegen


@intrinsic
def std_str_to_unicode(typingctx, unicode_t=None):
    def codegen(context, builder, sig, args):
        return gen_std_str_to_unicode(context, builder, args[0], True)
    return string_type(std_str_type), codegen


@intrinsic
def alloc_str_list(typingctx, n_t=None):
    def codegen(context, builder, sig, args):
        nitems = args[0]
        list_type = types.List(string_type)
        result = numba.targets.listobj.ListInstance.allocate(context, builder, list_type, nitems)
        result.size = nitems
        return impl_ret_new_ref(context, builder, list_type, result.value)
    return types.List(string_type)(types.intp), codegen


# XXX using list of list string instead of array of list string since Numba's
# arrays can't store lists
list_string_array_type = types.List(types.List(string_type))


@intrinsic
def alloc_list_list_str(typingctx, n_t=None):
    def codegen(context, builder, sig, args):
        nitems = args[0]
        list_type = list_string_array_type
        result = numba.targets.listobj.ListInstance.allocate(context, builder, list_type, nitems)
        result.size = nitems
        return impl_ret_new_ref(context, builder, list_type, result.value)
    return list_string_array_type(types.intp), codegen


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
    length = context.get_constant(types.intp, len(pyval))

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="init_string_const")
    ret = builder.call(fn, [cstr, length])
    return ret


@lower_cast(types.StringLiteral, StringType)
def const_to_string_type(context, builder, fromty, toty, val):
    cstr = context.insert_const_string(builder.module, fromty.literal_value)
    length = context.get_constant(types.intp, len(fromty.literal_value))

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.IntType(64)])
    fn = builder.module.get_or_insert_function(fnty, name="init_string_const")
    ret = builder.call(fn, [cstr, length])
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
    std_str = builder.call(fn, args)
    return gen_std_str_to_unicode(context, builder, std_str)


@lower_builtin(operator.add, std_str_type, std_str_type)
@lower_builtin("+", std_str_type, std_str_type)
def impl_string_concat(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_concat")
    return builder.call(fn, args)


@lower_builtin(operator.eq, std_str_type, std_str_type)
@lower_builtin('==', std_str_type, std_str_type)
def string_eq_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_equal")
    return builder.call(fn, args)


@lower_builtin(operator.eq, char_type, char_type)
@lower_builtin('==', char_type, char_type)
def char_eq_impl(context, builder, sig, args):
    def char_eq_impl(c1, c2):
        return c1 == c2
    new_sig = signature(sig.return_type, types.uint8, types.uint8)
    res = context.compile_internal(builder, char_eq_impl, new_sig, args)
    return res


@lower_builtin(operator.ne, std_str_type, std_str_type)
@lower_builtin('!=', std_str_type, std_str_type)
def string_neq_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_equal")
    return builder.not_(builder.call(fn, args))


@lower_builtin(operator.ge, std_str_type, std_str_type)
@lower_builtin('>=', std_str_type, std_str_type)
def string_ge_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_compare")
    comp_val = builder.call(fn, args)
    zero = context.get_constant(types.int32, 0)
    res = builder.icmp(lc.ICMP_SGE, comp_val, zero)
    return res


@lower_builtin(operator.gt, std_str_type, std_str_type)
@lower_builtin('>', std_str_type, std_str_type)
def string_gt_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_compare")
    comp_val = builder.call(fn, args)
    zero = context.get_constant(types.int32, 0)
    res = builder.icmp(lc.ICMP_SGT, comp_val, zero)
    return res


@lower_builtin(operator.le, std_str_type, std_str_type)
@lower_builtin('<=', std_str_type, std_str_type)
def string_le_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_compare")
    comp_val = builder.call(fn, args)
    zero = context.get_constant(types.int32, 0)
    res = builder.icmp(lc.ICMP_SLE, comp_val, zero)
    return res


@lower_builtin(operator.lt, std_str_type, std_str_type)
@lower_builtin('<', std_str_type, std_str_type)
def string_lt_impl(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_compare")
    comp_val = builder.call(fn, args)
    zero = context.get_constant(types.int32, 0)
    res = builder.icmp(lc.ICMP_SLT, comp_val, zero)
    return res


@lower_builtin("str.split", std_str_type, std_str_type)
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
    fn = builder.module.get_or_insert_function(fnty, name="std_str_to_int64")
    return builder.call(fn, (val,))

# # XXX handle unicode until Numba supports int(str)
# @lower_cast(string_type, types.int64)
# def cast_unicode_str_to_int64(context, builder, fromty, toty, val):
#     std_str = gen_unicode_to_std_str(context, builder, val)
#     return cast_str_to_int64(context, builder, std_str_type, toty, std_str)


@lower_cast(StringType, types.float64)
def cast_str_to_float64(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.DoubleType(), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_to_float64")
    return builder.call(fn, (val,))

# XXX handle unicode until Numba supports float(str)
@lower_cast(string_type, types.float64)
def cast_unicode_str_to_float64(context, builder, fromty, toty, val):
    std_str = gen_unicode_to_std_str(context, builder, val)
    return cast_str_to_float64(context, builder, std_str_type, toty, std_str)

# @lower_builtin(len, StringType)
# def len_string(context, builder, sig, args):
#     fnty = lir.FunctionType(lir.IntType(64),
#                             [lir.IntType(8).as_pointer()])
#     fn = builder.module.get_or_insert_function(fnty, name="get_str_len")
#     return (builder.call(fn, args))


@lower_builtin(compile_regex, std_str_type)
def lower_compile_regex(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                            [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="compile_regex")
    return builder.call(fn, args)


@lower_builtin(compile_regex, string_type)
def lower_compile_regex_unicode(context, builder, sig, args):
    val = args[0]
    std_val = gen_unicode_to_std_str(context, builder, val)
    return lower_compile_regex(context, builder, sig, [std_val])


@lower_builtin(contains_regex, std_str_type, regex_type)
def impl_string_contains_regex(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_contains_regex")
    return builder.call(fn, args)


@lower_builtin(contains_regex, string_type, regex_type)
def impl_unicode_string_contains_regex(context, builder, sig, args):
    val, reg = args
    std_val = gen_unicode_to_std_str(context, builder, val)
    return impl_string_contains_regex(
        context, builder, sig, [std_val, reg])


@lower_builtin(contains_noregex, std_str_type, std_str_type)
def impl_string_contains_noregex(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(1),
                            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(
        fnty, name="str_contains_noregex")
    return builder.call(fn, args)


@lower_builtin(contains_noregex, string_type, string_type)
def impl_unicode_string_contains_noregex(context, builder, sig, args):
    val, pat = args
    std_val = gen_unicode_to_std_str(context, builder, val)
    std_pat = gen_unicode_to_std_str(context, builder, pat)
    return impl_string_contains_noregex(
        context, builder, sig, [std_val, std_pat])

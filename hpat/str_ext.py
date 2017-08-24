from numba.extending import (box, unbox, typeof_impl, register_model, models,
                            NativeValue, lower_builtin)
from numba.targets.imputils import lower_constant
from numba import types, typing
from numba.typing.templates import (signature, AbstractTemplate, infer,
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

import hstr_ext
ll.add_symbol('init_string', hstr_ext.init_string)
ll.add_symbol('init_string_const', hstr_ext.init_string_const)
ll.add_symbol('get_c_str', hstr_ext.get_c_str)
ll.add_symbol('str_concat', hstr_ext.str_concat)
ll.add_symbol('str_equal', hstr_ext.str_equal)

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

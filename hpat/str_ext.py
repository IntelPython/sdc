from numba.extending import box, unbox, typeof_impl, register_model, models, NativeValue
from numba import types, typing
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

import hstr_ext
ll.add_symbol('init_string', hstr_ext.init_string)
ll.add_symbol('get_c_str', hstr_ext.get_c_str)

@unbox(StringType)
def unbox_string(typ, obj, c):
    """
    """
    lty = c.context.get_value_type(typ)
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
    return c.pyapi.string_as_string(c_str)

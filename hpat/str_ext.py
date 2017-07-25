from numba.extending import box
from numba import types, typing
from numba import cgutils
import hpat

class StringType(types.Opaque):
    def __init__(self):
        super(StringType, self).__init__(name='StringType')

@typing.typeof.typeof_impl.register(str)
def _typeof_str(val, c):
    return StringType

@box(StringType)
def box_str(typ, val, c):
    """
    """
    # interval = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    # lo_obj = c.pyapi.float_from_double(interval.lo)
    # hi_obj = c.pyapi.float_from_double(interval.hi)
#    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(DictIntInt))
#    res = c.pyapi.call_function_objargs(class_obj, (val,))
    # c.pyapi.decref(lo_obj)
    # c.pyapi.decref(hi_obj)
#    c.pyapi.decref(class_obj)
#    return res

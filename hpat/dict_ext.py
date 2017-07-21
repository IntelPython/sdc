import numba
from numba import types
from numba.extending import typeof_impl
from numba.extending import type_callable
from numba.extending import models, register_model
from numba.extending import lower_builtin
from numba import cgutils
from llvmlite import ir as lir
import llvmlite.binding as ll

class DictType(types.Opaque):
    def __init__(self, key_typ, val_typ):
        self.key_typ = key_typ
        self.val_typ = val_typ
        super(DictType, self).__init__(name='DictType')

dict_int_int_type = DictType(types.intp, types.intp)

class DictIntInt(object):
    def __new__(cls, *args):
        return {}

@typeof_impl.register(DictIntInt)
def typeof_index(val, c):
    print(val, c)
    return dict_int_int_type

@type_callable(DictIntInt)
def type_dict(context):
    def typer():
        return dict_int_int_type
    return typer

register_model(DictType)(models.OpaqueModel)

import hdict_ext
ll.add_symbol('init_dict_int_int', hdict_ext.init_dict_int_int)

@lower_builtin(DictIntInt)
def impl_dict_int_int(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [])
    fn = builder.module.get_or_insert_function(fnty, name="init_dict_int_int")
    return builder.call(fn, [])

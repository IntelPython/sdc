import numba
import hpat
from numba import types
from numba.typing.templates import (infer_global, AbstractTemplate, infer,
                    signature, AttributeTemplate, infer_getattr, bound_function)
from numba.extending import (typeof_impl, type_callable, models, register_model,
                                make_attribute_wrapper, lower_builtin, box, lower_getattr)
from numba import cgutils
from hpat.str_ext import string_type
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed

class SVC(object):
    def __init__(self):
        return

class SVCType(types.Type):
    def __init__(self):
        super(SVCType, self).__init__(
                                    name='SVCType()')

svc_type = SVCType()

@typeof_impl.register(SVC)
def typeof_svc_val(val, c):
    return svc_type

@type_callable(SVC)
def type_svc_call(context):
    def typer():
        return svc_type
    return typer

register_model(SVCType)(models.OpaqueModel)

@infer_getattr
class SVCAttribute(AttributeTemplate):
    key = SVCType

    @bound_function("svc.train")
    def resolve_train(self, dict, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.none, *args)

from llvmlite import ir as lir
import llvmlite.binding as ll
import daal_wrapper
ll.add_symbol('svc_train', daal_wrapper.svc_train)

@lower_builtin(SVC)
def impl_svc_constructor(context, builder, sig, args):
    zero = context.get_constant(types.intp, 0)
    #return builder.bitcast(zero, lir.IntType(8).as_pointer())
    return context.get_dummy_value()

@lower_builtin("svc.train", svc_type, types.Array, types.Array)
def string_split_impl(context, builder, sig, args):
    X = context.make_array(sig.args[1])(context, builder, args[1])
    y = context.make_array(sig.args[2])(context, builder, args[2])

    zero = context.get_constant(types.intp, 0)
    one = context.get_constant(types.intp, 1)

    # num_features = builder.load(builder.gep(X.shape, [one]))
    # num_samples = builder.load(builder.gep(X.shape, [zero]))
    num_features = builder.extract_value(X.shape, 1)
    num_samples = builder.extract_value(X.shape, 0)

    call_args = [num_features, num_samples, X.data, y.data]

    # num_features, num_samples, X, y
    arg_typs = [lir.IntType(64), lir.IntType(64), lir.DoubleType().as_pointer(), lir.DoubleType().as_pointer()]
    fnty = lir.FunctionType(lir.VoidType(), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="svc_train")
    builder.call(fn, call_args)
    return context.get_dummy_value()

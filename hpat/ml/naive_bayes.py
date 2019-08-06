from llvmlite import ir as lir
import llvmlite.binding as ll
import numba
import hpat
from hpat.utils import debug_prints
from numba import types
from numba.typing.templates import (infer_global, AbstractTemplate, infer,
                                    signature, AttributeTemplate, infer_getattr, bound_function)
from numba.extending import (typeof_impl, type_callable, models, register_model,
                             make_attribute_wrapper, lower_builtin, box, lower_getattr)
from numba import cgutils, utils
from numba.targets.arrayobj import _empty_nd_impl
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed


class MultinomialNB(object):
    def __init__(self, nclasses=-1):
        self.n_classes = nclasses
        return


class MultinomialNBType(types.Type):
    def __init__(self):
        super(MultinomialNBType, self).__init__(
            name='MultinomialNBType()')


mnb_type = MultinomialNBType()


class MultinomialNBPayloadType(types.Type):
    def __init__(self):
        super(MultinomialNBPayloadType, self).__init__(
            name='MultinomialNBPayloadType()')


@typeof_impl.register(MultinomialNB)
def typeof_mnb_val(val, c):
    return mnb_type

# @type_callable(MultinomialNB)
# def type_mnb_call(context):
#     def typer(nclasses = None):
#         return mnb_type
#     return typer

# dummy function providing pysignature for MultinomialNB()


def MultinomialNB_dummy(n_classes=-1):
    return 1


@infer_global(MultinomialNB)
class MultinomialNBConstructorInfer(AbstractTemplate):
    def generic(self, args, kws):
        sig = signature(mnb_type, types.intp)
        pysig = utils.pysignature(MultinomialNB_dummy)
        sig.pysig = pysig
        return sig


@register_model(MultinomialNBType)
class MultinomialNBDataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        dtype = MultinomialNBPayloadType()
        members = [
            ('meminfo', types.MemInfoPointer(dtype)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(MultinomialNBPayloadType)
class MultinomialNBPayloadDataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('model', types.Opaque('daal_model')),
            ('n_classes', types.intp),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@infer_getattr
class MultinomialNBAttribute(AttributeTemplate):
    key = MultinomialNBType

    @bound_function("mnb.train")
    def resolve_train(self, dict, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.none, *args)

    @bound_function("mnb.predict")
    def resolve_predict(self, dict, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(types.Array(types.int32, 1, 'C'), *args)


try:
    import daal_wrapper
    ll.add_symbol('mnb_train', daal_wrapper.mnb_train)
    ll.add_symbol('mnb_predict', daal_wrapper.mnb_predict)
    ll.add_symbol('dtor_mnb', daal_wrapper.dtor_mnb)
except ImportError:
    if debug_prints():  # pragma: no cover
        print("daal import error")


@lower_builtin(MultinomialNB, types.intp)
def impl_mnb_constructor(context, builder, sig, args):

    dtype = MultinomialNBPayloadType()
    alloc_type = context.get_data_type(dtype)
    alloc_size = context.get_abi_sizeof(alloc_type)

    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = lir.FunctionType(lir.VoidType(),
                                  [llvoidptr, llsize, llvoidptr])
    dtor_fn = builder.module.get_or_insert_function(dtor_ftype, name="dtor_mnb")

    meminfo = context.nrt.meminfo_alloc_dtor(
        builder,
        context.get_constant(types.uintp, alloc_size),
        dtor_fn,
    )
    data_pointer = context.nrt.meminfo_data(builder, meminfo)
    data_pointer = builder.bitcast(data_pointer,
                                   alloc_type.as_pointer())

    mnb_payload = cgutils.create_struct_proxy(dtype)(context, builder)
    mnb_payload.n_classes = args[0]
    builder.store(mnb_payload._getvalue(),
                  data_pointer)

    mnb_struct = cgutils.create_struct_proxy(mnb_type)(context, builder)
    mnb_struct.meminfo = meminfo
    return mnb_struct._getvalue()


@lower_builtin("mnb.train", mnb_type, types.Array, types.Array)
def mnb_train_impl(context, builder, sig, args):
    X = context.make_array(sig.args[1])(context, builder, args[1])
    y = context.make_array(sig.args[2])(context, builder, args[2])

    zero = context.get_constant(types.intp, 0)
    one = context.get_constant(types.intp, 1)

    # num_features = builder.load(builder.gep(X.shape, [one]))
    # num_samples = builder.load(builder.gep(X.shape, [zero]))
    num_features = builder.extract_value(X.shape, 1)
    num_samples = builder.extract_value(X.shape, 0)

    # num_features, num_samples, X, y
    arg_typs = [lir.IntType(64), lir.IntType(64),
                lir.IntType(32).as_pointer(), lir.IntType(32).as_pointer(),
                lir.IntType(64).as_pointer()]
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="mnb_train")

    dtype = MultinomialNBPayloadType()
    inst_struct = context.make_helper(builder, mnb_type, args[0])
    data_pointer = context.nrt.meminfo_data(builder, inst_struct.meminfo)
    data_pointer = builder.bitcast(data_pointer,
                                   context.get_data_type(dtype).as_pointer())

    mnb_struct = cgutils.create_struct_proxy(dtype)(context, builder, builder.load(data_pointer))

    call_args = [num_features, num_samples, X.data, y.data,
                 mnb_struct._get_ptr_by_name('n_classes')]
    model = builder.call(fn, call_args)
    mnb_struct.model = model
    builder.store(mnb_struct._getvalue(), data_pointer)
    return context.get_dummy_value()


@lower_builtin("mnb.predict", mnb_type, types.Array)
def mnb_predict_impl(context, builder, sig, args):

    dtype = MultinomialNBPayloadType()
    inst_struct = context.make_helper(builder, mnb_type, args[0])
    data_pointer = context.nrt.meminfo_data(builder, inst_struct.meminfo)
    data_pointer = builder.bitcast(data_pointer,
                                   context.get_data_type(dtype).as_pointer())
    mnb_struct = cgutils.create_struct_proxy(dtype)(context, builder, builder.load(data_pointer))

    p = context.make_array(sig.args[1])(context, builder, args[1])

    num_features = builder.extract_value(p.shape, 1)
    num_samples = builder.extract_value(p.shape, 0)

    ret_arr = _empty_nd_impl(context, builder, sig.return_type, [num_samples])

    call_args = [mnb_struct.model, num_features, num_samples, p.data, ret_arr.data, mnb_struct.n_classes]

    # model, num_features, num_samples, p, ret
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(64),
                lir.IntType(32).as_pointer(), lir.IntType(32).as_pointer(),
                lir.IntType(64)]
    fnty = lir.FunctionType(lir.VoidType(), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="mnb_predict")
    builder.call(fn, call_args)

    return impl_ret_new_ref(context, builder, sig.return_type, ret_arr._getvalue())

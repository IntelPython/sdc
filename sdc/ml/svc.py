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


from llvmlite import ir as lir
import llvmlite.binding as ll
import numba
import sdc
from sdc.utils import debug_prints
from numba import types
from numba.typing.templates import (infer_global, AbstractTemplate, infer,
                                    signature, AttributeTemplate, infer_getattr, bound_function)
from numba.extending import (typeof_impl, type_callable, models, register_model,
                             make_attribute_wrapper, lower_builtin, box, lower_getattr)
from numba import cgutils, utils
from numba.targets.arrayobj import _empty_nd_impl
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed


class SVC(object):
    def __init__(self, nclasses=-1):
        self.n_classes = nclasses
        return


class SVCType(types.Type):
    def __init__(self):
        super(SVCType, self).__init__(
            name='SVCType()')


svc_type = SVCType()


class SVCPayloadType(types.Type):
    def __init__(self):
        super(SVCPayloadType, self).__init__(
            name='SVCPayloadType()')


@typeof_impl.register(SVC)
def typeof_svc_val(val, c):
    return svc_type

# @type_callable(SVC)
# def type_svc_call(context):
#     def typer(nclasses = None):
#         return svc_type
#     return typer

# dummy function providing pysignature for SVC()


def SVC_dummy(n_classes=-1):
    return 1


@infer_global(SVC)
class SVCConstructorInfer(AbstractTemplate):
    def generic(self, args, kws):
        sig = signature(svc_type, types.intp)
        pysig = utils.pysignature(SVC_dummy)
        sig.pysig = pysig
        return sig


@register_model(SVCType)
class SVCDataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        dtype = SVCPayloadType()
        members = [
            ('meminfo', types.MemInfoPointer(dtype)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(SVCPayloadType)
class SVCPayloadDataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('model', types.Opaque('daal_model')),
            ('n_classes', types.intp),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@infer_getattr
class SVCAttribute(AttributeTemplate):
    key = SVCType

    @bound_function("svc.train")
    def resolve_train(self, dict, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.none, *args)

    @bound_function("svc.predict")
    def resolve_predict(self, dict, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(types.Array(types.float64, 1, 'C'), *args)


try:
    import daal_wrapper
    ll.add_symbol('svc_train', daal_wrapper.svc_train)
    ll.add_symbol('svc_predict', daal_wrapper.svc_predict)
    ll.add_symbol('dtor_svc', daal_wrapper.dtor_svc)
except ImportError:
    if debug_prints():  # pragma: no cover
        print("daal import error")


@lower_builtin(SVC, types.intp)
def impl_svc_constructor(context, builder, sig, args):

    dtype = SVCPayloadType()
    alloc_type = context.get_data_type(dtype)
    alloc_size = context.get_abi_sizeof(alloc_type)

    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = lir.FunctionType(lir.VoidType(),
                                  [llvoidptr, llsize, llvoidptr])
    dtor_fn = builder.module.get_or_insert_function(dtor_ftype, name="dtor_svc")

    meminfo = context.nrt.meminfo_alloc_dtor(
        builder,
        context.get_constant(types.uintp, alloc_size),
        dtor_fn,
    )
    data_pointer = context.nrt.meminfo_data(builder, meminfo)
    data_pointer = builder.bitcast(data_pointer,
                                   alloc_type.as_pointer())

    svc_payload = cgutils.create_struct_proxy(dtype)(context, builder)
    svc_payload.n_classes = args[0]
    builder.store(svc_payload._getvalue(),
                  data_pointer)

    svc_struct = cgutils.create_struct_proxy(svc_type)(context, builder)
    svc_struct.meminfo = meminfo
    return svc_struct._getvalue()


@lower_builtin("svc.train", svc_type, types.Array, types.Array)
def svc_train_impl(context, builder, sig, args):
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
                lir.DoubleType().as_pointer(), lir.DoubleType().as_pointer(),
                lir.IntType(64).as_pointer()]
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="svc_train")

    dtype = SVCPayloadType()
    inst_struct = context.make_helper(builder, svc_type, args[0])
    data_pointer = context.nrt.meminfo_data(builder, inst_struct.meminfo)
    data_pointer = builder.bitcast(data_pointer,
                                   context.get_data_type(dtype).as_pointer())

    svc_struct = cgutils.create_struct_proxy(dtype)(context, builder, builder.load(data_pointer))

    call_args = [num_features, num_samples, X.data, y.data,
                 svc_struct._get_ptr_by_name('n_classes')]
    model = builder.call(fn, call_args)
    svc_struct.model = model
    builder.store(svc_struct._getvalue(), data_pointer)
    return context.get_dummy_value()


@lower_builtin("svc.predict", svc_type, types.Array)
def svc_predict_impl(context, builder, sig, args):

    dtype = SVCPayloadType()
    inst_struct = context.make_helper(builder, svc_type, args[0])
    data_pointer = context.nrt.meminfo_data(builder, inst_struct.meminfo)
    data_pointer = builder.bitcast(data_pointer,
                                   context.get_data_type(dtype).as_pointer())
    svc_struct = cgutils.create_struct_proxy(dtype)(context, builder, builder.load(data_pointer))

    p = context.make_array(sig.args[1])(context, builder, args[1])

    num_features = builder.extract_value(p.shape, 1)
    num_samples = builder.extract_value(p.shape, 0)

    ret_arr = _empty_nd_impl(context, builder, sig.return_type, [num_samples])

    call_args = [svc_struct.model, num_features, num_samples, p.data, ret_arr.data, svc_struct.n_classes]

    # model, num_features, num_samples, p, ret
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(64),
                lir.DoubleType().as_pointer(), lir.DoubleType().as_pointer(),
                lir.IntType(64)]
    fnty = lir.FunctionType(lir.VoidType(), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="svc_predict")
    builder.call(fn, call_args)

    return impl_ret_new_ref(context, builder, sig.return_type, ret_arr._getvalue())

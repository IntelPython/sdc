# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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

import ctypes as ct

import numba
from numba.core import cgutils, types
from numba.extending import (
    intrinsic,
    overload,
    overload_method,
    type_callable,
    lower_builtin,
    infer_getattr,
)
from numba.core.typing.templates import signature, bound_function, AttributeTemplate
from llvmlite import ir as lir
import llvmlite.binding as ll

from sdc.extensions.sdc_string_view_type import StdStringViewType
from sdc import str_arr_ext
from sdc import hstr_ext
from sdc.str_arr_ext import decode_utf8
from sdc.extensions.sdc_hashmap_ext import load_native_func

load_native_func('string_view_create', hstr_ext)
load_native_func('string_view_create_with_data', hstr_ext)
load_native_func('string_view_len', hstr_ext)
load_native_func('string_view_get_data_ptr', hstr_ext)
load_native_func('string_view_set_data', hstr_ext)
load_native_func('string_view_to_int', hstr_ext)
load_native_func('string_view_to_float64', hstr_ext)


@intrinsic
def string_view_create(typingctx):
    ret_type = StdStringViewType()

    def codegen(context, builder, sig, args):
        nrt_table = context.nrt.get_nrt_api(builder)
        str_view_ctinfo = cgutils.create_struct_proxy(ret_type)(
            context, builder)
        fnty = lir.FunctionType(lir.VoidType(),
                                [str_view_ctinfo.meminfo.type.as_pointer(),     # meminfo to fill
                                 lir.IntType(8).as_pointer(),                   # NRT API func table
                                 ])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="string_view_create")
        builder.call(fn,
                     [str_view_ctinfo._get_ptr_by_name('meminfo'),
                      nrt_table])
        str_view_ctinfo.data_ptr = context.nrt.meminfo_data(builder, str_view_ctinfo.meminfo)
        return str_view_ctinfo._getvalue()

    return ret_type(), codegen


@intrinsic
def string_view_create_with_data(typingctx, data, size):
    ret_type = StdStringViewType()

    def codegen(context, builder, sig, args):
        data_val, size_val = args

        nrt_table = context.nrt.get_nrt_api(builder)
        str_view_ctinfo = cgutils.create_struct_proxy(ret_type)(
            context, builder)
        fnty = lir.FunctionType(lir.VoidType(),
                                [str_view_ctinfo.meminfo.type.as_pointer(),     # meminfo to fill
                                 lir.IntType(8).as_pointer(),                   # NRT API func table
                                 lir.IntType(8).as_pointer(),                   # char ptr to store in string view
                                 lir.IntType(64)                                # size of data to point to in bytes
                                 ])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="string_view_create_with_data")
        builder.call(fn,
                     [str_view_ctinfo._get_ptr_by_name('meminfo'),
                      nrt_table,
                      data_val,
                      size_val])
        str_view_ctinfo.data_ptr = context.nrt.meminfo_data(builder, str_view_ctinfo.meminfo)
        return str_view_ctinfo._getvalue()

    return ret_type(data, size), codegen


@intrinsic
def string_view_len(typingctx, str_view):
    ret_type = types.int64

    def codegen(context, builder, sig, args):
        str_view_ctinfo = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0])
        fnty = lir.FunctionType(lir.IntType(64),
                                [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="string_view_len")
        return builder.call(fn, [str_view_ctinfo.data_ptr])

    return ret_type(str_view), codegen


@overload(len)
def len_string_view_ovld(str_view):
    if not isinstance(str_view, StdStringViewType):
        return None

    def len_string_view_impl(str_view):
        return string_view_len(str_view)
    return len_string_view_impl


@intrinsic
def string_view_get_data_ptr(typingctx, str_view):
    ret_type = types.voidptr

    def codegen(context, builder, sig, args):
        str_view_ctinfo = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0])
        fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                                [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="string_view_get_data_ptr")
        return builder.call(fn, [str_view_ctinfo.data_ptr])

    return ret_type(str_view), codegen


@intrinsic
def string_view_print(typingctx, str_view):

    # load hashmap_dump here as otherwise module import will fail
    # since it's included in debug build only
    load_native_func('string_view_print', hstr_ext)

    ret_type = types.void

    def codegen(context, builder, sig, args):
        str_view_ctinfo = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0])
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="string_view_print")
        builder.call(fn, [str_view_ctinfo.data_ptr])

    return ret_type(str_view), codegen


@intrinsic
def string_view_set_data(typingctx, str_view, data, size):
    ret_type = types.voidptr

    def codegen(context, builder, sig, args):
        new_data_val, new_data_size = args[1:]
        str_view_ctinfo = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0])
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(64)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="string_view_set_data")
        return builder.call(fn,
                            [str_view_ctinfo.data_ptr,
                             new_data_val,
                             new_data_size])

    return ret_type(str_view, data, size), codegen


@intrinsic
def string_view_to_int(typingctx, str_view, base):
    ret_type = types.Tuple([types.bool_, types.int64])

    def codegen(context, builder, sig, args):
        str_view_val, base_val = args
        str_view_ctinfo = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=str_view_val)
        fnty = lir.FunctionType(lir.IntType(8),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(64),
                                 lir.IntType(64).as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="string_view_to_int")
        res_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        status = builder.call(fn,
                              [str_view_ctinfo.data_ptr,
                               base_val,
                               res_ptr])
        status_as_bool = context.cast(builder, status, types.int8, types.bool_)
        return context.make_tuple(builder, ret_type, [status_as_bool, builder.load(res_ptr)])

    return ret_type(str_view, base), codegen


@overload(int)
def string_view_to_int_ovld(x, base=10):
    if not isinstance(x, StdStringViewType):
        return None

    def string_view_to_int_impl(x, base=10):
        # FIXME: raise from numba compiled code will cause leak of string_view (no decref emitted)
        status, res = string_view_to_int(x, base)
        if status:
            raise ValueError("invalid string for conversion with int()")
        return res
    return string_view_to_int_impl


@intrinsic
def string_view_to_float64(typingctx, str_view):
    ret_type = types.Tuple([types.bool_, types.float64])

    def codegen(context, builder, sig, args):
        str_view_val, = args
        str_view_ctinfo = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=str_view_val)
        fnty = lir.FunctionType(lir.IntType(8),
                                [lir.IntType(8).as_pointer(),
                                 lir.DoubleType().as_pointer()])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="string_view_to_float64")
        res_ptr = cgutils.alloca_once(builder, lir.DoubleType())
        status = builder.call(fn,
                              [str_view_ctinfo.data_ptr,
                               res_ptr])
        status_as_bool = context.cast(builder, status, types.int8, types.bool_)
        return context.make_tuple(builder, ret_type, [status_as_bool, builder.load(res_ptr)])

    return ret_type(str_view), codegen


@overload(float)
def string_view_to_float_ovld(x):
    if not isinstance(x, StdStringViewType):
        return None

    def string_view_to_float_impl(x):
        status, res = string_view_to_float64(x)
        if status:
            raise ValueError("invalid string for conversion with float()")
        return res
    return string_view_to_float_impl


@overload(str)
def string_view_str_ovld(str_view):
    if not isinstance(str_view, StdStringViewType):
        return None

    def string_view_str_impl(str_view):
        str_view_data_ptr = string_view_get_data_ptr(str_view)
        return decode_utf8(str_view_data_ptr, len(str_view))

    return string_view_str_impl


def install_string_view_delegating_methods(nbtype):
    # TO-DO: generalize?
    from numba.core.registry import CPUDispatcher
    from numba.core import utils

    # need to do refresh, as unicode templates may not be avaialble yet
    typingctx = CPUDispatcher.targetdescr.typing_context
    typingctx.refresh()

    # filter only methods from all attribute templates registered for nbtype
    method_templates = list(typingctx._get_attribute_templates(nbtype))
    method_templates = [x for x in method_templates if getattr(x, 'is_method', None)]
    method_names = [x._attr for x in method_templates]

    # for all unicode methods register corresponding StringView overload
    # that delegates to it via creating a temporary unicode string
    for this_name, this_template in zip(method_names, method_templates):
        pysig_str = str(utils.pysignature(this_template._overload_func))
        pysig_params = utils.pysignature(this_template._overload_func).parameters.keys()
        self_param_name = list(pysig_params)[0]
        method_param_names = list(pysig_params)[1:]
        inner_call_params = ', '.join([f'{x}={x}' for x in method_param_names])

        from textwrap import dedent
        func_name = f'string_view_{this_name}'
        text = dedent(f"""
        @overload_method(StdStringViewType, '{this_name}')
        def {func_name}_ovld{pysig_str}:
            if not isinstance({self_param_name}, StdStringViewType):
                return None
            def _impl{pysig_str}:
                return str({self_param_name}).{this_name}({inner_call_params})
            return _impl
        """)
        global_vars, local_vars = {'StdStringViewType': StdStringViewType,
                                   'overload_method': overload_method}, {}
        exec(text, global_vars, local_vars)


install_string_view_delegating_methods(types.unicode_type)

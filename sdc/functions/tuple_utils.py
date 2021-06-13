# -*- coding: utf-8 -*-
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

import numba
import numpy as np
import operator
import pandas as pd

from numba import types, prange
from numba.core import cgutils
from numba.extending import (typeof_impl, NativeValue, intrinsic, box, unbox, lower_builtin, type_callable)
from numba.core.errors import TypingError
from numba.core.typing.templates import signature, AttributeTemplate, AbstractTemplate, infer_getattr
from numba.core.imputils import impl_ret_untracked, call_getiter, impl_ret_borrowed
from numba.core.imputils import (impl_ret_new_ref, impl_ret_borrowed, iternext_impl, RefType)
from numba.core.boxing import box_array, unbox_array, box_tuple

import llvmlite.llvmpy.core as lc

from sdc.datatypes.indexes import *
from sdc.utilities.sdc_typing_utils import SDCLimitation
from sdc.utilities.utils import sdc_overload, sdc_overload_attribute, sdc_overload_method, BooleanLiteral
from sdc.utilities.sdc_typing_utils import (
        TypeChecker,
        check_signed_integer,
        _check_dtype_param_type,
        sdc_pandas_index_types,
        check_types_comparable,
    )
from sdc.functions import numpy_like
from sdc.hiframes.api import fix_df_array, fix_df_index
from sdc.hiframes.boxing import _infer_index_type, _unbox_index_data
from sdc.datatypes.common_functions import hpat_arrays_append
from sdc.extensions.indexes.indexes_generic import *

from sdc.datatypes.indexes.multi_index_type import MultiIndexIteratorType
from numba.core.extending import register_jitable
from numba import literal_unroll
from numba.typed import Dict, List
from sdc.str_arr_type import StringArrayType
from sdc.datatypes.sdc_typeref import SdcTypeRef


@intrinsic
def sdc_tuple_map(typingctx, func, data, *args):

    if not isinstance(func, (types.Dispatcher, types.Function)):
        assert False, f"sdc_tuple_map's arg 'func' is expected to be " \
                      f"numba compiled function or a dispatcher, given: {func}"

    if not isinstance(data, (types.Tuple, types.UniTuple)):
        assert False, f"sdc_tuple_map's arg 'data' is expected to be a tuple, given: {data}"

    nargs = len(args)
    tuple_len = len(data)

    func_arg_types = [(typ, ) + args for typ in data]
    ret_tuple_types = []
    for i in range(tuple_len):
        res_sig = func.get_call_type(typingctx, func_arg_types[i], {})
        ret_tuple_types.append(res_sig.return_type)
    ret_type = types.Tuple(ret_tuple_types)
    ret_sig = ret_type(func, data, types.StarArgTuple.from_types(args))

    ### FIXME: this works with single overload for decorated function only
    ### but this isn't necessary, just need to find out corresponding template
    if isinstance(func, types.Function):
        assert len(func.templates) == 1, "Function template has multiple overloads"

    def codegen(context, builder, sig, args):

        tup_val = args[1]  # main tuple which elements are mapped
        other_val = []
        for i in range(0, nargs):
            other_val.append(
                builder.extract_value(args[2], i)
            )

        mapped_values = []
        for i in range(tuple_len):
            tup_elem = builder.extract_value(tup_val, i)
            input_args = [tup_elem] + other_val
            call_sig = signature(ret_tuple_types[i], *func_arg_types[i])

            if isinstance(func, types.Dispatcher):
                py_func = func.dispatcher.py_func
            else:
                # for function overloads get pyfunc from compiled impl
                target_disp = func.templates[0](context.typing_context)
                py_func = target_disp._get_impl(call_sig.args, {})[0].py_func

            mapped_values.append(
                context.compile_internal(builder,
                                         py_func,
                                         call_sig,
                                         input_args)
            )
        res = context.make_tuple(builder, ret_type, mapped_values)
        return res

    return ret_sig, codegen


@intrinsic
def sdc_tuple_map_elementwise(typingctx, func, lhs, rhs, *args):

    if not isinstance(func, (types.Dispatcher, types.Function)):
        assert False, f"sdc_tuple_map_elementwise's arg 'func' is expected to be " \
                      f"numba compiled function or a dispatcher, given: {func}"

    if not (isinstance(lhs, (types.Tuple, types.UniTuple))
            and isinstance(rhs, (types.Tuple, types.UniTuple))):
        assert False, f"sdc_tuple_map_elementwise's args are expected to be " \
                      f"tuples, given: lhs={lhs}, rhs={rhs}"

    assert len(lhs) == len(rhs), f"lhs and rhs tuples have different sizes: lhs={lhs}, rhs={rhs}"

    nargs = len(args)
    tuple_len = len(lhs)

    func_arg_types = [x for x in zip(lhs, rhs, *args)]
    ret_tuple_types = []
    for i in range(tuple_len):
        res_sig = func.get_call_type(typingctx, func_arg_types[i], {})
        ret_tuple_types.append(res_sig.return_type)
    ret_type = types.Tuple(ret_tuple_types)
    ret_sig = ret_type(func, lhs, rhs, types.StarArgTuple.from_types(args))

    if isinstance(func, types.Function):
        assert len(func.templates) == 1, "Function template has multiple overloads"

    def codegen(context, builder, sig, args):
        lhs_val = args[1]
        rhs_val = args[2]
        other_vals = []
        for i in range(0, nargs):
            other_vals.append(
                builder.extract_value(args[3], i)
            )

        mapped_values = []
        for i in range(tuple_len):
            lhs_elem = builder.extract_value(lhs_val, i)
            rhs_elem = builder.extract_value(rhs_val, i)
            other_elems = []
            for other_tup in other_vals:
                other_elems.append(
                    builder.extract_value(other_tup, i)
                )

            input_args = [lhs_elem, rhs_elem] + other_elems
            call_sig = signature(ret_tuple_types[i], *func_arg_types[i])

            if isinstance(func, types.Dispatcher):
                py_func = func.dispatcher.py_func
            else:
                # for function overloads get pyfunc from compiled impl
                target_disp = func.templates[0](context.typing_context)
                py_func = target_disp._get_impl(call_sig.args, {})[0].py_func

            mapped_values.append(
                context.compile_internal(builder,
                                         py_func,
                                         call_sig,
                                         input_args)
            )
        res = context.make_tuple(builder, ret_type, mapped_values)
        return res

    return ret_sig, codegen


@intrinsic
def sdc_tuple_unzip(typingctx, data_type):
    """ This function gets tuple of pairs and repacks them into two tuples, holding
    first and seconds elements, i.e. ((a, b), (c, d), (e, f)) -> ((a, c, e), (b, d, f)). """

    _func_name = 'sdc_tuple_unzip'
    _given_args_str = f'Given: data_type={data_type}'
    assert isinstance(data_type, (types.Tuple, types.UniTuple)), \
           f"{_func_name} expects tuple as argument. {_given_args_str}"

    data_len = len(data_type)
    assert data_len > 0, f"{_func_name}: empty tuple not allowed. {_given_args_str}"

    for x in data_type:
        assert isinstance(x, (types.Tuple, types.UniTuple)) and len(x) == len(data_type[0]), \
        f"{_func_name}: non-supported tuple elements types. {_given_args_str}"

    ty_firsts, ty_seconds = map(lambda x: types.Tuple.from_types(x),
                              zip(*data_type))
    ret_type = types.Tuple([ty_firsts, ty_seconds])

    def codegen(context, builder, sig, args):
        data_val, = args

        all_firsts = []
        all_seconds = []
        for i in range(data_len):
            tup_element_i = builder.extract_value(data_val, i)
            first_i = builder.extract_value(tup_element_i, 0)
            second_i = builder.extract_value(tup_element_i, 1)

            all_firsts.append(first_i)
            all_seconds.append(second_i)

            if context.enable_nrt:
                context.nrt.incref(builder, ty_firsts[i], first_i)
                context.nrt.incref(builder, ty_seconds[i], second_i)

        first_tup = context.make_tuple(builder, ty_firsts, all_firsts)
        second_tup = context.make_tuple(builder, ty_seconds, all_seconds)
        return context.make_tuple(builder, ret_type, [first_tup, second_tup])

    return ret_type(data_type), codegen

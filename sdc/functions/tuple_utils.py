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

from textwrap import dedent

from numba import types
from numba.extending import intrinsic
from numba.core.typing.templates import (signature, )
from numba.typed.dictobject import build_map

from sdc.utilities.utils import sdc_overload


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

    # codegen below uses first func template to get the dispatcher, so
    # for now deny compilation for overloaded func-s that have multiple overloads
    # (using the jitted function dispatcher as func will work anyway)
    # TO-DO: improve and upstream to Numba
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
                # for function overloads get pyfunc from compiled impl (this
                # hardcodes the first available template)
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


def sdc_tuple_zip(x, y):
    pass


@sdc_overload(sdc_tuple_zip)
def sdc_tuple_zip_ovld(x, y):
    """ This function combines tuple of pairs from two input tuples x and y, preserving
        literality of elements in them. """

    if not (isinstance(x, types.BaseAnonymousTuple) and isinstance(y, types.BaseAnonymousTuple)):
        return None

    res_size = min(len(x), len(y))
    func_impl_name = 'sdc_tuple_zip_impl'
    tup_elements = ', '.join([f"(x[{i}], y[{i}])" for i in range(res_size)])
    func_text = dedent(f"""
    def {func_impl_name}(x, y):
        return ({tup_elements}{',' if res_size else ''})
    """)
    use_globals, use_locals = {}, {}
    exec(func_text, use_globals, use_locals)
    return use_locals[func_impl_name]

    # FIXME_Numba#6533: alternatively we could have used sdc_tuple_map_elementwise
    # to avoid another use of exec, but due to @intrinsic-s not supporting
    # prefer_literal option below implementation looses literaly of args!
    # from sdc.functions.tuple_utils import sdc_tuple_map_elementwise
    # def sdc_tuple_zip_impl(x, y):
    #     return sdc_tuple_map_elementwise(
    #         lambda a, b: (a, b),
    #         x,
    #         y
    #     )
    #
    # return sdc_tuple_zip_impl


@intrinsic
def literal_dict_ctor(typingctx, items):

    tup_size = len(items)
    key_order = {p[0].literal_value: i for i, p in enumerate(items)}
    ret_type = types.LiteralStrKeyDict(dict(items), key_order)

    def codegen(context, builder, sig, args):
        items_val = args[0]

        # extract elements from the input tuple and repack into a list of variables required by build_map
        repacked_items = []
        for i in range(tup_size):
            elem = builder.extract_value(items_val, i)
            elem_first = builder.extract_value(elem, 0)
            elem_second = builder.extract_value(elem, 1)
            repacked_items.append((elem_first, elem_second))
        d = build_map(context, builder, ret_type, items, repacked_items)
        return d

    return ret_type(items), codegen


@sdc_overload(dict)
def dict_from_tuples_ovld(x):

    accepted_tuple_types = (types.Tuple, types.UniTuple)
    if not isinstance(x, types.BaseAnonymousTuple):
        return None

    def check_tuple_element(ty):
        return (isinstance(ty, accepted_tuple_types)
                and len(ty) == 2
                and isinstance(ty[0], types.StringLiteral))

    # below checks that elements are tuples with size 2 and first element is literal string
    if not (len(x) != 0 and all(map(check_tuple_element, x))):
        assert False, f"Creating LiteralStrKeyDict not supported from pairs of: {x}"

    def dict_from_tuples_impl(x):
        return literal_dict_ctor(x)
    return dict_from_tuples_impl

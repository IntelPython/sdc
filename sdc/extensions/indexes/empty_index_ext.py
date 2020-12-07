# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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
import pandas as pd

from numba import types
from numba.core import cgutils
from numba.extending import (NativeValue, intrinsic, box, unbox, )
from numba.core.typing.templates import signature

from sdc.datatypes.indexes import EmptyIndexType
from sdc.utilities.sdc_typing_utils import sdc_pandas_index_types
from sdc.utilities.utils import sdc_overload, sdc_overload_attribute, sdc_overload_method
from sdc.utilities.sdc_typing_utils import TypeChecker


@intrinsic
def init_empty_index(typingctx, name=None):
    name = types.none if name is None else name
    is_named = False if name is types.none else True

    def codegen(context, builder, sig, args):
        name_val, = args
        # create series struct and store values
        index_struct = cgutils.create_struct_proxy(
            sig.return_type)(context, builder)

        if is_named:
            if isinstance(name, types.StringLiteral):
                index_struct.name = numba.cpython.unicode.make_string_from_constant(
                    context, builder, types.unicode_type, name.literal_value)
            else:
                index_struct.name = name_val

        if context.enable_nrt and is_named:
                context.nrt.incref(builder, sig.args[1], name_val)

        return index_struct._getvalue()

    ret_typ = EmptyIndexType(is_named)
    sig = signature(ret_typ, name)
    return sig, codegen


@box(EmptyIndexType)
def box_empty_index(typ, val, c):

    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    empty_index = cgutils.create_struct_proxy(
        typ)(c.context, c.builder, val)

    data = c.pyapi.list_new(c.context.get_constant(types.int64, 0))
    if typ.is_named:
        name = c.pyapi.from_native_value(types.unicode_type, empty_index.name)
    else:
        name = c.pyapi.make_none()

    res = c.pyapi.call_method(pd_class_obj, "Index", (data, name))

    c.pyapi.decref(data)
    c.pyapi.decref(name)
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(EmptyIndexType)
def unbox_empty_index(typ, val, c):

    index_struct = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    if typ.is_named:
        name_obj = c.pyapi.object_getattr_string(val, "name")
        index_struct.name = numba.cpython.unicode.unbox_unicode_str(
            types.unicode_type, name_obj, c).value
        c.pyapi.decref(name_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(index_struct._getvalue(), is_error=is_error)


@sdc_overload_method(EmptyIndexType, 'take')
def pd_empty_index_take_overload(self, indexes):
    if not isinstance(self, EmptyIndexType):
        return None

    _func_name = 'Method take().'
    ty_checker = TypeChecker(_func_name)

    valid_indexes_types = (types.Array, types.List) + sdc_pandas_index_types
    if not (isinstance(indexes, valid_indexes_types) and isinstance(indexes.dtype, types.Integer)):
        ty_checker.raise_exc(indexes, 'array/list of integers or integer index', 'indexes')

    def pd_empty_index_take_impl(self, indexes):
        return init_empty_index(name=self._name)

    return pd_empty_index_take_impl


@sdc_overload(len)
def pd_empty_index_len_overload(self):
    if not isinstance(self, EmptyIndexType):
        return None

    def pd_empty_index_len_impl(self):
        return 0

    return pd_empty_index_len_impl

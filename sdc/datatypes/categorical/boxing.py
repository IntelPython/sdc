# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
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

from numba.extending import box, unbox, NativeValue
from numba.targets import boxing
from numba.targets.imputils import lower_constant
from numba.targets import arrayobj
from numba import types

from . import pandas_support
from .types import (
    CategoricalDtypeType,
    Categorical,
)


@box(CategoricalDtypeType)
def box_CategoricalDtype(typ, val, c):
    pd_dtype = pandas_support.as_dtype(typ)
    return c.pyapi.unserialize(c.pyapi.serialize_object(pd_dtype))


@unbox(CategoricalDtypeType)
def unbox_CategoricalDtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


@box(Categorical)
def box_Categorical(typ, val, c):
    pandas_module_name = c.context.insert_const_string(c.builder.module, "pandas")
    pandas_module = c.pyapi.import_module_noblock(pandas_module_name)

    constructor = c.pyapi.object_getattr_string(pandas_module, "Categorical")

    empty_list = c.pyapi.list_new(c.context.get_constant(types.intp, 0))
    args = c.pyapi.tuple_pack([empty_list])
    categorical = c.pyapi.call(constructor, args)

    dtype = box_CategoricalDtype(typ.pd_dtype, val, c)
    c.pyapi.object_setattr_string(categorical, "_dtype", dtype)

    codes = boxing.box_array(typ.codes, val, c)
    c.pyapi.object_setattr_string(categorical, "_codes", codes)

    c.pyapi.decref(codes)
    c.pyapi.decref(dtype)
    c.pyapi.decref(args)
    c.pyapi.decref(empty_list)
    c.pyapi.decref(constructor)
    c.pyapi.decref(pandas_module)
    return categorical


@unbox(Categorical)
def unbox_Categorical(typ, val, c):
    codes = c.pyapi.object_getattr_string(val, "codes")
    native_value = boxing.unbox_array(typ.codes, codes, c)
    c.pyapi.decref(codes)
    return native_value


@lower_constant(Categorical)
def constant_Categorical(context, builder, ty, pyval):
    """
    Create a constant Categorical.
    """
    return arrayobj.constant_array(context, builder, ty.codes, pyval.codes)

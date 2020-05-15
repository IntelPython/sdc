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


import numba
from numba.extending import (box, unbox, typeof_impl, register_model, models,
                             NativeValue, lower_builtin, lower_cast, overload,
                             type_callable, overload_method, intrinsic)
from numba import types
from numba.core.boxing import box_array, unbox_array

import numpy as np


class PDCategoricalDtype(types.Opaque):
    def __init__(self, _categories):
        self.categories = _categories
        name = 'PDCategoricalDtype({})'.format(self.categories)
        super(PDCategoricalDtype, self).__init__(name=name)


class CategoricalArray(types.Array):
    def __init__(self, dtype):
        self.dtype = dtype
        super(CategoricalArray, self).__init__(
            dtype, 1, 'C', name='CategoricalArray({})'.format(dtype))


@unbox(CategoricalArray)
def unbox_categorical_array(typ, val, c):
    arr_obj = c.pyapi.object_getattr_string(val, "codes")
    # c.pyapi.print_object(arr_obj)
    dtype = get_categories_int_type(typ.dtype)
    native_val = unbox_array(types.Array(dtype, 1, 'C'), arr_obj, c)
    c.pyapi.decref(arr_obj)
    return native_val


def get_categories_int_type(cat_dtype):
    dtype = types.int64
    n_cats = len(cat_dtype.categories)
    if n_cats < np.iinfo(np.int8).max:
        dtype = types.int8
    elif n_cats < np.iinfo(np.int16).max:
        dtype = types.int16
    elif n_cats < np.iinfo(np.int32).max:
        dtype = types.int32
    return dtype


@box(CategoricalArray)
def box_categorical_array(typ, val, c):
    dtype = typ.dtype
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    # categories list e.g. ['A', 'B', 'C']
    item_objs = _get_cat_obj_items(dtype.categories, c)
    n = len(item_objs)
    list_obj = c.pyapi.list_new(c.context.get_constant(types.intp, n))
    for i in range(n):
        idx = c.context.get_constant(types.intp, i)
        c.pyapi.incref(item_objs[i])
        c.pyapi.list_setitem(list_obj, idx, item_objs[i])


    int_dtype = get_categories_int_type(dtype)
    arr = box_array(types.Array(int_dtype, 1, 'C'), val, c)

    pdcat_cls_obj = c.pyapi.object_getattr_string(pd_class_obj, "Categorical")
    cat_arr = c.pyapi.call_method(pdcat_cls_obj, "from_codes", (arr, list_obj))
    c.pyapi.decref(pdcat_cls_obj)
    c.pyapi.decref(arr)
    c.pyapi.decref(list_obj)
    for obj in item_objs:
        c.pyapi.decref(obj)

    c.pyapi.decref(pd_class_obj)
    return cat_arr


def _get_cat_obj_items(categories, c):
    assert len(categories) > 0
    val = categories[0]
    if isinstance(val, str):
        return [c.pyapi.string_from_constant_string(item) for item in categories]

    dtype = numba.typeof(val)
    return [c.box(dtype, c.context.get_constant(dtype, item)) for item in categories]

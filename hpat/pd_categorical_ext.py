import operator
import numba
from numba.extending import (box, unbox, typeof_impl, register_model, models,
                             NativeValue, lower_builtin, lower_cast, overload,
                             type_callable, overload_method)
from numba.targets.imputils import lower_constant, impl_ret_new_ref, impl_ret_untracked
from numba import types, typing
from numba.targets.boxing import box_array, unbox_array

import numpy as np
import pandas as pd


class PDCategoricalDtype(types.Opaque):
    def __init__(self, _categories):
        self.categories = _categories
        name = 'PDCategoricalDtype({})'.format(self.categories)
        super(PDCategoricalDtype, self).__init__(name=name)


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


def box_categorical_series_dtype_fix(dtype, val, c, pd_class_obj):
    #

    # categories list e.g. ['A', 'B', 'C']
    item_objs = _get_cat_obj_items(dtype.categories, c)
    list_obj = c.pyapi.list_pack(item_objs)

    # call pd.api.types.CategoricalDtype(['A', 'B', 'C'])
    # api_obj = c.pyapi.object_getattr_string(pd_class_obj, "api")
    # types_obj = c.pyapi.object_getattr_string(api_obj, "types")
    # pd_dtype = c.pyapi.call_method(types_obj, "CategoricalDtype", (list_obj,))
    # c.pyapi.decref(api_obj)
    # c.pyapi.decref(types_obj)

    int_dtype = get_categories_int_type(dtype)
    arr = box_array(types.Array(int_dtype, 1, 'C'), val, c)

    pdcat_cls_obj = c.pyapi.object_getattr_string(pd_class_obj, "Categorical")
    cat_arr = c.pyapi.call_method(pdcat_cls_obj, "from_codes", (arr, list_obj))
    c.pyapi.decref(pdcat_cls_obj)
    c.pyapi.decref(arr)
    c.pyapi.decref(list_obj)
    for obj in item_objs:
        c.pyapi.decref(obj)

    return cat_arr


def _get_cat_obj_items(categories, c):
    assert len(categories) > 0
    val = categories[0]
    if isinstance(val, str):
        return [c.pyapi.string_from_constant_string(item) for item in categories]

    dtype = numba.typeof(val)
    return [c.box(dtype, c.context.get_constant(dtype, item)) for item in categories]

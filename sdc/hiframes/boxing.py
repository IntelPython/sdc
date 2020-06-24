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



import pandas as pd
import pandas.api.types
import numpy as np
import numba
from numba.extending import (typeof_impl, unbox, register_model, models,
                             NativeValue, box, intrinsic)
from numba import types
from numba.core import cgutils
from numba.np import numpy_support
from numba.core.typing import signature
from numba.core.boxing import box_array, unbox_array, box_list, unbox_none
from numba.core.boxing import _NumbaTypeHelper
from numba.cpython import listobj

from sdc.hiframes.pd_dataframe_type import DataFrameType
from sdc.str_ext import string_type, list_string_array_type
from sdc.str_arr_ext import (string_array_type, unbox_str_series, box_str_arr)
from sdc.datatypes.categorical.types import CategoricalDtypeType, Categorical
from sdc.datatypes.categorical.boxing import unbox_Categorical, box_Categorical
from sdc.hiframes.pd_series_ext import SeriesType
from sdc.hiframes.pd_series_type import _get_series_array_type

from sdc.hiframes.pd_dataframe_ext import get_structure_maps

from .. import hstr_ext
import llvmlite.binding as ll
from llvmlite import ir as lir
from llvmlite.llvmpy.core import Type as LLType
from sdc.datatypes.range_index_type import RangeIndexType
from sdc.extensions.indexes.range_index_ext import box_range_index, unbox_range_index
from sdc.str_arr_type import StringArrayType
ll.add_symbol('array_size', hstr_ext.array_size)
ll.add_symbol('array_getptr1', hstr_ext.array_getptr1)


@typeof_impl.register(pd.DataFrame)
def typeof_pd_dataframe(val, c):

    col_names = tuple(val.columns.tolist())
    # TODO: support other types like string and timestamp
    col_types = get_hiframes_dtypes(val)
    index_type = _infer_index_type(val.index)
    column_loc, _, _ = get_structure_maps(col_types, col_names)

    return DataFrameType(col_types, index_type, col_names, True, column_loc=column_loc)


# register series types for import
@typeof_impl.register(pd.Series)
def typeof_pd_series(val, c):
    index_type = _infer_index_type(val.index)
    is_named = val.name is not None
    return SeriesType(
        _infer_series_dtype(val), index=index_type, is_named=is_named)


@unbox(DataFrameType)
def unbox_dataframe(typ, val, c):
    """unbox dataframe to an empty DataFrame struct
    columns will be extracted later if necessary.
    """
    n_cols = len(typ.columns)
    column_strs = [numba.cpython.unicode.make_string_from_constant(
        c.context, c.builder, string_type, a) for a in typ.columns]
    # create dataframe struct and store values
    dataframe = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    errorptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)

    col_list_type = types.List(string_type)
    ok, inst = listobj.ListInstance.allocate_ex(c.context, c.builder, col_list_type, n_cols)

    with c.builder.if_else(ok, likely=True) as (if_ok, if_not_ok):
        with if_ok:
            inst.size = c.context.get_constant(types.intp, n_cols)
            for i, column_str in enumerate(column_strs):
                inst.setitem(c.context.get_constant(types.intp, i), column_str, incref=False)
            dataframe.columns = inst.value

        with if_not_ok:
            c.builder.store(cgutils.true_bit, errorptr)

    # If an error occurred, drop the whole native list
    with c.builder.if_then(c.builder.load(errorptr)):
        c.context.nrt.decref(c.builder, col_list_type, inst.value)

    _, data_typs_map, types_order = get_structure_maps(typ.data, typ.columns)

    for col_typ in types_order:
        type_id, col_indices = data_typs_map[col_typ]
        n_type_cols = len(col_indices)
        list_type = types.List(col_typ)
        ok, inst = listobj.ListInstance.allocate_ex(c.context, c.builder, list_type, n_type_cols)

        with c.builder.if_else(ok, likely=True) as (if_ok, if_not_ok):
            with if_ok:
                inst.size = c.context.get_constant(types.intp, n_type_cols)
                for i, col_idx in enumerate(col_indices):
                    series_obj = c.pyapi.object_getattr_string(val, typ.columns[col_idx])
                    arr_obj = c.pyapi.object_getattr_string(series_obj, "values")
                    ty_series = typ.data[col_idx]
                    if isinstance(ty_series, types.Array):
                        native_val = unbox_array(typ.data[col_idx], arr_obj, c)
                    elif ty_series == string_array_type:
                        native_val = unbox_str_series(string_array_type, series_obj, c)

                    inst.setitem(c.context.get_constant(types.intp, i), native_val.value, incref=False)

                dataframe.data = c.builder.insert_value(dataframe.data, inst.value, type_id)

            with if_not_ok:
                c.builder.store(cgutils.true_bit, errorptr)

        # If an error occurred, drop the whole native list
        with c.builder.if_then(c.builder.load(errorptr)):
            c.context.nrt.decref(c.builder, list_type, inst.value)

    index_obj = c.pyapi.object_getattr_string(val, "index")
    dataframe.index = _unbox_index_data(typ.index, index_obj, c).value
    c.pyapi.decref(index_obj)

    dataframe.parent = val

    # increase refcount of stored values
    if c.context.enable_nrt:
        # TODO: other objects?
        for var in column_strs:
            c.context.nrt.incref(c.builder, string_type, var)

    return NativeValue(dataframe._getvalue(), is_error=c.builder.load(errorptr))


def get_hiframes_dtypes(df):
    """get hiframe data types for a pandas dataframe
    """
    col_names = df.columns.tolist()
    hi_typs = [_get_series_array_type(_infer_series_dtype(df[cname]))
               for cname in col_names]
    return tuple(hi_typs)


def _infer_series_dtype(S):
    if S.dtype == np.dtype('O'):
        # XXX assuming the whole column is strings if 1st val is string
        # TODO: handle NA as 1st value
        i = 0
        while i < len(S) and (S.iloc[i] is np.nan or S.iloc[i] is None):
            i += 1
        if i == len(S):
            raise ValueError(
                "object dtype infer out of bounds for {}".format(S.name))

        first_val = S.iloc[i]
        if isinstance(first_val, list):
            return _infer_series_list_dtype(S)
        elif isinstance(first_val, str):
            return string_type
        else:
            raise ValueError(
                "object dtype infer: data type for column {} not supported".format(S.name))
    elif isinstance(S.dtype, pd.CategoricalDtype):
        return numba.typeof(S.dtype)
    # regular numpy types
    try:
        return numpy_support.from_dtype(S.dtype)
    except NotImplementedError:
        raise ValueError("np dtype infer: data type for column {} not supported".format(S.name))


def _infer_series_list_dtype(S):
    for i in range(len(S)):
        first_val = S.iloc[i]
        if not isinstance(first_val, list):
            raise ValueError(
                "data type for column {} not supported".format(S.name))
        if len(first_val) > 0:
            # TODO: support more types
            if isinstance(first_val[0], str):
                return types.List(string_type)
            else:
                raise ValueError(
                    "data type for column {} not supported".format(S.name))
    raise ValueError(
        "data type for column {} not supported".format(S.name))


def _infer_index_type(index):
    """ Deduces native Numba type used to represent index Python object """
    if isinstance(index, pd.RangeIndex):
        # depending on actual index value unbox to diff types: none-index if it matches
        # positions or to RangeIndexType in general case
        if (index.start == 0 and index.step == 1 and index.name is None):
            return types.none
        else:
            if index.name is None:
                return RangeIndexType()
            else:
                return RangeIndexType(is_named=True)

    # for unsupported pandas indexes we explicitly unbox to None
    if isinstance(index, pd.DatetimeIndex):
        return types.none
    if index.dtype == np.dtype('O'):
        # TO-DO: should we check that all elements are strings?
        if len(index) > 0 and isinstance(index[0], str):
            return string_array_type
        else:
            return types.none

    numba_index_type = numpy_support.from_dtype(index.dtype)
    return types.Array(numba_index_type, 1, 'C')


@box(DataFrameType)
def box_dataframe(typ, val, c):
    context = c.context
    builder = c.builder

    col_names = typ.columns
    arr_typs = typ.data

    dataframe = cgutils.create_struct_proxy(typ)(context, builder, value=val)

    pyapi = c.pyapi
    # gil_state = pyapi.gil_ensure()  # acquire GIL

    mod_name = context.insert_const_string(c.builder.module, "pandas")
    class_obj = pyapi.import_module_noblock(mod_name)
    df_dict = pyapi.dict_new()

    arrays_list_objs = {}
    for cname, arr_typ in zip(col_names, arr_typs):
        # df['cname'] = boxed_arr
        # TODO: datetime.date, DatetimeIndex?
        name_str = context.insert_const_string(c.builder.module, cname)
        cname_obj = pyapi.string_from_string(name_str)

        col_loc = typ.column_loc[cname]
        type_id, col_id = col_loc.type_id, col_loc.col_id

        # dataframe.data looks like a tuple(list(array))
        # e.g. ([array(int64, 1d, C), array(int64, 1d, C)], [array(float64, 1d, C)])
        arrays_list_obj = arrays_list_objs.get(type_id)
        if arrays_list_obj is None:
            list_typ = types.List(arr_typ)
            # extracting list from the tuple
            list_val = builder.extract_value(dataframe.data, type_id)
            # getting array from the list to box it then
            arrays_list_obj = box_list(list_typ, list_val, c)
            arrays_list_objs[type_id] = arrays_list_obj

        # PyList_GetItem returns borrowed reference
        arr_obj = pyapi.list_getitem(arrays_list_obj, col_id)
        pyapi.dict_setitem(df_dict, cname_obj, arr_obj)

        pyapi.decref(cname_obj)

    df_obj = pyapi.call_method(class_obj, "DataFrame", (df_dict,))
    pyapi.decref(df_dict)

    # set df.index if necessary
    if typ.index != types.none:
        index_obj = _box_index_data(typ.index, dataframe.index, c)
        pyapi.object_setattr_string(df_obj, 'index', index_obj)
        pyapi.decref(index_obj)

    for arrays_list_obj in arrays_list_objs.values():
        pyapi.decref(arrays_list_obj)

    pyapi.decref(class_obj)
    # pyapi.gil_release(gil_state)    # release GIL
    return df_obj


@intrinsic
def unbox_dataframe_column(typingctx, df, i=None):

    def codegen(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        c = numba.pythonapi._UnboxContext(context, builder, pyapi)

        df_typ = sig.args[0]
        col_ind = sig.args[1].literal_value
        data_typ = df_typ.data[col_ind]
        col_name = df_typ.columns[col_ind]
        # TODO: refcounts?

        dataframe = cgutils.create_struct_proxy(
            sig.args[0])(context, builder, value=args[0])
        series_obj = c.pyapi.object_getattr_string(dataframe.parent, col_name)
        arr_obj = c.pyapi.object_getattr_string(series_obj, "values")

        # TODO: support column of tuples?
        native_val = _unbox_series_data(
            data_typ.dtype, data_typ, arr_obj, c)

        c.pyapi.decref(series_obj)
        c.pyapi.decref(arr_obj)
        c.context.nrt.incref(builder, df_typ.index, dataframe.index)

        # assign array and set unboxed flag
        dataframe.data = builder.insert_value(
            dataframe.data, native_val.value, col_ind)
        return dataframe._getvalue()

    return signature(df, df, i), codegen


def _unbox_index_data(index_typ, index_obj, c):
    """ Unboxes Pandas index object basing on the native type inferred previously.
        Params:
            index_typ: native Numba type the object is to be unboxed into
            index_obj: Python object to be unboxed
            c: LLVM context object
        Returns: LLVM instructions to generate native value
    """
    if isinstance(index_typ, RangeIndexType):
        return unbox_range_index(index_typ, index_obj, c)

    if index_typ == string_array_type:
        return unbox_str_series(index_typ, index_obj, c)

    if isinstance(index_typ, types.Array):
        index_data = c.pyapi.object_getattr_string(index_obj, "_data")
        res = unbox_array(index_typ, index_data, c)
        c.pyapi.decref(index_data)
        return res

    if isinstance(index_typ, types.NoneType):
        return unbox_none(index_typ, index_obj, c)

    assert False, f"_unbox_index_data: unexpected index type({index_typ}) while unboxing"


@unbox(SeriesType)
def unbox_series(typ, val, c):
    arr_obj = c.pyapi.object_getattr_string(val, "values")
    series = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    series.data = _unbox_series_data(typ.dtype, typ.data, arr_obj, c).value

    index_obj = c.pyapi.object_getattr_string(val, "index")
    series.index = _unbox_index_data(typ.index, index_obj, c).value

    if typ.is_named:
        name_obj = c.pyapi.object_getattr_string(val, "name")
        series.name = numba.cpython.unicode.unbox_unicode_str(
            string_type, name_obj, c).value
        c.pyapi.decref(name_obj)

    c.pyapi.decref(arr_obj)
    c.pyapi.decref(index_obj)
    return NativeValue(series._getvalue())


def _unbox_series_data(dtype, data_typ, arr_obj, c):
    if data_typ == string_array_type:
        return unbox_str_series(string_array_type, arr_obj, c)
    elif data_typ == list_string_array_type:
        return _unbox_array_list_str(arr_obj, c)
    elif isinstance(dtype, CategoricalDtypeType):
        return unbox_Categorical(data_typ, arr_obj, c)

    # TODO: error handling like Numba callwrappers.py
    return unbox_array(data_typ, arr_obj, c)


@box(SeriesType)
def box_series(typ, val, c):
    """
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype = typ.dtype

    series = cgutils.create_struct_proxy(
        typ)(c.context, c.builder, val)

    arr = _box_series_data(dtype, typ.data, series.data, c)

    if typ.index is types.none:
        index = c.pyapi.make_none()
    else:
        index = _box_index_data(typ.index, series.index, c)

    if typ.is_named:
        name = c.pyapi.from_native_value(string_type, series.name)
    else:
        name = c.pyapi.make_none()

    dtype = c.pyapi.make_none()  # TODO: dtype
    res = c.pyapi.call_method(
        pd_class_obj, "Series", (arr, index, dtype, name))

    c.pyapi.decref(arr)
    c.pyapi.decref(index)
    c.pyapi.decref(dtype)
    c.pyapi.decref(name)
    c.pyapi.decref(pd_class_obj)
    return res


def _box_series_data(dtype, data_typ, val, c):

    if isinstance(dtype, types.BaseTuple):
        np_dtype = np.dtype(
            ','.join(str(t) for t in dtype.types), align=True)
        dtype = numba.np.numpy_support.from_dtype(np_dtype)

    if dtype == string_type:
        arr = box_str_arr(string_array_type, val, c)
    elif isinstance(dtype, CategoricalDtypeType):
        arr = box_Categorical(data_typ, val, c)
    elif dtype == types.List(string_type):
        arr = box_list(list_string_array_type, val, c)
    else:
        arr = box_array(data_typ, val, c)

    if isinstance(dtype, types.Record):
        o_str = c.context.insert_const_string(c.builder.module, "O")
        o_str = c.pyapi.string_from_string(o_str)
        arr = c.pyapi.call_method(arr, "astype", (o_str,))

    return arr


def _box_index_data(index_typ, val, c):
    """ Boxes native value used to represent Pandas index into appropriate Python object.
        Params:
            index_typ: Numba type of native value
            val: native value
            c: LLVM context object
        Returns: Python object native value is boxed into
    """
    assert isinstance(index_typ, (RangeIndexType, StringArrayType, types.Array, types.NoneType))

    if isinstance(index_typ, RangeIndexType):
        index = box_range_index(index_typ, val, c)
    elif isinstance(index_typ, types.Array):
        index = box_array(index_typ, val, c)
    elif isinstance(index_typ, StringArrayType):
        index = box_str_arr(string_array_type, val, c)
    else:  # index_typ is types.none
        index = c.pyapi.make_none()

    return index


def _unbox_array_list_str(obj, c):
    #
    typ = list_string_array_type
    # from unbox_list
    errorptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    listptr = cgutils.alloca_once(c.builder, c.context.get_value_type(typ))

    # get size of array
    arr_size_fnty = LLType.function(c.pyapi.py_ssize_t, [c.pyapi.pyobj])
    arr_size_fn = c.pyapi._get_function(arr_size_fnty, name="array_size")
    size = c.builder.call(arr_size_fn, [obj])
    # cgutils.printf(c.builder, 'size %d\n', size)

    _python_array_obj_to_native_list(typ, obj, c, size, listptr, errorptr)

    return NativeValue(c.builder.load(listptr),
                       is_error=c.builder.load(errorptr))


def _python_array_obj_to_native_list(typ, obj, c, size, listptr, errorptr):
    """
    Construct a new native list from a Python array of objects.
    copied from _python_list_to_native but list_getitem is converted to array
    getitem.
    """
    def check_element_type(nth, itemobj, expected_typobj):
        typobj = nth.typeof(itemobj)
        # Check if *typobj* is NULL
        with c.builder.if_then(
                cgutils.is_null(c.builder, typobj),
                likely=False,
        ):
            c.builder.store(cgutils.true_bit, errorptr)
            loop.do_break()
        # Mandate that objects all have the same exact type
        type_mismatch = c.builder.icmp_signed('!=', typobj, expected_typobj)

        with c.builder.if_then(type_mismatch, likely=False):
            c.builder.store(cgutils.true_bit, errorptr)
            c.pyapi.err_format(
                "PyExc_TypeError",
                "can't unbox heterogeneous list: %S != %S",
                expected_typobj, typobj,
            )
            c.pyapi.decref(typobj)
            loop.do_break()
        c.pyapi.decref(typobj)

    # Allocate a new native list
    ok, list = listobj.ListInstance.allocate_ex(c.context, c.builder, typ, size)
    # Array getitem call
    arr_get_fnty = LLType.function(LLType.pointer(c.pyapi.pyobj), [c.pyapi.pyobj, c.pyapi.py_ssize_t])
    arr_get_fn = c.pyapi._get_function(arr_get_fnty, name="array_getptr1")

    with c.builder.if_else(ok, likely=True) as (if_ok, if_not_ok):
        with if_ok:
            list.size = size
            zero = lir.Constant(size.type, 0)
            with c.builder.if_then(c.builder.icmp_signed('>', size, zero),
                                   likely=True):
                # Traverse Python list and unbox objects into native list
                with _NumbaTypeHelper(c) as nth:
                    # Note: *expected_typobj* can't be NULL
                    # TODO: enable type checking when emty list item in
                    # list(list(str)) case can be handled
                    # expected_typobj = nth.typeof(c.builder.load(
                    #                 c.builder.call(arr_get_fn, [obj, zero])))
                    with cgutils.for_range(c.builder, size) as loop:
                        itemobj = c.builder.call(arr_get_fn, [obj, loop.index])
                        # extra load since we have ptr to object
                        itemobj = c.builder.load(itemobj)
                        # c.pyapi.print_object(itemobj)
                        # check_element_type(nth, itemobj, expected_typobj)
                        # XXX we don't call native cleanup for each
                        # list element, since that would require keeping
                        # of which unboxings have been successful.
                        native = c.unbox(typ.dtype, itemobj)
                        with c.builder.if_then(native.is_error, likely=False):
                            c.builder.store(cgutils.true_bit, errorptr)
                            loop.do_break()
                        # The object (e.g. string) is stored so incref=True
                        list.setitem(loop.index, native.value, incref=True)
                    # c.pyapi.decref(expected_typobj)
            if typ.reflected:
                list.parent = obj
            # Stuff meminfo pointer into the Python object for
            # later reuse.
            with c.builder.if_then(c.builder.not_(c.builder.load(errorptr)),
                                   likely=False):
                c.pyapi.object_set_private_data(obj, list.meminfo)
            list.set_dirty(False)
            c.builder.store(list.value, listptr)

        with if_not_ok:
            c.builder.store(cgutils.true_bit, errorptr)

    # If an error occurred, drop the whole native list
    with c.builder.if_then(c.builder.load(errorptr)):
        c.context.nrt.decref(c.builder, typ, list.value)

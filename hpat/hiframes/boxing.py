
import pandas as pd
import numpy as np
import datetime
import numba
from numba.extending import (typeof_impl, unbox, register_model, models,
    NativeValue, box)
from numba import numpy_support, types, cgutils
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate, CallableTemplate
from numba.targets.boxing import box_array, unbox_array, box_list
from numba.targets.imputils import lower_builtin
from numba.targets.boxing import _NumbaTypeHelper
from numba.targets import listobj

import hpat
from hpat.hiframes.api import PandasDataFrameType
from hpat.hiframes.pd_timestamp_ext import (datetime_date_type,
    unbox_datetime_date_array, box_datetime_date_array)
from hpat.str_ext import string_type, list_string_array_type
from hpat.str_arr_ext import (string_array_type, unbox_str_series, box_str_arr)
from hpat.hiframes.pd_categorical_ext import (PDCategoricalDtype,
    box_categorical_series_dtype_fix)
from hpat.hiframes.pd_series_ext import SeriesType, arr_to_series_type

import hstr_ext
import llvmlite.binding as ll
from llvmlite import ir as lir
from llvmlite.llvmpy.core import Type as LLType
ll.add_symbol('array_size', hstr_ext.array_size)
ll.add_symbol('array_getptr1', hstr_ext.array_getptr1)


@typeof_impl.register(pd.DataFrame)
def typeof_pd_dataframe(val, c):
    col_names = val.columns.tolist()
    # TODO: support other types like string and timestamp
    col_types = get_hiframes_dtypes(val)
    return PandasDataFrameType(col_names, col_types)

register_model(PandasDataFrameType)(models.OpaqueModel)


# register series types for import
@typeof_impl.register(pd.Series)
def typeof_pd_str_series(val, c):
    # TODO: handle NA as 1st value
    if len(val) > 0 and isinstance(val.values[0], str):  # and isinstance(val[-1], str):
        arr_typ = string_array_type
    elif len(val) > 0 and isinstance(val.values[0], datetime.date):
        # XXX: using .values to check date type since DatetimeIndex returns
        # Timestamp which is subtype of datetime.date
        return SeriesType(datetime_date_type)
    else:
        arr_typ = numba.typing.typeof._typeof_ndarray(val.values, c)

    return arr_to_series_type(arr_typ)


@typeof_impl.register(pd.Index)
def typeof_pd_index(val, c):
    if len(val) > 0 and isinstance(val[0], datetime.date):
        return SeriesType(datetime_date_type)
    else:
        raise NotImplementedError("unsupported pd.Index type")

@unbox(PandasDataFrameType)
def unbox_df(typ, val, c):
    """unbox dataframe to an Opaque pointer
    columns will be extracted later if necessary.
    """
    # XXX: refcount?
    return NativeValue(val)

def get_hiframes_dtypes(df):
    """get hiframe data types for a pandas dataframe
    """
    pd_typ_list = df.dtypes.tolist()
    col_names = df.columns.tolist()
    hi_typs = []
    for cname, typ in zip(col_names, pd_typ_list):
        if typ == np.dtype('O'):
            # XXX assuming the whole column is strings if 1st val is string
            first_val = df[cname].iloc[0]
            if isinstance(first_val, list):
                typ = _infer_series_list_type(df[cname], cname)
                hi_typs.append(typ)
                continue
            if isinstance(first_val, str):
                hi_typs.append(string_type)
                continue
            else:
                raise ValueError("data type for column {} not supported".format(cname))
        try:
            t = numpy_support.from_dtype(typ)
            hi_typs.append(t)
        except NotImplementedError:
            raise ValueError("data type for column {} not supported".format(cname))

    return hi_typs


def _infer_series_list_type(S, cname):
    for i in range(len(S)):
        first_val = S.iloc[i]
        if not isinstance(first_val, list):
            raise ValueError(
                "data type for column {} not supported".format(cname))
        if len(first_val) > 0:
            # TODO: support more types
            if isinstance(first_val[0], str):
                return types.List(string_type)
            else:
                raise ValueError(
                    "data type for column {} not supported".format(cname))
    raise ValueError(
            "data type for column {} not supported".format(cname))

def box_df(names, arrs):
    return pd.DataFrame()

@infer_global(box_df)
class BoxDfTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) % 2 == 0, "name and column pairs expected"
        col_names = [a.literal_value for a in args[:len(args)//2]]
        col_types =  [a.dtype for a in args[len(args)//2:]]
        df_typ = PandasDataFrameType(col_names, col_types)
        return signature(df_typ, *args)


@lower_builtin(box_df, types.Literal, types.VarArg(types.Any))
def lower_box_df(context, builder, sig, args):
    assert len(sig.args) % 2 == 0, "name and column pairs expected"
    n_cols = len(sig.args)//2
    col_names = [a.literal_value for a in sig.args[:n_cols]]
    col_arrs = [a for a in args[n_cols:]]
    arr_typs = [a for a in sig.args[n_cols:]]
    # use dtypes from return type since it contains Categorical dtype
    dtypes = sig.return_type.col_types

    pyapi = context.get_python_api(builder)
    env_manager = context.get_env_manager(builder)
    c = numba.pythonapi._BoxContext(context, builder, pyapi, env_manager)
    gil_state = pyapi.gil_ensure()  # acquire GIL

    mod_name = context.insert_const_string(c.builder.module, "pandas")
    class_obj = pyapi.import_module_noblock(mod_name)
    res = pyapi.call_method(class_obj, "DataFrame", ())
    for cname, arr, arr_typ, dtype in zip(col_names, col_arrs, arr_typs, dtypes):
        # df['cname'] = boxed_arr
        # TODO: datetime.date, DatetimeIndex?
        if dtype == string_type:
            arr_obj = box_str_arr(arr_typ, arr, c)
        elif isinstance(dtype, PDCategoricalDtype):
            arr_obj = box_categorical_series_dtype_fix(dtype, arr, c, class_obj)
            context.nrt.incref(builder, arr_typ, arr)
        elif dtype == types.List(string_type):
            arr_obj = box_list(list_string_array_type, arr, c)
            context.nrt.incref(builder, arr_typ, arr)  # TODO required?
            # pyapi.print_object(arr_obj)
        else:
            arr_obj = box_array(arr_typ, arr, c)
            # TODO: is incref required?
            context.nrt.incref(builder, arr_typ, arr)
        name_str = context.insert_const_string(c.builder.module, cname)
        cname_obj = pyapi.string_from_string(name_str)
        pyapi.object_setitem(res, cname_obj, arr_obj)
        # pyapi.decref(arr_obj)
        pyapi.decref(cname_obj)

    pyapi.decref(class_obj)
    pyapi.gil_release(gil_state)    # release GIL
    return res

@box(PandasDataFrameType)
def box_df_dummy(typ, val, c):
    return val


def unbox_df_column(df, col_name, dtype):
    return df[col_name]

@infer_global(unbox_df_column)
class UnBoxDfCol(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        df_typ, col_ind_const, dtype_typ = args[0], args[1], args[2]
        if isinstance(dtype_typ, types.Literal):
            if dtype_typ.literal_value == 12:  # FIXME dtype for dt64
                out_typ = types.Array(types.NPDatetime('ns'), 1, 'C')
            elif dtype_typ.literal_value == 11:  # FIXME dtype for str
                out_typ = string_array_type
            elif dtype_typ.literal_value == 13:  # FIXME dtype for list(str)
                out_typ = list_string_array_type
            else:
                raise ValueError("invalid input dataframe dtype {}".format(dtype_typ.literal_value))
        else:
            out_typ = types.Array(dtype_typ.dtype, 1, 'C')
        # FIXME: last arg should be types.DType?
        return signature(out_typ, *args)


@lower_builtin(unbox_df_column, PandasDataFrameType, types.Literal, types.Any)
def lower_unbox_df_column(context, builder, sig, args):
    # FIXME: last arg should be types.DType?
    pyapi = context.get_python_api(builder)
    c = numba.pythonapi._UnboxContext(context, builder, pyapi)

    # TODO: refcounts?
    col_ind = sig.args[1].literal_value
    col_name = sig.args[0].col_names[col_ind]
    series_obj = c.pyapi.object_getattr_string(args[0], col_name)
    arr_obj = c.pyapi.object_getattr_string(series_obj, "values")

    if isinstance(sig.args[2], types.Literal) and sig.args[2].literal_value == 11:  # FIXME: str code
        native_val = unbox_str_series(string_array_type, arr_obj, c)
    elif isinstance(sig.args[2], types.Literal) and sig.args[2].literal_value == 13:  # FIXME: list(str) code
        native_val = _unbox_array_list_str(arr_obj, c)
    else:
        if isinstance(sig.args[2], types.Literal) and sig.args[2].literal_value == 12:  # FIXME: dt64 code
            dtype = types.NPDatetime('ns')
        else:
            dtype = sig.args[2].dtype
        # TODO: error handling like Numba callwrappers.py
        native_val = unbox_array(types.Array(dtype=dtype, ndim=1, layout='C'), arr_obj, c)

    c.pyapi.decref(series_obj)
    c.pyapi.decref(arr_obj)
    return native_val.value


@unbox(SeriesType)
def unbox_series(typ, val, c):
    arr_obj = c.pyapi.object_getattr_string(val, "values")
    series = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    series.data = _unbox_series_data(typ.dtype, typ.data, val, c, arr_obj)
    # TODO: handle index and name
    c.pyapi.decref(arr_obj)
    return NativeValue(series._getvalue())

def _unbox_series_data(dtype, data_typ, val, c, arr_obj):
    if dtype == string_type:
        return unbox_str_series(string_array_type, arr_obj, c).value
    elif dtype == datetime_date_type:
        return unbox_datetime_date_array(data_typ, val, c).value

    # TODO: error handling like Numba callwrappers.py
    return unbox_array(types.Array(dtype, 1, 'C'), arr_obj, c).value


@box(SeriesType)
def box_series(typ, val, c):
    """
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype = typ.dtype

    series = cgutils.create_struct_proxy(
            typ)(c.context, c.builder, val)

    arr = _box_series_data(dtype, typ.data, series.data, c, pd_class_obj)

    if typ.index is types.none:
        index = c.pyapi.make_none()
    else:
        # TODO: index-specific boxing like RangeIndex() etc.
        index = _box_series_data(
            typ.index.dtype, typ.index, series.index, c, pd_class_obj)

    if typ.is_named:
        name = c.pyapi.from_native_value(string_type, series.name)
    else:
        name = c.pyapi.make_none()

    dtype = c.pyapi.make_none()  # TODO: dtype
    res = c.pyapi.call_method(
        pd_class_obj, "Series", (arr, index, dtype, name))

    c.pyapi.decref(pd_class_obj)
    return res


def _box_series_data(dtype, data_typ, val, c, pd_class_obj):

    if isinstance(dtype, types.BaseTuple):
        np_dtype = np.dtype(
            ','.join(str(t) for t in dtype.types), align=True)
        dtype = numba.numpy_support.from_dtype(np_dtype)

    if dtype == string_type:
        arr = box_str_arr(string_array_type, val, c)
    elif dtype == datetime_date_type:
        arr = box_datetime_date_array(data_typ, val, c)
    elif isinstance(dtype, PDCategoricalDtype):
        arr = box_categorical_series_dtype_fix(dtype, val, c, pd_class_obj)
    elif dtype == types.List(string_type):
        arr = box_list(list_string_array_type, val, c)
    else:
        arr = box_array(data_typ, val, c)

    if isinstance(dtype, types.Record):
        o_str = c.context.insert_const_string(c.builder.module, "O")
        o_str = c.pyapi.string_from_string(o_str)
        arr = c.pyapi.call_method(arr, "astype", (o_str,))

    return arr


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
                    expected_typobj = nth.typeof(c.builder.load(
                                    c.builder.call(arr_get_fn, [obj, zero])))
                    with cgutils.for_range(c.builder, size) as loop:
                        itemobj = c.builder.call(arr_get_fn, [obj, loop.index])
                        # extra load since we have ptr to object
                        itemobj = c.builder.load(itemobj)
                        #c.pyapi.print_object(itemobj)
                        check_element_type(nth, itemobj, expected_typobj)
                        # XXX we don't call native cleanup for each
                        # list element, since that would require keeping
                        # of which unboxings have been successful.
                        native = c.unbox(typ.dtype, itemobj)
                        with c.builder.if_then(native.is_error, likely=False):
                            c.builder.store(cgutils.true_bit, errorptr)
                            loop.do_break()
                        # The reference is borrowed so incref=False
                        list.setitem(loop.index, native.value, incref=False)
                    c.pyapi.decref(expected_typobj)
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

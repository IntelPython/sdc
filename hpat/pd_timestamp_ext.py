import operator
import numba
from numba import types
from numba.extending import (typeof_impl, type_callable, models, register_model, NativeValue,
                             make_attribute_wrapper, lower_builtin, box, unbox, lower_cast,
                             lower_getattr, infer_getattr, overload_method, overload, intrinsic)
from numba import cgutils
from numba.targets.arrayobj import make_array, _empty_nd_impl, store_item, basic_indexing
from numba.targets.boxing import unbox_array, box_array
from numba.typing.templates import (infer_getattr, AttributeTemplate,
    bound_function, signature, infer_global, AbstractTemplate,
    ConcreteTemplate)

import numpy as np
import ctypes
import inspect
import hpat.str_ext
import hpat.utils

from llvmlite import ir as lir

import pandas as pd
# TODO: make pandas optional, not import this file if no pandas
# pandas_present = True
# try:
#     import pandas as pd
# except ImportError:
#     pandas_present = False

import datetime
import hdatetime_ext
import llvmlite.binding as ll
ll.add_symbol('parse_iso_8601_datetime', hdatetime_ext.parse_iso_8601_datetime)
ll.add_symbol('convert_datetimestruct_to_datetime', hdatetime_ext.convert_datetimestruct_to_datetime)
ll.add_symbol('np_datetime_date_array_from_packed_ints', hdatetime_ext.np_datetime_date_array_from_packed_ints)

date_fields = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'nanosecond']
timedelta_fields = ['days', 'seconds', 'microseconds', 'nanoseconds']

#--------------------------------------------------------------

class PANDAS_DATETIMESTRUCT(ctypes.Structure):
    _fields_ = [("year", ctypes.c_longlong),
                ("month", ctypes.c_int),
                ("day", ctypes.c_int),
                ("hour", ctypes.c_int),
                ("min", ctypes.c_int),
                ("sec", ctypes.c_int),
                ("us", ctypes.c_int),
                ("ps", ctypes.c_int),
                ("as", ctypes.c_int)]

class PandasDtsType(types.Type):
    def __init__(self):
        super(PandasDtsType, self).__init__(
            name='PandasDtsType()')

pandas_dts_type = PandasDtsType()

@typeof_impl.register(PANDAS_DATETIMESTRUCT)
def typeof_pandas_dts(val, c):
    return pandas_dts_type

@register_model(PandasDtsType)
class PandasDtsModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
                ("year", types.int64),
                ("month", types.int32),
                ("day", types.int32),
                ("hour", types.int32),
                ("min", types.int32),
                ("sec", types.int32),
                ("us", types.int32),
                ("ps", types.int32),
                ("as", types.int32),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)

make_attribute_wrapper(PandasDtsType, 'year', 'year')
make_attribute_wrapper(PandasDtsType, 'month', 'month')
make_attribute_wrapper(PandasDtsType, 'day', 'day')

@type_callable(PANDAS_DATETIMESTRUCT)
def type_pandas_dts(context):
    def typer():
        return pandas_dts_type
    return typer

@lower_builtin(PANDAS_DATETIMESTRUCT)
def impl_ctor_pandas_dts(context, builder, sig, args):
    typ = sig.return_type
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    return ts._getvalue()


#-- builtin operators for dt64 ----------------------------------------------
# TODO: move to Numba

class CompDT64(ConcreteTemplate):
    cases = signature(
        types.boolean, types.NPDatetime('ns'), types.NPDatetime('ns'))

@infer_global(operator.lt)
class CmpOpLt(CompDT64):
    key = operator.lt

@infer_global(operator.le)
class CmpOpLe(CompDT64):
    key = operator.le

@infer_global(operator.gt)
class CmpOpGt(CompDT64):
    key = operator.gt

@infer_global(operator.ge)
class CmpOpGe(CompDT64):
    key = operator.ge

@infer_global(operator.eq)
class CmpOpEq(CompDT64):
    key = operator.eq

@infer_global(operator.ne)
class CmpOpNe(CompDT64):
    key = operator.ne

class MinMaxBaseDT64(numba.typing.builtins.MinMaxBase):

    def _unify_minmax(self, tys):
        for ty in tys:
            if not ty == types.NPDatetime('ns'):
                return
        return self.context.unify_types(*tys)

@infer_global(max)
class Max(MinMaxBaseDT64):
    pass


@infer_global(min)
class Min(MinMaxBaseDT64):
    pass

#---------------------------------------------------------------

# datetime.date implementation that uses a single int to store year/month/day
class DatetimeDateType(types.Type):
    def __init__(self):
        super(DatetimeDateType, self).__init__(
            name='DatetimeDateType()')
        self.bitwidth = 64

datetime_date_type = DatetimeDateType()

@typeof_impl.register(datetime.date)
def typeof_pd_timestamp(val, c):
    return datetime_date_type

register_model(DatetimeDateType)(models.IntegerModel)

@infer_getattr
class DatetimeAttribute(AttributeTemplate):
    key = DatetimeDateType

    def generic_resolve(self, typ, attr):
        return types.int64

@lower_getattr(DatetimeDateType, 'year')
def datetime_get_year(context, builder, typ, val):
    return builder.lshr(val, lir.Constant(lir.IntType(64), 32))

@lower_getattr(DatetimeDateType, 'month')
def datetime_get_month(context, builder, typ, val):
    return builder.and_(builder.lshr(val, lir.Constant(lir.IntType(64), 16)), lir.Constant(lir.IntType(64), 0xFFFF))

@lower_getattr(DatetimeDateType, 'day')
def datetime_get_day(context, builder, typ, val):
    return builder.and_(val, lir.Constant(lir.IntType(64), 0xFFFF))

@numba.njit
def convert_datetime_date_array_to_native(x):
    return np.array([(val.day + (val.month << 16) + (val.year << 32)) for val in x], dtype=np.int64)

@numba.njit
def datetime_date_ctor(y, m, d):
    return datetime.date(y, m, d)

@unbox(DatetimeDateType)
def unbox_datetime_date(typ, val, c):
    year_obj = c.pyapi.object_getattr_string(val, "year")
    month_obj = c.pyapi.object_getattr_string(val, "month")
    day_obj = c.pyapi.object_getattr_string(val, "day")

    yll = c.pyapi.long_as_longlong(year_obj)
    mll = c.pyapi.long_as_longlong(month_obj)
    dll = c.pyapi.long_as_longlong(day_obj)

    nopython_date = c.builder.add(dll,
        c.builder.add(c.builder.shl(yll, lir.Constant(lir.IntType(64), 32)),
                        c.builder.shl(mll, lir.Constant(lir.IntType(64), 16))))

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(nopython_date, is_error=is_error)


def unbox_datetime_date_array(typ, val, c):
    #
    n = object_length(c, val)
    #cgutils.printf(c.builder, "len %d\n", n)
    arr_typ = types.Array(types.intp, 1, 'C')
    out_arr = _empty_nd_impl(c.context, c.builder, arr_typ, [n])

    with cgutils.for_range(c.builder, n) as loop:
        dt_date = sequence_getitem(c, val, loop.index)
        int_date = unbox_datetime_date(datetime_date_type, dt_date, c).value
        dataptr, shapes, strides = basic_indexing(
            c.context, c.builder, arr_typ, out_arr, (types.intp,), (loop.index,))
        store_item(c.context, c.builder, arr_typ, int_date, dataptr)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(out_arr._getvalue(), is_error=is_error)

def int_to_datetime_date_python(ia):
    return datetime.date(ia >> 32, (ia >> 16) & 0xffff, ia & 0xffff)

def int_array_to_datetime_date(ia):
    return np.vectorize(int_to_datetime_date_python)(ia)

def box_datetime_date_array(typ, val, c):
    ary = box_array(types.Array(types.int64, 1, 'C'), val, c)
    hpat_name = c.context.insert_const_string(c.builder.module, 'hpat')
    hpat_mod = c.pyapi.import_module_noblock(hpat_name)
    pte_mod = c.pyapi.object_getattr_string(hpat_mod, 'pd_timestamp_ext')
    iatdd = c.pyapi.object_getattr_string(pte_mod, 'int_array_to_datetime_date')
    res = c.pyapi.call_function_objargs(iatdd, [ary])
    return res

# TODO: move to utils or Numba

def object_length(c, obj):
    """
    len(obj)
    """
    pyobj_lltyp = c.context.get_argument_type(types.pyobject)
    fnty = lir.FunctionType(lir.IntType(64), [pyobj_lltyp])
    fn = c.builder.module.get_or_insert_function(fnty, name="PyObject_Length")
    return c.builder.call(fn, (obj,))

def sequence_getitem(c, obj, ind):
    """
    seq[ind]
    """
    pyobj_lltyp = c.context.get_argument_type(types.pyobject)
    fnty = lir.FunctionType(pyobj_lltyp, [pyobj_lltyp, lir.IntType(64)])
    fn = c.builder.module.get_or_insert_function(fnty, name="PySequence_GetItem")
    return c.builder.call(fn, (obj, ind))


@box(DatetimeDateType)
def box_datetime_date(typ, val, c):
    year_obj = c.pyapi.long_from_longlong(c.builder.lshr(val, lir.Constant(lir.IntType(64), 32)))
    month_obj = c.pyapi.long_from_longlong(c.builder.and_(c.builder.lshr(val, lir.Constant(lir.IntType(64), 16)), lir.Constant(lir.IntType(64), 0xFFFF)))
    day_obj = c.pyapi.long_from_longlong(c.builder.and_(val, lir.Constant(lir.IntType(64), 0xFFFF)))

    dt_obj = c.pyapi.unserialize(c.pyapi.serialize_object(datetime.date))
    res = c.pyapi.call_function_objargs(dt_obj, (year_obj, month_obj, day_obj))
    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    return res

@type_callable(datetime.date)
def type_timestamp(context):
    def typer(year, month, day):
        # TODO: check types
        return datetime_date_type
    return typer

@lower_builtin(datetime.date, types.int64, types.int64, types.int64)
def impl_ctor_timestamp(context, builder, sig, args):
    typ = sig.return_type
    year, month, day = args
    nopython_date = builder.add(day,
                    builder.add(builder.shl(year, lir.Constant(lir.IntType(64), 32)),
                                builder.shl(month, lir.Constant(lir.IntType(64), 16))))
    return nopython_date

@intrinsic
def datetime_date_to_int(typingctx, dt_date_tp):
    assert dt_date_tp == datetime_date_type
    def codegen(context, builder, sig, args):
        return args[0]
    return signature(types.int64, datetime_date_type), codegen

@intrinsic
def int_to_datetime_date(typingctx, dt_date_tp):
    assert dt_date_tp == types.intp
    def codegen(context, builder, sig, args):
        return args[0]
    return signature(datetime_date_type, types.int64), codegen

def set_df_datetime_date(df, cname, arr):
    df[cname] = arr

@infer_global(set_df_datetime_date)
class SetDfDTInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        assert isinstance(args[1], types.Const)
        return signature(types.none, *args)

SetDfDTInfer.support_literals = True

@lower_builtin(set_df_datetime_date, types.Any, types.Const, types.Array)
def set_df_datetime_date_lower(context, builder, sig, args):
    #
    col_name = sig.args[1].value
    data_arr = make_array(sig.args[2])(context, builder, args[2])
    num_elems = builder.extract_value(data_arr.shape, 0)

    pyapi = context.get_python_api(builder)
    gil_state = pyapi.gil_ensure()  # acquire GIL

    dt_class = pyapi.unserialize(pyapi.serialize_object(datetime.date))

    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [lir.IntType(64).as_pointer(), lir.IntType(64),
                                                lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="np_datetime_date_array_from_packed_ints")
    py_arr = builder.call(fn, [data_arr.data, num_elems, dt_class])

    # get column as string obj
    cstr = context.insert_const_string(builder.module, col_name)
    cstr_obj = pyapi.string_from_string(cstr)

    # set column array
    pyapi.object_setitem(args[0], cstr_obj, py_arr)

    pyapi.decref(py_arr)
    pyapi.decref(cstr_obj)

    pyapi.gil_release(gil_state)    # release GIL

    return context.get_dummy_value()


# datetime.date implementation that uses multiple ints to store year/month/day
# class DatetimeDateType(types.Type):
#     def __init__(self):
#         super(DatetimeDateType, self).__init__(
#             name='DatetimeDateType()')

# datetime_date_type = DatetimeDateType()

# @typeof_impl.register(datetime.date)
# def typeof_pd_timestamp(val, c):
#     return datetime_date_type


# @register_model(DatetimeDateType)
# class DatetimeDateModel(models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [
#             ('year', types.int64),
#             ('month', types.int64),
#             ('day', types.int64),
#         ]
#         models.StructModel.__init__(self, dmm, fe_type, members)

# make_attribute_wrapper(DatetimeDateType, 'year', 'year')
# make_attribute_wrapper(DatetimeDateType, 'month', 'month')
# make_attribute_wrapper(DatetimeDateType, 'day', 'day')

# @unbox(DatetimeDateType)
# def unbox_datetime_date(typ, val, c):
#     year_obj = c.pyapi.object_getattr_string(val, "year")
#     month_obj = c.pyapi.object_getattr_string(val, "month")
#     day_obj = c.pyapi.object_getattr_string(val, "day")

#     dt_date = cgutils.create_struct_proxy(typ)(c.context, c.builder)
#     dt_date.year = c.pyapi.long_as_longlong(year_obj)
#     dt_date.month = c.pyapi.long_as_longlong(month_obj)
#     dt_date.day = c.pyapi.long_as_longlong(day_obj)

#     c.pyapi.decref(year_obj)
#     c.pyapi.decref(month_obj)
#     c.pyapi.decref(day_obj)

#     is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#     return NativeValue(dt_date._getvalue(), is_error=is_error)

# @box(DatetimeDateType)
# def box_datetime_date(typ, val, c):
#     dt_date = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
#     year_obj = c.pyapi.long_from_longlong(dt_date.year)
#     month_obj = c.pyapi.long_from_longlong(dt_date.month)
#     day_obj = c.pyapi.long_from_longlong(dt_date.day)
#     dt_obj = c.pyapi.unserialize(c.pyapi.serialize_object(datetime.date))
#     res = c.pyapi.call_function_objargs(dt_obj, (year_obj, month_obj, day_obj))
#     c.pyapi.decref(year_obj)
#     c.pyapi.decref(month_obj)
#     c.pyapi.decref(day_obj)
#     return res

# @type_callable(datetime.date)
# def type_timestamp(context):
#     def typer(year, month, day):
#         # TODO: check types
#         return datetime_date_type
#     return typer

# @lower_builtin(datetime.date, types.int64, types.int64, types.int64)
# def impl_ctor_timestamp(context, builder, sig, args):
#     typ = sig.return_type
#     year, month, day = args
#     ts = cgutils.create_struct_proxy(typ)(context, builder)
#     ts.year = year
#     ts.month = month
#     ts.day = day
#     return ts._getvalue()

#------------------------------------------------------------------------

class PandasTimestampType(types.Type):
    def __init__(self):
        super(PandasTimestampType, self).__init__(
            name='PandasTimestampType()')

pandas_timestamp_type = PandasTimestampType()

@typeof_impl.register(pd.Timestamp)
def typeof_pd_timestamp(val, c):
    return pandas_timestamp_type

@typeof_impl.register(datetime.datetime)
def typeof_datetime_datetime(val, c):
    return pandas_timestamp_type

ts_field_typ = types.int64


@register_model(PandasTimestampType)
class PandasTimestampModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('year', ts_field_typ),
            ('month', ts_field_typ),
            ('day', ts_field_typ),
            ('hour', ts_field_typ),
            ('minute', ts_field_typ),
            ('second', ts_field_typ),
            ('microsecond', ts_field_typ),
            ('nanosecond', ts_field_typ),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)

make_attribute_wrapper(PandasTimestampType, 'year', 'year')
make_attribute_wrapper(PandasTimestampType, 'month', 'month')
make_attribute_wrapper(PandasTimestampType, 'day', 'day')
make_attribute_wrapper(PandasTimestampType, 'hour', 'hour')
make_attribute_wrapper(PandasTimestampType, 'minute', 'minute')
make_attribute_wrapper(PandasTimestampType, 'second', 'second')
make_attribute_wrapper(PandasTimestampType, 'microsecond', 'microsecond')
make_attribute_wrapper(PandasTimestampType, 'nanosecond', 'nanosecond')

@overload_method(PandasTimestampType, 'date')
def overload_pd_timestamp_date(ptt):
    def pd_timestamp_date_impl(ptt):
        return datetime.date(ptt.year, ptt.month, ptt.day)
    return pd_timestamp_date_impl

@overload_method(PandasTimestampType, 'isoformat')
def overload_pd_timestamp_isoformat(ts_typ, sep=None):
    if sep is None:
        def timestamp_isoformat_impl(ts):
            assert ts.nanosecond == 0 # TODO: handle nanosecond (timestamps.pyx)
            _time = str_2d(ts.hour) + ':' + str_2d(ts.minute) + ':' + str_2d(ts.second)
            res = str(ts.year) + '-' + str_2d(ts.month) + '-' + str_2d(ts.day) + 'T' + _time
            return res
    else:
        def timestamp_isoformat_impl(ts, sep):
            assert ts.nanosecond == 0 # TODO: handle nanosecond (timestamps.pyx)
            _time = str_2d(ts.hour) + ':' + str_2d(ts.minute) + ':' + str_2d(ts.second)
            res = str(ts.year) + '-' + str_2d(ts.month) + '-' + str_2d(ts.day) + sep + _time
            return res

    return timestamp_isoformat_impl

# TODO: support general string formatting
@numba.njit
def str_2d(a):
    res = str(a)
    if len(res) == 1:
        return '0' + res
    return res

@overload(str)
def ts_str_overload(in_typ):
    if in_typ == pandas_timestamp_type:
        return lambda a: a.isoformat(' ')

@unbox(PandasTimestampType)
def unbox_pandas_timestamp(typ, val, c):
    year_obj = c.pyapi.object_getattr_string(val, "year")
    month_obj = c.pyapi.object_getattr_string(val, "month")
    day_obj = c.pyapi.object_getattr_string(val, "day")
    hour_obj = c.pyapi.object_getattr_string(val, "hour")
    minute_obj = c.pyapi.object_getattr_string(val, "minute")
    second_obj = c.pyapi.object_getattr_string(val, "second")
    microsecond_obj = c.pyapi.object_getattr_string(val, "microsecond")
    nanosecond_obj = c.pyapi.object_getattr_string(val, "nanosecond")

    pd_timestamp = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    pd_timestamp.year = c.pyapi.long_as_longlong(year_obj)
    pd_timestamp.month = c.pyapi.long_as_longlong(month_obj)
    pd_timestamp.day = c.pyapi.long_as_longlong(day_obj)
    pd_timestamp.hour = c.pyapi.long_as_longlong(hour_obj)
    pd_timestamp.minute = c.pyapi.long_as_longlong(minute_obj)
    pd_timestamp.second = c.pyapi.long_as_longlong(second_obj)
    pd_timestamp.microsecond = c.pyapi.long_as_longlong(microsecond_obj)
    pd_timestamp.nanosecond = c.pyapi.long_as_longlong(nanosecond_obj)

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(microsecond_obj)
    c.pyapi.decref(nanosecond_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(pd_timestamp._getvalue(), is_error=is_error)

@box(PandasTimestampType)
def box_pandas_timestamp(typ, val, c):
    pdts = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    year_obj = c.pyapi.long_from_longlong(pdts.year)
    month_obj = c.pyapi.long_from_longlong(pdts.month)
    day_obj = c.pyapi.long_from_longlong(pdts.day)
    hour_obj = c.pyapi.long_from_longlong(pdts.hour)
    minute_obj = c.pyapi.long_from_longlong(pdts.minute)
    second_obj = c.pyapi.long_from_longlong(pdts.second)
    us_obj = c.pyapi.long_from_longlong(pdts.microsecond)
    ns_obj = c.pyapi.long_from_longlong(pdts.nanosecond)

    pdts_obj = c.pyapi.unserialize(c.pyapi.serialize_object(pd.Timestamp))
    res = c.pyapi.call_function_objargs(pdts_obj, (year_obj, month_obj, day_obj, hour_obj, minute_obj, second_obj, us_obj, ns_obj))
    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(us_obj)
    c.pyapi.decref(ns_obj)
    return res

@type_callable(pd.Timestamp)
def type_timestamp(context):
    def typer(year, month, day, hour, minute, second, us, ns):
        # TODO: check types
        return pandas_timestamp_type
    return typer

@type_callable(pd.Timestamp)
def type_timestamp(context):
    def typer(datetime_type):
        # TODO: check types
        return pandas_timestamp_type
    return typer

@type_callable(datetime.datetime)
def type_timestamp(context):
    def typer(year, month, day):  # how to handle optional hour, minute, second, us, ns?
        # TODO: check types
        return pandas_timestamp_type
    return typer

@lower_builtin(pd.Timestamp, types.int64, types.int64, types.int64, types.int64,
                types.int64, types.int64, types.int64, types.int64)
def impl_ctor_timestamp(context, builder, sig, args):
    typ = sig.return_type
    year, month, day, hour, minute, second, us, ns = args
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    ts.year = year
    ts.month = month
    ts.day = day
    ts.hour = hour
    ts.minute = minute
    ts.second = second
    ts.microsecond = us
    ts.nanosecond = ns
    return ts._getvalue()

@lower_builtin(pd.Timestamp, pandas_timestamp_type)
def impl_ctor_ts_ts(context, builder, sig, args):
    typ = sig.return_type
    rhs = args[0]
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    rhsproxy = cgutils.create_struct_proxy(typ)(context, builder)
    rhsproxy._setvalue(rhs)
    cgutils.copy_struct(ts, rhsproxy)
    return ts._getvalue()

#              , types.int64, types.int64, types.int64, types.int64, types.int64)
@lower_builtin(datetime.datetime, types.int64, types.int64, types.int64)
def impl_ctor_datetime(context, builder, sig, args):
    typ = sig.return_type
    year, month, day = args
    #year, month, day, hour, minute, second, us, ns = args
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    ts.year = year
    ts.month = month
    ts.day = day
    ts.hour = lir.Constant(lir.IntType(64), 0)
    ts.minute = lir.Constant(lir.IntType(64), 0)
    ts.second = lir.Constant(lir.IntType(64), 0)
    ts.microsecond = lir.Constant(lir.IntType(64), 0)
    ts.nanosecond = lir.Constant(lir.IntType(64), 0)
    #ts.hour = hour
    #ts.minute = minute
    #ts.second = second
    #ts.microsecond = us
    #ts.nanosecond = ns
    return ts._getvalue()



@lower_cast(types.NPDatetime('ns'), types.int64)
def dt64_to_integer(context, builder, fromty, toty, val):
    # dt64 is stored as int64 so just return value
    return val

@numba.njit
def convert_timestamp_to_datetime64(ts):
    year = ts.year - 1970
    days = year * 365
    if days >= 0:
        year += 1
        days += year // 4
        year += 68
        days -= year // 100
        year += 300
        days += year // 400
    else:
        year -= 2
        days += year // 4
        year -= 28
        days -= year // 100
        days += year // 400
    leapyear = (ts.year % 400 == 0) or (ts.year %4 == 0 and ts.year %100 != 0)
    month_len = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leapyear:
        month_len[1] = 29

    for i in range(ts.month - 1):
        days += month_len[i]

    days += ts.day - 1

    return ((((days * 24 + ts.hour) * 60 + ts.minute) * 60 + ts.second) * 1000000 + ts.microsecond) * 1000 + ts.nanosecond

@numba.njit
def convert_datetime64_to_timestamp(dt64):
    # pandas 0.23 np_datetime.c:762
    perday = 24 * 60 * 60 * 1000 * 1000 * 1000

    if dt64 >= 0:
        in_day = dt64 % perday
        dt64 = dt64 // perday
    else:
        in_day = (perday - 1) + (dt64 + 1) % perday
        dt64 = dt64 // perday - (0 if (dt64 % perday == 0) else 1)

    # pandas 0.23np_datetime.c:173
    days400years = 146097
    days = dt64 - 10957
    if days >= 0:
        year = 400 * (days // days400years)
        days = days % days400years
    else:
        year = 400 * ((days - (days400years - 1)) // days400years)
        days = days % days400years
        if days < 0:
            days += days400years

    if days >= 366:
        year += 100 * ((days - 1) // 36524)
        days = (days - 1) % 36524
        if days >= 365:
            year += 4 * ((days + 1) // 1461)
            days = (days + 1) % 1461
            if days >= 366:
                year += (days - 1) // 365
                days = (days - 1) % 365

    year = year + 2000
    # pandas 0.23 np_datetime.c:237
    leapyear = (year % 400 == 0) or (year %4 == 0 and year %100 != 0)
    month_len = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leapyear:
        month_len[1] = 29

    for i in range(12):
        if days < month_len[i]:
            month = i + 1
            day = days + 1
            break
        else:
            days = days - month_len[i]

    return pd.Timestamp(year, month, day,
                        in_day // (60 * 60 * 1000000000), #hour
                        (in_day // (60 * 1000000000)) % 60, #minute
                        (in_day // 1000000000) % 60, #second
                        (in_day // 1000) % 1000000, #microsecond
                        in_day % 1000) #nanosecond

#-----------------------------------------------------------

def myref(val):
    pass

@type_callable(myref)
def type_myref(context):
    def typer(val):
        return types.voidptr
    return typer

#-----------------------------------------------------------

def integer_to_timedelta64(val):
    return np.timedelta64(val)

@type_callable(integer_to_timedelta64)
def type_int_to_timedelta64(context):
    def typer(val):
        return types.NPTimedelta('ns')
    return typer

@lower_builtin(integer_to_timedelta64, types.int64)
def impl_int_to_timedelta64(context, builder, sig, args):
    return args[0]

#-----------------------------------------------------------

def integer_to_dt64(val):
    return np.datetime64(val)

@type_callable(integer_to_dt64)
def type_int_to_dt64(context):
    def typer(val):
        return types.NPDatetime('ns')
    return typer

@lower_builtin(integer_to_dt64, types.int64)
def impl_int_to_dt64(context, builder, sig, args):
    return args[0]

#-----------------------------------------------------------

def dt64_to_integer(val):
    return int(val)

@type_callable(dt64_to_integer)
def type_dt64_to_int(context):
    def typer(val):
        return types.int64
    return typer

@lower_builtin(dt64_to_integer, types.NPDatetime('ns'))
def impl_dt64_to_int(context, builder, sig, args):
    return args[0]

#-----------------------------------------------------------
def timedelta64_to_integer(val):
    return int(val)

@type_callable(timedelta64_to_integer)
def type_dt64_to_int(context):
    def typer(val):
        return types.int64
    return typer

@lower_builtin(timedelta64_to_integer, types.NPTimedelta('ns'))
def impl_dt64_to_int(context, builder, sig, args):
    return args[0]

#-----------------------------------------------------------

@lower_builtin(myref, types.int32)
@lower_builtin(myref, types.int64)
def impl_myref_int32(context, builder, sig, args):
    typ = types.voidptr
    val = args[0]
    assert isinstance(val, lir.instructions.LoadInstr)
    return builder.bitcast(val.operands[0], lir.IntType(8).as_pointer())

@lower_builtin(myref, PandasDtsType)
def impl_myref_pandas_dts_type(context, builder, sig, args):
    typ = types.voidptr
    val = args[0]
    assert isinstance(val, lir.instructions.LoadInstr)
    return builder.bitcast(val.operands[0], lir.IntType(8).as_pointer())

# tslib_so = inspect.getfile(pd._libs.tslib)
# tslib_cdll = ctypes.CDLL(tslib_so)
# func_parse_iso = tslib_cdll.parse_iso_8601_datetime
# func_parse_iso.restype = ctypes.c_int32
# func_parse_iso.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
# func_dts_to_dt = tslib_cdll.pandas_datetimestruct_to_datetime
# func_dts_to_dt.restype = ctypes.c_int64
# func_dts_to_dt.argtypes = [ctypes.c_int, ctypes.c_void_p]

sig = types.intp(
                types.voidptr,             # C str
                types.intp,             # len(str)
                types.voidptr,          # struct ptr
                types.voidptr,  # int ptr
                types.voidptr,  # int ptr
                )
parse_iso_8601_datetime = types.ExternalFunction("parse_iso_8601_datetime", sig)
sig = types.intp(
                types.intp,             # fr magic number
                types.voidptr,          # struct ptr
                types.voidptr,  # out int ptr
                )
convert_datetimestruct_to_datetime = types.ExternalFunction("convert_datetimestruct_to_datetime", sig)

@numba.njit(locals={'arg1': numba.int32, 'arg3': numba.int32, 'arg4': numba.int32})
def parse_datetime_str(str):
    arg0 = hpat.str_ext.getpointer(str)
    arg1 = len(str)
    arg2 = PANDAS_DATETIMESTRUCT()
    arg3 = np.int32(13)
    arg4 = np.int32(13)
    arg2ref = myref(arg2)
    retval = parse_iso_8601_datetime(arg0, arg1, arg2ref, myref(arg3), myref(arg4))
    out = 0
    retval2 = convert_datetimestruct_to_datetime(10, arg2ref, myref(out))
    return integer_to_dt64(out)

#     retval = func_parse_iso(arg0, arg1, arg2ref, myref(arg3), myref(arg4))
#     # "10" is magic enum value for PANDAS_FR_ns (nanosecond date time unit)
# #        return func_dts_to_dt(10, arg2ref)
#     return integer_to_dt64(func_dts_to_dt(10, arg2ref))

#----------------------------------------------------------------------------------------------


# XXX: code for timestamp series getitem in regular Numba

# @infer
# class GetItemTimestampSeries(AbstractTemplate):
#     key = "getitem"
#
#     def generic(self, args, kws):
#         assert not kws
#         [ary, idx] = args
#         if isinstance(ary, TimestampSeriesType):
#             if isinstance(idx, types.SliceType):
#                 return signature(timestamp_series_type, *args)
#             else:
#                 assert isinstance(idx, types.Integer)
#                 return signature(pandas_timestamp_type, *args)
# from numba.targets.listobj import ListInstance
# from llvmlite import ir as lir
# import llvmlite.binding as ll
# #import hdatetime_ext
# #ll.add_symbol('dt_to_timestamp', hdatetime_ext.dt_to_timestamp)
#
# @lower_builtin('getitem', TimestampSeriesType, types.Integer)
# def lower_timestamp_series_getitem(context, builder, sig, args):
#     #print("lower_timestamp_series_getitem", sig, type(sig), args, type(args), sig.return_type)
#     old_ret = sig.return_type
#     sig.return_type = types.NPDatetime('ns')
#     # If the return type is a view then just use standard array getitem.
#     if isinstance(sig.return_type, types.Buffer):
#         return getitem_arraynd_intp(context, builder, sig, args)
#     else:
#         # The standard getitem_arraynd_intp should return a NPDatetime here
#         # that then needs to be converted into a Pandas Timestamp.
#         unconverted = getitem_arraynd_intp(context, builder, sig, args)
#         sig.return_type = old_ret
#         ret = context.make_helper(builder, pandas_timestamp_type)
#         resptr = builder.bitcast(ret._getpointer(), lir.IntType(8).as_pointer())
#         dt_to_datetime_fnty = lir.FunctionType(lir.VoidType(),
#                                                [lir.IntType(64), lir.IntType(8).as_pointer()])
#         dt_to_datetime_fn = builder.module.get_or_insert_function(dt_to_datetime_fnty, name="dt_to_timestamp")
#         builder.call(dt_to_datetime_fn, [unconverted, resptr])
#         res = ret._getvalue()
#         return impl_ret_untracked(context, builder, PandasTimestampType, res)

import numba
import hpat
from numba import types
from numba.typing.templates import infer_global, AbstractTemplate, infer, signature, AttributeTemplate, infer_getattr
import numba.typing.typeof
from numba.extending import (typeof_impl, type_callable, models, register_model, NativeValue,
                             make_attribute_wrapper, lower_builtin, box, unbox, lower_getattr)
from numba import cgutils
from numba.targets.boxing import unbox_array
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed, impl_ret_untracked
from numba.targets.arrayobj import getitem_arraynd_intp
import pandas as pd
import numpy as np


class PandasTimestampType(types.Type):
    def __init__(self):
        super(PandasTimestampType, self).__init__(
            name='PandasTimestampType()')

pandas_timestamp_type = PandasTimestampType()

@typeof_impl.register(pd.Timestamp)
def typeof_pd_timestamp(val, c):
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

# long_from_signed_int
    pd_timestamp = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    #import pdb
    #pdb.set_trace()
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

@type_callable(pd.Timestamp)
def type_timestamp(context):
    def typer(year, month, day, hour, minute, second, us, ns):
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

@numba.njit(nopython=True)
def convert_datetime64_to_timestamp(dt64):
    perday = 24 * 60 * 60 * 1000 * 1000 * 1000

    if dt64 > 0:
        in_day = dt64 % perday
        dt64 = dt64 // perday
    else:
        in_day = (perday - 1) + (dt64 + 1) % perday
        dt64 = dt64 // perday - (0 if (dt64 % perday == 0) else 1)

    days400years = 146097
    days = dt64 - 10957
    if days >= 0:
        year = 400 * (days // days400years)
        days = days % days400years
    else:
        years = 400 * ((days - (days400years - 1)) // days400years)
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

#----------------------------------------------------------------------------------------------

class TimestampSeriesType(types.Array):
    def __init__(self):
        super(TimestampSeriesType, self).__init__(dtype=types.NPDatetime('ns'), ndim=1, layout='C')

timestamp_series_type = TimestampSeriesType()

@register_model(TimestampSeriesType)
class TimestampSeriesModel(models.ArrayModel):
    pass

@unbox(TimestampSeriesType)
def unbox_timestamp_series(typ, val, c):
    getvalues = c.pyapi.object_getattr_string(val, "values")
    return unbox_array(types.Array(dtype=types.NPDatetime('ns'), ndim=1, layout='C'), getvalues, c)

@typeof_impl.register(pd.Series)
def typeof_pd_timestamp_series(val, c):
    if len(val) > 0 and isinstance(val[0], pd.Timestamp):
        return timestamp_series_type


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

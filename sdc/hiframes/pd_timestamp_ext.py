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


from numba import types
from numba.extending import (typeof_impl, type_callable, models, register_model, NativeValue,
                             make_attribute_wrapper, lower_builtin, box, unbox, lower_cast,
                             lower_getattr, infer_getattr, overload_method, overload, intrinsic)
from numba import cgutils

from llvmlite import ir as lir

import pandas as pd

import datetime
from .. import hdatetime_ext
import llvmlite.binding as ll
ll.add_symbol('parse_iso_8601_datetime', hdatetime_ext.parse_iso_8601_datetime)
ll.add_symbol('convert_datetimestruct_to_datetime', hdatetime_ext.convert_datetimestruct_to_datetime)
ll.add_symbol('np_datetime_date_array_from_packed_ints', hdatetime_ext.np_datetime_date_array_from_packed_ints)


# ---------------------------------------------------------------

# datetime.date implementation that uses a single int to store year/month/day


class DatetimeDateType(types.Type):
    def __init__(self):
        super(DatetimeDateType, self).__init__(
            name='DatetimeDateType()')
        self.bitwidth = 64


datetime_date_type = DatetimeDateType()


register_model(DatetimeDateType)(models.IntegerModel)


@box(DatetimeDateType)
def box_datetime_date(typ, val, c):
    year_obj = c.pyapi.long_from_longlong(c.builder.lshr(val, lir.Constant(lir.IntType(64), 32)))
    month_obj = c.pyapi.long_from_longlong(
        c.builder.and_(
            c.builder.lshr(
                val, lir.Constant(
                    lir.IntType(64), 16)), lir.Constant(
                lir.IntType(64), 0xFFFF)))
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

# ------------------------------------------------------------------------

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


@overload_method(PandasTimestampType, 'date')
def overload_pd_timestamp_date(ptt):
    def pd_timestamp_date_impl(ptt):
        return datetime.date(ptt.year, ptt.month, ptt.day)
    return pd_timestamp_date_impl


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
@lower_builtin(datetime.datetime, types.IntegerLiteral, types.IntegerLiteral, types.IntegerLiteral)
def impl_ctor_datetime(context, builder, sig, args):
    typ = sig.return_type
    year, month, day = args
    ts = cgutils.create_struct_proxy(typ)(context, builder)
    ts.year = year
    ts.month = month
    ts.day = day
    ts.hour = lir.Constant(lir.IntType(64), 0)
    ts.minute = lir.Constant(lir.IntType(64), 0)
    ts.second = lir.Constant(lir.IntType(64), 0)
    ts.microsecond = lir.Constant(lir.IntType(64), 0)
    ts.nanosecond = lir.Constant(lir.IntType(64), 0)
    return ts._getvalue()

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
from glob import glob
import pandas as pd
import numpy as np

import pdb

class PandasTimestampType(types.Type):
    def __init__(self):
        super(PandasTimestampType, self).__init__(
            name='PandasTimestampType()')


pandas_timestamp_type = PandasTimestampType()


@typeof_impl.register(pd._libs.tslib.Timestamp)
def typeof_pd_timestamp(val, c):
    return pandas_timestamp_type

field_typ1 = types.int64
field_typ2 = types.int64

#field_typ1 = types.int16
#field_typ2 = types.int8

@register_model(PandasTimestampType)
class PandasTimestampModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
#            ('value', types.int64),
#            ('nanosecond', types.int64),
            ('year', field_typ1),
            ('month', field_typ2),
            ('day', field_typ2),
            ('hour', field_typ2),
            ('minute', field_typ2),
            ('second', field_typ2),
            ('microsecond', field_typ1),
            ('nanosecond', field_typ1),
#            ('tz', field_typ),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)
    
#make_attribute_wrapper(PandasTimestampType, 'value', 'value')
#make_attribute_wrapper(PandasTimestampType, 'nanosecond', 'nanosecond')
make_attribute_wrapper(PandasTimestampType, 'year', 'year')
make_attribute_wrapper(PandasTimestampType, 'month', 'month')
make_attribute_wrapper(PandasTimestampType, 'day', 'day')
make_attribute_wrapper(PandasTimestampType, 'hour', 'hour')
make_attribute_wrapper(PandasTimestampType, 'minute', 'minute')
make_attribute_wrapper(PandasTimestampType, 'second', 'second')
make_attribute_wrapper(PandasTimestampType, 'microsecond', 'microsecond')
make_attribute_wrapper(PandasTimestampType, 'nanosecond', 'nanosecond')
#make_attribute_wrapper(PandasTimestampType, 'tz', 'tz')

#@overload_attribute(PandasTimestampType, "minute")
#def get_minute(ts):
#    def getter(ts):
#        return ts
#    return getter

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
#    tz_obj = c.pyapi.object_getattr_string(val, "tz")

# long_from_signed_int
    pd_timestamp = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    pd_timestamp.year = c.pyapi.long_from_long(year_obj)
    pd_timestamp.month = c.pyapi.long_from_long(month_obj)
    pd_timestamp.day = c.pyapi.long_from_long(day_obj)
    pd_timestamp.hour = c.pyapi.long_from_long(hour_obj)
    pd_timestamp.minute = c.pyapi.long_from_long(minute_obj)
    pd_timestamp.second = c.pyapi.long_from_long(second_obj)
    pd_timestamp.microsecond = c.pyapi.long_from_long(microsecond_obj)
    pd_timestamp.nanosecond = c.pyapi.long_from_long(nanosecond_obj)
    #pd_timestamp.tz = c.pyapi.long_from_long(tz_obj)
#    pd_timestamp.tz = 0

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    c.pyapi.decref(hour_obj)
    c.pyapi.decref(minute_obj)
    c.pyapi.decref(second_obj)
    c.pyapi.decref(microsecond_obj)
    c.pyapi.decref(nanosecond_obj)
#    c.pyapi.decref(tz_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(pd_timestamp._getvalue(), is_error=is_error)

#----------------------------------------------------------------------------------------------

class TimestampSeriesType(types.Array):
    def __init__(self):
        super(TimestampSeriesType, self).__init__(dtype=types.NPDatetime('ns'), ndim=1, layout='C')

timestamp_series_type = TimestampSeriesType()

@register_model(TimestampSeriesType)
class TimestampSeriesModel(models.ArrayModel):
    pass

def get_values(x):
    print("get_values", x, type(x))
    return x.values

@unbox(TimestampSeriesType)
def unbox_timestamp_series(typ, val, c):
    print("unbox_timestamp_series", typ, type(val), c, type(c), c.pyapi, type(c.pyapi))
    getvalues = c.pyapi.object_getattr_string(val, "values")
    return unbox_array(types.Array(dtype=types.NPDatetime('ns'), ndim=1, layout='C'), getvalues, c)

@typeof_impl.register(pd.Series)
def typeof_pd_timestamp_series(val, c):
    print("typeof_pd_timestamp_series", type(val), c, type(c), val[0], type(val[0]), val.dtype, type(val.dtype))
    if len(val) > 0 and isinstance(val[0], pd._libs.tslib.Timestamp):
        print("found")
        return timestamp_series_type

@infer
class GetItemStringArray(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        if isinstance(ary, TimestampSeriesType):
            if isinstance(idx, types.SliceType):
                return signature(timestamp_series_type, *args)
            else:
                assert isinstance(idx, types.Integer)
                return signature(pandas_timestamp_type, *args)

from numba.targets.listobj import ListInstance
from llvmlite import ir as lir
import llvmlite.binding as ll
#import tsseries_ext
#ll.add_symbol('getitem_timestamp_series', tsseries_ext.getitem_timestamp_series)

@lower_builtin('getitem', TimestampSeriesType, types.Integer)
def lower_timestamp_series_getitem(context, builder, sig, args):
    pdb.set_trace()
    print("lower_timestamp_series_getitem", sig, type(sig), args, type(args), sig.return_type)
    old_ret = sig.return_type
    sig.return_type = types.NPDatetime('ns')
    # If the return type is a view then just use standard array getitem.
    if isinstance(sig.return_type, types.Buffer):
        return getitem_arraynd_intp(context, builder, sig, args) 
    else:
        # The standard getitem_arraynd_intp should return a NPDatetime here
        # that then needs to be converted into a Pandas Timestamp.
        unconverted = getitem_arraynd_intp(context, builder, sig, args)
        sig.return_type = old_ret
        ret = context.make_helper(builder, pandas_timestamp_type)
        res = ret._getvalue()
        return impl_ret_untracked(context, builder, PandasTimestampType, res)

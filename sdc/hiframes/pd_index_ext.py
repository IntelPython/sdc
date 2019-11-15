# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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


import operator
import pandas as pd
import numpy as np
import numba
from numba import types
from numba.extending import (models, register_model, lower_cast, infer_getattr,
                             type_callable, infer, overload, make_attribute_wrapper, box)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
                                    AttributeTemplate, bound_function)
from numba.targets.boxing import box_array

import sdc
from sdc.str_ext import string_type
import sdc.hiframes
from sdc.hiframes.pd_series_ext import (is_str_series_typ, string_array_type,
                                         SeriesType)
from sdc.hiframes.pd_timestamp_ext import pandas_timestamp_type, datetime_date_type
from sdc.hiframes.datetime_date_ext import array_datetime_date

_dt_index_data_typ = types.Array(types.NPDatetime('ns'), 1, 'C')
_timedelta_index_data_typ = types.Array(types.NPTimedelta('ns'), 1, 'C')


class DatetimeIndexType(types.IterableType):
    """Temporary type class for DatetimeIndex objects.
    """

    def __init__(self, is_named=False):
        # TODO: support other properties like freq/tz/dtype/yearfirst?
        self.is_named = is_named
        super(DatetimeIndexType, self).__init__(
            name="DatetimeIndex(is_named = {})".format(is_named))

    def copy(self):
        # XXX is copy necessary?
        return DatetimeIndexType(self.is_named)

    @property
    def key(self):
        # needed?
        return self.is_named

    def unify(self, typingctx, other):
        # needed?
        return super(DatetimeIndexType, self).unify(typingctx, other)

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timestamp
        return types.iterators.ArrayIterator(_dt_index_data_typ)


# @typeof_impl.register(pd.DatetimeIndex)

@register_model(DatetimeIndexType)
class DatetimeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', _dt_index_data_typ),
            ('name', string_type),
        ]
        super(DatetimeIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DatetimeIndexType, 'data', '_data')
make_attribute_wrapper(DatetimeIndexType, 'name', '_name')


@box(DatetimeIndexType)
def box_dt_index(typ, val, c):
    """
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    dt_index = numba.cgutils.create_struct_proxy(
        typ)(c.context, c.builder, val)

    arr = box_array(_dt_index_data_typ, dt_index.data, c)

    # TODO: support name boxing
    # if typ.is_named:
    #     name = c.pyapi.from_native_value(string_type, series.name)
    # else:
    #     name = c.pyapi.make_none()

    res = c.pyapi.call_method(pd_class_obj, "DatetimeIndex", (arr,))

    c.pyapi.decref(pd_class_obj)
    return res


@infer_getattr
class DatetimeIndexAttribute(AttributeTemplate):
    key = DatetimeIndexType

    def resolve_values(self, ary):
        return _dt_index_data_typ

    def resolve_date(self, ary):
        return array_datetime_date

    @bound_function("dt_index.max")
    def resolve_max(self, ary, args, kws):
        assert not kws
        return signature(pandas_timestamp_type, *args)

    @bound_function("dt_index.min")
    def resolve_min(self, ary, args, kws):
        assert not kws
        return signature(pandas_timestamp_type, *args)


# all datetimeindex fields return int64 same as Timestamp fields
def resolve_date_field(self, ary):
    # TODO: return Int64Index
    return SeriesType(types.int64)


for field in sdc.hiframes.pd_timestamp_ext.date_fields:
    setattr(DatetimeIndexAttribute, "resolve_" + field, resolve_date_field)


@overload(pd.DatetimeIndex)
def pd_datetimeindex_overload(data=None, freq=None, start=None, end=None,
                              periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
                              dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
                              verify_integrity=True):
    # TODO: check/handle other input
    if data is None:
        raise ValueError("data argument in pd.DatetimeIndex() expected")

    if data != string_array_type and not is_str_series_typ(data):
        return (lambda data=None, freq=None, start=None, end=None,
                periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
                dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
                verify_integrity=True: sdc.hiframes.api.init_datetime_index(
                    sdc.hiframes.api.ts_series_to_arr_typ(data), name))

    def f(data=None, freq=None, start=None, end=None,
          periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
          dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
          verify_integrity=True):
        S = sdc.hiframes.api.parse_datetimes_from_strings(data)
        return sdc.hiframes.api.init_datetime_index(S, name)

    return f


# ----------- Timedelta

# similar to DatetimeIndex
class TimedeltaIndexType(types.IterableType):
    """Temporary type class for TimedeltaIndex objects.
    """

    def __init__(self, is_named=False):
        # TODO: support other properties like unit/freq?
        self.is_named = is_named
        super(TimedeltaIndexType, self).__init__(
            name="TimedeltaIndexType(is_named = {})".format(is_named))

    def copy(self):
        # XXX is copy necessary?
        return TimedeltaIndexType(self.is_named)

    @property
    def key(self):
        # needed?
        return self.is_named

    def unify(self, typingctx, other):
        # needed?
        return super(TimedeltaIndexType, self).unify(typingctx, other)

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timedelta
        return types.iterators.ArrayIterator(_timedelta_index_data_typ)


@register_model(TimedeltaIndexType)
class TimedeltaIndexTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', _timedelta_index_data_typ),
            ('name', string_type),
        ]
        super(TimedeltaIndexTypeModel, self).__init__(dmm, fe_type, members)


@infer_getattr
class TimedeltaIndexAttribute(AttributeTemplate):
    key = TimedeltaIndexType

    def resolve_values(self, ary):
        return _timedelta_index_data_typ

    # TODO: support pd.Timedelta
    # @bound_function("timedelta_index.max")
    # def resolve_max(self, ary, args, kws):
    #     assert not kws
    #     return signature(pandas_timestamp_type, *args)

    # @bound_function("timedelta_index.min")
    # def resolve_min(self, ary, args, kws):
    #     assert not kws
    #     return signature(pandas_timestamp_type, *args)


# all datetimeindex fields return int64 same as Timestamp fields
def resolve_timedelta_field(self, ary):
    # TODO: return Int64Index
    return types.Array(types.int64, 1, 'C')


for field in sdc.hiframes.pd_timestamp_ext.timedelta_fields:
    setattr(TimedeltaIndexAttribute, "resolve_" + field, resolve_timedelta_field)

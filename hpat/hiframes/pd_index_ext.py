import operator
import pandas as pd
import numpy as np
import numba
from numba import types
from numba.extending import (models, register_model, lower_cast, infer_getattr,
    type_callable, infer, overload, make_attribute_wrapper)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
    AttributeTemplate, bound_function)

import hpat
from hpat.str_ext import string_type
import hpat.hiframes
from hpat.hiframes.pd_series_ext import string_series_type, string_array_type

_dt_index_data_typ = types.Array(types.NPDatetime('ns'), 1, 'C')


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


#@typeof_impl.register(pd.DatetimeIndex)

@register_model(DatetimeIndexType)
class DatetimeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', _dt_index_data_typ),
            ('name', string_type),
        ]
        super(DatetimeIndexModel, self).__init__(dmm, fe_type, members)


@infer_getattr
class DatetimeIndexAttribute(AttributeTemplate):
    key = DatetimeIndexType

    def resolve_values(self, ary):
        return _dt_index_data_typ


@overload(pd.DatetimeIndex)
def pd_datetimeindex_overload(data=None, freq=None, start=None, end=None,
        periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
        verify_integrity=True):
    # TODO: check/handle other input
    if data is None:
        raise ValueError("data argument in pd.DatetimeIndex() expected")

    if data not in (string_array_type, string_series_type):
        return (lambda data=None, freq=None, start=None, end=None,
        periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
        verify_integrity=True: hpat.hiframes.api.init_datetime_index(
            hpat.hiframes.api.ts_series_to_arr_typ(data), name))

    def f(data=None, freq=None, start=None, end=None,
        periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
        verify_integrity=True):
        S = hpat.hiframes.api.parse_datetimes_from_strings(data)
        return hpat.hiframes.api.init_datetime_index(S, name)

    return f

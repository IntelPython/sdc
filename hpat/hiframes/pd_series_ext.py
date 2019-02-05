import operator
import pandas as pd
import numpy as np
import numba
from numba import types
from numba.extending import (models, register_model, lower_cast, infer_getattr,
    type_callable, infer, overload, make_attribute_wrapper)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
    AttributeTemplate, bound_function)
from numba.typing.arraydecl import (get_array_index_type, _expand_integer,
    ArrayAttribute, SetItemBuffer)
from numba.typing.npydecl import (Numpy_rules_ufunc, NumpyRulesArrayOperator,
    NumpyRulesInplaceArrayOperator, NumpyRulesUnaryArrayOperator,
    NdConstructorLike)
import hpat
from hpat.str_ext import string_type, list_string_array_type
from hpat.str_arr_ext import (string_array_type, offset_typ, char_typ,
    str_arr_payload_type, StringArrayType, GetItemStringArray)
from hpat.hiframes.pd_timestamp_ext import pandas_timestamp_type, datetime_date_type
from hpat.hiframes.pd_categorical_ext import PDCategoricalDtype, get_categories_int_type
from hpat.hiframes.rolling import supported_rolling_funcs
import datetime


class SeriesType(types.IterableType):
    """Temporary type class for Series objects.
    """
    def __init__(self, dtype, data=None, index=None, is_named=False):
        # keeping data array in type since operators can make changes such
        # as making array unaligned etc.
        data = _get_series_array_type(dtype) if data is None else data
        self.dtype = dtype
        self.data = data
        if index is None:
            index = types.none
        self.index = index
        # keep is_named in type to enable boxing
        self.is_named = is_named
        super(SeriesType, self).__init__(
            name="series({}, {}, {}, {})".format(dtype, data, index, is_named))

    def copy(self, dtype=None):
        # XXX is copy necessary?
        index = None if self.index is None else self.index.copy()
        dtype = dtype if dtype is not None else self.dtype
        data = _get_series_array_type(dtype)
        return SeriesType(dtype, data, index)

    @property
    def key(self):
        # needed?
        return self.dtype, self.data, self.index, self.is_named

    def unify(self, typingctx, other):
        if isinstance(other, SeriesType):
            new_index = None
            if self.index is not None:
                new_index = self.index.unify(other.index)
            if other.index is not None:
                new_index = other.index.unify(self.index)

            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                return SeriesType(
                    self.dtype, self.data.unify(other.data), new_index)

        # XXX: unify Series/Array as Array
        return super(SeriesType, self).unify(typingctx, other)

    def can_convert_to(self, typingctx, other):
        # same as types.Array
        if (isinstance(other, SeriesType) and other.dtype == self.dtype):
            # TODO: index?
            return self.data.can_convert_to(other.data)

    def is_precise(self):
        # same as types.Array
        return self.dtype.is_precise()

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timestamp
        return types.iterators.ArrayIterator(self.data)


def _get_series_array_type(dtype):
    """get underlying array type of series based on its dtype
    """
    # list(list(str))
    if dtype == types.List(string_type):
        return list_string_array_type
    # string array
    elif dtype == string_type:
        return string_array_type

    # categorical
    if isinstance(dtype, PDCategoricalDtype):
        dtype = get_categories_int_type(dtype)

    # use recarray data layout for series of tuples
    if isinstance(dtype, types.BaseTuple):
        if any(not isinstance(t, types.Number) for t in dtype.types):
            # TODO: support more types. what types can be in recarrays?
            raise ValueError("series tuple dtype {} includes non-numerics".format(dtype))
        np_dtype = np.dtype(
            ','.join(str(t) for t in dtype.types), align=True)
        dtype = numba.numpy_support.from_dtype(np_dtype)

    # TODO: other types?
    # regular numpy array
    return types.Array(dtype, 1, 'C')


string_series_type = SeriesType(string_type)
# TODO: create a separate DatetimeIndex type from Series
dt_index_series_type = SeriesType(types.NPDatetime('ns'))
timedelta_index_series_type = SeriesType(types.NPTimedelta('ns'))
date_series_type = SeriesType(datetime_date_type)

# register_model(SeriesType)(models.ArrayModel)
# need to define model since fix_df_array overload goes to native code
@register_model(SeriesType)
class SeriesModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.data),
            ('index', fe_type.index),
            ('name', string_type),
        ]
        super(SeriesModel, self).__init__(dmm, fe_type, members)

make_attribute_wrapper(SeriesType, 'data', '_data')
make_attribute_wrapper(SeriesType, 'index', '_index')
make_attribute_wrapper(SeriesType, 'name', '_name')


class BoxedSeriesType(types.Type):
    """Series type before unboxing. Using a different type to avoid data model
    issues and confusion.
    """
    def __init__(self, dtype):
        self.dtype = dtype
        name = "BoxedSeriesType({})".format(dtype)
        super(BoxedSeriesType, self).__init__(name)

# register_model(BoxedSeriesType)(models.OpaqueModel)
register_model(BoxedSeriesType)(SeriesModel)

class UnBoxedSeriesType(types.Type):
    """Series type before boxing. Using a different type to avoid data model
    issues and confusion.
    """
    def __init__(self, dtype, data, index):
        self.dtype = dtype
        self.data = data
        self.index = index
        name = "UnBoxedSeriesType({})".format(dtype)
        super(UnBoxedSeriesType, self).__init__(name)

register_model(UnBoxedSeriesType)(SeriesModel)


def series_to_array_type(typ, replace_boxed=False):
    # XXX: Boxed series variable types shouldn't be replaced in hiframes_typed
    # it results in cast error for call dummy_unbox_series
    if isinstance(typ, BoxedSeriesType) and not replace_boxed:
        return typ
    return _get_series_array_type(typ.dtype)


def is_series_type(typ):
    # XXX: UnBoxedSeriesType only used in unboxing
    assert not isinstance(typ, UnBoxedSeriesType)
    return isinstance(typ, (SeriesType, BoxedSeriesType))

def arr_to_series_type(arr):
    series_type = None
    if isinstance(arr, types.Array):
        series_type = SeriesType(arr.dtype, arr)
    elif arr == string_array_type:
        # StringArray is readonly
        series_type = string_series_type
    elif arr == list_string_array_type:
        series_type = SeriesType(types.List(string_type))
    return series_type

def arr_to_boxed_series_type(arr):
    series_type = None
    if isinstance(arr, types.Array):
        series_type = BoxedSeriesType(arr.dtype)
    elif arr == string_array_type:
        series_type = BoxedSeriesType(string_type)
    return series_type


def if_series_to_array_type(typ, replace_boxed=False):
    if isinstance(typ, (SeriesType, BoxedSeriesType)):
        return series_to_array_type(typ, replace_boxed)

    if isinstance(typ, (types.Tuple, types.UniTuple)):
        return types.Tuple(
            [if_series_to_array_type(t, replace_boxed) for t in typ.types])
    if isinstance(typ, types.List):
        return types.List(if_series_to_array_type(typ.dtype, replace_boxed))
    if isinstance(typ, types.Set):
        return types.Set(if_series_to_array_type(typ.dtype, replace_boxed))
    # TODO: other types that can have Series inside?
    return typ

def if_arr_to_series_type(typ):
    if isinstance(typ, types.Array) or typ in (string_array_type,
                                                    list_string_array_type):
        return arr_to_series_type(typ)
    if isinstance(typ, (types.Tuple, types.UniTuple)):
        return types.Tuple([if_arr_to_series_type(t) for t in typ.types])
    if isinstance(typ, types.List):
        return types.List(if_arr_to_series_type(typ.dtype))
    if isinstance(typ, types.Set):
        return types.Set(if_arr_to_series_type(typ.dtype))
    # TODO: other types that can have Arrays inside?
    return typ

def if_series_to_unbox(typ):
    if isinstance(typ, SeriesType):
        return UnBoxedSeriesType(typ.dtype, typ.data, typ.index)

    if isinstance(typ, (types.Tuple, types.UniTuple)):
        return types.Tuple(
            [if_series_to_unbox(t) for t in typ.types])
    if isinstance(typ, types.List):
        return types.List(if_series_to_unbox(typ.dtype))
    if isinstance(typ, types.Set):
        return types.Set(if_series_to_unbox(typ.dtype))
    # TODO: other types that can have Series inside?
    return typ

@lower_cast(string_array_type, UnBoxedSeriesType)
@lower_cast(list_string_array_type, UnBoxedSeriesType)
@lower_cast(types.Array, UnBoxedSeriesType)
def cast_string_series_unbox(context, builder, fromty, toty, val):
    return val

@lower_cast(string_series_type, string_array_type)
@lower_cast(string_array_type, string_series_type)
def cast_string_series(context, builder, fromty, toty, val):
    return val

@lower_cast(SeriesType, types.Array)
@lower_cast(types.Array, SeriesType)
def cast_series(context, builder, fromty, toty, val):
    return val

# --------------------------------------------------------------------------- #
# --- typing similar to arrays adopted from arraydecl.py, npydecl.py -------- #


@infer_getattr
class SeriesAttribute(AttributeTemplate):
    key = SeriesType

    def resolve_values(self, ary):
        return series_to_array_type(ary, True)

    def resolve_str(self, ary):
        assert ary.dtype in (string_type, types.List(string_type))
        # TODO: add dtype to series_str_methods_type
        return series_str_methods_type

    def resolve_dt(self, ary):
        assert ary.dtype == types.NPDatetime('ns')
        return series_dt_methods_type

    def resolve_date(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return date_series_type

    def resolve_iat(self, ary):
        return SeriesIatType(ary)

    def resolve_iloc(self, ary):
        # TODO: support iat/iloc differences
        return SeriesIatType(ary)

    def resolve_loc(self, ary):
        # TODO: support iat/iloc differences
        return SeriesIatType(ary)

    def resolve_year(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return types.Array(types.int64, 1, 'C')

    def resolve_month(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return types.Array(types.int64, 1, 'C')

    def resolve_day(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return types.Array(types.int64, 1, 'C')

    def resolve_hour(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return types.Array(types.int64, 1, 'C')

    def resolve_minute(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return types.Array(types.int64, 1, 'C')

    def resolve_second(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return types.Array(types.int64, 1, 'C')

    def resolve_microsecond(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return types.Array(types.int64, 1, 'C')

    def resolve_nanosecond(self, ary):
        if isinstance(ary.dtype, types.scalars.NPDatetime):
            return types.Array(types.int64, 1, 'C')

    def resolve_days(self, ary):
        if isinstance(ary.dtype, types.scalars.NPTimedelta):
            return types.Array(types.int64, 1, 'C')

    def resolve_seconds(self, ary):
        if isinstance(ary.dtype, types.scalars.NPTimedelta):
            return types.Array(types.int64, 1, 'C')

    def resolve_microseconds(self, ary):
        if isinstance(ary.dtype, types.scalars.NPTimedelta):
            return types.Array(types.int64, 1, 'C')

    def resolve_nanoseconds(self, ary):
        if isinstance(ary.dtype, types.scalars.NPTimedelta):
            return types.Array(types.int64, 1, 'C')

    @bound_function("array.astype")
    def resolve_astype(self, ary, args, kws):
        # TODO: handle other types like datetime etc.
        dtype, = args
        if isinstance(dtype, types.Function) and dtype.typing_key == str:
            ret_type = string_series_type
            sig = signature(ret_type, *args)
        else:
            resolver = ArrayAttribute.resolve_astype.__wrapped__
            sig = resolver(self, ary.data, args, kws)
            sig.return_type = if_arr_to_series_type(sig.return_type)
        return sig

    @bound_function("array.copy")
    def resolve_copy(self, ary, args, kws):
        # TODO: copy other types like list(str)
        dtype = ary.dtype
        if dtype == string_type:
            ret_type = string_series_type
            sig = signature(ret_type, *args)
        else:
            resolver = ArrayAttribute.resolve_copy.__wrapped__
            sig = resolver(self, ary.data, args, kws)
            sig.return_type = if_arr_to_series_type(sig.return_type)
        return sig

    @bound_function("series.rolling")
    def resolve_rolling(self, ary, args, kws):
        return signature(SeriesRollingType(ary.dtype), *args)

    @bound_function("array.argsort")
    def resolve_argsort(self, ary, args, kws):
        resolver = ArrayAttribute.resolve_argsort.__wrapped__
        sig = resolver(self, ary.data, args, kws)
        sig.return_type = if_arr_to_series_type(sig.return_type)
        return sig

    @bound_function("series.sort_values")
    def resolve_sort_values(self, ary, args, kws):
        return signature(ary, *args)

    @bound_function("array.take")
    def resolve_take(self, ary, args, kws):
        resolver = ArrayAttribute.resolve_take.__wrapped__
        sig = resolver(self, ary.data, args, kws)
        sig.return_type = if_arr_to_series_type(sig.return_type)
        return sig

    @bound_function("series.quantile")
    def resolve_quantile(self, ary, args, kws):
        # TODO: fix quantile output type if not float64
        return signature(types.float64, *args)

    @bound_function("series.count")
    def resolve_count(self, ary, args, kws):
        return signature(types.intp, *args)

    @bound_function("series.nunique")
    def resolve_nunique(self, ary, args, kws):
        return signature(types.intp, *args)

    @bound_function("series.unique")
    def resolve_unique(self, ary, args, kws):
        # unique returns ndarray for some reason
        arr_typ = series_to_array_type(ary)
        return signature(arr_typ, *args)

    @bound_function("series.describe")
    def resolve_describe(self, ary, args, kws):
        # TODO: return namedtuple or labeled Series
        return signature(string_type, *args)

    @bound_function("series.fillna")
    def resolve_fillna(self, ary, args, kws):
        out = ary
        # output is None for inplace case
        if 'inplace' in kws and kws['inplace'] == types.literal(True):
            out = types.none
        return signature(out, *args)

    @bound_function("series.dropna")
    def resolve_dropna(self, ary, args, kws):
        out = ary
        # output is None for inplace case
        if 'inplace' in kws and kws['inplace'] == types.literal(True):
            out = types.none
        return signature(out, *args)

    @bound_function("series.shift")
    def resolve_shift(self, ary, args, kws):
        # TODO: support default period argument
        out = ary
        # integers are converted to float64 to store NaN
        if isinstance(ary.dtype, types.Integer):
            out.dtype = types.float64
        return signature(out, *args)

    @bound_function("series.pct_change")
    def resolve_pct_change(self, ary, args, kws):
        # TODO: support default period argument
        out = ary
        # integers are converted to float64 to store NaN
        if isinstance(ary.dtype, types.Integer):
            out.dtype = types.float64
        return signature(out, *args)

    def _resolve_map_func(self, ary, args, kws):
        dtype = ary.dtype
        # getitem returns Timestamp for dt_index and series(dt64)
        if dtype == types.NPDatetime('ns'):
            dtype = pandas_timestamp_type
        code = args[0].literal_value.code
        f_ir = numba.ir_utils.get_ir_of_code({'np': np}, code)
        f_typemap, f_return_type, f_calltypes = numba.compiler.type_inference_stage(
                self.context, f_ir, (dtype,), None)

        return signature(SeriesType(f_return_type), *args)

    @bound_function("series.map")
    def resolve_map(self, ary, args, kws):
        return self._resolve_map_func(ary, args, kws)

    @bound_function("series.apply")
    def resolve_apply(self, ary, args, kws):
        # TODO: handle apply differences: extra args, np ufuncs etc.
        return self._resolve_map_func(ary, args, kws)

    def _resolve_combine_func(self, ary, args, kws):
        dtype1 = ary.dtype
        # getitem returns Timestamp for dt_index and series(dt64)
        if dtype1 == types.NPDatetime('ns'):
            dtype1 = pandas_timestamp_type
        dtype2 = args[0].dtype
        if dtype2 == types.NPDatetime('ns'):
            dtype2 = pandas_timestamp_type
        code = args[1].literal_value.code
        f_ir = numba.ir_utils.get_ir_of_code({'np': np}, code)
        f_typemap, f_return_type, f_calltypes = numba.compiler.type_inference_stage(
                self.context, f_ir, (dtype1,dtype2,), None)
        return signature(SeriesType(f_return_type), *args)

    @bound_function("series.combine")
    def resolve_combine(self, ary, args, kws):
        return self._resolve_combine_func(ary, args, kws)

    @bound_function("series.abs")
    def resolve_abs(self, ary, args, kws):
        # call np.abs(A) to get return type
        arr_typ = series_to_array_type(ary)
        all_args = tuple([arr_typ]+list(args))
        ret_typ = self.context.resolve_function_type(np.abs, all_args, kws).return_type
        ret_typ = if_arr_to_series_type(ret_typ)
        return signature(ret_typ, *args)

    def _resolve_cov_func(self, ary, args, kws):
        # array is valid since hiframes_typed calls this after type replacement
        assert len(args) == 1 and isinstance(args[0], (SeriesType, types.Array))
        assert isinstance(ary.dtype, types.Number)
        assert isinstance(args[0].dtype, types.Number)
        is_complex_op = isinstance(ary.dtype, types.Complex) or isinstance(args[0].dtype, types.Complex)
        ret_typ = types.complex128 if is_complex_op else types.float64
        return signature(ret_typ, *args)

    @bound_function("series.cov")
    def resolve_cov(self, ary, args, kws):
        return self._resolve_cov_func(ary, args, kws)

    @bound_function("series.corr")
    def resolve_corr(self, ary, args, kws):
        return self._resolve_cov_func(ary, args, kws)

    @bound_function("series.append")
    def resolve_append(self, ary, args, kws):
        # TODO: ignore_index
        assert not kws
        arr_typ = if_series_to_array_type(ary)
        other, = args
        if isinstance(other, (SeriesType, types.Array)):
            all_arrs = types.Tuple((arr_typ, if_series_to_array_type(other)))
        elif isinstance(other, types.BaseTuple):
            all_arrs = types.Tuple((arr_typ, *[if_series_to_array_type(a) for a in other.types]))
        elif isinstance(other, (types.List, types.Set)):
            # add only one value from the list for typing since it shouldn't
            # matter for np.concatenate typing
            all_arrs = types.Tuple((arr_typ, if_series_to_array_type(other.dtype)))
        else:
            raise ValueError("Invalid input for Series.append (Series, or tuple/list of Series expected)")

        # TODO: list
        # call np.concatenate to handle type promotion e.g. int, float -> float
        ret_typ = self.context.resolve_function_type(np.concatenate, (all_arrs,), kws).return_type
        ret_typ = if_arr_to_series_type(ret_typ)
        return signature(ret_typ, *args)

    @bound_function("series.isna")
    def resolve_isna(self, ary, args, kws):
        assert not kws
        assert not args
        return signature(SeriesType(types.boolean))

    # alias of isna
    @bound_function("series.isnull")
    def resolve_isnull(self, ary, args, kws):
        assert not kws
        assert not args
        return signature(SeriesType(types.boolean))

    @bound_function("series.notna")
    def resolve_notna(self, ary, args, kws):
        assert not kws
        assert not args
        return signature(SeriesType(types.boolean))

    @bound_function("series.nlargest")
    def resolve_nlargest(self, ary, args, kws):
        assert not kws
        return signature(ary, *args)

    @bound_function("series.nsmallest")
    def resolve_nsmallest(self, ary, args, kws):
        assert not kws
        return signature(ary, *args)

    @bound_function("series.head")
    def resolve_head(self, ary, args, kws):
        assert not kws
        return signature(ary, *args)

    @bound_function("series.median")
    def resolve_median(self, ary, args, kws):
        assert not kws
        dtype = ary.dtype
        # median converts integer output to float
        dtype = types.float64 if isinstance(dtype, types.Integer) else dtype
        return signature(dtype, *args)

    @bound_function("series.idxmin")
    def resolve_idxmin(self, ary, args, kws):
        assert not kws
        return signature(types.intp, *args)

    @bound_function("series.idxmax")
    def resolve_idxmax(self, ary, args, kws):
        assert not kws
        return signature(types.intp, *args)

    @bound_function("series.max")
    def resolve_max(self, ary, args, kws):
        assert not kws
        dtype = ary.dtype
        dtype = pandas_timestamp_type if isinstance(dtype, numba.types.scalars.NPDatetime) else dtype
        return signature(dtype, *args)

    @bound_function("series.min")
    def resolve_min(self, ary, args, kws):
        assert not kws
        dtype = ary.dtype
        dtype = pandas_timestamp_type if isinstance(dtype, numba.types.scalars.NPDatetime) else dtype
        return signature(dtype, *args)

# TODO: use ops logic from pandas/core/ops.py
# # called from numba/numpy_support.py:resolve_output_type
# # similar to SmartArray (targets/smartarray.py)
# @type_callable('__array_wrap__')
# def type_series_array_wrap(context):
#     def typer(input_type, result):
#         if isinstance(input_type, SeriesType):
#             return input_type.copy(dtype=result.dtype,
#                                    ndim=result.ndim,
#                                    layout=result.layout)

#     return typer

class SeriesStrMethodType(types.Type):
    def __init__(self):
        name = "SeriesStrMethodType"
        super(SeriesStrMethodType, self).__init__(name)

series_str_methods_type = SeriesStrMethodType()


@infer_getattr
class SeriesStrMethodAttribute(AttributeTemplate):
    key = SeriesStrMethodType

    @bound_function("strmethod.contains")
    def resolve_contains(self, ary, args, kws):
        return signature(SeriesType(types.bool_), *args)

    @bound_function("strmethod.len")
    def resolve_len(self, ary, args, kws):
        return signature(SeriesType(types.int64), *args)

    @bound_function("strmethod.replace")
    def resolve_replace(self, ary, args, kws):
        return signature(SeriesType(string_type), *args)

    @bound_function("strmethod.split")
    def resolve_split(self, ary, args, kws):
        return signature(SeriesType(types.List(string_type)), *args)

    @bound_function("strmethod.get")
    def resolve_get(self, ary, args, kws):
        # XXX only list(list(str)) supported
        return signature(SeriesType(string_type), *args)


class SeriesDtMethodType(types.Type):
    def __init__(self):
        name = "SeriesDtMethodType"
        super(SeriesDtMethodType, self).__init__(name)

series_dt_methods_type = SeriesDtMethodType()


@infer_getattr
class SeriesDtMethodAttribute(AttributeTemplate):
    key = SeriesDtMethodType

# all date fields return int64 same as Timestamp fields
def resolve_date_field(self, ary):
    return SeriesType(types.int64)

for field in hpat.hiframes.pd_timestamp_ext.date_fields:
    setattr(SeriesDtMethodAttribute, "resolve_" + field, resolve_date_field)


class SeriesRollingType(types.Type):
    def __init__(self, dtype):
        self.dtype = dtype
        name = "SeriesRollingType({})".format(dtype)
        super(SeriesRollingType, self).__init__(name)


@infer_getattr
class SeriesRollingAttribute(AttributeTemplate):
    key = SeriesRollingType

    @bound_function("rolling.apply")
    def resolve_apply(self, ary, args, kws):
        # result is always float64 (see Pandas window.pyx:roll_generic)
        return signature(SeriesType(types.float64), *args)

    @bound_function("rolling.cov")
    def resolve_cov(self, ary, args, kws):
        return signature(SeriesType(types.float64), *args)

    @bound_function("rolling.corr")
    def resolve_corr(self, ary, args, kws):
        return signature(SeriesType(types.float64), *args)

# similar to install_array_method in arraydecl.py
def install_rolling_method(name, generic):
    my_attr = {"key": "rolling." + name, "generic": generic}
    temp_class = type("Rolling_" + name, (AbstractTemplate,), my_attr)
    def rolling_attribute_attachment(self, ary):
        return types.BoundFunction(temp_class, ary)

    setattr(SeriesRollingAttribute, "resolve_" + name, rolling_attribute_attachment)

def rolling_generic(self, args, kws):
    # output is always float64
    return signature(SeriesType(types.float64), *args)

for fname in supported_rolling_funcs:
    install_rolling_method(fname, rolling_generic)

class SeriesIatType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesIatType({})".format(stype)
        super(SeriesIatType, self).__init__(name)

@infer_global(operator.getitem)
class GetItemSeriesIat(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        # iat[] is the same as regular getitem
        if isinstance(args[0], SeriesIatType):
            return GetItemSeries.generic(self, (args[0].stype, args[1]), kws)

@infer
@infer_global(operator.eq)
@infer_global(operator.ne)
@infer_global(operator.ge)
@infer_global(operator.gt)
@infer_global(operator.le)
@infer_global(operator.lt)
class SeriesCompEqual(AbstractTemplate):
    key = '=='
    def generic(self, args, kws):
        from hpat.str_arr_ext import is_str_arr_typ
        assert not kws
        [va, vb] = args
        # if one of the inputs is string array
        if va == string_series_type or vb == string_series_type:
            # inputs should be either string array or string
            assert is_str_arr_typ(va) or va == string_type
            assert is_str_arr_typ(vb) or vb == string_type
            return signature(SeriesType(types.boolean), va, vb)

        if ((va == dt_index_series_type and vb == string_type)
                or (vb == dt_index_series_type and va == string_type)):
            return signature(SeriesType(types.boolean), va, vb)

@infer
class CmpOpNEqSeries(SeriesCompEqual):
    key = '!='

@infer
class CmpOpGESeries(SeriesCompEqual):
    key = '>='

@infer
class CmpOpGTSeries(SeriesCompEqual):
    key = '>'

@infer
class CmpOpLESeries(SeriesCompEqual):
    key = '<='

@infer
class CmpOpLTSeries(SeriesCompEqual):
    key = '<'

# @infer_global(operator.getitem)
# class GetItemBuffer(AbstractTemplate):
#     key = operator.getitem

#     def generic(self, args, kws):
#         assert not kws
#         [ary, idx] = args
#         import pdb; pdb.set_trace()
#         if not isinstance(ary, SeriesType):
#             return
#         out = get_array_index_type(ary, idx)
#         # check result to be dt64 since it might be sliced array
#         # replace result with Timestamp
#         if out is not None and out.result == types.NPDatetime('ns'):
#             return signature(pandas_timestamp_type, ary, out.index)

def install_array_method(name, generic):
    # taken from arraydecl.py, Series instead of Array
    my_attr = {"key": "array." + name, "generic": generic}
    temp_class = type("Series_" + name, (AbstractTemplate,), my_attr)
    def array_attribute_attachment(self, ary):
        return types.BoundFunction(temp_class, ary)

    setattr(SeriesAttribute, "resolve_" + name, array_attribute_attachment)

def generic_expand_cumulative_series(self, args, kws):
    # taken from arraydecl.py, replaced Array with Series
    assert not args
    assert not kws
    assert isinstance(self.this, SeriesType)
    return_type = SeriesType(_expand_integer(self.this.dtype))
    return signature(return_type, recvr=self.this)

# replacing cumsum/cumprod since arraydecl.py definition uses types.Array
for fname in ["cumsum", "cumprod"]:
    install_array_method(fname, generic_expand_cumulative_series)

# TODO: add itemsize, strides, etc. when removed from Pandas
_not_series_array_attrs = ['flat', 'ctypes', 'itemset', 'reshape', 'sort', 'flatten']

# use ArrayAttribute for attributes not defined in SeriesAttribute
for attr, func in numba.typing.arraydecl.ArrayAttribute.__dict__.items():
    if (attr.startswith('resolve_')
            and attr not in SeriesAttribute.__dict__
            and attr not in _not_series_array_attrs):
        setattr(SeriesAttribute, attr, func)

@infer_global(operator.getitem)
class GetItemSeries(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        assert not kws
        [in_arr, in_idx] = args
        is_arr_series = False
        is_idx_series = False
        is_arr_dt_index = False

        if not isinstance(in_arr, SeriesType) and not isinstance(in_idx, SeriesType):
            return None

        if isinstance(in_arr, SeriesType):
            in_arr = series_to_array_type(in_arr)
            is_arr_series = True
            if in_arr.dtype == types.NPDatetime('ns'):
                is_arr_dt_index = True

        if isinstance(in_idx, SeriesType):
            in_idx = series_to_array_type(in_idx)
            is_idx_series = True

        # TODO: dt_index
        if in_arr == string_array_type:
            # XXX fails due in overload
            # compile_internal version results in symbol not found!
            # sig = self.context.resolve_function_type(
            #     operator.getitem, (in_arr, in_idx), kws)
            # HACK to get avoid issues for now
            if isinstance(in_idx, (types.Integer, types.IntegerLiteral)):
                sig = string_type(in_arr, in_idx)
            else:
                sig = GetItemStringArray.generic(self, (in_arr, in_idx), kws)
        elif in_arr == list_string_array_type:
            # mimic array indexing for list
            if (isinstance(in_idx, types.Array) and in_idx.ndim == 1
                    and isinstance(
                        in_idx.dtype, (types.Integer, types.Boolean))):
                sig = signature(in_arr, in_arr, in_idx)
            else:
                sig = numba.typing.collections.GetItemSequence.generic(
                    self, (in_arr, in_idx), kws)
        else:
            out = get_array_index_type(in_arr, in_idx)
            sig = signature(out.result, in_arr, out.index)

        if sig is not None:
            arg1 = sig.args[0]
            arg2 = sig.args[1]
            if is_arr_series:
                sig.return_type = if_arr_to_series_type(sig.return_type)
                arg1 = if_arr_to_series_type(arg1)
            if is_idx_series:
                arg2 = if_arr_to_series_type(arg2)
            sig.args = (arg1, arg2)
            # dt_index and Series(dt64) should return Timestamp
            if is_arr_dt_index and sig.return_type == types.NPDatetime('ns'):
                sig.return_type = pandas_timestamp_type
        return sig

@infer
class SetItemSeries(SetItemBuffer):
    key = "setitem"

    def generic(self, args, kws):
        assert not kws
        series, idx, val = args
        if not isinstance(series, SeriesType):
            return None
        ary = series_to_array_type(series)
        # TODO: strings, dt_index
        res = super(SetItemSeries, self).generic((ary, idx, val), kws)
        if res is not None:
            new_series = if_arr_to_series_type(res.args[0])
            res.args = (new_series, res.args[1], res.args[2])
            return res

@infer
class SetItemSeriesIat(SetItemSeries):
    key = "setitem"

    def generic(self, args, kws):
        # iat[] is the same as regular setitem
        if isinstance(args[0], SeriesIatType):
            return SetItemSeries.generic(self, (args[0].stype, args[1], args[2]), kws)


def series_op_generic(cls, self, args, kws):
    # return if no Series
    if not any(isinstance(arg, SeriesType) for arg in args):
        return None
    # convert args to array
    new_args = tuple(if_series_to_array_type(arg) for arg in args)
    sig = super(cls, self).generic(new_args, kws)
    # convert back to Series
    if sig is not None:
        sig.return_type = if_arr_to_series_type(sig.return_type)
        sig.args = tuple(if_arr_to_series_type(a) for a in sig.args)
    return sig

class SeriesOpUfuncs(NumpyRulesArrayOperator):
    def generic(self, args, kws):
        return series_op_generic(SeriesOpUfuncs, self, args, kws)

def install_series_method(op, name, generic):
    # taken from arraydecl.py, Series instead of Array
    my_attr = {"key": op, "generic": generic}
    temp_class = type("Series_" + name, (SeriesOpUfuncs,), my_attr)
    def array_attribute_attachment(self, ary):
        return types.BoundFunction(temp_class, ary)

    setattr(SeriesAttribute, "resolve_" + name, array_attribute_attachment)

explicit_binop_funcs = {
    operator.add: 'add',
    operator.sub: 'sub',
    operator.mul: 'mul',
    operator.truediv: 'div',
    operator.truediv: 'truediv',
    operator.floordiv: 'floordiv',
    operator.mod: 'mod',
    operator.pow: 'pow',
    operator.lt: 'lt',
    operator.gt: 'gt',
    operator.le: 'le',
    operator.ge: 'ge',
    operator.ne: 'ne',
    operator.eq: 'eq',
    }

def ex_binop_generic(self, args, kws):
    return SeriesOpUfuncs.generic(self, (self.this,) + args, kws)

for op, fname in explicit_binop_funcs.items():
    install_series_method(op, fname, ex_binop_generic)

class SeriesInplaceOpUfuncs(NumpyRulesInplaceArrayOperator):
    def generic(self, args, kws):
        return series_op_generic(SeriesInplaceOpUfuncs, self, args, kws)

class SeriesUnaryOpUfuncs(NumpyRulesUnaryArrayOperator):
    def generic(self, args, kws):
        return series_op_generic(SeriesUnaryOpUfuncs, self, args, kws)

# TODO: change class name to Series in install_operations
SeriesOpUfuncs.install_operations()
SeriesInplaceOpUfuncs.install_operations()
SeriesUnaryOpUfuncs.install_operations()

class Series_Numpy_rules_ufunc(Numpy_rules_ufunc):
    def generic(self, args, kws):
        return series_op_generic(Series_Numpy_rules_ufunc, self, args, kws)

# copied from npydecl.py since deleted
_aliases = set(["bitwise_not", "mod", "abs"])
if np.divide == np.true_divide:
    _aliases.add("divide")

for func in numba.typing.npydecl.supported_ufuncs:
    name = func.__name__
    #_numpy_ufunc(func)
    class typing_class(Series_Numpy_rules_ufunc):
        key = func

    typing_class.__name__ = "resolve_series_{0}".format(name)

    if not name in _aliases:
        infer_global(func, types.Function(typing_class))

@infer_global(len)
class LenSeriesType(AbstractTemplate):
    def generic(self, args, kws):
        if not kws and len(args) == 1 and isinstance(args[0], SeriesType):
            return signature(types.intp, *args)

# @infer_global(np.empty_like)
# @infer_global(np.zeros_like)
# @infer_global(np.ones_like)
# class SeriesLikeTyper(NdConstructorLike):
#     def generic(self):
#         typer = super(SeriesLikeTyper, self).generic()
#         def wrapper(*args, **kws):
#             new_args = tuple(if_series_to_array_type(arg) for arg in args)
#             new_kws = {n:if_series_to_array_type(t) for n,t in kws.items()}
#             return typer(*new_args, **new_kws)
#         return wrapper

#@infer_global(np.full_like)

def type_sub(context):
    def typer(val1, val2):
        if(val1 == dt_index_series_type and val2 == pandas_timestamp_type):
            return timedelta_index_series_type
    return typer

type_callable('-')(type_sub)
type_callable(operator.sub)(type_sub)

@overload(pd.Series)
def pd_series_overload(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):

    if index is not None:
        return (lambda data=None, index=None, dtype=None, name=None, copy=False,
        fastpath=False: hpat.hiframes.api.init_series(
            hpat.hiframes.api.fix_df_array(data),
            hpat.hiframes.api.fix_df_array(index),
            name
        ))

    return (lambda data=None, index=None, dtype=None, name=None, copy=False,
        fastpath=False: hpat.hiframes.api.init_series(
            hpat.hiframes.api.fix_df_array(data), index, name))

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
import numpy as np
import pandas as pd
import llvmlite.llvmpy.core as lc

import numba
from numba import types, cgutils
from numba.extending import (
    models,
    register_model,
    lower_cast,
    lower_builtin,
    infer_getattr,
    type_callable,
    infer,
    overload)
from numba.typing.arraydecl import (get_array_index_type, _expand_integer, ArrayAttribute, SetItemBuffer)
from numba.typing.npydecl import (
    Numpy_rules_ufunc,
    NumpyRulesArrayOperator,
    NumpyRulesInplaceArrayOperator,
    NumpyRulesUnaryArrayOperator,
    NdConstructorLike)
from numba.typing.templates import (infer_global, AbstractTemplate, signature, AttributeTemplate, bound_function)
from numba.targets.imputils import (impl_ret_new_ref, iternext_impl, RefType)
from numba.targets.arrayobj import (make_array, _getitem_array1d)

import sdc
from sdc.datatypes.hpat_pandas_stringmethods_types import StringMethodsType
from sdc.hiframes.pd_categorical_ext import (PDCategoricalDtype, CategoricalArray)
from sdc.hiframes.pd_timestamp_ext import (pandas_timestamp_type, datetime_date_type)
from sdc.hiframes.rolling import supported_rolling_funcs
from sdc.hiframes.split_impl import (SplitViewStringMethodsType,
                                     string_array_split_view_type,
                                     GetItemStringArraySplitView)
from sdc.str_arr_ext import (
    string_array_type,
    iternext_str_array,
    offset_typ,
    char_typ,
    str_arr_payload_type,
    StringArrayType,
    GetItemStringArray)
from sdc.str_ext import string_type, list_string_array_type
from sdc.hiframes.pd_series_type import (SeriesType, _get_series_array_type)


def is_str_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == string_type


def is_dt64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPDatetime('ns')


def is_timedelta64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPTimedelta('ns')


def is_datetime_date_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == datetime_date_type


def series_to_array_type(typ, replace_boxed=False):
    return typ.data
    # return _get_series_array_type(typ.dtype)


def is_series_type(typ):
    return isinstance(typ, SeriesType)


def arr_to_series_type(arr):
    series_type = None
    if isinstance(arr, types.Array):
        series_type = SeriesType(arr.dtype, arr)
    elif arr == string_array_type:
        # StringArray is readonly
        series_type = SeriesType(string_type)
    elif arr == list_string_array_type:
        series_type = SeriesType(types.List(string_type))
    elif arr == string_array_split_view_type:
        series_type = SeriesType(types.List(string_type),
                                 string_array_split_view_type)
    return series_type


def if_series_to_array_type(typ, replace_boxed=False):
    if isinstance(typ, SeriesType):
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
                                               list_string_array_type, string_array_split_view_type):
        return arr_to_series_type(typ)
    if isinstance(typ, (types.Tuple, types.UniTuple)):
        return types.Tuple([if_arr_to_series_type(t) for t in typ.types])
    if isinstance(typ, types.List):
        return types.List(if_arr_to_series_type(typ.dtype))
    if isinstance(typ, types.Set):
        return types.Set(if_arr_to_series_type(typ.dtype))
    # TODO: other types that can have Arrays inside?
    return typ


# TODO remove this cast?
@lower_cast(SeriesType, string_array_type)
@lower_cast(string_array_type, SeriesType)
def cast_string_series(context, builder, fromty, toty, val):
    return val

# cast Series(int8) to Series(cat) for init_series() in test_csv_cat1
# TODO: separate array type for Categorical data
@lower_cast(SeriesType, types.Array)
@lower_cast(types.Array, SeriesType)
@lower_cast(SeriesType, SeriesType)
def cast_series(context, builder, fromty, toty, val):
    return val

# --------------------------------------------------------------------------- #
# --- typing similar to arrays adopted from arraydecl.py, npydecl.py -------- #


@infer_getattr
class SeriesAttribute(AttributeTemplate):
    key = SeriesType

    # PR135. This needs to be commented out
    def resolve_values(self, ary):
        return series_to_array_type(ary, True)

    # PR135. This needs to be commented out
    def resolve_T(self, ary):
        return series_to_array_type(ary, True)

# PR135. This needs to be commented out
    # def resolve_shape(self, ary):
    #     return types.Tuple((types.int64,))

# PR171. This needs to be commented out
#     def resolve_index(self, ary):
#         return ary.index

    # def resolve_str(self, ary):
    #     assert ary.dtype in (string_type, types.List(string_type))
    #     return StringMethodsType(ary)

    def resolve_dt(self, ary):
        assert ary.dtype == types.NPDatetime('ns')
        return series_dt_methods_type

# PR135. This needs to be commented out
    def resolve_iat(self, ary):
        return SeriesIatType(ary)

# PR135. This needs to be commented out
    def resolve_iloc(self, ary):
        # TODO: support iat/iloc differences
        return SeriesIatType(ary)

# PR135. This needs to be commented out
    def resolve_loc(self, ary):
        # TODO: support iat/iloc differences
        return SeriesIatType(ary)

    # @bound_function("array.astype")
    # def resolve_astype(self, ary, args, kws):
    #     # TODO: handle other types like datetime etc.
    #     dtype, = args
    #     if ((isinstance(dtype, types.Function) and dtype.typing_key == str)
    #             or (isinstance(dtype, types.StringLiteral) and dtype.literal_value == 'str')):
    #         ret_type = SeriesType(string_type, index=ary.index)
    #         sig = signature(ret_type, *args)
    #     else:
    #         resolver = ArrayAttribute.resolve_astype.__wrapped__
    #         sig = resolver(self, ary.data, args, kws)
    #         sig.return_type = if_arr_to_series_type(sig.return_type)
    #     return sig

    # @bound_function("array.copy")
    # def resolve_copy(self, ary, args, kws):
    #     # TODO: copy other types like list(str)
    #     dtype = ary.dtype
    #     if dtype == string_type:
    #         ret_type = SeriesType(string_type)
    #         sig = signature(ret_type, *args)
    #     else:
    #         resolver = ArrayAttribute.resolve_copy.__wrapped__
    #         sig = resolver(self, ary.data, args, kws)
    #         sig.return_type = if_arr_to_series_type(sig.return_type)
    #     return sig

    @bound_function("series.rolling")
    def resolve_rolling(self, ary, args, kws):
        return signature(SeriesRollingType(ary.dtype), *args)

    # @bound_function("array.argsort")
    # def resolve_argsort(self, ary, args, kws):
    #     resolver = ArrayAttribute.resolve_argsort.__wrapped__
    #     sig = resolver(self, ary.data, args, kws)
    #     sig.return_type = if_arr_to_series_type(sig.return_type)
    #     return sig

    # @bound_function("series.sort_values")
    # def resolve_sort_values(self, ary, args, kws):
    #     # output will have permuted input index
    #     out_index = ary.index
    #     if out_index == types.none:
    #         out_index = types.Array(types.intp, 1, 'C')
    #     out = SeriesType(ary.dtype, ary.data, out_index)
    #     return signature(out, *args)

#     @bound_function("array.take")
#     def resolve_take(self, ary, args, kws):
#         resolver = ArrayAttribute.resolve_take.__wrapped__
#         sig = resolver(self, ary.data, args, kws)
#         sig.return_type = if_arr_to_series_type(sig.return_type)
#         return sig

    # @bound_function("series.quantile")
    # def resolve_quantile(self, ary, args, kws):
    #     # TODO: fix quantile output type if not float64
    #     return signature(types.float64, *args)

    # @bound_function("series.count")
    # def resolve_count(self, ary, args, kws):
    #     return signature(types.intp, *args)

    # @bound_function("series.nunique")
    # def resolve_nunique(self, ary, args, kws):
    #     return signature(types.intp, *args)

    # @bound_function("series.unique")
    # def resolve_unique(self, ary, args, kws):
    #     # unique returns ndarray for some reason
    #     arr_typ = series_to_array_type(ary)
    #     return signature(arr_typ, *args)

    @bound_function("series.describe")
    def resolve_describe(self, ary, args, kws):
        # TODO: return namedtuple or labeled Series
        return signature(string_type, *args)

    # PR135. This needs to be commented out (for the new impl to be called)
    @bound_function("series.fillna")
    def resolve_fillna(self, ary, args, kws):
        out = ary
        # output is None for inplace case
        if 'inplace' in kws and kws['inplace'] == types.literal(True):
            out = types.none
        return signature(out, *args)

    # PR135. This needs to be commented out (for new-style impl to be called)
    @bound_function("series.dropna")
    def resolve_dropna(self, ary, args, kws):
        out = ary
        # output is None for inplace case
        if 'inplace' in kws and kws['inplace'] == types.literal(True):
            out = types.none
        return signature(out, *args)

    # @bound_function("series.shift")
    # def resolve_shift(self, ary, args, kws):
    #     # TODO: support default period argument
    #     out = ary
    #     # integers are converted to float64 to store NaN
    #     if isinstance(ary.dtype, types.Integer):
    #         out = out.copy(dtype=types.float64)
    #     return signature(out, *args)

    @bound_function("series.pct_change")
    def resolve_pct_change(self, ary, args, kws):
        # TODO: support default period argument
        out = ary
        # integers are converted to float64 to store NaN
        if isinstance(ary.dtype, types.Integer):
            out = out.copy(dtype=types.float64)
        return signature(out, *args)

    def _resolve_map_func(self, ary, args, kws):
        dtype = ary.dtype
        # getitem returns Timestamp for dt_index and series(dt64)
        if dtype == types.NPDatetime('ns'):
            dtype = pandas_timestamp_type
        t = args[0].get_call_type(self.context, (dtype,), {})
        return signature(SeriesType(t.return_type), *args)

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
        t = args[1].get_call_type(self.context, (dtype1, dtype2,), {})
        return signature(SeriesType(t.return_type), *args)

    @bound_function("series.combine")
    def resolve_combine(self, ary, args, kws):
        return self._resolve_combine_func(ary, args, kws)

    # @bound_function("series.abs")
    # def resolve_abs(self, ary, args, kws):
    #     # call np.abs(A) to get return type
    #     arr_typ = series_to_array_type(ary)
    #     all_args = tuple([arr_typ] + list(args))
    #     ret_typ = self.context.resolve_function_type(np.abs, all_args, kws).return_type
    #     ret_typ = if_arr_to_series_type(ret_typ)
    #     return signature(ret_typ, *args)

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

# PR135. This needs to be commented out
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

    # @bound_function("series.isna")
    # def resolve_isna(self, ary, args, kws):
    #     assert not kws
    #     assert not args
    #     return signature(SeriesType(types.boolean))

    # alias of isna
    # @bound_function("series.isnull")
    # def resolve_isnull(self, ary, args, kws):
    #     assert not kws
    #     assert not args
    #     return signature(SeriesType(types.boolean))

    # @bound_function("series.notna")
    # def resolve_notna(self, ary, args, kws):
    #     assert not kws
    #     assert not args
    #     return signature(SeriesType(types.boolean))

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

#     @bound_function("series.median")
#     def resolve_median(self, ary, args, kws):
#         assert not kws
#         dtype = ary.dtype
#         # median converts integer output to float
#         dtype = types.float64 if isinstance(dtype, types.Integer) else dtype
#         return signature(dtype, *args)

    # @bound_function("series.idxmin")
    # def resolve_idxmin(self, ary, args, kws):
    #     assert not kws
    #     return signature(types.intp, *args)

    # @bound_function("series.idxmax")
    # def resolve_idxmax(self, ary, args, kws):
    #     assert not kws
    #     return signature(types.intp, *args)

    # @bound_function("series.max")
    # def resolve_max(self, ary, args, kws):
    #     assert not kws
    #     dtype = ary.dtype
    #     dtype = (pandas_timestamp_type
    #              if isinstance(dtype, types.NPDatetime) else dtype)
    #     return signature(dtype, *args)

    # @bound_function("series.min")
    # def resolve_min(self, ary, args, kws):
    #     assert not kws
    #     dtype = ary.dtype
    #     dtype = (pandas_timestamp_type
    #              if isinstance(dtype, types.NPDatetime) else dtype)
    #     return signature(dtype, *args)

    @bound_function("series.value_counts")
    def resolve_value_counts(self, ary, args, kws):
        # output is int series with original data as index
        out = SeriesType(
            types.int64, types.Array(types.int64, 1, 'C'), ary.data)
        return signature(out, *args)

    # @bound_function("series.rename")
    # def resolve_rename(self, ary, args, kws):
    #     # TODO: support index rename, kws
    #     assert len(args) == 1 and isinstance(
    #         args[0], (types.UnicodeType, types.StringLiteral))
    #     out = SeriesType(ary.dtype, ary.data, ary.index, True)
    #     return signature(out, *args)


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

str2str_methods = ['capitalize', 'swapcase', 'title']
"""
    Functions which are still overloaded by HPAT compiler pipeline
"""

str2str_methods_excluded = [
    'upper', 'center', 'endswith', 'find', 'isupper', 'len', 'ljust',
    'lower', 'lstrip', 'rjust', 'rstrip', 'startswith', 'strip', 'zfill',
    'isspace', 'islower', 'isalpha', 'isalnum', 'istitle'
]
"""
    Functions which are used from Numba directly by calling from StringMethodsType

    Test: HPAT_CONFIG_PIPELINE_HPAT=0 python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_str2str
"""

# class SeriesStrMethodType(types.Type):
#     def __init__(self):
#         name = "SeriesStrMethodType"
#         super(SeriesStrMethodType, self).__init__(name)


# series_str_methods_type = SeriesStrMethodType


@infer_getattr
class SeriesStrMethodAttribute(AttributeTemplate):
    key = StringMethodsType

    @bound_function("strmethod.contains")
    def resolve_contains(self, ary, args, kws):
        return signature(SeriesType(types.bool_), *args)

    # @bound_function("strmethod.len")
    # def resolve_len(self, ary, args, kws):
    #     return signature(SeriesType(types.int64), *args)

    @bound_function("strmethod.replace")
    def resolve_replace(self, ary, args, kws):
        return signature(SeriesType(string_type), *args)

    @bound_function("strmethod.split")
    def resolve_split(self, ary, args, kws):
        out = SeriesType(types.List(string_type))
        if (len(args) == 1 and isinstance(args[0], types.StringLiteral)
                and len(args[0].literal_value) == 1):
            out = SeriesType(types.List(string_type), string_array_split_view_type)
        return signature(out, *args)

    @bound_function("strmethod.get")
    def resolve_get(self, ary, args, kws):
        # XXX only list(list(str)) supported
        return signature(SeriesType(string_type), *args)

    def generic_resolve(self, s_str, func_name):
        if sdc.config.config_pipeline_hpat_default and func_name in str2str_methods:
            template_key = 'strmethod.' + func_name
            out_typ = SeriesType(string_type)

            class MethodTemplate(AbstractTemplate):
                key = template_key

                def generic(self, args, kws):
                    return signature(out_typ, *args)

            return types.BoundFunction(MethodTemplate, s_str)

        if func_name in str2str_methods_excluded:
            return

        raise NotImplementedError('Series.str.{} is not supported yet'.format(func_name))


@infer_getattr
class SplitViewSeriesStrMethodAttribute(AttributeTemplate):
    key = SplitViewStringMethodsType

    @bound_function('strmethod.get')
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

    def resolve_date(self, ary):
        return SeriesType(datetime_date_type)  # TODO: name, index


# all date fields return int64 same as Timestamp fields
def resolve_date_field(self, ary):
    return SeriesType(types.int64)


for field in sdc.hiframes.pd_timestamp_ext.date_fields:
    setattr(SeriesDtMethodAttribute, "resolve_" + field, resolve_date_field)


if sdc.config.config_pipeline_hpat_default:
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

else:
    from sdc.datatypes.hpat_pandas_series_rolling_types import SeriesRollingType

class SeriesIatType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesIatType({})".format(stype)
        super(SeriesIatType, self).__init__(name)


# PR135. This needs to be commented out
if sdc.config.config_pipeline_hpat_default:
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
        from sdc.str_arr_ext import is_str_arr_typ
        assert not kws
        [va, vb] = args
        # if one of the inputs is string array
        if is_str_series_typ(va) or is_str_series_typ(vb):
            # inputs should be either string array or string
            assert is_str_arr_typ(va) or va == string_type
            assert is_str_arr_typ(vb) or vb == string_type
            return signature(SeriesType(types.boolean), va, vb)

        if ((is_dt64_series_typ(va) and vb == string_type)
                or (is_dt64_series_typ(vb) and va == string_type)):
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


if sdc.config.config_pipeline_hpat_default:
    @infer_global(operator.getitem)
    class GetItemBuffer(AbstractTemplate):
        key = operator.getitem

        def generic(self, args, kws):
            assert not kws
            [ary, idx] = args
            if not isinstance(ary, SeriesType):
                return
            out = get_array_index_type(ary, idx)
            # check result to be dt64 since it might be sliced array
            # replace result with Timestamp
            if out is not None and out.result == types.NPDatetime('ns'):
                return signature(pandas_timestamp_type, ary, out.index)


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


# replacing cumprod since arraydecl.py definition uses types.Array
install_array_method('cumprod', generic_expand_cumulative_series)

# TODO: add itemsize, strides, etc. when removed from Pandas
_not_series_array_attrs = ['flat', 'ctypes', 'itemset', 'reshape', 'sort', 'flatten',
                           'resolve_cumsum',
                           'resolve_shift', 'resolve_sum', 'resolve_copy', 'resolve_corr', 'resolve_mean',
                           'resolve_take', 'resolve_max', 'resolve_min', 'resolve_nunique',
                           'resolve_argsort', 'resolve_sort_values', 'resolve_pct_change',
                           'resolve_prod', 'resolve_count', 'resolve_dropna', 'resolve_fillna', 'resolve_astype']

# disable using of some Array attributes in non-hpat pipeline only
if not sdc.config.config_pipeline_hpat_default:
    for attr in ['resolve_std', 'resolve_var']:
        _not_series_array_attrs.append(attr)

_non_hpat_pipeline_attrs = [
    'resolve_append', 'resolve_combine', 'resolve_corr', 'resolve_cov',
    'resolve_dropna', 'resolve_fillna', 'resolve_head', 'resolve_nlargest',
    'resolve_nsmallest', 'resolve_pct_change', 'resolve_loc', 'resolve_iloc',
    'resolve_iat', 'resolve_rolling', 'resolve_value_counts'
]

# use ArrayAttribute for attributes not defined in SeriesAttribute
for attr, func in numba.typing.arraydecl.ArrayAttribute.__dict__.items():
    if (attr.startswith('resolve_')
            and attr not in SeriesAttribute.__dict__
            and attr not in _not_series_array_attrs):
        setattr(SeriesAttribute, attr, func)

# remove some attributes from SeriesAttribute for non-hpat pipeline
if not sdc.config.config_pipeline_hpat_default:
    for attr in _non_hpat_pipeline_attrs:
        if attr in SeriesAttribute.__dict__:
            delattr(SeriesAttribute, attr)

# PR135. This needs to be commented out
if sdc.config.config_pipeline_hpat_default:
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
                # TODO: split view
                # mimic array indexing for list
                if (isinstance(in_idx, types.Array) and in_idx.ndim == 1
                        and isinstance(
                            in_idx.dtype, (types.Integer, types.Boolean))):
                    sig = signature(in_arr, in_arr, in_idx)
                else:
                    sig = numba.typing.collections.GetItemSequence.generic(
                        self, (in_arr, in_idx), kws)
            elif in_arr == string_array_split_view_type:
                sig = GetItemStringArraySplitView.generic(
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

if sdc.config.config_pipeline_hpat_default:
    @infer_global(operator.setitem)
    class SetItemSeries(SetItemBuffer):
        def generic(self, args, kws):
            assert not kws
            series, idx, val = args
            if not isinstance(series, SeriesType):
                return None
            # TODO: handle any of args being Series independently
            ary = series_to_array_type(series)
            is_idx_series = False
            if isinstance(idx, SeriesType):
                idx = series_to_array_type(idx)
                is_idx_series = True
            is_val_series = False
            if isinstance(val, SeriesType):
                val = series_to_array_type(val)
                is_val_series = True
            # TODO: strings, dt_index
            res = super(SetItemSeries, self).generic((ary, idx, val), kws)
            if res is not None:
                new_series = if_arr_to_series_type(res.args[0])
                idx = res.args[1]
                val = res.args[2]
                if is_idx_series:
                    idx = if_arr_to_series_type(idx)
                if is_val_series:
                    val = if_arr_to_series_type(val)
                res.args = (new_series, idx, val)
                return res

if sdc.config.config_pipeline_hpat_default:
    @infer_global(operator.setitem)
    class SetItemSeriesIat(SetItemSeries):
        def generic(self, args, kws):
            # iat[] is the same as regular setitem
            if isinstance(args[0], SeriesIatType):
                return SetItemSeries.generic(self, (args[0].stype, args[1], args[2]), kws)


inplace_ops = [
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.itruediv,
    operator.ifloordiv,
    operator.imod,
    operator.ipow,
    operator.ilshift,
    operator.irshift,
    operator.iand,
    operator.ior,
    operator.ixor,
]


def series_op_generic(cls, self, args, kws):
    # return if no Series
    if not any(isinstance(arg, SeriesType) for arg in args):
        return None
    # convert args to array
    new_args = tuple(if_series_to_array_type(arg) for arg in args)
    sig = super(cls, self).generic(new_args, kws)
    # convert back to Series
    if sig is not None:
        # if A += B and A is array but B is Series, the output is Array
        # TODO: other cases?
        if not (self.key in inplace_ops and isinstance(args[0], types.Array)):
            sig.return_type = if_arr_to_series_type(sig.return_type)
        sig.args = args
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
    # 'add': operator.add,
    # 'sub': operator.sub,
    # 'mul': operator.mul,
    # 'div': operator.truediv,
    # 'truediv': operator.truediv,
    # 'floordiv': operator.floordiv,
    # 'mod': operator.mod,
    # 'pow': operator.pow,
    # 'lt': operator.lt,
    # 'gt': operator.gt,
    # 'le': operator.le,
    # 'ge': operator.ge,
    # 'ne': operator.ne,
    # 'eq': operator.eq,
}


def ex_binop_generic(self, args, kws):
    return SeriesOpUfuncs.generic(self, (self.this,) + args, kws)


for fname, op in explicit_binop_funcs.items():
    install_series_method(op, fname, ex_binop_generic)


class SeriesInplaceOpUfuncs(NumpyRulesInplaceArrayOperator):
    def generic(self, args, kws):
        return series_op_generic(SeriesInplaceOpUfuncs, self, args, kws)


class SeriesUnaryOpUfuncs(NumpyRulesUnaryArrayOperator):
    def generic(self, args, kws):
        return series_op_generic(SeriesUnaryOpUfuncs, self, args, kws)


if sdc.config.config_pipeline_hpat_default:
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
    # _numpy_ufunc(func)

    class typing_class(Series_Numpy_rules_ufunc):
        key = func

    typing_class.__name__ = "resolve_series_{0}".format(name)

    if name not in _aliases:
        infer_global(func, types.Function(typing_class))


# @infer_global(len)
# class LenSeriesType(AbstractTemplate):
#     def generic(self, args, kws):
#         if not kws and len(args) == 1 and isinstance(args[0], SeriesType):
#             return signature(types.intp, *args)

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

# @infer_global(np.full_like)

# TODO: handle all timedelta args


def type_sub(context):
    def typer(val1, val2):
        if is_dt64_series_typ(val1) and val2 == pandas_timestamp_type:
            return SeriesType(types.NPTimedelta('ns'))

        from sdc.hiframes.pd_index_ext import DatetimeIndexType
        if isinstance(val1, DatetimeIndexType) and val2 == pandas_timestamp_type:
            from sdc.hiframes.pd_index_ext import TimedeltaIndexType
            return TimedeltaIndexType(False)
    return typer


type_callable('-')(type_sub)
type_callable(operator.sub)(type_sub)


@overload(pd.Series)
def pd_series_overload(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):

    is_index_none = isinstance(index, types.NoneType) or index is None

    def hpat_pandas_series_ctor_impl(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):

        '''' use binop here as otherwise Numba's dead branch pruning doesn't work
        TODO: replace with 'if not is_index_none' when resolved '''
        if is_index_none == False:  # noqa
            fix_index = sdc.hiframes.api.fix_df_array(index)
        else:
            fix_index = index

        return sdc.hiframes.api.init_series(sdc.hiframes.api.fix_df_array(data), fix_index, name)

    return hpat_pandas_series_ctor_impl

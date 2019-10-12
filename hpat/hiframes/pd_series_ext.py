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

import datetime
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
    overload,
    make_attribute_wrapper)
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

import hpat
from hpat.hiframes.pd_categorical_ext import (PDCategoricalDtype, CategoricalArray)
from hpat.hiframes.pd_timestamp_ext import (pandas_timestamp_type, datetime_date_type)
from hpat.hiframes.rolling import supported_rolling_funcs
from hpat.hiframes.split_impl import (string_array_split_view_type, GetItemStringArraySplitView)
from hpat.str_arr_ext import (
    string_array_type,
    iternext_str_array,
    offset_typ,
    char_typ,
    str_arr_payload_type,
    StringArrayType,
    GetItemStringArray)
from hpat.str_ext import string_type, list_string_array_type


class SeriesType(types.IterableType):
    """Temporary type class for Series objects.
    """

    def __init__(self, dtype, data=None, index=None, is_named=False):
        # keeping data array in type since operators can make changes such
        # as making array unaligned etc.
        data = _get_series_array_type(dtype) if data is None else data
        # convert Record to tuple (for tuple output of map)
        # TODO: handle actual Record objects in Series?
        self.dtype = (types.Tuple(list(dict(dtype.members).values()))
                      if isinstance(dtype, types.Record) else dtype)
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
        index = types.none if self.index == types.none else self.index.copy()
        dtype = dtype if dtype is not None else self.dtype
        data = _get_series_array_type(dtype)
        return SeriesType(dtype, data, index)

    @property
    def key(self):
        # needed?
        return self.dtype, self.data, self.index, self.is_named

    @property
    def ndim(self):
        return self.data.ndim

    def unify(self, typingctx, other):
        if isinstance(other, SeriesType):
            new_index = types.none
            if self.index != types.none and other.index != types.none:
                new_index = self.index.unify(typingctx, other.index)
            elif other.index != types.none:
                new_index = other.index
            elif self.index != types.none:
                new_index = self.index

            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                return SeriesType(
                    self.dtype,
                    self.data.unify(typingctx, other.data),
                    new_index)

        # XXX: unify Series/Array as Array
        return super(SeriesType, self).unify(typingctx, other)

    def can_convert_to(self, typingctx, other):
        # same as types.Array
        if (isinstance(other, SeriesType) and other.dtype == self.dtype):
            # called for overload selection sometimes
            # TODO: fix index
            if self.index == types.none and other.index == types.none:
                return self.data.can_convert_to(typingctx, other.data)
            if self.index != types.none and other.index != types.none:
                return max(self.data.can_convert_to(typingctx, other.data),
                           self.index.can_convert_to(typingctx, other.index))

    def is_precise(self):
        # same as types.Array
        return self.dtype.is_precise()

    @property
    def iterator_type(self):
        # TODO: fix timestamp
        return SeriesIterator(self)


class SeriesIterator(types.SimpleIteratorType):
    """
    Type class for iterator over dataframe series.
    """

    def __init__(self, series_type):
        self.series_type = series_type
        self.array_type = series_type.data

        name = f'iter({self.series_type.data})'
        yield_type = series_type.dtype
        super(SeriesIterator, self).__init__(name, yield_type)

    @property
    def _iternext(self):
        if isinstance(self.array_type, StringArrayType):
            return iternext_str_array
        elif isinstance(self.array_type, types.Array):
            return iternext_series_array


@register_model(SeriesIterator)
class SeriesIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('array', fe_type.series_type.data)]

        models.StructModel.__init__(self, dmm, fe_type, members)


def _get_series_array_type(dtype):
    """get underlying default array type of series based on its dtype
    """
    # list(list(str))
    if dtype == types.List(string_type):
        # default data layout is list but split view is used if possible
        return list_string_array_type
    # string array
    elif dtype == string_type:
        return string_array_type

    # categorical
    if isinstance(dtype, PDCategoricalDtype):
        return CategoricalArray(dtype)

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


def is_str_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == string_type


def is_dt64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPDatetime('ns')


def is_timedelta64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPTimedelta('ns')


def is_datetime_date_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == datetime_date_type


# register_model(SeriesType)(models.ArrayModel)
# need to define model since fix_df_array overload goes to native code
@register_model(SeriesType)
class SeriesModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        name_typ = string_type if fe_type.is_named else types.none
        members = [
            ('data', fe_type.data),
            ('index', fe_type.index),
            ('name', name_typ),
        ]
        super(SeriesModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesType, 'data', '_data')
make_attribute_wrapper(SeriesType, 'index', '_index')
make_attribute_wrapper(SeriesType, 'name', '_name')


@lower_builtin('getiter', SeriesType)
def getiter_series(context, builder, sig, args):
    """
    Getting iterator for the Series type

    :param context: context descriptor
    :param builder: llvmlite IR Builder
    :param sig: iterator signature
    :param args: tuple with iterator arguments, such as instruction, operands and types
    :param result: iternext result
    :return: reference to iterator
    """

    arraytype = sig.args[0].data

    # Create instruction to get array to iterate
    zero_member_pointer = context.get_constant(types.intp, 0)
    zero_member = context.get_constant(types.int32, 0)
    alloca = args[0].operands[0]
    gep_result = builder.gep(alloca, [zero_member_pointer, zero_member])
    array = builder.load(gep_result)

    # TODO: call numba getiter with gep_result for array
    iterobj = context.make_helper(builder, sig.return_type)
    zero_index = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero_index)

    iterobj.index = indexptr
    iterobj.array = array

    if context.enable_nrt:
        context.nrt.incref(builder, arraytype, array)

    result = iterobj._getvalue()
    # Note: a decref on the iterator will dereference all internal MemInfo*
    out = impl_ret_new_ref(context, builder, sig.return_type, result)
    return out


# TODO: call it from numba.targets.arrayobj, need separate function in numba
def iternext_series_array(context, builder, sig, args, result):
    """
    Implementation of iternext() for the ArrayIterator type

    :param context: context descriptor
    :param builder: llvmlite IR Builder
    :param sig: iterator signature
    :param args: tuple with iterator arguments, such as instruction, operands and types
    :param result: iternext result
    """

    [iterty] = sig.args
    [iter] = args
    arrayty = iterty.array_type

    if arrayty.ndim != 1:
        raise NotImplementedError("iterating over %dD array" % arrayty.ndim)

    iterobj = context.make_helper(builder, iterty, value=iter)
    ary = make_array(arrayty)(context, builder, value=iterobj.array)

    nitems, = cgutils.unpack_tuple(builder, ary.shape, count=1)

    index = builder.load(iterobj.index)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        value = _getitem_array1d(context, builder, arrayty, ary, index,
                                 wraparound=False)
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


@lower_builtin('iternext', SeriesIterator)
@iternext_impl(RefType.BORROWED)
def iternext_series(context, builder, sig, args, result):
    """
    Iternext implementation depending on Array type

    :param context: context descriptor
    :param builder: llvmlite IR Builder
    :param sig: iterator signature
    :param args: tuple with iterator arguments, such as instruction, operands and types
    :param result: iternext result
    """
    iternext_func = sig.args[0]._iternext
    iternext_func(context=context, builder=builder, sig=sig, args=args, result=result)


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
    # def resolve_shape(self, ary):
    #     return types.Tuple((types.int64,))

# PR171. This needs to be commented out
#     def resolve_index(self, ary):
#         return ary.index

    def resolve_str(self, ary):
        assert ary.dtype in (string_type, types.List(string_type))
        # TODO: add dtype to series_str_methods_type
        return series_str_methods_type

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

    @bound_function("array.astype")
    def resolve_astype(self, ary, args, kws):
        # TODO: handle other types like datetime etc.
        dtype, = args
        if ((isinstance(dtype, types.Function) and dtype.typing_key == str)
                or (isinstance(dtype, types.StringLiteral) and dtype.literal_value == 'str')):
            ret_type = SeriesType(string_type)
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
            ret_type = SeriesType(string_type)
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
        # output will have permuted input index
        out_index = ary.index
        if out_index == types.none:
            out_index = types.Array(types.intp, 1, 'C')
        out = SeriesType(ary.dtype, ary.data, out_index)
        return signature(out, *args)

#     @bound_function("array.take")
#     def resolve_take(self, ary, args, kws):
#         resolver = ArrayAttribute.resolve_take.__wrapped__
#         sig = resolver(self, ary.data, args, kws)
#         sig.return_type = if_arr_to_series_type(sig.return_type)
#         return sig

    @bound_function("series.quantile")
    def resolve_quantile(self, ary, args, kws):
        # TODO: fix quantile output type if not float64
        return signature(types.float64, *args)

    @bound_function("series.count")
    def resolve_count(self, ary, args, kws):
        return signature(types.intp, *args)

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
            out = out.copy(dtype=types.float64)
        return signature(out, *args)

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
        code = args[0].literal_value.code
        _globals = {'np': np}
        # XXX hack in hiframes_typed to make globals available
        if hasattr(args[0].literal_value, 'globals'):
            # TODO: use code.co_names to find globals actually used?
            _globals = args[0].literal_value.globals

        f_ir = numba.ir_utils.get_ir_of_code(_globals, code)
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
            self.context, f_ir, (dtype1, dtype2,), None)
        return signature(SeriesType(f_return_type), *args)

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

    @bound_function("series.rename")
    def resolve_rename(self, ary, args, kws):
        # TODO: support index rename, kws
        assert len(args) == 1 and isinstance(
            args[0], (types.UnicodeType, types.StringLiteral))
        out = SeriesType(ary.dtype, ary.data, ary.index, True)
        return signature(out, *args)


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

str2str_methods = ('capitalize', 'lower', 'lstrip', 'rstrip',
                   'strip', 'swapcase', 'title', 'upper')


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
        if func_name not in str2str_methods:
            raise ValueError("Series.str.{} is not supported yet".format(
                func_name))

        template_key = 'strmethod.' + func_name
        out_typ = SeriesType(string_type)

        class MethodTemplate(AbstractTemplate):
            key = template_key

            def generic(self, args, kws):
                return signature(out_typ, *args)

        return types.BoundFunction(MethodTemplate, s_str)


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


# PR135. This needs to be commented out
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
_not_series_array_attrs = ['flat', 'ctypes', 'itemset', 'reshape', 'sort', 'flatten',
                           'resolve_take', 'resolve_max', 'resolve_min', 'resolve_nunique']

# use ArrayAttribute for attributes not defined in SeriesAttribute
for attr, func in numba.typing.arraydecl.ArrayAttribute.__dict__.items():
    if (attr.startswith('resolve_')
            and attr not in SeriesAttribute.__dict__
            and attr not in _not_series_array_attrs):
        setattr(SeriesAttribute, attr, func)


# PR135. This needs to be commented out
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

        from hpat.hiframes.pd_index_ext import DatetimeIndexType
        if isinstance(val1, DatetimeIndexType) and val2 == pandas_timestamp_type:
            from hpat.hiframes.pd_index_ext import TimedeltaIndexType
            return TimedeltaIndexType(False)
    return typer


type_callable('-')(type_sub)
type_callable(operator.sub)(type_sub)


@overload(pd.Series)
def pd_series_overload(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):

    if index is not None:
        def hpat_pandas_series_index_ctor_impl(
                data=None,
                index=None,
                dtype=None,
                name=None,
                copy=False,
                fastpath=False):
            return hpat.hiframes.api.init_series(
                hpat.hiframes.api.fix_df_array(data),
                hpat.hiframes.api.fix_df_array(index),
                name)

        return hpat_pandas_series_index_ctor_impl

    def hpat_pandas_series_ctor_impl(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        return hpat.hiframes.api.init_series(hpat.hiframes.api.fix_df_array(data), index, name)

    return hpat_pandas_series_ctor_impl

from hpat.datatypes.hpat_pandas_series_functions import *

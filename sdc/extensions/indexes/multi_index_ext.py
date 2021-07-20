# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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

import numba
import numpy as np
import operator
import pandas as pd

from numba import types, prange
from numba.core import cgutils
from numba.extending import (typeof_impl, NativeValue, intrinsic, box, unbox, lower_builtin, type_callable)
from numba.core.errors import TypingError
from numba.core.typing.templates import signature, AttributeTemplate, infer_getattr
from numba.core.imputils import (impl_ret_borrowed, iternext_impl, RefType)
from numba.core.boxing import unbox_array, box_tuple

import llvmlite.llvmpy.core as lc

from sdc.datatypes.indexes import *
from sdc.utilities.sdc_typing_utils import SDCLimitation
from sdc.utilities.utils import sdc_overload, sdc_overload_attribute, sdc_overload_method, BooleanLiteral
from sdc.utilities.sdc_typing_utils import (
        TypeChecker,
        sdc_pandas_index_types,
        sdc_pandas_df_column_types,
        check_types_comparable,
    )
from sdc.functions import numpy_like
from sdc.functions.tuple_utils import sdc_tuple_map, sdc_tuple_map_elementwise, sdc_tuple_unzip
from sdc.hiframes.api import fix_df_array, fix_df_index
from sdc.hiframes.boxing import _infer_index_type, _unbox_index_data
from sdc.extensions.indexes.indexes_generic import *

from sdc.datatypes.indexes.multi_index_type import MultiIndexIteratorType
from numba.core.extending import register_jitable
from numba import literal_unroll
from numba.typed import Dict, List
from sdc.datatypes.sdc_typeref import MultiIndexTypeRef


@typeof_impl.register(pd.MultiIndex)
def typeof_multi_index(val, c):
    levels = tuple(_infer_index_type(x) for x in val.levels)
    codes = tuple(numba.typeof(x) for x in val.codes)  # note this produces readonly array(int8, 1d, C)
    is_named = val.name is not None

    return MultiIndexType(types.Tuple.from_types(levels),
                          types.Tuple.from_types(codes),
                          is_named=is_named)


@box(MultiIndexType)
def box_multi_index(typ, val, c):

    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    multi_index = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    py_levels = box_tuple(typ.levels, multi_index.levels, c)
    py_codes = box_tuple(typ.codes, multi_index.codes, c)

    # dtype and copy params are not stored so use default values
    dtype = c.pyapi.make_none()
    copy = c.pyapi.bool_from_bool(
        c.context.get_constant(types.bool_, False)
    )
    sortorder = c.pyapi.make_none()

    if typ.is_named:
        name = c.pyapi.from_native_value(types.unicode_type, multi_index.name)
    else:
        name = c.pyapi.make_none()

    # build MultiIndex names from names of boxed levels (if python level has name attribute)
    # TO-DO: refactor this to use native indexes names when all index have it (e.g. StringIndexType)
    nlevels = len(typ.levels)
    py_nlevels = c.pyapi.tuple_size(py_levels)
    py_names = c.pyapi.list_new(py_nlevels)
    for i in range(nlevels):
        level_type = typ.levels[i]
        if isinstance(level_type, sdc_old_index_types):
            py_level_name = c.pyapi.make_none()
        else:
            py_level_obj = c.pyapi.tuple_getitem(py_levels, i)
            py_level_name = c.pyapi.object_getattr_string(py_level_obj, 'name')
        c.pyapi.list_setitem(py_names, c.context.get_constant(types.intp, i), py_level_name)
        # FIXME: check decref is needed for pe_level_obj?

    res = c.pyapi.call_method(pd_class_obj, "MultiIndex",
                              (py_levels, py_codes, sortorder, py_names, dtype, copy, name))

    c.pyapi.decref(py_levels)
    c.pyapi.decref(py_codes)
    c.pyapi.decref(sortorder)
    c.pyapi.decref(py_names)
    c.pyapi.decref(dtype)
    c.pyapi.decref(copy)
    c.pyapi.decref(name)
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(MultiIndexType)
def unbox_int64_index(typ, val, c):

    nlevels = len(typ.levels)
    levels_types = typ.levels_types
    codes_types = typ.codes_types
    multi_index = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    py_levels_data = c.pyapi.object_getattr_string(val, "levels")
    native_levels_data = []
    for i in range(nlevels):
        idx = c.pyapi.long_from_ulonglong(c.context.get_constant(types.int64, i))
        level_data = c.pyapi.object_getitem(py_levels_data, idx)
        native_levels_data.append(
            _unbox_index_data(levels_types[i], level_data, c).value
        )
        c.pyapi.decref(level_data)
    c.pyapi.decref(py_levels_data)
    multi_index.levels = c.context.make_tuple(c.builder, typ.levels, native_levels_data)

    py_codes_data = c.pyapi.object_getattr_string(val, "codes")
    native_codes_data = []
    for i in range(nlevels):
        idx = c.pyapi.long_from_ulonglong(c.context.get_constant(types.int64, i))
        code_data = c.pyapi.object_getitem(py_codes_data, idx)
        native_codes_data.append(
            unbox_array(codes_types[i], code_data, c).value
        )
        c.pyapi.decref(code_data)
    c.pyapi.decref(py_codes_data)
    multi_index.codes = c.context.make_tuple(c.builder, typ.codes, native_codes_data)

    if typ.is_named:
        name_obj = c.pyapi.object_getattr_string(val, "name")
        multi_index.name = numba.cpython.unicode.unbox_unicode_str(
            types.unicode_type, name_obj, c).value
        c.pyapi.decref(name_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(multi_index._getvalue(), is_error=is_error)


@intrinsic
def init_multi_index(typingctx, levels, codes):

    if not (isinstance(levels, (types.Tuple, types.UniTuple)) and
            isinstance(codes, (types.Tuple, types.UniTuple))):
        return None

    def is_valid_level_type(typ):
        return isinstance(typ, sdc_pandas_index_types)

    def is_valid_code_type(typ):
        return (isinstance(typ, types.Array) and isinstance(typ.dtype, types.Integer))

    if not all(map(is_valid_level_type, levels)):
        return None

    if not all(map(is_valid_code_type, codes)):
        return None

    def codegen(context, builder, sig, args):
        levels_val, codes_val = args
        # create series struct and store values
        multi_index = cgutils.create_struct_proxy(
            sig.return_type)(context, builder)

        multi_index.levels = levels_val
        multi_index.codes = codes_val
        multi_index.name = context.get_dummy_value()

        if context.enable_nrt:
            context.nrt.incref(builder, sig.args[0], levels_val)
            context.nrt.incref(builder, sig.args[1], codes_val)

        return multi_index._getvalue()

    ret_typ = MultiIndexType(levels, codes, is_named=False)  # pandas ctor always creates unnamed indexes
    sig = signature(ret_typ, levels, codes)
    return sig, codegen


def _sdc_multi_index_ctor_typer(typing_ctx, *args):

    _func_name = '_sdc_multi_index_ctor_typer'
    # this types subsequent call to sdc_pandas_multi_index_ctor function with signature:
    # args = (levels, codes, sortorder=None, names=None, dtype=None, copy=False, name=None)

    assert len(args) >= 2, f"{_func_name}: Expecting 2 or more positional args, given: {args}"

    levels, codes = args[:2]
    if not (isinstance(levels, (types.Tuple, types.UniTuple))
            and isinstance(codes, (types.Tuple, types.UniTuple))):
        raise TypingError(f"{_func_name}: levels and codes args must be tuples, given: levels={levels}, codes={codes}")

    nlevels = len(levels)
    ty_codes = types.Tuple.from_types(
        [typing_ctx._resolve_user_function_type(
            fix_df_array, (typ,), {}).return_type for typ in codes]
    )

    if len(args) >= 2 and not (isinstance(args[2], (types.NoneType, types.Omitted)) or args[2] is None):
        assert False, f"{_func_name}: argument sortorder is not supported, given: {args[2]}"
    if len(args) >= 3 and not (isinstance(args[3], (types.NoneType, types.Omitted)) or args[3] is None):
        assert False, f"{_func_name}: argument names is not supported, given: {args[3]}"
    if len(args) >= 4 and not (isinstance(args[4], (types.NoneType, types.Omitted)) or args[4] is None):
        assert False, f"{_func_name}: argument dtype is not supported, given: {args[4]}"
    if len(args) >= 5 and not (isinstance(args[5], (types.Boolean, types.Omitted)) or args[5] is False):
        assert False, f"{_func_name}: argument copy is not supported, given: {args[5]}"

    # if ctor args provide list of levels names via name argument
    # update type information for elements in ty_levels (so that levels are named indexes)
    name = args[6] if len(args) >= 6 and not args[6] is None else types.none
    if not isinstance(name, (types.NoneType, types.Omitted)):
        assert (isinstance(name, types.Tuple)
                and all(map(lambda x: isinstance(x, (types.StringLiteral, types.UnicodeType, types.NoneType)), name))
                or isinstance(name, types.UniTuple)
                and isinstance(name.dtype, (types.UnicodeType, types.NoneType))), \
                f"{_func_name}: argument name must be tuple of strings, given: {args[6]}"
        assert len(name) == nlevels, \
               f"{_func_name}: Length of names must match number of levels in MultiIndex, given: {args[6]}"

        ty_levels = types.Tuple.from_types(
            [typing_ctx._resolve_user_function_type(
                _multi_index_create_level, (t1, t2), {}).return_type for t1, t2 in zip(levels, name)]
        )
    else:
        ty_levels = types.Tuple.from_types(
            [typing_ctx._resolve_user_function_type(
                _multi_index_create_level, (typ, types.none), {}).return_type for typ in levels]
        )

    return MultiIndexType(ty_levels, ty_codes, is_named=False)


# TO-DO: refactor: this allows SdcTypeRef to be callable and makes pd.MultiIndex.from_product
# work, but this typer handles only case when SdcTypeRef.instance_type is MultiIndexType
# but it may be reference to other type as well (e.g. ConcurrentDictType). Need differentiate
# SdcTypeRef-s for different types.
@type_callable(MultiIndexTypeRef)
def typing_sdctyperef(context):
    typing_ctx = context

    def typer(levels, codes, sortorder=None, names=None,
              dtype=None, copy=False, name=None):
        return _sdc_multi_index_ctor_typer(typing_ctx, levels, codes, sortorder,
                                           names, dtype, copy, name)

    return typer


# FIXME_Numba#7111: low-level api is used to implement typing and impl of MultiIndex ctor
# which is a workaround numba issue (once it's fixed @overload can be used instead)
@infer_getattr
class SdcTypeRefAttribute(AttributeTemplate):
    key = MultiIndexTypeRef

    def resolve___call__(self, instance):
        return type(instance)


def sdc_pandas_multi_index_ctor(levels, codes, sortorder=None, names=None,
                                dtype=None, copy=False, name=None):
    pass


@sdc_overload(sdc_pandas_multi_index_ctor)
def pd_multi_index_overload(levels, codes, sortorder=None, names=None,
                            dtype=None, copy=False, name=None):

    _func_name = 'pd.MultiIndex().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(sortorder, (types.Omitted, types.NoneType)) or sortorder is None):
        raise TypingError('{} Unsupported parameters. Given sortorder: {}'.format(_func_name, sortorder))

    if not (isinstance(names, (types.Omitted, types.NoneType)) or names is None):
        raise TypingError('{} Unsupported parameters. Given names: {}'.format(_func_name, names))

    if not (isinstance(dtype, (types.Omitted, types.NoneType)) or dtype is None):
        raise TypingError('{} Unsupported parameters. Given dtype: {}'.format(_func_name, dtype))

    if not (isinstance(copy, (types.Omitted, types.Boolean, types.BooleanLiteral)) or copy is False):
        raise TypingError('{} Unsupported parameters. Given copy: {}'.format(_func_name, copy))

    accepted_index_names = (types.NoneType, types.StringLiteral, types.UnicodeType)
    is_name_none = name is None or isinstance(name, (types.NoneType, types.Omitted))
    if not (isinstance(name, (types.Tuple, types.UniTuple))
            and all(map(lambda x: isinstance(x, accepted_index_names), name))
            or is_name_none):
        ty_checker.raise_exc(name, 'tuple of strings/nones or none', 'name')

    def pd_multi_index_ctor_impl(levels, codes, sortorder=None, names=None,
                                 dtype=None, copy=False, name=None):

        if len(levels) != len(codes):
            raise ValueError("Length of levels and codes must be the same.")
        if len(levels) == 0:
            raise ValueError("Must pass non-zero number of levels/codes")

        # if name is None then all level names are reset
        if is_name_none == True:  # noqa
            _names = sdc_tuple_map(
                lambda x: None,
                levels,
            )
        else:
            _names = name

        levels_and_codes_pairs = sdc_tuple_map_elementwise(
            _multi_index_create_levels_and_codes,
            levels,
            codes,
            _names
        )

        _levels, _codes = sdc_tuple_unzip(levels_and_codes_pairs)
        return init_multi_index(_levels, _codes)

    return pd_multi_index_ctor_impl


@lower_builtin(MultiIndexTypeRef, types.VarArg(types.Any))
def sdctyperef_call_impl(context, builder, sig, args):

    # FIXME: this hardcodes template number and selected dispatcher, refactor?
    call_sig = context.typing_context._resolve_user_function_type(
        sdc_pandas_multi_index_ctor,
        sig.args,
        {}
    )
    fnty = context.typing_context._lookup_global(sdc_pandas_multi_index_ctor)
    disp = fnty.templates[0](context.typing_context)._get_impl(call_sig.args, {})
    cres = disp[0].get_compile_result(call_sig)

    res = context.call_internal(
        builder,
        cres.fndesc,
        sig,
        args
    )

    return impl_ret_borrowed(context, builder, sig.return_type, res)


@sdc_overload(len)
def pd_multi_index_len_overload(self):
    if not isinstance(self, MultiIndexType):
        return None

    def pd_multi_index_len_impl(self):
        return len(self._codes[0])

    return pd_multi_index_len_impl


@intrinsic
def _multi_index_getitem_impl(typingctx, self, idx):
    if not isinstance(self, MultiIndexType):
        return None

    nlevels = self.nlevels
    levels_types = self.levels_types
    codes_types = self.codes_types
    ret_type = types.Tuple.from_types([index.dtype for index in levels_types])

    def codegen(context, builder, sig, args):
        self_val, idx_val = args
        self_ctinfo = context.make_helper(builder, self, self_val)

        res_elements = []
        for level_index in range(nlevels):
            level = builder.extract_value(self_ctinfo.levels, level_index)
            code = builder.extract_value(self_ctinfo.codes, level_index)
            element = context.compile_internal(
                builder,
                lambda index, code, i: index[code[i]],
                signature(levels_types[level_index].dtype, levels_types[level_index], codes_types[level_index], idx),
                [level, code, idx_val]
            )
            res_elements.append(element)

        return context.make_tuple(builder, ret_type, res_elements)

    return ret_type(self, idx), codegen


@sdc_overload(operator.getitem)
def pd_multi_index_getitem_overload(self, idx):
    if not isinstance(self, MultiIndexType):
        return None

    _func_name = 'Operator getitem().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(idx, (types.Integer, types.SliceType))
            or isinstance(idx, (types.Array, types.List)) and isinstance(idx.dtype, (types.Integer, types.Boolean))):
        ty_checker.raise_exc(idx, 'integer, slice, integer array or list', 'idx')

    if isinstance(idx, types.Integer):
        def pd_multi_index_getitem_idx_scalar_impl(self, idx):
            index_len = len(self)
            # FIXME_Numba#5801: Numba type unification rules make this float
            idx = types.int64((index_len + idx) if idx < 0 else idx)
            if (idx < 0 or idx >= index_len):
                raise IndexError("MultiIndex.getitem: index is out of bounds")

            return _multi_index_getitem_impl(self, idx)

        return pd_multi_index_getitem_idx_scalar_impl

    elif isinstance(idx, types.SliceType):
        def pd_multi_index_getitem_idx_slice_impl(self, idx):

            new_levels = self._levels
            new_codes = sdc_tuple_map(
                lambda arr_codes, taken_idxs: arr_codes[taken_idxs],
                self._codes,
                idx
            )
            return pd.MultiIndex(new_levels, new_codes)

        return pd_multi_index_getitem_idx_slice_impl

    elif isinstance(idx, types.Array) and isinstance(idx.dtype, types.Boolean):
        def pd_multi_index_getitem_idx_bool_array_impl(self, idx):

            new_levels = self._levels
            new_codes = sdc_tuple_map(
                lambda arr_codes, taken_idxs: numpy_like.getitem_by_mask(arr_codes, taken_idxs),
                self._codes,
                idx
            )
            return pd.MultiIndex(new_levels, new_codes)

        return pd_multi_index_getitem_idx_bool_array_impl

    elif isinstance(idx, types.Array) and isinstance(idx.dtype, types.Integer):
        def pd_multi_index_getitem_as_take_impl(self, idx):
            return self.take(idx)

        return pd_multi_index_getitem_as_take_impl


@sdc_overload_attribute(MultiIndexType, 'values')
def pd_multi_index_values_overload(self):
    if not isinstance(self, MultiIndexType):
        return None

    # FIXME: we return a list for now, as there's no arrays of tuples in numba, nor other
    # sequence container that is boxed to dtype=object numpy array. TO-DO: replace with other type?
    def pd_multi_index_values_impl(self):
        res = []
        for i in range(len(self)):
            res.append(self[i])
        return res

    return pd_multi_index_values_impl


@sdc_overload_attribute(MultiIndexType, 'dtype')
def pd_multi_index_dtype_overload(self):
    if not isinstance(self, MultiIndexType):
        return None

    mindex_dtype = self.dtype

    def pd_multi_index_dtype_impl(self):
        return mindex_dtype

    return pd_multi_index_dtype_impl


@sdc_overload_attribute(MultiIndexType, 'levels')
def pd_multi_index_levels_overload(self):
    if not isinstance(self, MultiIndexType):
        return None

    def pd_multi_index_levels_impl(self):
        return self._levels

    return pd_multi_index_levels_impl


@sdc_overload_attribute(MultiIndexType, 'codes')
def codespd_multi_index_levels_overload(self):
    if not isinstance(self, MultiIndexType):
        return None

    def pd_multi_index_codes_impl(self):
        return self._codes

    return pd_multi_index_codes_impl


@sdc_overload_method(MultiIndexType, 'take')
def pd_multi_index_take_overload(self, indexes):
    if not isinstance(self, MultiIndexType):
        return None

    _func_name = 'Method take().'
    ty_checker = TypeChecker(_func_name)

    valid_indexes_types = (types.Array, types.List, types.ListType) + sdc_pandas_index_types
    if not (isinstance(indexes, valid_indexes_types)
            and isinstance(indexes.dtype, (types.Integer, types.ListType))):
        ty_checker.raise_exc(indexes, 'array/list of integers or integer index', 'indexes')

    def pd_multi_index_take_impl(self, indexes):
        new_levels = self._levels
        new_codes = sdc_tuple_map(
            lambda idx, taken_idxs: sdc_indexes_take(idx, taken_idxs),
            self._codes,
            indexes)
        return pd.MultiIndex(new_levels, new_codes)

    return pd_multi_index_take_impl


@sdc_overload_attribute(MultiIndexType, 'nlevels')
def pd_multi_index_nlevels_overload(self):
    if not isinstance(self, MultiIndexType):
        return None

    nlevels_value = len(self.levels)

    def pd_multi_index_nlevels_impl(self):
        return nlevels_value

    return pd_multi_index_nlevels_impl


@sdc_overload_attribute(MultiIndexType, 'name')
def pd_multi_index_name_overload(self):
    if not isinstance(self, MultiIndexType):
        return None

    is_named_index = self.is_named

    def pd_multi_index_name_impl(self):
        if is_named_index == True:  # noqa
            return self._name
        else:
            return None

    return pd_multi_index_name_impl


@sdc_overload_attribute(MultiIndexType, 'names')
def pd_multi_index_names_overload(self):
    if not isinstance(self, MultiIndexType):
        return None

    def pd_multi_index_names_impl(self):
        levels_names = sdc_tuple_map(
            lambda x: x.name,
            self._levels
        )

        # this exploits undesired side-effect of literal_unroll - type-unification
        # of resulting list dtype that will be types.Optional(types.unicode_type)
        # as using typed.List of Optional values currently fails to compile
        res = []
        for i in literal_unroll(levels_names):
            res.append(i)
        return res

    return pd_multi_index_names_impl


# FIXME: move to a different file?
def cat_array_equal(A, codes_A, B, codes_B):
    pass


@sdc_overload(cat_array_equal)
def sdc_cat_array_equal_overload(A, codes_A, B, codes_B):

    def sdc_cat_array_equal_impl(A, codes_A, B, codes_B):
        if len(codes_A) != len(codes_B):
            return False

        # FIXME_Numba#5157: change to simple A == B when issue is resolved
        eq_res_size = len(codes_A)
        eq_res = np.empty(eq_res_size, dtype=types.bool_)
        for i in numba.prange(eq_res_size):
            eq_res[i] = A[codes_A[i]] == B[codes_B[i]]
        return np.all(eq_res)

    return sdc_cat_array_equal_impl


@intrinsic
def _multi_index_binop_helper(typingctx, self, other):
    """ This function gets two multi_index objects each represented as
    Tuple(levels) and Tuple(codes) and repacks these into Tuple of following
    elements (self_level_0, self_codes_0, other_level_0, other_codes_0), etc
    """

    nlevels = len(self.levels)
    if not len(self.levels) == len(other.levels):
        assert True, "Cannot flatten MultiIndex of different nlevels"

    elements_types = zip(self.levels, self.codes, other.levels, other.codes)
    ret_type = types.Tuple([types.Tuple.from_types(x) for x in elements_types])

    def codegen(context, builder, sig, args):
        self_val, other_val = args

        self_ctinfo = cgutils.create_struct_proxy(self)(
                    context, builder, value=self_val)
        self_levels = self_ctinfo.levels
        self_codes = self_ctinfo.codes

        other_ctinfo = cgutils.create_struct_proxy(other)(
                    context, builder, value=other_val)
        other_levels = other_ctinfo.levels
        other_codes = other_ctinfo.codes

        ret_tuples = []
        for i in range(nlevels):
            self_level_i = builder.extract_value(self_levels, i)
            self_codes_i = builder.extract_value(self_codes, i)
            other_level_i = builder.extract_value(other_levels, i)
            other_codes_i = builder.extract_value(other_codes, i)

            ret_tuples.append(
                context.make_tuple(builder,
                                   ret_type[i],
                                   [self_level_i, self_codes_i, other_level_i, other_codes_i])
            )

            if context.enable_nrt:
                context.nrt.incref(builder, ret_type[i][0], self_level_i)
                context.nrt.incref(builder, ret_type[i][1], self_codes_i)
                context.nrt.incref(builder, ret_type[i][2], other_level_i)
                context.nrt.incref(builder, ret_type[i][3], other_codes_i)

        res = context.make_tuple(builder, ret_type, ret_tuples)
        return res

    return ret_type(self, other), codegen


@sdc_overload_method(MultiIndexType, 'equals')
def pd_multi_index_equals_overload(self, other):
    if not isinstance(self, MultiIndexType):
        return None

    _func_name = 'Method equals().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(other, MultiIndexType) and self.dtype is other.dtype):
        ty_checker.raise_exc(other, 'pandas MultiIndex', 'other')

    def pd_multi_index_equals_impl(self, other):

        if self.nlevels != other.nlevels:
            return False

        self_and_other_data = _multi_index_binop_helper(self, other)
        tup_levels_cmp_res = sdc_tuple_map(
            lambda x: cat_array_equal(*x),
            self_and_other_data,
        )

        # np.all is not supported for Tuples and below compiles a bit faster
        # than 'np.all(np.array(list(tup_levels_cmp_res)))'
        for cmp_res in tup_levels_cmp_res:
            if not cmp_res:
                return False
        return True

    return pd_multi_index_equals_impl


# FIXME: move to another file?
def _build_index_map(self):
    pass


@sdc_overload(_build_index_map)
def _build_index_map_ovld(self):

    indexer_dtype = self.dtype
    indexer_value_type = types.ListType(types.int64)

    def _build_index_map(self):
        indexer_map = Dict.empty(indexer_dtype, indexer_value_type)
        for i in range(len(self)):
            val = self[i]
            index_list = indexer_map.get(val, None)
            if index_list is None:
                indexer_map[val] = List.empty_list(types.int64)
                indexer_map[val].append(i)
            else:
                index_list.append(i)

        return indexer_map

    return _build_index_map


@sdc_overload(operator.contains)
def pd_multi_index_contains_overload(self, label):
    if not isinstance(self, MultiIndexType):
        return None

    _func_name = 'Method contains().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(label, (types.Tuple, types.UniTuple)) and self.dtype is label):
        ty_checker.raise_exc(label, 'tuple ', 'val')

    def pd_multi_index_contains_impl(self, label):

        # build indexer_map (should already been built in index ctor?)
        indexer_map = _build_index_map(self)
        res = label in indexer_map
        return res

    return pd_multi_index_contains_impl


@sdc_overload(operator.eq)
def pd_multi_index_eq_overload(self, other):

    _func_name = 'Operator eq.'

    self_is_multi_index = isinstance(self, MultiIndexType)
    other_is_multi_index = isinstance(other, MultiIndexType)
    both_are_multi_indexes = self_is_multi_index and other_is_multi_index
    if not (both_are_multi_indexes and check_types_comparable(self, other)
            or (self_is_multi_index and other is getattr(self, 'dtype', types.none))
            or (self is getattr(other, 'dtype', types.none) and other_is_multi_index)):
        raise TypingError('{} Not allowed for non comparable types. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def pd_multi_index_eq_impl(self, other):

        if both_are_multi_indexes == True:  # noqa
            self_size = len(self)
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")

            if self.nlevels != other.nlevels:
                res = np.zeros(self_size, dtype=types.bool_)
            else:
                res = np.empty(self_size, dtype=types.bool_)
                for i in prange(self_size):
                    res[i] = self[i] == other[i]

        elif self_is_multi_index == True:  # noqa
            self_size = len(self)
            res = np.empty(self_size, dtype=types.bool_)
            for i in prange(self_size):
                res[i] = self[i] == other

        else:
            other_size = len(other)
            res = np.empty(other_size, dtype=types.bool_)
            for i in prange(other_size):
                res[i] = self == other[i]

        return list(res)  # FIXME_Numba#5157: result must be np.array, remove list when Numba is fixed

    return pd_multi_index_eq_impl


@sdc_overload_method(MultiIndexType, 'ravel')
def pd_multi_index_ravel_overload(self, order='C'):
    if not isinstance(self, MultiIndexType):
        return None

    _func_name = 'Method ravel().'

    if not (isinstance(order, (types.Omitted, types.StringLiteral, types.UnicodeType)) or order == 'C'):
        raise TypingError('{} Unsupported parameters. Given order: {}'.format(_func_name, order))

    def pd_multi_index_ravel_impl(self, order='C'):
        # np.ravel argument order is not supported in Numba
        if order != 'C':
            raise ValueError(f"Unsupported value for argument 'order' (only default 'C' is supported)")

        return self.values

    return pd_multi_index_ravel_impl


@sdc_overload(operator.ne)
def pd_multi_index_ne_overload(self, other):

    _func_name = 'Operator ne.'

    self_is_multi_index = isinstance(self, MultiIndexType)
    other_is_multi_index = isinstance(other, MultiIndexType)
    both_are_multi_indexes = self_is_multi_index and other_is_multi_index
    if not (both_are_multi_indexes and check_types_comparable(self, other)
            or (self_is_multi_index and other is getattr(self, 'dtype', types.none))
            or (self is getattr(other, 'dtype', types.none) and other_is_multi_index)):
        raise TypingError('{} Not allowed for non comparable types. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def pd_multi_index_ne_impl(self, other):

        eq_res = np.asarray(self == other)  # FIXME_Numba#5157: remove np.asarray and return as list
        return list(~eq_res)

    return pd_multi_index_ne_impl


@lower_builtin(operator.is_, MultiIndexType, MultiIndexType)
def pd_multi_index_is_overload(context, builder, sig, args):

    ty_lhs, ty_rhs = sig.args
    if ty_lhs != ty_rhs:
        return cgutils.false_bit

    lhs, rhs = args
    lhs_ptr = builder.ptrtoint(lhs.operands[0], cgutils.intp_t)
    rhs_ptr = builder.ptrtoint(rhs.operands[0], cgutils.intp_t)
    return builder.icmp_signed('==', lhs_ptr, rhs_ptr)


@lower_builtin('getiter', MultiIndexType)
def impl_conc_dict_getiter(context, builder, sig, args):
    index_type, = sig.args
    index_val, = args

    it = context.make_helper(builder, index_type.iterator_type)
    it.parent = index_val
    zero = context.get_constant(types.intp, 0)
    it.state = cgutils.alloca_once_value(builder, zero)

    res = it._getvalue()
    return impl_ret_borrowed(context, builder, index_type.iterator_type, res)


@lower_builtin('iternext', MultiIndexIteratorType)
@iternext_impl(RefType.BORROWED)
def impl_iterator_iternext(context, builder, sig, args, result):
    iter_type, = sig.args
    iter_val, = args

    index_type = iter_type.parent
    it = context.make_helper(builder, iter_type, iter_val)

    nitems = context.compile_internal(
        builder,
        lambda index: len(index),
        signature(types.int64, index_type),
        [it.parent]
    )

    index = builder.load(it.state)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        element = context.compile_internal(
            builder,
            lambda index, i: index[i],
            signature(index_type.dtype, index_type, types.int64),
            [it.parent, index]
        )
        result.yield_(element)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, it.state)


@sdc_overload_method(MultiIndexType, 'reindex')
def pd_multi_index_reindex_overload(self, target, method=None, level=None, limit=None, tolerance=None):
    if not isinstance(self, MultiIndexType):
        return None

    _func_name = 'Method reindex().'
    if not isinstance(target, sdc_pandas_index_types):
        raise SDCLimitation(f"{_func_name} Unsupported parameter. Given 'target': {target}")

    if not check_types_comparable(self, target):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, target={}'.format(_func_name, self, target))

    # TO-DO: check why compilation time is more than 10 seconds
    def pd_multi_index_reindex_impl(self, target, method=None, level=None, limit=None, tolerance=None):
        return sdc_indexes_reindex(self, target=target, method=method, level=level, tolerance=tolerance)

    return pd_multi_index_reindex_impl


# TO-DO: seems like this can be removed when indexes have map_positions property
@register_jitable
def _appender_build_map(index1, index2):
    res = {}
    for i, val in enumerate(index1):
        if val not in res:
            res[val] = i

    k, count = i, len(res)
    while k < i + len(index2):
        val = index2[k - i]
        if val not in res:
            res[val] = count
            count += 1
        k += 1

    return res


def _multi_index_append_level(A, codes_A, B, codes_B):
    pass


@sdc_overload(_multi_index_append_level)
def _multi_index_append_overload(A, codes_A, B, codes_B):

    def _multi_index_append_impl(A, codes_A, B, codes_B):

        appender_map = _appender_build_map(A, B)
        res_size = len(codes_A) + len(codes_B)
        res_level = fix_df_index(
            list(appender_map.keys())
        )

        res_codes = np.empty(res_size, dtype=np.int64)
        A_size = len(codes_A)
        for i in prange(res_size):
            if i < A_size:
                res_codes[i] = codes_A[i]
            else:
                res_codes[i] = appender_map[B[codes_B[i - A_size]]]

        return (res_level, res_codes)

    return _multi_index_append_impl


@sdc_overload_method(MultiIndexType, 'append')
def pd_multi_index_append_overload(self, other):
    if not isinstance(self, MultiIndexType):
        return None

    _func_name = 'Method append().'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(other, MultiIndexType)):
        ty_checker.raise_exc(other, 'pandas MultiIndex', 'other')

    if not check_types_comparable(self, other):
        raise TypingError('{} Not allowed for non comparable indexes. \
        Given: self={}, other={}'.format(_func_name, self, other))

    def pd_multi_index_append_impl(self, other):

        self_and_other_data = _multi_index_binop_helper(self, other)
        tup_append_level_res = sdc_tuple_map(
            lambda x: _multi_index_append_level(*x),
            self_and_other_data
        )

        new_levels, new_codes = sdc_tuple_unzip(tup_append_level_res)
        return pd.MultiIndex(
            levels=new_levels,
            codes=new_codes
        )

    return pd_multi_index_append_impl


def _multi_index_create_level(index_data, name):
    pass


@sdc_overload(_multi_index_create_level)
def _multi_index_create_level_ovld(index_data, name):

    def _multi_index_create_level_impl(index_data, name):
        index = fix_df_index(index_data)
        return sdc_indexes_rename(index, name)
    return _multi_index_create_level_impl


def _multi_index_create_levels_and_codes(level_data, codes_data, name):
    pass


@sdc_overload(_multi_index_create_levels_and_codes)
def _multi_index_create_levels_and_codes_ovld(level_data, codes_data, name):

    def _multi_index_create_levels_and_codes_impl(level_data, codes_data, name):
        level_data_fixed = fix_df_index(level_data)
        level = sdc_indexes_rename(level_data_fixed, name)
        codes = fix_df_array(codes_data)

        # to avoid additional overload make data verification checks inplace
        # these checks repeat those in MultiIndex::_verify_integrity
        if len(codes) and np.max(codes) >= len(level):
            raise ValueError(
                "On one of the levels code max >= length of level. "
                "NOTE: this index is in an inconsistent state"
            )
        if len(codes) and np.min(codes) < -1:
            raise ValueError(
                "On one of the levels code value < -1")

        # TO-DO: support is_unique for all indexes and use it here
        indexer_map = _build_index_map(level)
        if len(level) != len(indexer_map):
            raise ValueError("Level values must be unique")

        return (level, codes)

    return _multi_index_create_levels_and_codes_impl


@register_jitable
def next_codes_info(level_info, cumprod_list):
    _, codes = level_info
    cumprod_list.append(cumprod_list[-1] * len(codes))
    return codes, cumprod_list[-1]


@register_jitable
def next_codes_array(stats, res_size):
    codes_pattern, factor = stats
    span_i = res_size // factor                             # tiles whole array
    repeat_i = res_size // (len(codes_pattern) * span_i)    # repeats each element
    return np.array(list(np.repeat(codes_pattern, span_i)) * repeat_i)


def factorize_level(level):
    pass


@sdc_overload(factorize_level)
def factorize_level_ovld(level):

    level_dtype = level.dtype

    def factorize_level_impl(level):
        unique_labels = List.empty_list(level_dtype)
        res_size = len(level)
        codes = np.empty(res_size, types.int64)
        if not res_size:
            return unique_labels, codes

        indexer_map = Dict.empty(level_dtype, types.int64)
        for i in range(res_size):
            val = level[i]
            _code = indexer_map.get(val, -1)
            if _code == -1:
                new_code = len(unique_labels)
                indexer_map[val] = new_code
                unique_labels.append(val)
            else:
                new_code = _code

            codes[i] = new_code

        return unique_labels, codes

    return factorize_level_impl


def _make_level_unique(index):
    pass


@sdc_overload(_make_level_unique)
def _make_level_unique_ovld(index):

    def _make_level_unique_impl(index):
        indexer_map = _build_index_map(index)
        return list(indexer_map.keys())

    return _make_level_unique_impl


@sdc_overload_method(MultiIndexTypeRef, 'from_product', prefer_literal=False)
def pd_multi_index_from_product_overload(cls, iterables, sortorder=None, names=None):
    if cls.instance_type is not MultiIndexType:
        return

    _func_name = f'Method from_product()'
    valid_levels_data_types = sdc_pandas_index_types + sdc_pandas_df_column_types + (types.List, types.ListType)
    ty_checker = TypeChecker(_func_name)
    if not (isinstance(iterables, (types.List, types.ListType, types.UniTuple))
            and isinstance(iterables.dtype, valid_levels_data_types)
            or isinstance(iterables, types.Tuple)
            and all(map(lambda x: isinstance(x, valid_levels_data_types), iterables))):
        ty_checker.raise_exc(iterables, 'list or tuple of tuples ', 'iterables')

    if not (isinstance(sortorder, (types.Omitted, types.NoneType)) or sortorder is None):
        raise TypingError('{} Unsupported parameters. Given sortorder: {}'.format(_func_name, sortorder))

    if not (isinstance(names, (types.Omitted, types.NoneType)) or names is None):
        raise TypingError('{} Unsupported parameters. Given names: {}'.format(_func_name, names))

    def pd_multi_index_from_product_impl(cls, iterables, sortorder=None, names=None):

        # TO-DO: support indexes.unique() method and use it here
        levels_factorized = sdc_tuple_map(
            factorize_level,
            iterables
        )

        levels_names = sdc_tuple_map(
            sdc_indexes_get_name,
            iterables
        )

        index_levels = sdc_tuple_map(
            lambda x: fix_df_index(list(x[0])),
            levels_factorized
        )

        temp_cumprod_sizes = [1, ]
        codes_info = sdc_tuple_map(
            next_codes_info,
            levels_factorized,
            temp_cumprod_sizes
        )

        res_index_size = temp_cumprod_sizes[-1]
        index_codes = sdc_tuple_map(
            next_codes_array,
            codes_info,
            res_index_size
        )

        res = sdc_pandas_multi_index_ctor(
            index_levels,
            index_codes,
            name=levels_names
        )

        return res

    return pd_multi_index_from_product_impl


def _make_level_dict(index):
    pass


@sdc_overload(_make_level_dict)
def _make_level_dict_ovld(index):

    index_type = index

    def _make_level_dict_impl(index):
        return Dict.empty(index_type, types.int64)

    return _make_level_dict_impl


def _multi_index_get_new_code(level, val):

    _code = level.get(val, -1)
    if _code == -1:
        res = len(level)
        level[val] = res
    else:
        res = _code

    return types.int64(res)


def _multi_index_set_new_code(codes, new_code, i):
    codes[i] = new_code


@intrinsic
def _multi_index_append_value(typingctx, val, levels, codes, idx):

    nlevels = len(val)
    if not (nlevels == len(levels) and nlevels == len(codes)):
        assert True, f"Cannot append MultiIndex value to existing codes/levels.\n" \
                     f"Given: val={val}, levels={levels}, codes={codes}"

    def codegen(context, builder, sig, args):
        index_val, levels_val, codes_val, idx_val = args

        for i in range(nlevels):
            label = builder.extract_value(index_val, i)
            level_i = builder.extract_value(levels_val, i)
            codes_i = builder.extract_value(codes_val, i)

            new_code = context.compile_internal(
                builder,
                _multi_index_get_new_code,
                signature(types.int64, levels[i], val[i]),
                [level_i, label]
            )
            context.compile_internal(
                builder,
                _multi_index_set_new_code,
                signature(types.none, codes[i], types.int64, idx),
                [codes_i, new_code, idx_val]
            )

    return types.none(val, levels, codes, idx), codegen


@sdc_overload_method(MultiIndexTypeRef, 'from_tuples', prefer_literal=False)
def pd_multi_index_from_tuples_overload(cls, iterables):
    if cls.instance_type is not MultiIndexType:
        return

    _func_name = f'Method from_tuples()'
    ty_checker = TypeChecker(_func_name)

    if not (isinstance(iterables, (types.List, types.ListType))
            and isinstance(iterables.dtype, (types.Tuple, types.UniTuple))):
        ty_checker.raise_exc(iterables, f'list of tuples', 'iterables')

    def pd_multi_index_type_from_tuples_impl(cls, iterables):

        index_size = len(iterables)
        if not index_size:
            raise TypeError("Cannot infer number of levels from empty list")

        example_value = iterables[0]
        levels_dicts = sdc_tuple_map(
            _make_level_dict,
            example_value
        )
        index_codes = sdc_tuple_map(
            lambda _, size: np.empty(size, dtype=types.int64),
            example_value,
            index_size
        )

        for i in range(index_size):
            val = iterables[i]
            _multi_index_append_value(val, levels_dicts, index_codes, i)

        index_levels = sdc_tuple_map(
            lambda x: list(x.keys()),
            levels_dicts
        )

        res = pd.MultiIndex(
            levels=index_levels,
            codes=index_codes,
        )
        return res

    return pd_multi_index_type_from_tuples_impl

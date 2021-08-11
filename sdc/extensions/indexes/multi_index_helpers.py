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

from numba import types, prange
from numba.core import cgutils
from numba.core.typing.templates import signature
from numba.core.extending import (intrinsic, register_jitable, )
from numba.typed import Dict, List

from sdc.utilities.utils import sdc_overload
from sdc.hiframes.api import fix_df_array, fix_df_index
from sdc.extensions.indexes.indexes_generic import (
        sdc_indexes_rename,
        sdc_indexes_build_map_positions,
    )


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


# TO-DO: seems like this can be refactored when indexes have cached map_positions property
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
def _multi_index_append_level_overload(A, codes_A, B, codes_B):

    def _multi_index_append_level_impl(A, codes_A, B, codes_B):

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

    return _multi_index_append_level_impl


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
        indexer_map = sdc_indexes_build_map_positions(level)
        if len(level) != len(indexer_map):
            raise ValueError("Level values must be unique")

        return (level, codes)

    return _multi_index_create_levels_and_codes_impl


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


def _multi_index_alloc_level_dict(index):
    pass


@sdc_overload(_multi_index_alloc_level_dict)
def _make_level_dict_ovld(index):

    index_type = index

    def _make_level_dict_impl(index):
        return Dict.empty(index_type, types.int64)

    return _make_level_dict_impl


@intrinsic
def _multi_index_from_tuples_helper(typingctx, val, levels, codes, idx):

    nlevels = len(val)
    if not (nlevels == len(levels) and nlevels == len(codes)):
        assert True, f"Cannot append MultiIndex value to existing codes/levels.\n" \
                     f"Given: val={val}, levels={levels}, codes={codes}"

    def _get_code_for_label(seen_labels, label):

        _code = seen_labels.get(label, -1)
        if _code != -1:
            return _code

        res = len(seen_labels)
        seen_labels[label] = res
        return types.int64(res)

    def _set_code_by_position(codes, new_code, i):
        codes[i] = new_code

    def codegen(context, builder, sig, args):
        index_val, levels_val, codes_val, idx_val = args

        for i in range(nlevels):
            label = builder.extract_value(index_val, i)
            level_i = builder.extract_value(levels_val, i)
            codes_i = builder.extract_value(codes_val, i)

            new_code = context.compile_internal(
                builder,
                _get_code_for_label,
                signature(types.int64, levels[i], val[i]),
                [level_i, label]
            )
            context.compile_internal(
                builder,
                _set_code_by_position,
                signature(types.none, codes[i], types.int64, idx),
                [codes_i, new_code, idx_val]
            )

    return types.none(val, levels, codes, idx), codegen

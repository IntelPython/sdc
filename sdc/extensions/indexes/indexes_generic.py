# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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
import pandas as pd

from numba import types
from numba.typed import Dict
from numba.typed.typedobjectutils import _nonoptional

from sdc.utilities.sdc_typing_utils import sdc_pandas_index_types, sdc_old_index_types
from sdc.datatypes.indexes import *
from sdc.utilities.utils import sdc_overload_method, sdc_overload
from sdc.utilities.sdc_typing_utils import (
                        find_index_common_dtype,
                        sdc_indexes_wo_values_cache,
                    )
from sdc.hiframes.api import fix_df_index
from sdc.functions import numpy_like
from sdc.datatypes.common_functions import _sdc_internal_join


def sdc_numeric_indexes_equals(left, right):
    pass


@sdc_overload(sdc_numeric_indexes_equals)
def sdc_numeric_indexes_equals_ovld(left, right):

    if not (isinstance(left, sdc_pandas_index_types)
            or isinstance(right, sdc_pandas_index_types)):
        return None

    convert_A = not isinstance(left, types.Array)
    convert_B = not isinstance(right, types.Array)

    def sdc_numeric_indexes_equals_impl(left, right):
        left = left.values if convert_A == True else left  # noqa
        right = right.values if convert_B == True else right  # noqa

        return numpy_like.array_equal(left, right)

    return sdc_numeric_indexes_equals_impl


def sdc_indexes_attribute_dtype(self):
    pass


@sdc_overload(sdc_indexes_attribute_dtype)
def sdc_indexes_attribute_dtype_ovld(self):

    if not isinstance(self, sdc_pandas_index_types):
        return None

    index_dtype = self.data.dtype

    def sdc_indexes_attribute_dtype_impl(self):
        return index_dtype

    return sdc_indexes_attribute_dtype_impl


def sdc_indexes_operator_eq(self):
    pass


@sdc_overload(sdc_indexes_operator_eq)
def sdc_indexes_operator_eq_ovld(self, other):

    # TO-DO: this is for numeric indexes only now, extend to string-index when it's added
    use_self_values = isinstance(self, sdc_pandas_index_types) and not isinstance(self, types.Array)
    use_other_values = isinstance(other, sdc_pandas_index_types) and not isinstance(other, types.Array)
    one_operand_is_scalar = isinstance(self, types.Number) or isinstance(other, types.Number)

    def sdc_indexes_operator_eq_impl(self, other):

        if one_operand_is_scalar == False:  # noqa
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")

        left = self.values if use_self_values == True else self  # noqa
        right = other.values if use_other_values == True else other  # noqa
        return list(left == right)  # FIXME_Numba#5157: result must be np.array, remove list when Numba is fixed

    return sdc_indexes_operator_eq_impl


def sdc_indexes_reindex(self, target):
    pass


@sdc_overload(sdc_indexes_reindex)
def pd_indexes_reindex_overload(self, target, method=None, level=None, limit=None, tolerance=None):

    index_dtype = self.dtype
    def pd_indexes_index_reindex_impl(self, target, method=None, level=None, limit=None, tolerance=None):
        """ Simplified version of pandas.core.index.base.reindex """

        if (self is target or self.equals(target)):
            return target, None

        # build a dict of 'self' index values to their positions:
        map_index_to_position = Dict.empty(
            key_type=index_dtype,
            value_type=types.int32
        )

        # TO-DO: needs concurrent hash map
        for i, value in enumerate(self):
            if value in map_index_to_position:
                raise ValueError("cannot reindex from a duplicate axis")
            else:
                map_index_to_position[value] = i

        res_size = len(target)
        indexer = np.empty(res_size, dtype=np.int64)
        for i in numba.prange(res_size):
            val = target[i]
            if val in map_index_to_position:
                indexer[i] = map_index_to_position[val]
            else:
                indexer[i] = -1

        return target, indexer

    return pd_indexes_index_reindex_impl


def sdc_indexes_join_outer(left, right):
    pass


@sdc_overload(sdc_indexes_join_outer, jit_options={'parallel': False})
def pd_indexes_join_overload(left, right):
    """Function for joining arrays left and right in a way similar to pandas.join 'outer' algorithm"""

    # check that both operands are of types used for representing Pandas indexes
    if not (isinstance(left, sdc_pandas_index_types) and isinstance(right, sdc_pandas_index_types)
            and not isinstance(left, EmptyIndexType)
            and not isinstance(right, EmptyIndexType)):
        return None

    # for index types with dtype=int64 resulting index should be of Int64Index type
    if (isinstance(left, (PositionalIndexType, RangeIndexType, Int64IndexType))
          and isinstance(right, (PositionalIndexType, RangeIndexType, Int64IndexType))):

        def _convert_to_arrays_impl(left, right):

            if (left is right or left.equals(right)):
                return pd.Int64Index(left.values), None, None

            joined_data, indexer1, indexer2 = _sdc_internal_join(left.values, right.values)
            return pd.Int64Index(joined_data), indexer1, indexer2

        return _convert_to_arrays_impl

    # for joining with deprecated types.Array indexes (e.g. representing UInt64Index)
    # resulting index will be of numpy array type. TO-DO: remove once pd.Index overload
    # is supported and all indexes are represented with distinct types
    else:
        convert_left = isinstance(left, (PositionalIndexType, RangeIndexType, Int64IndexType))
        convert_right = isinstance(right, (PositionalIndexType, RangeIndexType, Int64IndexType))
        index_dtypes_match, res_index_dtype = find_index_common_dtype(left, right)
        def pd_indexes_join_array_indexes_impl(left, right):

            _left = left.values if convert_left == True else left  # noqa
            _right = right.values if convert_right == True else right  # noqa
            if (_left is _right
                    or numpy_like.array_equal(_left, _right)):
                if index_dtypes_match == False:  # noqa
                    joined_index = numpy_like.astype(_left, res_index_dtype)
                else:
                    joined_index = _left
                return joined_index, None, None

            return _sdc_internal_join(_left, _right)

        return pd_indexes_join_array_indexes_impl

    return None


def sdc_fix_indexes_join(joined, indexer1, indexer2):
    pass


@sdc_overload(sdc_fix_indexes_join)
def pd_fix_indexes_join_overload(joined, indexer1, indexer2):
    """ Wraps pandas index.join() into new function that returns indexers as arrays and not optional(array) """

    # This function is simply a workaround for problem with parfor lowering
    # broken by indexers typed as types.Optional(Array) - FIXME_Numba#XXXX: remove it
    # in all places whne parfor issue is fixed
    def pd_fix_indexes_join_impl(joined, indexer1, indexer2):
        if indexer1 is not None:
            _indexer1 = _nonoptional(indexer1)
        else:
            _indexer1 = np.arange(len(joined))

        if indexer2 is not None:
            _indexer2 = _nonoptional(indexer2)
        else:
            _indexer2 = _indexer1

        return joined, _indexer1, _indexer2

    return pd_fix_indexes_join_impl


def sdc_unify_index_types(left, right):
    pass


@sdc_overload(sdc_unify_index_types)
def sdc_unify_index_types_overload(left, right):
    """ For equal indexes of different dtypes produced index of common dtype """

    index_dtypes_match, numba_index_common_dtype = find_index_common_dtype(left, right)
    is_left_index_cached = not isinstance(left, sdc_indexes_wo_values_cache)
    is_left_index_array = isinstance(left, types.Array)
    is_right_index_cached = not isinstance(right, sdc_indexes_wo_values_cache)
    is_right_index_array = isinstance(right, types.Array)

    def sdc_unify_index_types_impl(left, right):
        if index_dtypes_match == True:  # noqa
            return left
        else:
            if is_left_index_cached == True:  # noqa
                index_data = left.values if is_left_index_array == False else left  # noqa
            elif is_right_index_cached == True:  # noqa
                index_data = right.values if is_right_index_array == False else right  # noqa
            else:
                # using numpy_like.astype but not index.astype since latter works differently
                index_data = numpy_like.astype(left, numba_index_common_dtype)

        return fix_df_index(index_data)

    return sdc_unify_index_types_impl


@sdc_overload(np.array)
def sdc_np_array_overload(A):
    """ Overload provides np.array(A) implementations for internal pandas index types """

    if not (isinstance(A, sdc_pandas_index_types)
            and not isinstance(A, sdc_old_index_types)):
        return None

    if isinstance(A, PositionalIndexType):
        return lambda A: np.arange(len(A))

    if isinstance(A, RangeIndexType):
        return lambda A: np.arange(A.start, A.stop, A.step)

    if isinstance(A, Int64IndexType):
        return lambda A: A._data

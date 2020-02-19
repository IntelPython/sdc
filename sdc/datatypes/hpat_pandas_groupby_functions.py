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

"""
| :class:`pandas.DataFrame.GroupBy` functions and operators implementations in Intel SDC
"""

import pandas
import numba
import numpy
import sdc

from numba import types
from numba import cgutils
from numba.extending import intrinsic
from numba.targets.registry import cpu_target
from numba.typed import List, Dict
from numba.typing import signature

from sdc.datatypes.common_functions import sdc_arrays_argsort, _sdc_asarray, _sdc_take
from sdc.datatypes.hpat_pandas_groupby_types import DataFrameGroupByType
from sdc.utilities.sdc_typing_utils import TypeChecker, kwsparams2list, sigparams2list
from sdc.utilities.utils import sdc_overload_method, sdc_overload_attribute
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data
from sdc.hiframes.pd_series_type import SeriesType


@intrinsic
def init_dataframe_groupby(typingctx, parent, column_id, data, sort):

    def codegen(context, builder, signature, args):
        parent_val, column_id_val, data_val, sort_val = args
        # create series struct and store values
        groupby_obj = cgutils.create_struct_proxy(
            signature.return_type)(context, builder)
        groupby_obj.parent = parent_val
        groupby_obj.col_id = column_id_val
        groupby_obj.data = data_val
        groupby_obj.sort = sort_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], parent_val)
            context.nrt.incref(builder, signature.args[1], column_id_val)
            context.nrt.incref(builder, signature.args[2], data_val)

        return groupby_obj._getvalue()

    ret_typ = DataFrameGroupByType(parent, column_id)
    sig = signature(ret_typ, parent, column_id, data, sort)
    return sig, codegen


def _sdc_pandas_groupby_generic_func_codegen(func_name, columns, func_params, defaults, impl_params):

    all_params_as_str = ', '.join(sigparams2list(func_params, defaults))
    extra_impl_params = ', '.join(kwsparams2list(impl_params))

    groupby_obj = f'{func_params[0]}'
    df = f'{groupby_obj}._parent'
    groupby_dict = f'{groupby_obj}._data'
    groupby_param_sort = f'{groupby_obj}._sort'
    column_names, column_ids = tuple(zip(*columns))

    func_lines = [
        f'def _groupby_{func_name}_impl({all_params_as_str}):',
        f'  group_keys = _sdc_asarray([key for key in {groupby_dict}])',
        f'  res_index_len = len(group_keys)',
        f'  if {groupby_param_sort}:',
        f'    argsorted_index = sdc_arrays_argsort(group_keys, kind=\'mergesort\')',
    ]

    # TODO: remove conversion from Numba typed.List to reflected one while creating group_arr_{i}
    func_lines.extend(['\n'.join([
        f'  result_data_{i} = numpy.empty(res_index_len, dtype=res_arrays_dtypes[{i}])',
        f'  for j in numba.prange(res_index_len):',
        f'    column_data_{i} = get_dataframe_data({df}, {column_ids[i]})',
        f'    group_arr_{i} = _sdc_take(column_data_{i}, list({groupby_dict}[group_keys[j]]))',
        f'    group_series_{i} = pandas.Series(group_arr_{i})',
        f'    idx = argsorted_index[j] if {groupby_param_sort} else j',
        f'    result_data_{i}[idx] = group_series_{i}.{func_name}({extra_impl_params})',
    ]) for i in range(len(columns))])

    data = ', '.join(f'\'{column_names[i]}\': result_data_{i}' for i in range(len(columns)))
    func_lines.extend(['\n'.join([
        f'  if {groupby_param_sort}:',
        f'    res_index = _sdc_take(group_keys, argsorted_index)',
        f'  else:',
        f'    res_index = group_keys',
        f'  return pandas.DataFrame({{{data}}}, index=res_index)'
    ])])

    func_text = '\n'.join(func_lines)
    global_vars = {'pandas': pandas,
                   'numpy': numpy,
                   'numba': numba,
                   '_sdc_asarray': _sdc_asarray,
                   '_sdc_take': _sdc_take,
                   'sdc_arrays_argsort': sdc_arrays_argsort,
                   'get_dataframe_data': get_dataframe_data}

    return func_text, global_vars


series_method_to_func = {
    'count': lambda S: S.count(),
    'max': lambda S: S.max(),
    'mean': lambda S: S.mean(),
    'median': lambda S: S.median(),
    'min': lambda S: S.min(),
    'prod': lambda S: S.prod(),
    'std': lambda S: S.std(),
    'sum': lambda S: S.sum(),
    'var': lambda S: S.var()
}


def _groupby_resolve_impl_func_type(series_dtype, method_name):
    """ Used for typing result value of functions implementing groupby methods,
        assuming that these implementation call method of Series.
    """
    ty_series = SeriesType(series_dtype)
    jitted_func = numba.njit(series_method_to_func[method_name])
    return cpu_target.typing_context.resolve_function_type(jitted_func, (ty_series, ), {})


def sdc_pandas_groupby_apply_func(self, func_name, func_args, defaults=None, impl_args=None):

    defaults = defaults if defaults else {}
    impl_args = impl_args if impl_args else {}

    df_column_types = self.parent.data
    df_column_names = self.parent.columns
    by_column_id = self.col_id.literal_value
    subject_columns = [(name, i) for i, name in enumerate(df_column_names) if i != by_column_id]

    # resolve types of result dataframe columns
    res_arrays_dtypes = tuple(
        _groupby_resolve_impl_func_type(
            ty_arr.dtype, func_name
            ).return_type for i, ty_arr in enumerate(df_column_types) if i != by_column_id)

    groupby_func_name = f'_groupby_{func_name}_impl'
    func_text, global_vars = _sdc_pandas_groupby_generic_func_codegen(
        func_name, subject_columns, func_args, defaults, impl_args)

    # capture result column types into generated func context
    global_vars['res_arrays_dtypes'] = res_arrays_dtypes

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    _groupby_method_impl = loc_vars[groupby_func_name]

    return _groupby_method_impl


@sdc_overload_method(DataFrameGroupByType, 'count')
def sdc_pandas_dataframe_groupby_count(self):

    method_name = 'GroupBy.count().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    method_args = ['self']
    applied_func_name = 'count'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args)


@sdc_overload_method(DataFrameGroupByType, 'max')
def sdc_pandas_dataframe_groupby_max(self):

    method_name = 'GroupBy.max().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    method_args = ['self']
    applied_func_name = 'max'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args)


@sdc_overload_method(DataFrameGroupByType, 'mean')
def sdc_pandas_dataframe_groupby_mean(self, *args):

    method_name = 'GroupBy.mean().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    method_args = ['self', '*args']
    applied_func_name = 'mean'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args)


@sdc_overload_method(DataFrameGroupByType, 'median')
def sdc_pandas_dataframe_groupby_median(self):

    method_name = 'GroupBy.median().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    method_args = ['self']
    applied_func_name = 'median'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args)


@sdc_overload_method(DataFrameGroupByType, 'min')
def sdc_pandas_dataframe_groupby_min(self):

    method_name = 'GroupBy.min().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    method_args = ['self']
    applied_func_name = 'min'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args)


@sdc_overload_method(DataFrameGroupByType, 'prod')
def sdc_pandas_dataframe_groupby_prod(self):

    method_name = 'GroupBy.prod().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    method_args = ['self']
    applied_func_name = 'prod'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args)


@sdc_overload_method(DataFrameGroupByType, 'std')
def sdc_pandas_dataframe_groupby_std(self, ddof=1, *args):

    method_name = 'GroupBy.std().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    if not isinstance(ddof, (types.Omitted, int, types.Integer)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    method_args = ['self', 'ddof', '*args']
    default_values = {'ddof': 1}
    impl_used_params = {'ddof': 'ddof'}

    applied_func_name = 'std'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args, default_values, impl_used_params)


@sdc_overload_method(DataFrameGroupByType, 'sum')
def sdc_pandas_dataframe_groupby_sum(self):

    method_name = 'GroupBy.sum().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    method_args = ['self']
    applied_func_name = 'sum'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args)


@sdc_overload_method(DataFrameGroupByType, 'var')
def sdc_pandas_dataframe_groupby_var(self, ddof=1, *args):

    method_name = 'GroupBy.var().'
    ty_checker = TypeChecker(method_name)
    ty_checker.check(self, DataFrameGroupByType)

    if not isinstance(ddof, (types.Omitted, int, types.Integer)):
        ty_checker.raise_exc(ddof, 'int', 'ddof')

    method_args = ['self', 'ddof', '*args']
    default_values = {'ddof': 1}
    impl_used_params = {'ddof': 'ddof'}

    applied_func_name = 'var'
    return sdc_pandas_groupby_apply_func(self, applied_func_name, method_args, default_values, impl_used_params)

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


import numpy as np
import numba
from numba import types
from numba.extending import (models, register_model, lower_cast, infer_getattr,
                             type_callable, infer, overload, make_attribute_wrapper, intrinsic,
                             lower_builtin, overload_method)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
                                    AttributeTemplate, bound_function)
import sdc
from sdc.hiframes.pd_series_ext import (SeriesType, _get_series_array_type,
                                         arr_to_series_type)
from sdc.hiframes.pd_dataframe_ext import DataFrameType
from sdc.hiframes.aggregate import get_agg_func


class DataFrameGroupByType(types.Type):  # TODO: IterableType over groups
    """Temporary type class for DataFrameGroupBy objects before transformation
    to aggregate node.
    """

    def __init__(self, df_type, keys, selection, as_index, explicit_select=False):

        self.df_type = df_type
        self.keys = keys
        self.selection = selection
        self.as_index = as_index
        self.explicit_select = explicit_select

        super(DataFrameGroupByType, self).__init__(
            name="DataFrameGroupBy({}, {}, {}, {}, {})".format(
                df_type, keys, selection, as_index, explicit_select))

    def copy(self):
        # XXX is copy necessary?
        return DataFrameGroupByType(self.df_type, self.keys, self.selection,
                                    self.as_index, self.explicit_select)


# dummy model since info is kept in type
# TODO: add df object to allow control flow?
register_model(DataFrameGroupByType)(models.OpaqueModel)


@overload_method(DataFrameType, 'groupby')
def df_groupby_overload(df, by=None, axis=0, level=None, as_index=True,
                        sort=True, group_keys=True, squeeze=False, observed=False):

    if by is None:
        raise ValueError("groupby 'by' argument required")

    def _impl(df, by=None, axis=0, level=None, as_index=True,
              sort=True, group_keys=True, squeeze=False, observed=False):
        return sdc.hiframes.pd_groupby_ext.groupby_dummy(df, by, as_index)

    return _impl


# a dummy groupby function that will be replace in dataframe_pass
def groupby_dummy(df, by, as_index):
    return 0


@infer_global(groupby_dummy)
class GroupbyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, by, as_index = args

        if isinstance(by, types.StringLiteral):
            keys = (by.literal_value,)
        elif hasattr(by, 'consts'):
            keys = by.consts

        selection = list(df.columns)
        for k in keys:
            selection.remove(k)

        if isinstance(as_index, sdc.utils.BooleanLiteral):
            as_index = as_index.literal_value
        else:
            # XXX as_index type is just bool when value not passed. Therefore,
            # we assume the default True value.
            # TODO: more robust fix or just check
            as_index = True

        out_typ = DataFrameGroupByType(
            df, keys, tuple(selection), as_index, False)
        return signature(out_typ, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(groupby_dummy, types.VarArg(types.Any))
def lower_groupby_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


@infer
class GetItemDataFrameGroupBy2(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        grpby, idx = args
        # df.groupby('A')['B', 'C']
        if isinstance(grpby, DataFrameGroupByType):
            if isinstance(idx, tuple):
                assert all(isinstance(c, str) for c in idx)
                selection = idx
            elif isinstance(idx, str):
                selection = (idx,)
            else:
                raise ValueError("invalid groupby selection {}".format(idx))
            ret_grp = DataFrameGroupByType(
                grpby.df_type, grpby.keys, selection, grpby.as_index, True)
            return signature(ret_grp, *args)


@infer_getattr
class DataframeGroupByAttribute(AttributeTemplate):
    key = DataFrameGroupByType

    def _get_agg_typ(self, grp, args, code):
        f_ir = numba.ir_utils.get_ir_of_code(
            {'np': np, 'numba': numba, 'sdc': sdc}, code)
        out_data = []
        out_columns = []
        # add key columns of not as_index
        if not grp.as_index:
            for k in grp.keys:
                out_columns.append(k)
                ind = grp.df_type.columns.index(k)
                out_data.append(grp.df_type.data[ind])

        # get output type for each selected column
        for c in grp.selection:
            out_columns.append(c)
            ind = grp.df_type.columns.index(c)
            data = grp.df_type.data[ind]
            _, out_dtype, _ = numba.typed_passes.type_inference_stage(
                self.context, f_ir, (data,), None)
            out_arr = _get_series_array_type(out_dtype)
            out_data.append(out_arr)

        out_res = DataFrameType(tuple(out_data), None, tuple(out_columns))
        # XXX output becomes series if single output and explicitly selected
        if len(grp.selection) == 1 and grp.explicit_select and grp.as_index:
            out_res = arr_to_series_type(out_data[0])
        return signature(out_res, *args)

    @bound_function("groupby.agg")
    def resolve_agg(self, grp, args, kws):
        code = args[0].literal_value.code
        return self._get_agg_typ(grp, args, code)

    @bound_function("groupby.aggregate")
    def resolve_aggregate(self, grp, args, kws):
        code = args[0].literal_value.code
        return self._get_agg_typ(grp, args, code)

    @bound_function("groupby.sum")
    def resolve_sum(self, grp, args, kws):
        func = get_agg_func(None, 'sum', None)
        return self._get_agg_typ(grp, args, func.__code__)

    @bound_function("groupby.count")
    def resolve_count(self, grp, args, kws):
        func = get_agg_func(None, 'count', None)
        return self._get_agg_typ(grp, args, func.__code__)

    @bound_function("groupby.mean")
    def resolve_mean(self, grp, args, kws):
        func = get_agg_func(None, 'mean', None)
        return self._get_agg_typ(grp, args, func.__code__)

    @bound_function("groupby.min")
    def resolve_min(self, grp, args, kws):
        func = get_agg_func(None, 'min', None)
        return self._get_agg_typ(grp, args, func.__code__)

    @bound_function("groupby.max")
    def resolve_max(self, grp, args, kws):
        func = get_agg_func(None, 'max', None)
        return self._get_agg_typ(grp, args, func.__code__)

    @bound_function("groupby.prod")
    def resolve_prod(self, grp, args, kws):
        func = get_agg_func(None, 'prod', None)
        return self._get_agg_typ(grp, args, func.__code__)

    @bound_function("groupby.var")
    def resolve_var(self, grp, args, kws):
        func = get_agg_func(None, 'var', None)
        return self._get_agg_typ(grp, args, func.__code__)

    @bound_function("groupby.std")
    def resolve_std(self, grp, args, kws):
        func = get_agg_func(None, 'std', None)
        return self._get_agg_typ(grp, args, func.__code__)


# a dummy pivot_table function that will be replace in dataframe_pass
def pivot_table_dummy(df, values, index, columns, aggfunc, _pivot_values):
    return 0


@infer_global(pivot_table_dummy)
class PivotTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, values, index, columns, aggfunc, _pivot_values = args

        if not (isinstance(values, types.StringLiteral)
                and isinstance(index, types.StringLiteral)
                and isinstance(columns, types.StringLiteral)):
            raise ValueError("pivot_table() only support string constants for"
                             "'values', 'index' and 'columns' arguments")

        values = values.literal_value
        index = index.literal_value
        columns = columns.literal_value

        # get output data type
        data = df.data[df.columns.index(values)]
        func = get_agg_func(None, aggfunc.literal_value, None)
        f_ir = numba.ir_utils.get_ir_of_code(
            {'np': np, 'numba': numba, 'sdc': sdc}, func.__code__)
        _, out_dtype, _ = numba.typed_passes.type_inference_stage(
            self.context, f_ir, (data,), None)
        out_arr_typ = _get_series_array_type(out_dtype)

        pivot_vals = _pivot_values.meta
        n_vals = len(pivot_vals)
        out_df = DataFrameType((out_arr_typ,) * n_vals, None, tuple(pivot_vals))

        return signature(out_df, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(pivot_table_dummy, types.VarArg(types.Any))
def lower_pivot_table_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


# a dummy crosstab function that will be replace in dataframe_pass
def crosstab_dummy(index, columns, _pivot_values):
    return 0


@infer_global(crosstab_dummy)
class CrossTabTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        index, columns, _pivot_values = args

        # TODO: support agg func other than frequency
        out_arr_typ = types.Array(types.int64, 1, 'C')

        pivot_vals = _pivot_values.meta
        n_vals = len(pivot_vals)
        out_df = DataFrameType((out_arr_typ,) * n_vals, None, tuple(pivot_vals))

        return signature(out_df, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(crosstab_dummy, types.VarArg(types.Any))
def lower_crosstab_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)

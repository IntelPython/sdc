import operator
import pandas as pd
import numpy as np
import numba
from numba import types, cgutils
from numba.extending import (models, register_model, lower_cast, infer_getattr,
    type_callable, infer, overload, make_attribute_wrapper, intrinsic,
    lower_builtin, overload_method)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
    AttributeTemplate, bound_function)
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed
import hpat
from hpat.hiframes.pd_series_ext import (SeriesType, _get_series_array_type,
    arr_to_series_type)
from hpat.str_ext import string_type
from hpat.hiframes.pd_dataframe_ext import DataFrameType
from hpat.hiframes.aggregate import get_agg_func


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
        return hpat.hiframes.pd_groupby_ext.groupby_dummy(df, by, as_index)

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

        if isinstance(as_index, hpat.utils.BooleanLiteral):
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
            {'np': np, 'numba': numba, 'hpat': hpat}, code)
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
            _, out_dtype, _ = numba.compiler.type_inference_stage(
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

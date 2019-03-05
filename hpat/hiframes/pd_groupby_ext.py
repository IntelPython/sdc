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
    arr_to_series_type, arr_to_series_type)
from hpat.str_ext import string_type
from hpat.hiframes.pd_dataframe_ext import DataFrameType


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

        # TODO: multi key
        if isinstance(by, types.StringLiteral):
            keys = (by.literal_value,)

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


@infer_global(operator.getitem)
class GetItemDataFrameGroupBy(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        grpby, idx = args
        # df.groupby('A')['B']
        if isinstance(grpby, DataFrameGroupByType):
            if isinstance(idx, types.StringLiteral):
                selection = (idx.literal_value,)
            ret_grp = DataFrameGroupByType(
                grpby.df_type, grpby.keys, selection, grpby.as_index, True)
            return signature(ret_grp, *args)

@infer_getattr
class DataframeGroupByAttribute(AttributeTemplate):
    key = DataFrameGroupByType

    @bound_function("groupby.agg")
    def resolve_agg(self, grp, args, kws):
        code = args[0].literal_value.code
        f_ir = numba.ir_utils.get_ir_of_code({'np': np}, code)
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
            in_typ = arr_to_series_type(data)
            _, out_dtype, _ = numba.compiler.type_inference_stage(
                self.context, f_ir, (in_typ,), None)
            out_arr = _get_series_array_type(out_dtype)
            out_data.append(out_arr)

        out_res = DataFrameType(tuple(out_data), None, tuple(out_columns))
        # XXX output becomes series if single output and explicitly selected
        if len(grp.selection) == 1 and grp.explicit_select and grp.as_index:
            out_res = arr_to_series_type(out_data[0])
        return signature(out_res, *args)

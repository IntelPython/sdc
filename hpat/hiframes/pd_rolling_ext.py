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
from hpat.hiframes.rolling import supported_rolling_funcs


class RollingType(types.Type):
    """Temporary type class for RollingType objects before transformation
    to rolling node.
    """

    def __init__(self, df_type, on, selection, explicit_select=False):

        self.df_type = df_type
        self.on = on
        self.selection = selection
        self.explicit_select = explicit_select

        super(RollingType, self).__init__(
            name="RollingType({}, {}, {}, {})".format(
                df_type, on, selection, explicit_select))

    def copy(self):
        # XXX is copy necessary?
        # TODO: key attribute?
        return RollingType(self.df_type, self.on, self.selection,
                           self.explicit_select)


# dummy model since info is kept in type
# TODO: add df object and win/center vals to allow control flow?
register_model(RollingType)(models.OpaqueModel)


@overload_method(DataFrameType, 'rolling')
def df_rolling_overload(df, window, min_periods=None, center=False,
                        win_type=None, on=None, axis=0, closed=None):

    def _impl(df, window, min_periods=None, center=False,
              win_type=None, on=None, axis=0, closed=None):
        return hpat.hiframes.pd_rolling_ext.rolling_dummy(
            df, window, center, on)

    return _impl


# a dummy rolling function that will be replace in dataframe_pass
def rolling_dummy(df, window, center, on):
    return 0


@infer_global(rolling_dummy)
class RollingTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, window, center, on = args

        if on == types.none:
            on = None
        else:
            if not isinstance(on, types.StringLiteral):
                raise ValueError(
                    "'on' argument to rolling() should be constant string")
            on = on.literal_value

        selection = list(df.columns)
        if on is not None:
            selection.remove(on)

        out_typ = RollingType(
            df, on, tuple(selection), False)
        return signature(out_typ, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(rolling_dummy, types.VarArg(types.Any))
def lower_rolling_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


@infer
class GetItemDataFrameRolling2(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        rolling, idx = args
        # df.rolling('A')['B', 'C']
        if isinstance(rolling, RollingType):
            if isinstance(idx, tuple):
                assert all(isinstance(c, str) for c in idx)
                selection = idx
            elif isinstance(idx, str):
                selection = (idx,)
            else:
                raise ValueError("invalid rolling selection {}".format(idx))
            ret_rolling = RollingType(
                rolling.df_type, rolling.on, selection, True)
            return signature(ret_rolling, *args)


@infer_getattr
class DataframeRollingAttribute(AttributeTemplate):
    key = RollingType

    def generic_resolve(self, rolling, func_name):
        if func_name not in supported_rolling_funcs:
            raise ValueError("only ({}) supported in rolling".format(
                ", ".join(supported_rolling_funcs)))
        template_key = 'rolling.' + func_name
        # output is always float64
        out_arr = types.Array(types.float64, 1, 'C')

        # TODO: handle Series case (explicit select)
        columns = rolling.selection

        # handle 'on' case
        if rolling.on is not None:
            columns = columns + (rolling.on,)
        # Pandas sorts the output column names _flex_binary_moment
        # line: res_columns = arg1.columns.union(arg2.columns)
        columns = tuple(sorted(columns))
        n_out_cols = len(columns)
        out_data = [out_arr] * n_out_cols
        if rolling.on is not None:
            # offset key's data type is preserved
            out_ind = columns.index(rolling.on)
            in_ind = rolling.df_type.columns.index(rolling.on)
            out_data[out_ind] = rolling.df_type.data[in_ind]
        out_typ = DataFrameType(tuple(out_data), None, columns)

        class MethodTemplate(AbstractTemplate):
            key = template_key

            def generic(self, args, kws):
                if func_name in ('cov', 'corr'):
                    if len(args) != 1:
                        raise ValueError("rolling {} requires one argument (other)".format(func_name))
                    # XXX pandas only accepts variable window cov/corr
                    # when both inputs have time index
                    if rolling.on is not None:
                        raise ValueError("variable window rolling {} not supported yet.".format(func_name))
                    # TODO: support variable window rolling cov/corr which is only
                    # possible in pandas with time index
                    other = args[0]
                    # df on df cov/corr returns common columns only (without
                    # pairwise flag)
                    # TODO: support pairwise arg
                    out_cols = tuple(sorted(set(columns) | set(other.columns)))
                    return signature(DataFrameType(
                        (out_arr,) * len(out_cols), None, out_cols), *args)
                return signature(out_typ, *args)

        return types.BoundFunction(MethodTemplate, rolling)

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

import numba
from numba.core import cgutils, types
from numba.core.rewrites import (register_rewrite, Rewrite)
from numba.core.ir_utils import (guard, find_callname)
from numba.core.ir import (Expr)
from numba.extending import overload
from numba.core.extending import intrinsic
from numba.core.typing import signature

from pandas import DataFrame
from sys import modules
from textwrap import dedent

from sdc.rewrites.ir_utils import (find_operations, is_dict,
                                   get_tuple_items, get_dict_items, remove_unused_recursively,
                                   get_call_parameters,
                                   declare_constant,
                                   import_function, make_call,
                                   insert_before)
from sdc.hiframes import pd_dataframe_ext as pd_dataframe_ext_module
from sdc.hiframes.pd_dataframe_type import DataFrameType, ColumnLoc
from sdc.hiframes.pd_dataframe_ext import get_structure_maps
from sdc.hiframes.api import fix_df_array, fix_df_index
from sdc.str_ext import string_type


@register_rewrite('before-inference')
class RewriteDataFrame(Rewrite):
    """
    Searches for calls of pandas.DataFrame and replace it with calls of init_dataframe.
    """

    _pandas_dataframe = ('DataFrame', 'pandas')

    _df_arg_list = ('data', 'index', 'columns', 'dtype', 'copy')

    def __init__(self, pipeline):
        self._pipeline = pipeline
        super().__init__(pipeline)

        self._reset()

    def match(self, func_ir, block, typemap, calltypes):
        self._reset()

        self._block = block
        self._func_ir = func_ir
        self._calls_to_rewrite = set()

        for stmt in find_operations(block=block, op_name='call'):
            expr = stmt.value
            fdef = guard(find_callname, func_ir, expr)
            if fdef == self._pandas_dataframe:
                args = get_call_parameters(call=expr, arg_names=self._df_arg_list)

                if self._match_dict_case(args, func_ir):
                    self._calls_to_rewrite.add(stmt)
                else:
                    pass  # Forward this case to pd_dataframe_overload which will handle it

        return len(self._calls_to_rewrite) > 0

    def apply(self):
        for stmt in self._calls_to_rewrite:
            args = get_call_parameters(call=stmt.value, arg_names=self._df_arg_list)
            old_data = args['data']
            args['data'], args['columns'] = self._extract_dict_args(args, self._func_ir)

            args_len = len(args['data'])
            func_name = f'init_dataframe_{args_len}'

            # injected_module = modules[pd_dataframe_ext_module.__name__]
            init_df = getattr(pd_dataframe_ext_module, func_name, None)
            if init_df is None:
                init_df_text = gen_init_dataframe_text(func_name, args_len)
                init_df = gen_init_dataframe_func(
                    func_name,
                    init_df_text,
                    {
                        'numba': numba,
                        'cgutils': cgutils,
                        'signature': signature,
                        'types': types,
                        'get_structure_maps': get_structure_maps,
                        'intrinsic': intrinsic,
                        'DataFrameType': DataFrameType,
                        'ColumnLoc': ColumnLoc,
                        'string_type': string_type,
                        'intrinsic': intrinsic,
                        'fix_df_array': fix_df_array,
                        'fix_df_index': fix_df_index
                    })

                setattr(pd_dataframe_ext_module, func_name, init_df)
                init_df.__module__ = pd_dataframe_ext_module.__name__
                init_df._defn.__module__ = pd_dataframe_ext_module.__name__

            init_df_stmt = import_function(init_df, self._block, self._func_ir)
            self._replace_call(stmt, init_df_stmt.target, args, self._block, self._func_ir)

            remove_unused_recursively(old_data, self._block, self._func_ir)
            self._pipeline.typingctx.refresh()

        return self._block

    def _reset(self):
        self._block = None
        self._func_ir = None
        self._calls_to_rewrite = None

    @staticmethod
    def _match_dict_case(args, func_ir):
        if 'data' in args and is_dict(args['data'], func_ir) and 'columns' not in args:
            return True

        return False

    @staticmethod
    def _extract_tuple_args(args, block, func_ir):
        data_args = get_tuple_items(args['data'], block, func_ir) if 'data' in args else None
        columns_args = get_tuple_items(args['columns'], block, func_ir) if 'columns' in args else None

        return data_args, columns_args

    @staticmethod
    def _extract_dict_args(args, func_ir):
        dict_items = get_dict_items(args['data'], func_ir)

        data_args = [item[1] for item in dict_items]
        columns_args = [item[0] for item in dict_items]

        return data_args, columns_args

    @staticmethod
    def _replace_call(stmt, new_call, args, block, func_ir):
        func = stmt.value

        data_args = args['data']
        columns_args = args['columns']
        index_args = args.get('index')

        if index_args is None:  # index arg was omitted
            none_stmt = declare_constant(None, block, func_ir, stmt.loc)
            index_args = none_stmt.target

        index_args = [index_args]

        all_args = data_args + index_args + columns_args
        call = Expr.call(new_call, all_args, {}, func.loc)

        stmt.value = call


def gen_init_dataframe_text(func_name, n_cols):
    args_col_data = ['c' + str(i) for i in range(n_cols)]
    args_col_names = ['n' + str(i) for i in range(n_cols)]
    params = ', '.join(args_col_data + ['index'] + args_col_names)
    suffix = ('' if n_cols == 0 else ', ')

    func_text = dedent(
        f'''
    @intrinsic
    def {func_name}(typingctx, {params}):
        """Create a DataFrame with provided columns data and index values.
        Takes 2n+1 args: n columns data, index data and n column names.
        Each column data is passed as separate argument to have compact LLVM IR.
        Used as as generic constructor for native DataFrame objects, which
        can be used with different input column types (e.g. lists), and
        resulting DataFrameType is deduced by applying transform functions
        (fix_df_array and fix_df_index) to input argument types.
        """

        n_cols = {n_cols}

        input_data_typs = ({', '.join(args_col_data) + suffix})
        fnty = typingctx.resolve_value_type(fix_df_array)
        fixed_col_sigs = []
        for i in range({n_cols}):
            to_sig = fnty.get_call_type(typingctx, (input_data_typs[i],), {{}})
            fixed_col_sigs.append(to_sig)
        data_typs = tuple(fixed_col_sigs[i].return_type for i in range({n_cols}))
        need_fix_cols = tuple(data_typs[i] != input_data_typs[i] for i in range({n_cols}))

        input_index_typ = index
        fnty = typingctx.resolve_value_type(fix_df_index)
        fixed_index_sig = fnty.get_call_type(typingctx, (input_index_typ,), {{}})
        index_typ = fixed_index_sig.return_type
        need_fix_index = index_typ != input_index_typ

        column_names = tuple(a.literal_value for a in ({', '.join(args_col_names) + suffix}))
        column_loc, data_typs_map, types_order = get_structure_maps(data_typs, column_names)
        col_needs_transform = tuple(not isinstance(data_typs[i], types.Array) for i in range(len(data_typs)))

        def codegen(context, builder, sig, args):
            {params}, = args
            data_arrs = [{', '.join(args_col_data) + suffix}]
            data_arrs_transformed = []
            for i, arr in enumerate(data_arrs):
                if need_fix_cols[i] == False:
                    data_arrs_transformed.append(arr)
                else:
                    res = context.compile_internal(builder, lambda a: fix_df_array(a), fixed_col_sigs[i], [arr])
                    data_arrs_transformed.append(res)

            # create dataframe struct and store values
            dataframe = cgutils.create_struct_proxy(
                sig.return_type)(context, builder)

            data_list_type = [types.List(typ) for typ in types_order]

            data_lists = []
            for typ_id, typ in enumerate(types_order):
                data_arrs_of_typ = [data_arrs_transformed[data_id] for data_id in data_typs_map[typ][1]]
                data_list_typ = context.build_list(builder, data_list_type[typ_id], data_arrs_of_typ)
                data_lists.append(data_list_typ)

            data_tup = context.make_tuple(
                builder, types.Tuple(data_list_type), data_lists)

            if need_fix_index == True:
                index = context.compile_internal(builder, lambda a: fix_df_index(a), fixed_index_sig, [index])

            dataframe.data = data_tup
            dataframe.index = index
            dataframe.parent = context.get_constant_null(types.pyobject)

            # increase refcount of stored values
            if context.enable_nrt:
                context.nrt.incref(builder, index_typ, index)
                for var, typ in zip(data_arrs_transformed, data_typs):
                    context.nrt.incref(builder, typ, var)

            return dataframe._getvalue()

        ret_typ = DataFrameType(data_typs, index_typ, column_names, column_loc=column_loc)
        sig = signature(ret_typ, {params})
        return sig, codegen
    ''')

    return func_text


def gen_init_dataframe_func(func_name, func_text, global_vars):

    loc_vars = {}
    exec(func_text, global_vars, loc_vars)
    return loc_vars[func_name]


@overload(DataFrame)
def pd_dataframe_overload(data, index=None, columns=None, dtype=None, copy=False):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.DataFrame

    Limitations
    -----------
    - Parameters `dtype` and `copy` are currently unsupported by Intel Scalable Dataframe Compiler.
    """

    ty_checker = TypeChecker('Method DataFrame')
    ty_checker.check(self, DataFrameType)

    if not isinstance(data, dict):
        ty_checker.raise_exc(pat, 'dict', 'data')

    if not isinstance(index, (types.Ommited, types.Array, StringArray, types.NoneType)) and index is not None:
        ty_checker.raise_exc(na, 'array-like', 'index')

    if not isinstance(columns, (types.Ommited, types.NoneType)) and columns is not None:
        ty_checker.raise_exc(na, 'None', 'columns')

    if not isinstance(dtype, (types.Ommited, types.NoneType)) and dtype is not None:
        ty_checker.raise_exc(na, 'None', 'dtype')

    if not isinstance(copy, (types.Ommited, types.NoneType)) and columns is not False:
        ty_checker.raise_exc(na, 'False', 'copy')

    return None

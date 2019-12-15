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


from numba.rewrites import (register_rewrite, Rewrite)
from numba.ir_utils import (guard, find_callname)
from numba.ir import (Expr)
from numba.extending import overload

from pandas import DataFrame

from sdc.rewrites.ir_utils import (find_operations, is_dict,
                                   get_tuple_items, get_dict_items, remove_unused_recursively,
                                   get_call_parameters,
                                   declare_constant,
                                   import_function, make_call,
                                   insert_before)
from sdc.hiframes.pd_dataframe_ext import (init_dataframe, DataFrameType)

from sdc.hiframes.api import fix_df_array

from sdc.config import config_pipeline_hpat_default

if not config_pipeline_hpat_default:
    @register_rewrite('before-inference')
    class RewriteDataFrame(Rewrite):
        """
        Searches for calls of pandas.DataFrame and replace it with calls of init_dataframe.
        """

        _pandas_dataframe = ('DataFrame', 'pandas')

        _df_arg_list = ('data', 'index', 'columns', 'dtype', 'copy')

        def __init__(self, pipeline):
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
            init_df_stmt = import_function(init_dataframe, self._block, self._func_ir)

            for stmt in self._calls_to_rewrite:
                args = get_call_parameters(call=stmt.value, arg_names=self._df_arg_list)

                old_data = args['data']

                args['data'], args['columns'] = self._extract_dict_args(args, self._func_ir)

                self._replace_call(stmt, init_df_stmt.target, args, self._block, self._func_ir)

                remove_unused_recursively(old_data, self._block, self._func_ir)

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

            if index_args is None:
                none_stmt = declare_constant(None, block, func_ir, stmt.loc)
                index_args = [none_stmt.target]
            else:
                index_args = RewriteDataFrame._replace_with_arrays([index_args], stmt, block, func_ir)

            data_args = RewriteDataFrame._replace_with_arrays(data_args, stmt, block, func_ir)
            all_args = data_args + index_args + columns_args

            call = Expr.call(new_call, all_args, {}, func.loc)

            stmt.value = call

        @staticmethod
        def _replace_with_arrays(args, stmt, block, func_ir):
            new_args = []

            for var in args:
                call_stmt = make_call(fix_df_array, [var], {}, block, func_ir, var.loc)
                insert_before(block, call_stmt, stmt)
                new_args.append(call_stmt.target)

            return new_args

    @overload(DataFrame)
    def pd_dataframe_overload(data, index=None, columns=None, dtype=None, copy=False):
        """
        Two-dimensional size-mutable, potentially heterogeneous tabular data
        structure with labeled axes (rows and columns). Arithmetic operations
        align on both row and column labels. Can be thought of as a dict-like
        container for Series objects. The primary pandas data structure.

        Parameters
        ----------
        data : dict
            Dict can contain Series, arrays, constants, or list-like objects

        index : array-like
            Index to use for resulting frame. Will default to RangeIndex if
            no indexing information part of input data and no index provided

        columns : Index or array-like
            Column labels to use for resulting frame. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided
            *unsupported*

        dtype : dtype, default None
            Data type to force. Only a single dtype is allowed. If None, infer
            *unsupported*

        copy : boolean, default False
            Copy data from inputs. Only affects DataFrame / 2d ndarray input
            *unsupported*
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

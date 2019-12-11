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

'''
| :class:`pandas.DataFrame` functions and operators implementations in Intel SDC
| Also, it contains Numba internal operators which are required for DataFrame type handling
'''

import operator
import pandas
import numpy
import numba

import sdc
from sdc.datatypes.hpat_pandas_series_functions import TypeChecker

from numba import types
from numba.extending import (overload, overload_method, overload_attribute)
from sdc.hiframes.pd_dataframe_ext import DataFrameType
from sdc.hiframes.pd_series_ext import SeriesType
from numba.errors import TypingError

if not sdc.config.use_default_dataframe:
    from sdc.datatypes.hpat_pandas_dataframe_types import DataFrameType

else:
    def sdc_pandas_dataframe_reduce_columns_series(df, name, params):
        saved_columns = df.columns
        n_cols = len(saved_columns)
        data_args = tuple('data{}'.format(i) for i in range(n_cols))
        space = []
        if len(params) > 0:
            space.append(', ')
        func_definition = 'def _reduce_impl(df{}{}):'.format("".join(space), ", ".join(
            str(key) + '=' + str(value) for key, value in params))
        func_lines = [func_definition]
        for i, d in enumerate(data_args):
            line = '  {} = sdc.hiframes.api.init_series(sdc.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}))'
            func_lines.append(line.format(d + '_S', i))
            func_lines.append('  {} = {}.{}({})'.format(d + '_O', d + '_S', name, ", ".join(
                str(key) for key, value in params)))
        func_lines.append("  return sdc.hiframes.pd_dataframe_ext.init_dataframe({}, None, {})\n".format(
            ", ".join(d + '_O._data' for d in data_args),
            ", ".join("'" + c + "'" for c in saved_columns)))

        loc_vars = {}
        func_text = '\n'.join(func_lines)
        exec(func_text, {'sdc': sdc, 'np': numpy}, loc_vars)
        _reduce_impl = loc_vars['_reduce_impl']

        return _reduce_impl

    def check_type(name, df, axis=None, skipna=None, level=None, numeric_only=None, ddof=1, min_count=0, n=5):
        ty_checker = TypeChecker('Method {}().'.format(name))
        ty_checker.check(df, DataFrameType)
        
        if not (isinstance(axis, types.Omitted) or axis is None):
            ty_checker.raise_exc(axis, 'unsupported', 'axis')

        if not (isinstance(skipna, (types.Omitted, types.NoneType, types.Boolean)) or skipna is None):
            ty_checker.raise_exc(skipna, 'bool', 'skipna')

        if not (isinstance(level, types.Omitted) or level is None):
            ty_checker.raise_exc(level, 'unsupported', 'level')

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is None):
            ty_checker.raise_exc(numeric_only, 'unsupported', 'numeric_only')

        if not (isinstance(ddof, types.Omitted) or ddof == 1):
            ty_checker.raise_exc(ddof, 'unsupported', 'ddof')
        
        if not (isinstance(min_count, types.Omitted) or min_count == 0):
            ty_checker.raise_exc(min_count, 'unsupported', 'min_count')

        if not (isinstance(n, (types.Omitted, types.Integer)) or n == 5):
            ty_checker.raise_exc(n, 'int64', 'n')

    @overload_method(DataFrameType, 'head')
    def head_overload(df, n=5):
        """
        Pandas DataFrame method :meth:`pandas.DataFrame.head` implementation.
        .. only:: developer
            Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_head1
        Parameters
        -----------
        self: :class:`pandas.DataFrame`
            input arg
        n: :obj:`int`, default 5
            input arg, default 5
        Returns
        -------
        :obj:`pandas.Series`
        returns: The first n rows of the caller object.
        """

        name = 'head'

        check_type(name, df, n=n)

        params = [('n', 5)]

        return sdc_pandas_dataframe_reduce_columns_series(df, name, params)

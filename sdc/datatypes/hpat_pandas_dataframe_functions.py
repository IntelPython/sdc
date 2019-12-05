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

import sdc

from numba import types
from numba.extending import (overload, overload_method, overload_attribute)
from sdc.hiframes.pd_dataframe_ext import DataFrameType
from numba.errors import TypingError
from sdc.datatypes.hpat_pandas_series_functions import TypeChecker

if not sdc.config.use_default_dataframe:
    from sdc.datatypes.hpat_pandas_dataframe_types import DataFrameType

    @overload_method(DataFrameType, 'count')
    def sdc_pandas_dataframe_count(self, axis=0, level=None, numeric_only=False):
        """
        Pandas DataFrame method :meth:`pandas.DataFrame.count` implementation.
        .. only:: developer
            Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_count
        Parameters
        -----------
        self: :class:`pandas.DataFrame`
            input arg
        axis:
            *unsupported*
        level:
            *unsupported*
        numeric_only:
            *unsupported*
        Returns
        -------
        :obj:`pandas.Series` or `pandas.DataFrame`
                returns: For each column/row the number of non-NA/null entries. If level is specified returns
                a DataFrame.
        """

        _func_name = 'Method pandas.dataframe.count().'

        if not isinstance(self, DataFrameType):
            raise TypingError('{} The object must be a pandas.dataframe. Given: {}'.format(_func_name, self))

        if not (isinstance(axis, types.Omitted) or axis == 0):
            raise TypingError("{} 'axis' unsupported. Given: {}".format(_func_name, axis))

        if not (isinstance(level, types.Omitted) or level is None):
            raise TypingError("{} 'level' unsupported. Given: {}".format(_func_name, axis))

        if not (isinstance(numeric_only, types.Omitted) or numeric_only is False):
            raise TypingError("{} 'numeric_only' unsupported. Given: {}".format(_func_name, axis))

        def sdc_pandas_dataframe_count_impl(self, axis=0, level=None, numeric_only=False):
            result_data = []
            result_index = []

            for dataframe_item in self._data:
                item_count = dataframe_item.count()
                item_name = dataframe_item._name
                result_data.append(item_count)
                result_index.append(item_name)

            return pandas.Series(data=result_data, index=result_index)

        return sdc_pandas_dataframe_count_impl

else:
    def sdc_pandas_dataframe_reduce_columns(df, name, params):
        saved_columns = df.columns
        n_cols = len(saved_columns)
        data_args = tuple('data{}'.format(i) for i in range(n_cols))
        all_params = ['df']
        for key, value in params:
            all_params.append('{}={}'.format(key, value))
        func_definition = 'def _reduce_impl({}):'.format(', '.join(all_params))
        func_lines = [func_definition]
        for i, d in enumerate(data_args):
            line = '  {} = hpat.hiframes.api.init_series(hpat.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}))'
            func_lines.append(line.format(d + '_S', i))
            func_lines.append('  {} = {}.{}()'.format(d + '_O', d + '_S', name))
        func_lines.append('  data = np.array(({},))'.format(
            ", ".join(d + '_O' for d in data_args)))
        func_lines.append('  index = hpat.str_arr_ext.StringArray(({},))'.format(
            ', '.join('"{}"'.format(c) for c in saved_columns)))
        func_lines.append('  return hpat.hiframes.api.init_series(data, index)')
        loc_vars = {}
        func_text = '\n'.join(func_lines)

        exec(func_text, {'hpat': sdc, 'np': numpy}, loc_vars)
        _reduce_impl = loc_vars['_reduce_impl']

        return _reduce_impl

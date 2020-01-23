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

import numpy
import pandas
from numba import prange, types
from numba.targets.registry import cpu_target

from sdc.hiframes.pd_series_ext import SeriesType
from sdc.utils import sdc_overload_method

from ..common_functions import TypeChecker


@sdc_overload_method(SeriesType, 'map')
def hpat_pandas_series_map(self, arg, na_action=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.map

    Limitations
    -----------
    String data types are not supported by Intel Scalable Dataframe Compiler.
    `arg` could be function or dict and could not be Series. The function should return scalar type.
    `na_action` is unsupported by Intel Scalable Dataframe Compiler.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_map.py
       :language: python
       :lines: 36-
       :caption: `map()` accepts a function.
       :name: ex_series_map

    .. command-output:: python ./series/series_map.py
       :cwd: ../../../examples

    .. seealso::

        :ref:`Series.map <pandas.Series.apply>`
            For applying more complex functions on a Series.
        :ref:`DataFrame.apply <pandas.DataFrame.apply>`
            Apply a function row-/column-wise.
        :ref:`DataFrame.applymap <pandas.DataFrame.applymap>`
            Apply a function elementwise on a whole DataFrame.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series -k map
    """

    ty_checker = TypeChecker("Method map().")
    ty_checker.check(self, SeriesType)

    if isinstance(arg, types.Callable):
        sig = arg.get_call_type(cpu_target.typing_context, [self.dtype], {})
        output_type = sig.return_type

        def impl(self, arg, na_action=None):
            input_arr = self._data
            length = len(input_arr)

            output_arr = numpy.empty(length, dtype=output_type)

            for i in prange(length):
                output_arr[i] = arg(input_arr[i])

            return pandas.Series(output_arr, index=self._index, name=self._name)

        return impl

    if isinstance(arg, types.DictType):
        output_type = self.dtype

        def impl(self, arg, na_action=None):
            input_arr = self._data
            length = len(input_arr)

            output_arr = numpy.empty(length, dtype=output_type)

            for i in prange(length):
                output_arr[i] = arg.get(input_arr[i], numpy.nan)

            return pandas.Series(output_arr, index=self._index, name=self._name)

        return impl

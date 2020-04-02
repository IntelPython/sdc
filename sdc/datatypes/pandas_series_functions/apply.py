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
from numba.core.registry import cpu_target

from sdc.hiframes.pd_series_ext import SeriesType
from sdc.utilities.utils import sdc_overload_method

from sdc.utilities.sdc_typing_utils import TypeChecker


@sdc_overload_method(SeriesType, 'apply')
def hpat_pandas_series_apply(self, func, convert_dtype=True, args=()):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.apply

    Limitations
    -----------
    - Parameters ``convert_dtype`` and ``args`` are currently unsupported by Intel Scalable Dataframe Compiler.
    - ``function`` returning a Series object is currently unsupported by Intel Scalable Dataframe Compiler.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/series_apply.py
        :language: python
        :lines: 33-
        :caption: Square the values by defining a function and passing it as an argument to `apply()`.
        :name: ex_series_apply

    .. command-output:: python ./series/series_apply.py
        :cwd: ../../../examples

    .. literalinclude:: ../../../examples/series/series_apply_lambda.py
        :language: python
        :lines: 33-
        :caption: Square the values by passing an anonymous function as an argument to `apply()`.
        :name: ex_series_apply_lambda

    .. command-output:: python ./series/series_apply_lambda.py
        :cwd: ../../../examples

    .. literalinclude:: ../../../examples/series/series_apply_log.py
        :language: python
        :lines: 33-
        :caption: Use a function from the Numpy library.
        :name: ex_series_apply_log

    .. command-output:: python ./series/series_apply_log.py
        :cwd: ../../../examples

    .. seealso::

        :ref:`Series.map <pandas.Series.map>`
            For element-wise operations.
        :ref:`Series.agg <pandas.Series.agg>`
            Only perform aggregating type operations.
        :ref:`Series.transform <pandas.transform>`
            Only perform transforming type operations.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    .. only:: developer
        Test: python -m sdc.runtests sdc.tests.test_series_apply
    """

    ty_checker = TypeChecker("Method apply().")
    ty_checker.check(self, SeriesType)
    ty_checker.check(func, types.Callable)

    func_args = [self.dtype]
    if not isinstance(args, types.Omitted):
        func_args.extend(args)

    sig = func.get_call_type(cpu_target.typing_context, func_args, {})
    output_type = sig.return_type

    def impl(self, func, convert_dtype=True, args=()):
        input_arr = self._data
        length = len(input_arr)

        output_arr = numpy.empty(length, dtype=output_type)

        for i in prange(length):
            # Numba issue https://github.com/numba/numba/issues/5065
            # output_arr[i] = func(input_arr[i], *args)
            output_arr[i] = func(input_arr[i])

        return pandas.Series(output_arr, index=self._index, name=self._name)

    return impl

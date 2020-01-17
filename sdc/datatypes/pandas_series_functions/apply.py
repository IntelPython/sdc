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

from sdc.hiframes.pd_series_ext import SeriesType
from sdc.utils import sdc_overload_method

from ..common_functions import TypeChecker


@sdc_overload_method(SeriesType, 'apply')
def hpat_pandas_series_apply(self, func, convert_dtype=True, args=()):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************

    Pandas API: pandas.Series.apply

    Limitations
    -----------
    `convert_dtype`, `args` and `**kwds` are currently unsupported by Intel Scalable Dataframe Compiler.

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

    def impl(self, func, convert_dtype=True, args=()):
        return pandas.Series(list(map(func, self._data)), index=self._index, name=self._name)

    return impl

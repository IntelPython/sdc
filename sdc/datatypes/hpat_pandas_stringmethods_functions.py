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

"""

| :class:`pandas.core.strings.StringMethods` functions and operators implementations in HPAT

    .. only:: developer

    This is autogenerated sources for all Unicode string functions supported by Numba.
    Currently tested 45 functions only. List of functions obtained automatically from
    `numba.types.misc.UnicodeType` class

    Example of the generated method (for method upper()):
    `hpat_pandas_stringmethods_upper_parallel_impl` is paralell version
    (required additional import mentioned in the body)

    @overload_method(StringMethodsType, 'upper')
    def hpat_pandas_stringmethods_upper(self):
        _func_name = 'Method stringmethods.upper().'

        if not isinstance(self, StringMethodsType):
            raise TypingError('{} The object must be a pandas.core.strings. Given: {}'.format(_func_name, self))

        def hpat_pandas_stringmethods_upper_parallel_impl(self):
            from numba.parfor import (init_prange, min_checker, internal_prange)

            init_prange()
            result = []
            item_count = len(self._data)
            min_checker(item_count)
            for i in internal_prange(item_count):
                item = self._data[i]
                item_method = item.upper()
                result.append(item_method)

            return pandas.Series(result)

        return hpat_pandas_stringmethods_upper_parallel_impl

        def hpat_pandas_stringmethods_upper_impl(self):
            result = []
            item_count = len(self._data)
            for i in range(item_count):
                item = self._data[i]
                item_method = item.upper()
                result.append(item_method)

            return pandas.Series(result)

        return hpat_pandas_stringmethods_upper_impl

    Test: python -m sdc.runtests sdc.tests.test_hiframes.TestHiFrames.test_str_split_filter

"""


import numpy
import pandas

import numba
from numba.extending import overload_method

from sdc.datatypes.hpat_pandas_stringmethods_types import StringMethodsType


_hpat_pandas_stringmethods_autogen_global_dict = {
    'pandas': pandas,
    'numpy': numpy,
    'numba': numba,
    'StringMethodsType': StringMethodsType
}

_hpat_pandas_stringmethods_functions_params = {
    'cat': ', others=None, sep=None, na_rep=None, join="left"',
    'center': ', width, fillchar=" "',
    'contains': ', pat, case=True, flags=0, na=numpy.nan, regex=True',
    'count': ', pat, flags=0',
    'decode': ', encoding, errors="strict"',
    'encode': ', encoding, errors="strict"',
    'endswith': ', pat, na=numpy.nan',
    'extractall': ', pat, flags=0',
    'extract': ', pat, flags=0, expand=True',
    'findall': ', pat, flags=0',
    'find': ', sub, start=0, end=None',
    'get': ', i',
    'get_dummies': ', sep="|"',
    'index': ', sub, start=0, end=None',
    'join': ', sep',
    'ljust': ', width, fillchar=" "',
    'lstrip': ', to_strip=None',
    'match': ', pat, case=True, flags=0, na=numpy.nan',
    'normalize': ', form',
    'pad': ', width, side="left", fillchar=" "',
    'partition': ', sep=" ", expand=True',
    'repeat': ', repeats',
    'replace': ', pat, repl, n=-1, case=None, flags=0, regex=True',
    'rfind': ', sub, start=0, end=None',
    'rindex': ', sub, start=0, end=None',
    'rjust': ', width, fillchar=" "',
    'rpartition': ', sep=" ", expand=True',
    'rsplit': ', pat=None, n=-1, expand=False',
    'rstrip': ', to_strip=None',
    'slice_replace': ', start=None, stop=None, repl=None',
    'slice': ', start=None, stop=None, step=None',
    'split': ', pat=None, n=-1, expand=False',
    'startswith': ', pat, na=numpy.nan',
    'strip': ', to_strip=None',
    'translate': ', table',
    'wrap': ', width',
    'zfill': ', width',
}

_hpat_pandas_stringmethods_functions_template = """
# @overload_method(StringMethodsType, '{methodname}')
def hpat_pandas_stringmethods_{methodname}(self{methodparams}):
    \"\"\"
    Pandas Series method :meth:`pandas.core.strings.StringMethods.{methodname}()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests sdc.tests.test_strings.TestStrings.test_str2str
          python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_str2str
          python -m sdc.runtests sdc.tests.test_hiframes.TestHiFrames.test_str_get
          python -m sdc.runtests sdc.tests.test_hiframes.TestHiFrames.test_str_replace_noregex
          python -m sdc.runtests sdc.tests.test_hiframes.TestHiFrames.test_str_split
          python -m sdc.runtests sdc.tests.test_hiframes.TestHiFrames.test_str_contains_regex

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg
    other: {methodparams}
        input arguments decription in
        https://pandas.pydata.org/pandas-docs/version/0.25/reference/series.html#string-handling

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    \"\"\"

    if not isinstance(self, StringMethodsType):
        raise TypingError('Method {methodname}(). The object must be a pandas.core.strings. Given: ' % self)

    def hpat_pandas_stringmethods_{methodname}_impl(self{methodparams}):
        item_count = len(self._data)
        result = [''] * item_count
        # result = numba.typed.List.empty_list(numba.types.unicode_type)

        for it in range(item_count):
            item = self._data._data[it]
            if len(item) > 0:
                result[it] = item.{methodname}({methodparams_call})
            else:
                result[it] = item

        return pandas.Series(result, name=self._data._name)

    return hpat_pandas_stringmethods_{methodname}_impl
"""


def _hpat_pandas_stringmethods_autogen(method_name):
    """"
    The function generates a function for 'method_name' from source text that is created on the fly.
    """

    params = ""
    params_call = ""

    # get function parameters by name
    params_dict = _hpat_pandas_stringmethods_functions_params.get(method_name)
    if params_dict is not None:
        params = params_dict

    if len(params) > 0:
        """
        Translate parameters string for method

        For example:
            parameters for split(): ', pat=None, n=-1, expand=False'
                    translate into: 'pat, n, expand'
        """

        params_call_splitted = params.split(',')
        params_call_list = []
        for item in params_call_splitted:
            params_call_list.append(item.split("=")[0])
        params_call = ",".join(params_call_list)
        if len(params_call) > 1:
            params_call = params_call[2:]

    sourcecode = _hpat_pandas_stringmethods_functions_template.format(methodname=method_name,
                                                                      methodparams=params,
                                                                      methodparams_call=params_call)
    exec(sourcecode, _hpat_pandas_stringmethods_autogen_global_dict)

    global_dict_name = 'hpat_pandas_stringmethods_{methodname}'.format(methodname=method_name)
    return _hpat_pandas_stringmethods_autogen_global_dict[global_dict_name]


# _hpat_pandas_stringmethods_autogen_methods = sorted(dir(numba.types.misc.UnicodeType.__getattribute__.__qualname__))
_hpat_pandas_stringmethods_autogen_methods = ['upper', 'lower', 'lstrip', 'rstrip', 'strip']
"""
    This is the list of function which are autogenerated to be used from Numba directly.
"""

_hpat_pandas_stringmethods_autogen_exceptions = ['split', 'len', 'get', 'replace']

for method_name in _hpat_pandas_stringmethods_autogen_methods:
    if not (method_name.startswith('__') or method_name in _hpat_pandas_stringmethods_autogen_exceptions):
        overload_method(StringMethodsType, method_name)(_hpat_pandas_stringmethods_autogen(method_name))

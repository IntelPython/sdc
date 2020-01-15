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

    @sdc_overload_method(StringMethodsType, 'upper')
    def hpat_pandas_stringmethods_upper(self):

        ty_checker = TypeChecker('Method stringmethods.upper().')
        ty_checker.check(self, StringMethodsType)

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
from numba.types import (Boolean, Integer, NoneType,
                         Omitted, StringLiteral, UnicodeType)

from sdc.datatypes.common_functions import TypeChecker
from sdc.datatypes.hpat_pandas_stringmethods_types import StringMethodsType
from sdc.utils import sdc_overload_method

_hpat_pandas_stringmethods_autogen_global_dict = {
    'pandas': pandas,
    'numpy': numpy,
    'numba': numba,
    'StringMethodsType': StringMethodsType,
    'TypeChecker': TypeChecker
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
# @sdc_overload_method(StringMethodsType, '{methodname}')
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

    ty_checker = TypeChecker('Method {methodname}().')
    ty_checker.check(self, StringMethodsType)

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

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_{methodname}_impl
"""


@sdc_overload_method(StringMethodsType, 'center')
def hpat_pandas_stringmethods_center(self, width, fillchar=' '):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.center

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_center.py
       :language: python
       :lines: 27-
       :caption: Filling left and right side of strings in the Series with an additional character
       :name: ex_series_str_center

    .. command-output:: python ./series/str/series_str_center.py
       :cwd: ../../../examples

    .. todo:: Add support of 32-bit Unicode for `str.center()`

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.center()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_center

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg
    width: :obj:`int`
        Minimum width of resulting string
    fillchar: :obj:`str`
        Additional character for filling, default is whitespace

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method center().')
    ty_checker.check(self, StringMethodsType)

    if not isinstance(width, Integer):
        ty_checker.raise_exc(width, 'int', 'width')

    accepted_types = (Omitted, StringLiteral, UnicodeType)
    if not isinstance(fillchar, accepted_types) and fillchar != ' ':
        ty_checker.raise_exc(fillchar, 'str', 'fillchar')

    def hpat_pandas_stringmethods_center_impl(self, width, fillchar=' '):
        item_count = len(self._data)
        result = [''] * item_count
        for idx, item in enumerate(self._data._data):
            result[idx] = item.center(width, fillchar)

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_center_impl


@sdc_overload_method(StringMethodsType, 'endswith')
def hpat_pandas_stringmethods_endswith(self, pat, na=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.endswith

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_endswith.py
       :language: python
       :lines: 27-
       :caption: Test if the end of each string element matches a string
       :name: ex_series_str_endswith

    .. command-output:: python ./series/str/series_str_endswith.py
       :cwd: ../../../examples

    .. todo::
        - Add support of matching the end of each string by a pattern
        - Add support of parameter ``na``

    .. seealso::
        `str.endswith <https://docs.python.org/3/library/stdtypes.html#str.endswith>`_
            Python standard library string method.
        :ref:`Series.str.startswith <pandas.Series.str.startswith>`
            Same as endswith, but tests the start of string.
        :ref:`Series.str.contains <pandas.Series.str.contains>`
            Tests if string element contains a pattern.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.endswith()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_endswith

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg
    pat: :obj:`str`
        Character sequence
    na: :obj:`bool`
        Object shown if element tested is not a string
        *unsupported*

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method endswith().')
    ty_checker.check(self, StringMethodsType)

    if not isinstance(pat, (StringLiteral, UnicodeType)):
        ty_checker.raise_exc(pat, 'str', 'pat')

    if not isinstance(na, (Boolean, NoneType, Omitted)) and na is not None:
        ty_checker.raise_exc(na, 'bool', 'na')

    def hpat_pandas_stringmethods_endswith_impl(self, pat, na=None):
        if na is not None:
            msg = 'Method endswith(). The object na\n expected: None'
            raise ValueError(msg)

        item_endswith = len(self._data)
        result = numpy.empty(item_endswith, numba.types.boolean)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.endswith(pat)

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_endswith_impl


@sdc_overload_method(StringMethodsType, 'find')
def hpat_pandas_stringmethods_find(self, sub, start=0, end=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.find

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_find.py
       :language: python
       :lines: 27-
       :caption: Return lowest indexes in each strings in the Series
       :name: ex_series_str_find

    .. command-output:: python ./series/str/series_str_find.py
       :cwd: ../../../examples

    .. todo:: Add support of parameters ``start`` and ``end``

    .. seealso::
        :ref:`Series.str.rfind <pandas.Series.str.rfind>`
            Return highest indexes in each strings.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.find()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_find

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg
    sub: :obj:`str`
        Substring being searched
    start: :obj:`int`
        Left edge index
        *unsupported*
    end: :obj:`int`
        Right edge index
        *unsupported*

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method find().')
    ty_checker.check(self, StringMethodsType)

    if not isinstance(sub, (StringLiteral, UnicodeType)):
        ty_checker.raise_exc(sub, 'str', 'sub')

    accepted_types = (Integer, NoneType, Omitted)
    if not isinstance(start, accepted_types) and start != 0:
        ty_checker.raise_exc(start, 'None, int', 'start')

    if not isinstance(end, accepted_types) and end is not None:
        ty_checker.raise_exc(end, 'None, int', 'end')

    def hpat_pandas_stringmethods_find_impl(self, sub, start=0, end=None):
        if start != 0:
            raise ValueError('Method find(). The object start\n expected: 0')
        if end is not None:
            raise ValueError('Method find(). The object end\n expected: None')

        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.int64)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.find(sub)

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_find_impl


@sdc_overload_method(StringMethodsType, 'isupper')
def hpat_pandas_stringmethods_isupper(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.isupper

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_isupper.py
       :language: python
       :lines: 27-
       :caption: Check whether all characters in each string are uppercase
       :name: ex_series_str_isupper

    .. command-output:: python ./series/str/series_str_isupper.py
       :cwd: ../../../examples

    .. seealso::
        :ref:`Series.str.isalpha <pandas.Series.str.isalpha>`
            Check whether all characters are alphabetic.
        :ref:`Series.str.isnumeric <pandas.Series.str.isnumeric>`
            Check whether all characters are numeric.
        :ref:`Series.str.isalnum <pandas.Series.str.isalnum>`
            Check whether all characters are alphanumeric.
        :ref:`Series.str.isdigit <pandas.Series.str.isdigit>`
            Check whether all characters are digits.
        :ref:`Series.str.isdecimal <pandas.Series.str.isdecimal>`
            Check whether all characters are decimal.
        :ref:`Series.str.isspace <pandas.Series.str.isspace>`
            Check whether all characters are whitespace.
        :ref:`Series.str.islower <pandas.Series.str.islower>`
            Check whether all characters are lowercase.
        :ref:`Series.str.istitle <pandas.Series.str.istitle>`
            Check whether all characters are titlecase.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.isupper()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_str2str

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method isupper().')
    ty_checker.check(self, StringMethodsType)

    def hpat_pandas_stringmethods_isupper_impl(self):
        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.boolean)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.isupper()

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_isupper_impl


@sdc_overload_method(StringMethodsType, 'len')
def hpat_pandas_stringmethods_len(self):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.len

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_len.py
       :language: python
       :lines: 27-
       :caption: Compute the length of each element in the Series
       :name: ex_series_str_len

    .. command-output:: python ./series/str/series_str_len.py
       :cwd: ../../../examples

    .. seealso::
        `str.len`
            Python built-in function returning the length of an object.
        :ref:`Series.size <pandas.Series.size>`
            Returns the length of the Series.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.len()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_str_len1

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method len().')
    ty_checker.check(self, StringMethodsType)

    def hpat_pandas_stringmethods_len_impl(self):
        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.int64)
        for idx, item in enumerate(self._data._data):
            result[idx] = len(item)

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_len_impl


@sdc_overload_method(StringMethodsType, 'ljust')
def hpat_pandas_stringmethods_ljust(self, width, fillchar=' '):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.ljust

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_ljust.py
       :language: python
       :lines: 27-
       :caption: Filling right side of strings in the Series with an additional character
       :name: ex_series_str_ljust

    .. command-output:: python ./series/str/series_str_ljust.py
       :cwd: ../../../examples

    .. todo:: Add support of 32-bit Unicode for `str.ljust()`

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.ljust()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_ljust

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg
    width: :obj:`int`
        Minimum width of resulting string
    fillchar: :obj:`str`
        Additional character for filling, default is whitespace

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method ljust().')
    ty_checker.check(self, StringMethodsType)

    if not isinstance(width, Integer):
        ty_checker.raise_exc(width, 'int', 'width')

    accepted_types = (Omitted, StringLiteral, UnicodeType)
    if not isinstance(fillchar, accepted_types) and fillchar != ' ':
        ty_checker.raise_exc(fillchar, 'str', 'fillchar')

    def hpat_pandas_stringmethods_ljust_impl(self, width, fillchar=' '):
        item_count = len(self._data)
        result = [''] * item_count
        for idx, item in enumerate(self._data._data):
            result[idx] = item.ljust(width, fillchar)

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_ljust_impl


@sdc_overload_method(StringMethodsType, 'rjust')
def hpat_pandas_stringmethods_rjust(self, width, fillchar=' '):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.rjust

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_rjust.py
       :language: python
       :lines: 27-
       :caption: Filling left side of strings in the Series with an additional character
       :name: ex_series_str_rjust

    .. command-output:: python ./series/str/series_str_rjust.py
       :cwd: ../../../examples

    .. todo:: Add support of 32-bit Unicode for `str.rjust()`

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.rjust()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_rjust

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg
    width: :obj:`int`
        Minimum width of resulting string
    fillchar: :obj:`str`
        Additional character for filling, default is whitespace

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method rjust().')
    ty_checker.check(self, StringMethodsType)

    if not isinstance(width, Integer):
        ty_checker.raise_exc(width, 'int', 'width')

    accepted_types = (Omitted, StringLiteral, UnicodeType)
    if not isinstance(fillchar, accepted_types) and fillchar != ' ':
        ty_checker.raise_exc(fillchar, 'str', 'fillchar')

    def hpat_pandas_stringmethods_rjust_impl(self, width, fillchar=' '):
        item_count = len(self._data)
        result = [''] * item_count
        for idx, item in enumerate(self._data._data):
            result[idx] = item.rjust(width, fillchar)

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_rjust_impl


@sdc_overload_method(StringMethodsType, 'startswith')
def hpat_pandas_stringmethods_startswith(self, pat, na=None):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.startswith

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_startswith.py
       :language: python
       :lines: 27-
       :caption: Test if the start of each string element matches a string
       :name: ex_series_str_startswith

    .. command-output:: python ./series/str/series_str_startswith.py
       :cwd: ../../../examples

    .. todo::
        - Add support of matching the start of each string by a pattern
        - Add support of parameter ``na``

    .. seealso::
        `str.startswith <https://docs.python.org/3/library/stdtypes.html#str.startswith>`_
            Python standard library string method.
        :ref:`Series.str.endswith <pandas.Series.str.endswith>`
            Same as startswith, but tests the end of string.
        :ref:`Series.str.contains <pandas.Series.str.contains>`
            Tests if string element contains a pattern.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.startswith()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_startswith

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg
    pat: :obj:`str`
        Character sequence
    na: :obj:`bool`
        Object shown if element tested is not a string
        *unsupported*

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method startswith().')
    ty_checker.check(self, StringMethodsType)

    if not isinstance(pat, (StringLiteral, UnicodeType)):
        ty_checker.raise_exc(pat, 'str', 'pat')

    if not isinstance(na, (Boolean, NoneType, Omitted)) and na is not None:
        ty_checker.raise_exc(na, 'bool', 'na')

    def hpat_pandas_stringmethods_startswith_impl(self, pat, na=None):
        if na is not None:
            msg = 'Method startswith(). The object na\n expected: None'
            raise ValueError(msg)

        item_startswith = len(self._data)
        result = numpy.empty(item_startswith, numba.types.boolean)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.startswith(pat)

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_startswith_impl


@sdc_overload_method(StringMethodsType, 'zfill')
def hpat_pandas_stringmethods_zfill(self, width):
    """
    Intel Scalable Dataframe Compiler User Guide
    ********************************************
    Pandas API: pandas.Series.str.zfill

    Limitations
    -----------
    Series elements are expected to be Unicode strings. Elements cannot be NaN.

    Examples
    --------
    .. literalinclude:: ../../../examples/series/str/series_str_zfill.py
       :language: python
       :lines: 27-
       :caption: Pad strings in the Series by prepending '0' characters
       :name: ex_series_str_zfill

    .. command-output:: python ./series/str/series_str_zfill.py
       :cwd: ../../../examples

    .. todo:: Add support of 32-bit Unicode for `str.zfill()`

    .. seealso::
        :ref:`Series.str.rjust <pandas.Series.str.rjust>`
            Fills the left side of strings with an arbitrary character.
        :ref:`Series.str.ljust <pandas.Series.str.ljust>`
            Fills the right side of strings with an arbitrary character.
        :ref:`Series.str.pad <pandas.Series.str.pad>`
            Fills the specified sides of strings with an arbitrary character.
        :ref:`Series.str.center <pandas.Series.str.center>`
            Fills boths sides of strings with an arbitrary character.

    Intel Scalable Dataframe Compiler Developer Guide
    *************************************************

    Pandas Series method :meth:`pandas.core.strings.StringMethods.zfill()` implementation.

    Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

    .. only:: developer

    Test: python -m sdc.runtests -k sdc.tests.test_series.TestSeries.test_series_zfill

    Parameters
    ----------
    self: :class:`pandas.core.strings.StringMethods`
        input arg
    width: :obj:`int`
        Minimum width of resulting string

    Returns
    -------
    :obj:`pandas.Series`
         returns :obj:`pandas.Series` object
    """

    ty_checker = TypeChecker('Method zfill().')
    ty_checker.check(self, StringMethodsType)

    if not isinstance(width, Integer):
        ty_checker.raise_exc(width, 'int', 'width')

    def hpat_pandas_stringmethods_zfill_impl(self, width):
        item_count = len(self._data)
        result = [''] * item_count
        for idx, item in enumerate(self._data._data):
            result[idx] = item.zfill(width)

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_zfill_impl


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


sdc_pandas_series_str_docstring_template = """
        Intel Scalable Dataframe Compiler User Guide
        ********************************************
        Pandas API: pandas.Series.str.{method_name}

        Limitations
        -----------
        Series elements are expected to be Unicode strings. Elements cannot be NaN.

        Examples
        --------
        .. literalinclude:: ../../../examples/series/str/series_str_{method_name}.py
           :language: python
           :lines: 27-
           :caption: {caption}
           :name: ex_series_str_{method_name}

        .. command-output:: python ./series/str/series_str_{method_name}.py
           :cwd: ../../../examples

        .. seealso::
            :ref:`Series.str.isalpha <pandas.Series.str.isalpha>`
                Check whether all characters are alphabetic.
            :ref:`Series.str.isnumeric <pandas.Series.str.isnumeric>`
                Check whether all characters are numeric.
            :ref:`Series.str.isalnum <pandas.Series.str.isalnum>`
                Check whether all characters are alphanumeric.
            :ref:`Series.str.isdigit <pandas.Series.str.isdigit>`
                Check whether all characters are digits.
            :ref:`Series.str.isdecimal <pandas.Series.str.isdecimal>`
                Check whether all characters are decimal.
            :ref:`Series.str.isspace <pandas.Series.str.isspace>`
                Check whether all characters are whitespace.
            :ref:`Series.str.islower <pandas.Series.str.islower>`
                Check whether all characters are lowercase.
            :ref:`Series.str.isupper <pandas.Series.str.isupper>`
                Check whether all characters are uppercase.
            :ref:`Series.str.istitle <pandas.Series.str.istitle>`
                Check whether all characters are titlecase.

        Intel Scalable Dataframe Compiler Developer Guide
        *************************************************

        Pandas Series method :meth:`pandas.core.strings.StringMethods.{method_name}()` implementation.

        Note: Unicode type of list elements are supported only. Numpy.NaN is not supported as elements.

        .. only:: developer

        Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_{method_name}_str

        Parameters
        ----------
        self: :class:`pandas.core.strings.StringMethods`
            input arg

        Returns
        -------
        :obj:`pandas.Series`
             returns :obj:`pandas.Series` object
"""


@sdc_overload_method(StringMethodsType, 'istitle')
def hpat_pandas_stringmethods_istitle(self):

    ty_checker = TypeChecker('Method istitle().')
    ty_checker.check(self, StringMethodsType)

    def hpat_pandas_stringmethods_istitle_impl(self):
        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.boolean)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.istitle()

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_istitle_impl


@sdc_overload_method(StringMethodsType, 'isspace')
def hpat_pandas_stringmethods_isspace(self):

    ty_checker = TypeChecker('Method isspace().')
    ty_checker.check(self, StringMethodsType)

    def hpat_pandas_stringmethods_isspace_impl(self):
        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.boolean)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.isspace()

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_isspace_impl


@sdc_overload_method(StringMethodsType, 'isalpha')
def hpat_pandas_stringmethods_isalpha(self):

    ty_checker = TypeChecker('Method isalpha().')
    ty_checker.check(self, StringMethodsType)

    def hpat_pandas_stringmethods_isalpha_impl(self):
        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.boolean)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.isalpha()

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_isalpha_impl


@sdc_overload_method(StringMethodsType, 'islower')
def hpat_pandas_stringmethods_islower(self):

    ty_checker = TypeChecker('Method islower().')
    ty_checker.check(self, StringMethodsType)

    def hpat_pandas_stringmethods_islower_impl(self):
        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.boolean)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.islower()

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_islower_impl


@sdc_overload_method(StringMethodsType, 'isalnum')
def hpat_pandas_stringmethods_isalnum(self):

    ty_checker = TypeChecker('Method isalnum().')
    ty_checker.check(self, StringMethodsType)

    def hpat_pandas_stringmethods_isalnum_impl(self):
        item_count = len(self._data)
        result = numpy.empty(item_count, numba.types.boolean)
        for idx, item in enumerate(self._data._data):
            result[idx] = item.isalnum()

        return pandas.Series(result, self._data._index, name=self._data._name)

    return hpat_pandas_stringmethods_isalnum_impl


stringmethods_funcs = {
    'istitle': {'method': hpat_pandas_stringmethods_istitle,
                'caption': 'Check if each word start with an upper case letter'},
    'isspace': {'method': hpat_pandas_stringmethods_isspace,
                'caption': 'Check if all the characters in the text are whitespaces'},
    'isalpha': {'method': hpat_pandas_stringmethods_isalpha,
                'caption': 'Check whether all characters in each string are alphabetic'},
    'islower': {'method': hpat_pandas_stringmethods_islower,
                'caption': 'Check if all the characters in the text are alphanumeric'},
    'isalnum': {'method': hpat_pandas_stringmethods_isalnum,
                'caption': 'Check if all the characters in the text are alphanumeric'}
}


for name, data in stringmethods_funcs.items():
    data['method'].__doc__ = sdc_pandas_series_str_docstring_template.format(**{'method_name': name,
                                                                                'caption': data['caption']})


# _hpat_pandas_stringmethods_autogen_methods = sorted(dir(numba.types.misc.UnicodeType.__getattribute__.__qualname__))
_hpat_pandas_stringmethods_autogen_methods = ['upper', 'lower', 'lstrip', 'rstrip', 'strip']
"""
    This is the list of function which are autogenerated to be used from Numba directly.
"""

_hpat_pandas_stringmethods_autogen_exceptions = ['split', 'get', 'replace']

for method_name in _hpat_pandas_stringmethods_autogen_methods:
    if not (method_name.startswith('__') or method_name in _hpat_pandas_stringmethods_autogen_exceptions):
        sdc_overload_method(StringMethodsType, method_name)(_hpat_pandas_stringmethods_autogen(method_name))

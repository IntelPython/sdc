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


# -*- coding: utf-8 -*-

import gc
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import re
import unittest

from numba.tests.support import MemoryLeakMixin

import sdc
from sdc.str_arr_ext import StringArray, create_str_arr_from_list, getitem_str_offset, get_data_ptr_ind
from sdc.str_ext import std_str_to_unicode, unicode_to_std_str
from sdc.extensions.sdc_string_view_ext import (
    string_view_create,
    string_view_get_data_ptr,
    string_view_set_data,
    string_view_create_with_data,
)
from sdc.tests.gen_test_data import ParquetGenerator
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import skip_numba_jit
from sdc.utilities.utils import get_ctypes_ptr


class TestStrings(TestCase):

    def test_pass_return(self):
        def test_impl(_str):
            return _str
        hpat_func = self.jit(test_impl)

        # pass single string and return
        arg = 'test_str'
        self.assertEqual(hpat_func(arg), test_impl(arg))

        # pass string list and return
        arg = ['test_str1', 'test_str2']
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_const(self):
        def test_impl():
            return 'test_str'
        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    @skip_numba_jit
    def test_str2str(self):
        str2str_methods = ['capitalize', 'casefold', 'lower', 'lstrip',
                           'rstrip', 'strip', 'swapcase', 'title', 'upper']
        for method in str2str_methods:
            func_text = "def test_impl(_str):\n"
            func_text += "  return _str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = self.jit(test_impl)

            arg = ' \tbbCD\t '
            self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_equality(self):
        arg = 'test_str'

        def test_impl(_str):
            return (_str == 'test_str')
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(arg), test_impl(arg))

        def test_impl(_str):
            return (_str != 'test_str')
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_concat(self):
        def test_impl(_str):
            return (_str + 'test_str')
        hpat_func = self.jit(test_impl)

        arg = 'a_'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_split(self):
        def test_impl(_str):
            return _str.split('/')
        hpat_func = self.jit(test_impl)

        arg = 'aa/bb/cc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_replace(self):
        def test_impl(_str):
            return _str.replace('/', ';')
        hpat_func = self.jit(test_impl)

        arg = 'aa/bb/cc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_getitem_int(self):
        def test_impl(_str):
            return _str[3]
        hpat_func = self.jit(test_impl)

        arg = 'aa/bb/cc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_int_cast(self):
        def test_impl(_str):
            return int(_str)
        hpat_func = self.jit(test_impl)

        arg = '12'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_float_cast(self):
        def test_impl(_str):
            return float(_str)
        hpat_func = self.jit(test_impl)

        arg = '12.2'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_str_cast(self):
        def test_impl(a):
            return str(a)
        sdc_func = self.jit(test_impl)

        tested_values = [
            np.int32(45),
            43,
            np.float32(1.4),
            np.float64(1.4),
            4.5,
            np.float64(np.nan)
        ]
        for val in tested_values:
            with self.subTest(val=val):
                result_ref = test_impl(val)
                result = sdc_func(val)
                # XXX: use startswith since hpat output can have extra characters
                self.assertTrue(result.startswith(result_ref),
                                f"result={result} not started with {result_ref}")

    def test_re_sub(self):
        def test_impl(_str):
            p = re.compile('ab*')
            return p.sub('ff', _str)
        hpat_func = self.jit(test_impl)

        arg = 'aabbcc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_regex_std(self):
        def test_impl(_str, _pat):
            return sdc.str_ext.contains_regex(
                _str, sdc.str_ext.compile_regex(_pat))
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func('What does the fox say',
                                   r'd.*(the |fox ){2}'), True)
        self.assertEqual(hpat_func('What does the fox say', r'[kz]u*'), False)

    def test_replace_regex_std(self):
        def test_impl(_str, pat, val):
            s = unicode_to_std_str(_str)
            e = sdc.str_ext.compile_regex(unicode_to_std_str(pat))
            val = unicode_to_std_str(val)
            out = sdc.str_ext.str_replace_regex(s, e, val)
            return std_str_to_unicode(out)
        hpat_func = self.jit(test_impl)

        _str = 'What does the fox say'
        pat = r'd.*(the |fox ){2}'
        val = 'does the cat '
        self.assertEqual(
            hpat_func(_str, pat, val),
            _str.replace(re.compile(pat).search(_str).group(), val)
        )

    # string array tests
    def test_string_array_constructor(self):
        # create StringArray and return as list of strings
        def test_impl():
            return StringArray(['ABC', 'BB', 'CDEF'])
        hpat_func = self.jit(test_impl)

        self.assertTrue(np.array_equal(hpat_func(), ['ABC', 'BB', 'CDEF']))

    @skip_numba_jit
    def test_string_array_comp(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A == 'ABC'
            return B.sum()
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(), 1)

    @skip_numba_jit
    def test_string_series(self):
        def test_impl(ds):
            rs = ds == 'one'
            return ds, rs
        hpat_func = self.jit(test_impl)

        df = pd.DataFrame(
            {
                'A': [1, 2, 3] * 33,
                'B': ['one', 'two', 'three'] * 33
            }
        )
        ds, rs = hpat_func(df.B)
        gc.collect()
        self.assertTrue(isinstance(ds, pd.Series) and isinstance(rs, pd.Series))
        self.assertTrue(ds[0] == 'one' and ds[2] == 'three' and rs[0] and not rs[2])

    @skip_numba_jit
    def test_string_array_bool_getitem(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A == 'ABC'
            C = A[B]
            return len(C) == 1 and C[0] == 'ABC'
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(), True)

    @skip_numba_jit
    def test_string_NA_box(self):
        # create `example.parquet` file
        ParquetGenerator.gen_pq_test()

        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.five
        hpat_func = self.jit(test_impl)

        # XXX just checking isna() since Pandas uses None in this case
        # instead of nan for some reason
        np.testing.assert_array_equal(hpat_func().isna(), test_impl().isna())

    # test utf8 decode
    @skip_numba_jit
    def test_decode_empty1(self):
        def test_impl(S):
            return S[0]
        hpat_func = self.jit(test_impl)

        S = pd.Series([''])
        self.assertEqual(hpat_func(S), test_impl(S))

    @skip_numba_jit
    def test_decode_single_ascii_char1(self):
        def test_impl(S):
            return S[0]
        hpat_func = self.jit(test_impl)

        S = pd.Series(['A'])
        self.assertEqual(hpat_func(S)[0], test_impl(S))

    @skip_numba_jit
    def test_decode_ascii1(self):
        def test_impl(S):
            return S[0]
        hpat_func = self.jit(test_impl)

        S = pd.Series(['Abc12', 'bcd', '345'])
        self.assertEqual(hpat_func(S), test_impl(S))

    @unittest.skip("non ascii decode not implement")
    def test_decode_unicode1(self):
        def test_impl(S):
            return S[0]
        hpat_func = self.jit(test_impl)

        S = pd.Series(["¡Y tú quién te crees?", "🐍⚡", "大处着眼，小处着手。"])
        self.assertEqual(hpat_func(S), test_impl(S))

    @unittest.skip("non ascii decode not implement")
    def test_decode_unicode2(self):
        # test strings that start with ascii
        def test_impl(S):
            return S[0], S[1], S[2]
        hpat_func = self.jit(test_impl)

        S = pd.Series(['abc¡Y tú quién te crees?',
                       'dd2🐍⚡', '22 大处着眼，小处着手。', ])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_encode_unicode1(self):
        def test_impl():
            return pd.Series(['¡Y tú quién te crees?',
                              '🐍⚡', '大处着眼，小处着手。', ])
        hpat_func = self.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    @unittest.skip("TODO: explore np array of strings")
    def test_box_np_arr_string(self):
        def test_impl(A):
            return A[0]
        hpat_func = self.jit(test_impl)

        A = np.array(['AA', 'B'])
        self.assertEqual(hpat_func(A), test_impl(A))

    @unittest.skip("No glob support on windows yet. Segfault on Linux if no files found by pattern")
    def test_glob(self):
        def test_impl():
            glob.glob("*py")
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())

    def test_set_string(self):
        def test_impl():
            s = sdc.set_ext.init_set_string()
            s.add('ff')
            for v in s:
                pass
            return v
        hpat_func = self.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())


class TestStringView(MemoryLeakMixin, TestCase):

    def _generate_test_func_for_method(self, func, check_func=None):

        def test_impl(*args):
            data = args[0]
            str_view = string_view_create()
            res = []
            for x in data:
                string_view_set_data(str_view, x._data, x._length)  # reset string view to point to data
                res.append(func(str_view, *args[1:]))               # apply func to string view

            expected = None if check_func is None else [check_func(x, *args[1:]) for x in data]
            return res, expected

        return self.jit(test_impl)

    def _generate_get_ctor_params(self):

        @self.jit
        def get_ctor_params(str_arr, idx):
            start_offset = getitem_str_offset(str_arr, idx)
            data_ptr = get_ctypes_ptr(get_data_ptr_ind(str_arr, start_offset))
            size = getitem_str_offset(str_arr, idx + 1) - start_offset
            return data_ptr, size

        return get_ctor_params

    def test_string_view_get_data_ptr(self):

        tested_method = self.jit(lambda x: string_view_get_data_ptr(x))
        checkup_method = self.jit(lambda x: x._data)
        sdc_func = self._generate_test_func_for_method(tested_method, checkup_method)

        data_strs = ['abca', '123', '', ]
        result, expected = sdc_func(data_strs)
        self.assertEqual(result, expected)

    def test_string_view_len(self):

        tested_method = self.jit(lambda x: len(x))
        sdc_func = self._generate_test_func_for_method(tested_method, tested_method)

        data_strs = ['abca', '123', '', ]
        result, expected = sdc_func(data_strs)
        self.assertEqual(result, expected)

    def test_string_view_toint(self):

        tested_method = self.jit(lambda x: int(x))
        sdc_func = self._generate_test_func_for_method(tested_method, tested_method)

        data_strs = ['500', '11', '10000', ]
        result, expected = sdc_func(data_strs)
        self.assertEqual(result, expected)

    def test_string_view_toint_param_base(self):

        tested_method = self.jit(lambda x, base: int(x, base))
        sdc_func = self._generate_test_func_for_method(tested_method, None)

        data_strs = ['0x500', '0XA8', 'FFF', ]
        result, _ = sdc_func(data_strs, 16)
        # Numba cannot compile int(x, 16) for unicode data, so evaluate expected res here
        expected = [tested_method.py_func(x, 16) for x in data_strs]
        self.assertEqual(result, expected)

    @skip_numba_jit("Numba memleak check is failed since impl raises exception")
    def test_string_view_toint_negative(self):

        tested_method = self.jit(lambda x, base: int(x, base))
        sdc_func = self._generate_test_func_for_method(tested_method, None)

        data_strs = ['F23G', 'FF A', '', ' C1']
        for x in data_strs:
            with self.subTest(x=x):
                with self.assertRaises(ValueError) as raises:
                    sdc_func([x, ], 16)
                msg = 'invalid string for conversion with int()'
                self.assertIn(msg, str(raises.exception))

    def test_string_view_tofloat(self):

        tested_method = self.jit(lambda x: float(x))
        sdc_func = self._generate_test_func_for_method(tested_method, tested_method)

        data_strs = ['0.500', '-1.001', '1.32E+10', '-5e-2']
        result, expected = sdc_func(data_strs)
        self.assertEqual(result, expected)

    def test_string_view_tounicode(self):
        get_str_view_data = self._generate_get_ctor_params()

        def test_impl(x):
            # make native encoded unicode string via StringArrayType
            data_as_str_arr = create_str_arr_from_list([x, ])
            data_ptr, size = get_str_view_data(data_as_str_arr, 0)

            # actual test: create string view pointing to data and convert it back to unicode
            str_view = string_view_create_with_data(data_ptr, size)
            as_unicode = str(str_view)
            check_equal = as_unicode == data_as_str_arr[0]  # for extending the lifetime of str_arr
            return as_unicode, check_equal
        sdc_func = self.jit(test_impl)

        data_to_test = ['大处着眼', 'вавыа', '🐍⚡',  "dfad123/", ]
        for data in data_to_test:
            with self.subTest(data=data):
                result = sdc_func(data)
                self.assertEqual(result[0], data)
                self.assertEqual(result[1], True)

    def test_string_view_unicode_methods(self):
        get_str_view_data = self._generate_get_ctor_params()

        def reference_impl(x, meth_name, args):
            func = getattr(x, meth_name)
            return func(*args)

        def gen_test_impl(self, meth_name):
            from textwrap import dedent
            test_impl_text = dedent(f"""
                def test_impl(x, args):
                    data_as_str_arr = create_str_arr_from_list([x, ])
                    data_ptr, size = get_str_view_data(data_as_str_arr, 0)
                    str_view = string_view_create_with_data(data_ptr, size)
                    res = str_view.{meth_name}(*args)
                    extra_use = len(data_as_str_arr)  # for extending the lifetime of str_arr
                    return res, extra_use
            """)

            global_vars, local_vars = {'create_str_arr_from_list': create_str_arr_from_list,
                                       'get_str_view_data': get_str_view_data,
                                       'string_view_create_with_data': string_view_create_with_data}, {}
            exec(test_impl_text, global_vars, local_vars)
            test_impl = self.jit(local_vars['test_impl'])
            return test_impl

        tested_methods = ['find', 'count', 'split']
        data_to_test = ['大处着眼', 'ва12выа', '🐍⚡',  "dfad123/", ]

        call_args = ('12', )
        for method in tested_methods:
            test_impl = gen_test_impl(self, method)
            for data in data_to_test:
                with self.subTest(method=method, data=data):
                    result_ref = reference_impl(data, method, call_args)
                    result = test_impl(data, call_args)[0]
                    self.assertEqual(result, result_ref)


if __name__ == "__main__":
    unittest.main()

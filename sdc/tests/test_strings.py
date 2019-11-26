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


# -*- coding: utf-8 -*-

import unittest
import platform
import sdc
import numpy as np
import pandas as pd
import glob
import gc
import re
import pyarrow.parquet as pq
from sdc.str_arr_ext import StringArray
from sdc.str_ext import unicode_to_std_str, std_str_to_unicode
from sdc.tests.gen_test_data import ParquetGenerator
from sdc.tests.test_utils import TestCase

class TestStrings(TestCase):

    def test_pass_return(self):
        def test_impl(_str):
            return _str
        hpat_func = sdc.jit(test_impl)

        # pass single string and return
        arg = 'test_str'
        self.assertEqual(hpat_func(arg), test_impl(arg))

        # pass string list and return
        arg = ['test_str1', 'test_str2']
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_const(self):
        def test_impl():
            return 'test_str'
        hpat_func = sdc.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_str2str(self):
        str2str_methods = ['capitalize', 'casefold', 'lower', 'lstrip',
                           'rstrip', 'strip', 'swapcase', 'title', 'upper']
        for method in str2str_methods:
            func_text = "def test_impl(_str):\n"
            func_text += "  return _str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            hpat_func = sdc.jit(test_impl)

            arg = ' \tbbCD\t '
            self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_equality(self):
        arg = 'test_str'

        def test_impl(_str):
            return (_str == 'test_str')
        hpat_func = sdc.jit(test_impl)

        self.assertEqual(hpat_func(arg), test_impl(arg))

        def test_impl(_str):
            return (_str != 'test_str')
        hpat_func = sdc.jit(test_impl)

        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_concat(self):
        def test_impl(_str):
            return (_str + 'test_str')
        hpat_func = sdc.jit(test_impl)

        arg = 'a_'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_split(self):
        def test_impl(_str):
            return _str.split('/')
        hpat_func = sdc.jit(test_impl)

        arg = 'aa/bb/cc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_replace(self):
        def test_impl(_str):
            return _str.replace('/', ';')
        hpat_func = sdc.jit(test_impl)

        arg = 'aa/bb/cc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_getitem_int(self):
        def test_impl(_str):
            return _str[3]
        hpat_func = sdc.jit(test_impl)

        arg = 'aa/bb/cc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_int_cast(self):
        def test_impl(_str):
            return int(_str)
        hpat_func = sdc.jit(test_impl)

        arg = '12'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_float_cast(self):
        def test_impl(_str):
            return float(_str)
        hpat_func = sdc.jit(test_impl)

        arg = '12.2'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_str_cast(self):
        def test_impl(a):
            return str(a)
        hpat_func = sdc.jit(test_impl)

        for arg in [np.int32(45), 43, np.float32(1.4), 4.5]:
            py_res = test_impl(arg)
            h_res = hpat_func(arg)
            # XXX: use startswith since hpat output can have extra characters
            self.assertTrue(h_res.startswith(py_res))

    def test_re_sub(self):
        def test_impl(_str):
            p = re.compile('ab*')
            return p.sub('ff', _str)
        hpat_func = sdc.jit(test_impl)

        arg = 'aabbcc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_regex_std(self):
        def test_impl(_str, _pat):
            return sdc.str_ext.contains_regex(
                _str, sdc.str_ext.compile_regex(_pat))
        hpat_func = sdc.jit(test_impl)

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
        hpat_func = sdc.jit(test_impl)

        _str = 'What does the fox say'
        pat = r'd.*(the |fox ){2}'
        val = 'does the cat '
        self.assertEqual(
            hpat_func(_str, pat, val),
            _str.replace(re.compile(pat).search(_str).group(), val)
        )

    def test_replace_noregex_std(self):
        def test_impl(_str, pat, val):
            s = unicode_to_std_str(_str)
            e = unicode_to_std_str(pat)
            val = unicode_to_std_str(val)
            out = sdc.str_ext.str_replace_noregex(s, e, val)
            return std_str_to_unicode(out)
        hpat_func = sdc.jit(test_impl)

        _str = 'What does the fox say'
        pat = 'does the fox'
        val = 'does the cat'
        self.assertEqual(
            hpat_func(_str, pat, val),
            _str.replace(pat, val)
        )

    # string array tests
    def test_string_array_constructor(self):
        # create StringArray and return as list of strings
        def test_impl():
            return StringArray(['ABC', 'BB', 'CDEF'])
        hpat_func = sdc.jit(test_impl)

        self.assertTrue(np.array_equal(hpat_func(), ['ABC', 'BB', 'CDEF']))

    def test_string_array_comp(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A == 'ABC'
            return B.sum()
        hpat_func = sdc.jit(test_impl)

        self.assertEqual(hpat_func(), 1)

    def test_string_series(self):
        def test_impl(ds):
            rs = ds == 'one'
            return ds, rs
        hpat_func = sdc.jit(test_impl)

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

    def test_string_array_bool_getitem(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A == 'ABC'
            C = A[B]
            return len(C) == 1 and C[0] == 'ABC'
        hpat_func = sdc.jit(test_impl)

        self.assertEqual(hpat_func(), True)

    def test_string_NA_box(self):
        # create `example.parquet` file
        ParquetGenerator.gen_pq_test()

        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.five
        hpat_func = sdc.jit(test_impl)

        # XXX just checking isna() since Pandas uses None in this case
        # instead of nan for some reason
        np.testing.assert_array_equal(hpat_func().isna(), test_impl().isna())

    # test utf8 decode
    def test_decode_empty1(self):
        def test_impl(S):
            return S[0]
        hpat_func = sdc.jit(test_impl)

        S = pd.Series([''])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_decode_single_ascii_char1(self):
        def test_impl(S):
            return S[0]
        hpat_func = sdc.jit(test_impl)

        S = pd.Series(['A'])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_decode_ascii1(self):
        def test_impl(S):
            return S[0]
        hpat_func = sdc.jit(test_impl)

        S = pd.Series(['Abc12', 'bcd', '345'])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_decode_unicode1(self):
        def test_impl(S):
            return S[0], S[1], S[2]
        hpat_func = sdc.jit(test_impl)

        S = pd.Series(['¡Y tú quién te crees?',
                       '🐍⚡', '大处着眼，小处着手。', ])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_decode_unicode2(self):
        # test strings that start with ascii
        def test_impl(S):
            return S[0], S[1], S[2]
        hpat_func = sdc.jit(test_impl)

        S = pd.Series(['abc¡Y tú quién te crees?',
                       'dd2🐍⚡', '22 大处着眼，小处着手。', ])
        self.assertEqual(hpat_func(S), test_impl(S))

    def test_encode_unicode1(self):
        def test_impl():
            return pd.Series(['¡Y tú quién te crees?',
                              '🐍⚡', '大处着眼，小处着手。', ])
        hpat_func = sdc.jit(test_impl)

        pd.testing.assert_series_equal(hpat_func(), test_impl())

    @unittest.skip("TODO: explore np array of strings")
    def test_box_np_arr_string(self):
        def test_impl(A):
            return A[0]
        hpat_func = sdc.jit(test_impl)

        A = np.array(['AA', 'B'])
        self.assertEqual(hpat_func(A), test_impl(A))

    @unittest.skipIf(platform.system() == 'Windows', "no glob support on windows yet")
    def test_glob(self):
        def test_impl():
            glob.glob("*py")
        hpat_func = sdc.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())

    def test_set_string(self):
        def test_impl():
            s = sdc.set_ext.init_set_string()
            s.add('ff')
            for v in s:
                pass
            return v
        hpat_func = sdc.jit(test_impl)

        self.assertEqual(hpat_func(), test_impl())

    def test_dict_string(self):
        def test_impl():
            s = sdc.dict_ext.dict_unicode_type_unicode_type_init()
            s['aa'] = 'bb'
            return s['aa'], ('aa' in s)
        hpat_func = sdc.jit(test_impl)

        self.assertEqual(hpat_func(), ('bb', True))


if __name__ == "__main__":
    unittest.main()

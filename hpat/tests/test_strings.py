import unittest
import hpat
import numpy as np
import pandas as pd
import glob
import gc
import pyarrow.parquet as pq
from hpat.str_arr_ext import StringArray


class TestString(unittest.TestCase):
    def test_pass_return(self):
        def test_impl(_str):
            return _str
        hpat_func = hpat.jit(test_impl)
        # pass single string and return
        arg = 'test_str'
        self.assertEqual(hpat_func(arg), test_impl(arg))
        # pass string list and return
        arg = ['test_str1', 'test_str2']
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_const(self):
        def test_impl():
            return 'test_str'
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_equality(self):
        def test_impl(_str):
            return (_str=='test_str')
        hpat_func = hpat.jit(test_impl)
        arg = 'test_str'
        self.assertEqual(hpat_func(arg), test_impl(arg))
        def test_impl(_str):
            return (_str!='test_str')
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_concat(self):
        def test_impl(_str):
            return (_str+'test_str')
        hpat_func = hpat.jit(test_impl)
        arg = 'a_'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_split(self):
        def test_impl(_str):
            return _str.split('/')
        hpat_func = hpat.jit(test_impl)
        arg = 'aa/bb/cc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_getitem_int(self):
        def test_impl(_str):
            return _str[3]
        hpat_func = hpat.jit(test_impl)
        arg = 'aa/bb/cc'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_int_cast(self):
        def test_impl(_str):
            return int(_str)
        hpat_func = hpat.jit(test_impl)
        arg = '12'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_float_cast(self):
        def test_impl(_str):
            return float(_str)
        hpat_func = hpat.jit(test_impl)
        arg = '12.2'
        self.assertEqual(hpat_func(arg), test_impl(arg))

    def test_string_str_cast(self):
        def test_impl(a):
            return str(a)
        hpat_func = hpat.jit(test_impl)
        for arg in [np.int32(45), 43, np.float32(1.4), 4.5]:
            py_res = test_impl(arg)
            h_res = hpat_func(arg)
            # XXX: use startswith since hpat output can have extra characters
            self.assertTrue(h_res.startswith(py_res))

    def test_regex(self):
        def test_impl(_str, _pat):
            return hpat.str_ext.contains_regex(_str, hpat.str_ext.compile_regex(_pat))
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func('What does the fox say', r'd.*(the |fox ){2}'), True)
        self.assertEqual(hpat_func('What does the fox say', r'[kz]u*'), False)


    # string array tests
    def test_string_array_constructor(self):
        # create StringArray and return as list of strings
        def test_impl():
            return StringArray(['ABC', 'BB', 'CDEF'])
        hpat_func = hpat.jit(test_impl)
        self.assertTrue(np.array_equal(hpat_func(), ['ABC', 'BB', 'CDEF']))

    def test_string_array_comp(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A=='ABC'
            return B.sum()
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), 1)

    def test_string_series(self):
        def test_impl(ds):
            rs = ds == 'one'
            return ds, rs
        hpat_func = hpat.jit(test_impl)
        df = pd.DataFrame({'A': [1,2,3]*33, 'B': ['one', 'two', 'three']*33})
        ds, rs = hpat_func(df.B)
        gc.collect()
        self.assertTrue(isinstance(ds, pd.Series) and isinstance(rs, pd.Series))
        self.assertTrue(ds[0] == 'one' and ds[2] == 'three' and rs[0] == True and rs[2] == False)

    def test_string_array_bool_getitem(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A=='ABC'
            C = A[B]
            return len(C) == 1 and C[0] == 'ABC'
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), True)

    def test_string_NA_box(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.five
        hpat_func = hpat.jit(test_impl)
        # XXX just checking isna() since Pandas uses None in this case
        # instead of nan for some reason
        np.testing.assert_array_equal(hpat_func().isna(), test_impl().isna())

    @unittest.skip("TODO: explore np array of strings")
    def test_box_np_arr_string(self):
        def test_impl(A):
            return A[0]
        hpat_func = hpat.jit(test_impl)
        A = np.array(['AA', 'B'])
        self.assertEqual(hpat_func(A), test_impl(A))

    @unittest.skip("TODO: crashes, llvm ir is invalid?")
    def test_glob(self):
        def test_impl():
            glob.glob("*py")

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_set_string(self):
        def test_impl():
            s = hpat.set_ext.init_set_string()
            s.add('ff')
            for v in s:
                pass
            return v

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())


if __name__ == "__main__":
    unittest.main()

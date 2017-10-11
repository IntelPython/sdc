import unittest
import hpat
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

    # string array tests
    def test_string_array_constructor(self):
        # create StringArray and return as list of strings
        def test_impl():
            return StringArray(['ABC', 'BB', 'CDEF'])
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), ['ABC', 'BB', 'CDEF'])

    def test_string_array_comp(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A=='ABC'
            return B.sum()
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), 1)

if __name__ == "__main__":
    unittest.main()

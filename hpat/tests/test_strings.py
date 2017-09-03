import unittest
import hpat

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

if __name__ == "__main__":
    unittest.main()

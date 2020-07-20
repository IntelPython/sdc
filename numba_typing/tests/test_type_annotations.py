import unittest
import type_annotations
from typing import Union, Dict, List


class TestTypeAnnotations(unittest.TestCase):

    def test_get_func_annotations_exception(self):

        def foo(a: int, b, c: str = "string"):
            pass
        self.assertRaises(SyntaxError,  type_annotations.get_func_annotations, foo)

    def test_get_cls_annotations(self):
        class MyClass(object):
            x: int = 3
            y: str = "string"

            def __init__(self, x: str, y: int):
                self.x = x
                self.y = y

        self.assertEqual(type_annotations.get_cls_annotations(MyClass), ({'x': int, 'y': str}, {}))

    def test_get_func_annotations(self):
        def func_one(a: int, b: Union[int, float], c: str):
            pass
        with self.subTest("annotations"):
            self.assertEqual(type_annotations.get_func_annotations(func_one),
                             ({'a': int, 'b': [int, float], 'c': str}, {}))

        def func_two(a: int = 2, b: str = "string", c: List[int] = [1, 2, 3]):
            pass
        with self.subTest("annotations and all default values"):
            self.assertEqual(type_annotations.get_func_annotations(func_two),
                             ({'a': int, 'b': str, 'c': List[int]}, {'a': 2, 'b': 'string', 'c': [1, 2, 3]}))

        def func_three(a: Dict[int, str], b: str = "string", c: int = 1):
            pass
        with self.subTest("annotations and not all default values"):
            self.assertEqual(type_annotations.get_func_annotations(func_three),
                             ({'a': Dict[int, str], 'b': str, 'c': int}, {'b': 'string', 'c': 1}))


if __name__ == '__main__':
    unittest.main()

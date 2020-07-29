import unittest
import type_annotations
from typing import Union, Dict, List, TypeVar


class TestTypeAnnotations(unittest.TestCase):

    def test_get_func_annotations_exceptions(self):

        def foo(a: int, b, c: str = "string"):
            pass
        with self.assertRaises(SyntaxError) as raises:
            type_annotations.get_func_annotations(foo)
        self.assertIn('No annotation for parameter b', str(raises.exception))

    def test_get_cls_annotations(self):
        class TestClass(object):
            x: int = 3
            y: str = "string"

            def __init__(self, x, y):
                self.x = x
                self.y = y

        result = type_annotations.get_cls_annotations(TestClass)
        expected = ({'x': [int], 'y': [str]}, {})
        self.assertEqual(result, expected)

    def test_get_func_annotations(self):

        def func_one(a: int, b: Union[int, float], c: str):
            pass

        def func_two(a: int = 2, b: str = "string", c: List[int] = [1, 2, 3]):
            pass

        def func_three(a: Dict[int, str], b: str = "string", c: int = 1):
            pass

        expected_results = {
            func_one: ({'a': [int], 'b': [int, float], 'c': [str]}, {}),
            func_two: ({'a': [int], 'b': [str], 'c': [List[int]]}, {'a': 2, 'b': 'string', 'c': [1, 2, 3]}),
            func_three: ({'a': [Dict[int, str]], 'b': [str], 'c': [int]}, {'b': 'string', 'c': 1}),
        }
        for f, expected in expected_results.items():
            with self.subTest(func=f.__name__):
                self.assertEqual(type_annotations.get_func_annotations(f), expected)

    def test_product_annotations(self):

        S = TypeVar('S', float, str)
        annotations = ({'a': [int], 'b': [int, float], 'c': [S]}, {})
        result = type_annotations.product_annotations(annotations)
        expected = [[{'a': int, 'b': int, 'c': float}, {}],
                    [{'a': int, 'b': int, 'c': str}, {}],
                    [{'a': int, 'b': float, 'c': float}, {}],
                    [{'a': int, 'b': float, 'c': str}, {}]]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()

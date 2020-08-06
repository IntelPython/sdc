import unittest
import type_annotations
from typing import Union, Dict, List, TypeVar


def check_equal(result, expected):
    if len(result) != len(expected):
        return False
    for sig in result:
        if sig not in expected:
            return False
    return True


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

    def test_convert_to_sig_list(self):
        T = TypeVar('T', int, str)
        S = TypeVar('S', float, str)
        annotations = [{'a': [int], 'b': [int, float], 'c': [S]},
                       {'a': [int], 'b': [T], 'c': [S]},
                       {'a': [int, str], 'b': [int, float], 'c': [S]}]

        expected = [[{'a': int, 'b': int, 'c': S},
                     {'a': int, 'b': float, 'c': S}],
                    [{'a': int, 'b': T, 'c': S}],
                    [{'a': int, 'b': int, 'c': S},
                     {'a': int, 'b': float, 'c': S},
                     {'a': str, 'b': int, 'c': S},
                     {'a': str, 'b': float, 'c': S}]]

        for i in range(len(annotations)):
            with self.subTest(annotations=i):
                self.assertEqual(type_annotations.convert_to_sig_list(annotations[i]), expected[i])

    def test_get_typevars(self):
        T = TypeVar('T', int, str)
        S = TypeVar('S', float, str)
        types = [List[T], Dict[T, S], int, T, List[List[T]]]

        expected = [{T}, {T, S}, set(), {T}, {T}]

        for i in range(len(types)):
            with self.subTest(types=i):
                self.assertEqual(type_annotations.get_typevars(types[i]), expected[i])

    def test_add_vals_to_signature(self):
        signature = [{'a': Dict[float, int], 'b': int},
                     {'a': Dict[str, int], 'b': int},
                     {'a': Dict[float, str], 'b': int},
                     {'a': Dict[str, str], 'b': int}]
        vals = {'a': {'name': 3}, 'b': 3}

        result = type_annotations.add_vals_to_signature(signature, vals)
        expected = [[{'a': Dict[float, int], 'b': int}, {'a': {'name': 3}, 'b': 3}],
                    [{'a': Dict[str, int], 'b': int}, {'a': {'name': 3}, 'b': 3}],
                    [{'a': Dict[float, str], 'b': int}, {'a': {'name': 3}, 'b': 3}],
                    [{'a': Dict[str, str], 'b': int}, {'a': {'name': 3}, 'b': 3}]]
        self.assertEqual(result, expected)

    def test_exist_typevar(self):
        T = TypeVar('T', float, str)
        S = TypeVar('S', int, str)
        types = [List[List[T]], Dict[T, S], int, T, S]
        expected = [{True}, {False, True}, {False}, {True}, {False}]

        for i in range(len(types)):
            with self.subTest(types=i):
                self.assertEqual(type_annotations.exist_typevar(types[i], T), expected[i])

    def test_replace_typevar(self):
        T = TypeVar('T', int, str)
        S = TypeVar('S', float, str)

        types = [List[List[T]], Dict[T, S], T]
        expected = [List[List[int]], Dict[int, S], int]

        for i in range(len(types)):
            with self.subTest(types=i):
                self.assertEqual(type_annotations.replace_typevar(types[i], T, int), expected[i])

    def test_get_internal_typevars(self):

        T = TypeVar('T', int, str)
        S = TypeVar('S', float, bool)
        signature = {'a': T, 'b': Dict[T, S]}
        expected = [{'a': int, 'b': Dict[int, float]},
                    {'a': int, 'b': Dict[int, bool]},
                    {'a': str, 'b': Dict[str, float]},
                    {'a': str, 'b': Dict[str, bool]}]

        result = type_annotations.get_internal_typevars(signature)

        self.assertTrue(check_equal(result, expected))

    def test_update_sig(self):
        T = TypeVar('T', int, str)
        S = TypeVar('S', float, bool)

        sig = {'a': T, 'b': Dict[T, S]}
        expected = [{'a': T, 'b': Dict[T, float]},
                    {'a': T, 'b': Dict[T, bool]}]
        result = type_annotations.update_sig(sig, S)

        self.assertEqual(result, expected)

    def test_expand_typevars(self):
        T = TypeVar('T', int, str)
        S = TypeVar('S', float, bool)

        sig = {'a': T, 'b': Dict[T, S], 'c': int}
        unique_typevars = {T, S}
        expected = [{'a': int, 'b': Dict[int, float], 'c': int},
                    {'a': int, 'b': Dict[int, bool], 'c': int},
                    {'a': str, 'b': Dict[str, float], 'c': int},
                    {'a': str, 'b': Dict[str, bool], 'c': int}]

        result = type_annotations.expand_typevars(sig, unique_typevars)

        self.assertTrue(check_equal(result, expected))

    def test_product_annotations(self):

        T = TypeVar('T', int, str)
        S = TypeVar('S', float, bool)

        annotations = ({'a': [T], 'b': [Dict[T, S]],
                        'c': [T, bool], 'd': [int]}, {'d': 3})

        expected = [[{'a': int, 'b': Dict[int, float], 'c': int, 'd': int}, {'d': 3}],
                    [{'a': int, 'b': Dict[int, bool], 'c': int, 'd': int}, {'d': 3}],
                    [{'a': str, 'b': Dict[str, float], 'c': str, 'd': int}, {'d': 3}],
                    [{'a': str, 'b': Dict[str, bool], 'c': str, 'd': int}, {'d': 3}],
                    [{'a': int, 'b': Dict[int, float], 'c': bool, 'd': int}, {'d': 3}],
                    [{'a': int, 'b': Dict[int, bool], 'c': bool, 'd': int}, {'d': 3}],
                    [{'a': str, 'b': Dict[str, float], 'c': bool, 'd': int}, {'d': 3}],
                    [{'a': str, 'b': Dict[str, bool], 'c': bool, 'd': int}, {'d': 3}]]

        result = type_annotations.product_annotations(annotations)

        self.assertTrue(check_equal(result, expected))


if __name__ == '__main__':
    unittest.main()

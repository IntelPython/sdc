from local_variable_type_annotations import get_variable_annotations
import test_generics
import unittest
from typing import Any, Union, List, Tuple, Dict, Iterable, Iterator, Generic, TypeVar
import typing

T = TypeVar('T')
S = TypeVar('S', int, str)
G = Generic[T, S]


class A():
    ...


class B():
    ...


class C():
    ...


class TestAst(unittest.TestCase):
    maxDiff = None

    def test_get_variable_annotations_with_test_generics(self):
        result = get_variable_annotations(test_generics.qwe)
        expected_result = {'qwe_int': [int], 'qwe_list': [test_generics.List[int]],
                           'qwe_tuple': [test_generics.Tuple[str, int]], 'qwe_dict': [test_generics.Dict[str, str]],
                           'qwe_any': [test_generics.Any], 'qwe_int_op': [test_generics.typing.Optional[int]],
                           'qwe_int_float_un': [test_generics.Union[int, float]], 'T_func': [test_generics.T],
                           'S_func': [test_generics.S], 'G_func': [test_generics.G],
                           'User_func': [test_generics.UserId], 'class_func': [test_generics.A]}
        self.assertEqual(result, expected_result)

    def test_get_variable_annotations_standard_types(self):
        def test_func():
            t = 1
            t_int: int
            t_str: str
            t_float: float
            t_bool: bool
            t_bytes: bytes
            t_list: list
            t_dict: dict
            t_tuple: tuple
            t_set: set
        result = get_variable_annotations(test_func)
        expected_result = {'t_int': [int], 't_str': [str], 't_float': [float], 't_bool': [bool], 't_bytes': [bytes],
                           't_list': [list], 't_dict': [dict], 't_tuple': [tuple], 't_set': [set]}
        self.assertEqual(result, expected_result)

    def test_get_variable_annotations_generic_types(self):
        def test_func():
            t_any: Any
            t_union: Union[str, bytes]
            t_optional: typing.Optional[float]
            t_list: List[int]
            t_tuple: Tuple[str]
            t_dict: Dict[str, str]
            t_iterable: Iterable[float]
            t_iterator: Iterator[int]
            t_generic: G
            t_typevar_t: T
            t_typevar_s: S
            t_list_typevar: List[T]
            t_tuple_typevar: Tuple[S]
            t_dict_typevar: Dict[T, S]
        result = get_variable_annotations(test_func)
        expected_result = {'t_any': [Any], 't_union': [Union[str, bytes]], 't_optional': [typing.Optional[float]], 't_list': [List[int]],
                           't_tuple': [Tuple[str]], 't_dict': [Dict[str, str]], 't_iterable': [Iterable[float]],
                           't_iterator': [Iterator[int]], 't_generic': [G], 't_typevar_t': [T], 't_typevar_s': [S],
                           't_list_typevar': [List[T]], 't_tuple_typevar': [Tuple[S]], 't_dict_typevar': [Dict[T, S]]}
        self.assertEqual(result, expected_result)

    def test_get_variable_annotations_user_types(self):
        def test_func():
            t_class_a: A
            t_class_b: B
            t_class_c: C
        result = get_variable_annotations(test_func)
        expected_result = {'t_class_a': [A], 't_class_b': [B], 't_class_c': [C]}
        self.assertEqual(result, expected_result)

    def test_get_variable_annotations_non_locals(self):
        def foo():
            Q = TypeVar('Q')

            def bar():
                t_typevar: Q
            ...
            return bar
        test_func = foo()
        result = get_variable_annotations(test_func)
        expected_result = {'t_typevar': [Q]}
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()

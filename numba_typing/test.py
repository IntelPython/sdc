import ast_example
import unittest
from typing import Any, Union, Optional, List, Tuple, Dict, Iterable, Iterator, Generic, TypeVar


T = TypeVar('T')
S = TypeVar('S', int, str)
# G = Generic[T, S]


class TestAst(unittest.TestCase):
    maxDiff = None

    def test_get_variable_annotations_with_test_generics(self):
        result = ast_example.get_variable_annotations(ast_example.test_generics.qwe)
        expected_result = {'qwe_int': int, 'qwe_list': ast_example.test_generics.List[int],
                           'qwe_tuple': ast_example.test_generics.Tuple[str, int], 'qwe_dict': ast_example.test_generics.Dict[str, str],
                           'qwe_any': ast_example.test_generics.Any, 'qwe_int_op': ast_example.test_generics.typing.Optional[int],
                           'qwe_int_float_un': ast_example.test_generics.Union[int, float], 'T_func': ast_example.test_generics.T,
                           'S_func': ast_example.test_generics.S, 'G_func': ast_example.test_generics.G,
                           'User_func': ast_example.test_generics.UserId, 'class_func': ast_example.test_generics.A}
        self.assertEqual(result, expected_result)

    def test_get_variable_annotations_standard_types(self):
        def test_func():
            t_int: int
            t_str: str
            t_float: float
            t_bool: bool
            t_bytes: bytes
            t_list: list
            t_dict: dict
            t_tuple: tuple
            t_set: set
        result = ast_example.get_variable_annotations(test_func)
        expected_result = {'t_int': int, 't_str': str, 't_float': float, 't_bool': bool, 't_bytes': bytes,
                           't_list': list, 't_dict': dict, 't_tuple': tuple, 't_set': set}
        self.assertEqual(result, expected_result)

    def test_get_variable_annotations_generic_types(self):

        def test_func():
            # t_any: Any
            # t_union: Union[str, bytes]
            # t_optional: Optional[float]
            # t_list: List[int]
            # t_tuple: Tuple[str]
            # t_dict: Dict[str, str]
            # t_iterable: Iterable[float]
            # t_iterator: Iterator[int]
            # t_generic: G
            t_typevar: T
        result = ast_example.get_variable_annotations(test_func)
        # print(id(result['t_typevar']))
        # expected_result = {'t_any': Any, 't_union': Union[str, bytes], 't_optional': Optional[float], 't_list': List[int],
        #                    't_tuple': Tuple[str], 't_dict': Dict[str, str], 't_iterable': Iterable[float],
        #                    't_iterator': Iterator[int]}
        expected_result = {'t_typevar': T}
        # print(id(expected_result['t_typevar']))
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()

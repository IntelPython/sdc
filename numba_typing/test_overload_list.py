import overload_list
from overload_list import List, Dict
from overload_list import types
import unittest
import typing
from numba import njit, core
import re


T = typing.TypeVar('T')
K = typing.TypeVar('K')
S = typing.TypeVar('S', int, float)
UserType = typing.NewType('UserType', int)


def generator_test(param, values_dict, defaults_dict={}):

    def check_type(typ):
        if isinstance(typ, type):
            return typ.__name__
        return typ

    value_keys = ", ".join(f"{key}" if key not in defaults_dict.keys()
                           else f"{key} = {defaults_dict[key]}" for key in values_dict.keys())
    value_annotation = ", ".join(f"{key}: {check_type(val)}" if key not in defaults_dict.keys()
                                 else f"{key}: {check_type(val)} = {defaults_dict[key]}"
                                 for key, val in values_dict.items())
    value_type = ", ".join(f"{val}" for val in values_dict.values())
    return_value_keys = ", ".join("{}".format(key) for key in values_dict.keys())
    param_func = ", ".join(f"{val}" for val in param)
    test = f"""
def test_myfunc():
    def foo({value_keys}):
        ...

    @overload_list.overload_list(foo)
    def foo_ovld_list():

        def foo({value_annotation}):
            return ("{value_type}")

        return (foo,)

    @njit
    def jit_func({value_keys}):
        return foo({return_value_keys})

    return (jit_func({param_func}), ("{value_type}"))
"""
    loc = {}
    exec(test, globals(), loc)
    return loc


list_numba = List([1, 2, 3])
nested_list_numba = List([List([1, 2])])
dict_numba = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
dict_numba_1 = Dict.empty(key_type=types.int64, value_type=types.boolean)
dict_numba['qwe'] = 1
dict_numba_1[1] = True
list_type = types.ListType(types.int64)
list_in_dict_numba = Dict.empty(key_type=types.unicode_type, value_type=list_type)
list_in_dict_numba['qwe'] = List([3, 4, 5])
str_variable = 'qwe'
str_variable_1 = 'qaz'
user_type = UserType(1)


def run_test(case):
    run_generator = generator_test(*case)
    received, expected = run_generator['test_myfunc']()
    return (received, expected)


def run_test_with_error(case):
    run_generator = generator_test(*case)
    try:
        run_generator['test_myfunc']()
    except core.errors.TypingError as err:
        res = re.search(r'TypeError', err.msg)
        return res.group(0)


class TestOverload(unittest.TestCase):
    maxDiff = None

    def test_standart_types(self):
        test_cases = [([1], {'a': int}), ([1.0], {'a': float}), ([True], {'a': bool}), (['str_variable'], {'a': str})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_container_types(self):
        test_cases = [([[1, 2]], {'a': list}), ([(1.0, 2.0)], {'a': tuple}), (['dict_numba'], {'a': dict})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_typing_types(self):
        test_cases = [([[1.0, 2.0]], {'a': typing.List[float]}), (['list_numba'], {'a': typing.List[int]}),
                      ([(1, 2.0)], {'a': typing.Tuple[int, float]}), (['dict_numba_1'], {'a': typing.Dict[int, bool]}),
                      ([True, 'str_variable'], {'a': typing.Union[bool, str], 'b': typing.Union[bool, str]}),
                      ([1, False], {'a': typing.Any, 'b': typing.Any})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_nested_typing_types(self):
        test_cases = [(['nested_list_numba'], {'a': typing.List[typing.List[int]]}),
                      ([((1.0,),)], {'a': typing.Tuple[typing.Tuple[float]]})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_typevar_types(self):
        test_cases = [([1.0], {'a': 'T'}), ([False], {'a': 'T'}), (['list_numba', [1, 2]], {'a': 'T', 'b': 'T'}),
                      ([1, 2.0], {'a': 'T', 'b': 'K'}), ([1], {'a': 'S'}), ([1.0], {'a': 'S'}),
                      ([[True, True]], {'a': 'typing.List[T]'}), (['list_numba'], {'a': 'typing.List[T]'}),
                      ([('str_variable', 2)], {'a': 'typing.Tuple[T,K]'}),
                      (['dict_numba_1'], {'a': 'typing.Dict[K, T]'}), (['dict_numba'], {'a': 'typing.Dict[K, T]'}),
                      (['list_in_dict_numba'], {'a': 'typing.Dict[K, typing.List[T]]'})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_only_default_types(self):
        test_cases = [([], {'a': int}, {'a': 1}), ([], {'a': float}, {'a': 1.0}), ([], {'a': bool}, {'a': True}),
                      ([], {'a': str}, {'a': 'str_variable'})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_overriding_default_types(self):
        test_cases = [([5], {'a': int}, {'a': 1}), ([5.0], {'a': float}, {'a': 1.0}),
                      ([False], {'a': bool}, {'a': True}), (['str_variable_1'], {'a': str}, {'a': 'str_variable'})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_two_types(self):
        test_cases = [([5, 3.0], {'a': int, 'b': float}), ([5, 3.0], {'a': int, 'b': float}, {'b': 0.0}),
                      ([5], {'a': int, 'b': float}, {'b': 0.0}), ([], {'a': int, 'b': float}, {'a': 0, 'b': 0.0})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_three_types(self):
        test_cases = [([5, 3.0, 'str_variable_1'], {'a': int, 'b': float, 'c': str}),
                      ([5, 3.0], {'a': int, 'b': float, 'c': str}, {'c': 'str_variable'}),
                      ([5], {'a': int, 'b': float, 'c': str}, {'b': 0.0, 'c': 'str_variable'}),
                      ([], {'a': int, 'b': float, 'c': str}, {'a': 0, 'b': 0.0, 'c': 'str_variable'})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(*run_test(case))

    def test_type_error(self):
        test_cases = [([1], {'a': float}), ([], {'a': float}, {'a': 1}), ([1], {'a': typing.Iterable[int]}),
                      ([(1, 2, 3), (1.0, 2.0)], {'a': typing.Tuple[int, int],
                                                 'b':tuple}), ([1, 2.0], {'a': 'T', 'b': 'T'}),
                      ([1, True], {'a': 'T', 'b': 'S'})]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(run_test_with_error(case), 'TypeError')

    def test_attribute_error(self):
        def foo(a=0):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo(a: int):
                return (a,)

            return (foo,)

        @njit
        def jit_func():
            return foo()

        try:
            jit_func()
        except core.errors.TypingError as err:
            res = re.search(r'AttributeError', err.msg)
            self.assertEqual(res.group(0), 'AttributeError')

    def test_value_error(self):
        def foo(a=0):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo(a: int = 1):
                return (a,)

            return (foo,)

        @njit
        def jit_func():
            return foo()

        try:
            jit_func()
        except core.errors.TypingError as err:
            res = re.search(r'ValueError', err.msg)
            self.assertEqual(res.group(0), 'ValueError')


if __name__ == "__main__":
    unittest.main()

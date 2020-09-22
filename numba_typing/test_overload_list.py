import overload_list
from overload_list import List, Dict
from overload_list import types
import unittest
import typing
from numba import njit, core


T = typing.TypeVar('T')
K = typing.TypeVar('K')


class TestOverloadList(unittest.TestCase):
    maxDiff = None

    def test_myfunc_literal_type_default(self):
        def foo(a, b=0):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo_int_literal(a: int, b: int = 0):
                return ('literal', a, b)

            return (foo_int_literal,)

        @njit
        def jit_func(a):
            return foo(a, 2)

        self.assertEqual(jit_func(1), ('literal', 1, 2))

    def test_myfunc_tuple_type_error(self):
        def foo(a, b=(0, 0)):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo_tuple(a: typing.Tuple[int, int], b: tuple = (0, 0)):
                return ('tuple_', a, b)

            return (foo_tuple,)

        @njit
        def jit_func(a, b):
            return foo(a, b)

        self.assertRaises(core.errors.TypingError, jit_func, (1, 2, 3), ('3', False))


def generator_test(func_name, param, values_dict, defaults_dict={}):

    def check_type(typ):
        if isinstance(typ, type):
            return typ.__name__
        return typ

    value_keys = ", ".join("{}".format(key) for key in values_dict.keys())
    defaults_keys = ", ".join("{}".format(key) for key in defaults_dict.keys())
    value_str = ", ".join("{}: {}".format(key, check_type(val)) for key, val in values_dict.items())
    defaults_str = ", ".join("{} = {}".format(key, val) if not isinstance(
        val, str) else "{} = '{}'".format(key, val) for key, val in defaults_dict.items())
    defaults_str_type = ", ".join("{}: {} = {}".format(key, check_type(type(val)), val) if not isinstance(val, str)
                                  else "{}: {} = '{}'".format(key, check_type(type(val)), val)
                                  for key, val in defaults_dict.items())
    value_type = ", ".join("{}".format(val) for val in values_dict.values())
    defaults_type = ", ".join("{}".format(type(val)) for val in defaults_dict.values())
    param_qwe = ", ".join("{}".format(i) for i in param)
    test = f"""
def test_myfunc_{func_name}_type_default(self):
    def foo({value_keys},{defaults_str}):
        ...

    @overload_list.overload_list(foo)
    def foo_ovld_list():

        def foo_{func_name}({value_str},{defaults_str_type}):
            return ("{value_type}","{defaults_type}")

        return (foo_{func_name},)

    @njit
    def jit_func({value_keys},{defaults_str}):
        return foo({value_keys},{defaults_keys})

    self.assertEqual(jit_func({param_qwe}), ("{value_type}", "{defaults_type}"))
"""
    loc = {}
    exec(test, globals(), loc)
    return loc


L = List([1, 2, 3])
L_int = List([List([1, 2])])
L_float = List([List([List([3.0, 4.0])])])
L_f = List([1.0, 2.0])
D = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
D_1 = Dict.empty(key_type=types.int64, value_type=types.boolean)
D['qwe'] = 1
D['qaz'] = 2
D_1[1] = True
D_1[0] = False
list_type = types.ListType(types.int64)
D_list = Dict.empty(key_type=types.unicode_type, value_type=list_type)
D_list['qwe'] = List([3, 4, 5])
str_1 = 'qwe'
str_2 = 'qaz'
test_cases = [('int', [1, 2], {'a': int, 'b': int}), ('float', [1.0, 2.0], {'a': float, 'b': float}),
              ('bool', [True, True], {'a': bool, 'b': bool}), ('str', ['str_1', 'str_2'], {'a': str, 'b': str}),
              ('list', [[1, 2], [3, 4]], {'a': typing.List[int], 'b':list}),
              ('List_typed', [L, [3, 4]], {'a': typing.List[int], 'b':list}),
              ('tuple', [(1, 2.0), ('3', False)], {'a': typing.Tuple[int, float], 'b':tuple}),
              ('dict', ['D', 'D_1'], {'a': typing.Dict[str, int], 'b': typing.Dict[int, bool]}),
              ('union_1', [1, False], {'a': typing.Union[int, str], 'b': typing.Union[float, bool]}),
              ('union_2', ['str_1', False], {'a': typing.Union[int, str], 'b': typing.Union[float, bool]}),
              ('nested_list', ['L_int', 'L_float'], {'a': typing.List[typing.List[int]],
                                                     'b': typing.List[typing.List[typing.List[float]]]}),
              ('TypeVar_TT', ['L_f', [3.0, 4.0]], {'a': 'T', 'b': 'T'}),
              ('TypeVar_TK', [1.0, 2], {'a': 'T', 'b': 'K'}),
              ('TypeVar_ListT_T', ['L', 5], {'a': 'typing.List[T]', 'b': 'T'}),
              ('TypeVar_ListT_DictKT', ['L', 'D'], {'a': 'typing.List[T]', 'b': 'typing.Dict[K, T]'}),
              ('TypeVar_ListT_DictK_ListT', ['L', 'D_list'], {'a': 'typing.List[T]',
                                                              'b': 'typing.Dict[K, typing.List[T]]'})]

test_cases_default = [('int_defaults', [1], {'a': int}, {'b': 0}), ('float_defaults', [1.0], {'a': float}, {'b': 0.0}),
                      ('bool_defaults', [True], {'a': bool}, {'b': False}),
                      ('str_defaults', ['str_1'], {'a': str}, {'b': '0'}),
                      ('tuple_defaults', [(1, 2)], {'a': tuple}, {'b': (0, 0)})]


for name, val, annotation in test_cases:
    run_generator = generator_test(name, val, annotation)
    test_name = list(run_generator.keys())[0]
    setattr(TestOverloadList, test_name, run_generator[test_name])


for name, val, annotation, defaults in test_cases_default:
    run_generator = generator_test(name, val, annotation, defaults)
    test_name = list(run_generator.keys())[0]
    setattr(TestOverloadList, test_name, run_generator[test_name])


if __name__ == "__main__":
    unittest.main()

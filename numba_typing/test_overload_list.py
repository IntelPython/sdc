import overload_list
from overload_list import List, Dict
from overload_list import types
import unittest
import typing
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, njit, core
import warnings


def func(a, b):
    ...


T = typing.TypeVar('T')
K = typing.TypeVar('K')


@overload_list.overload_list(func)
def foo_ovld_list():

    def foo_int(a: int, b: int = 0):
        return ('int', a, b)

    def foo_float(a: float, b: float = 0.0):
        return ('float', a, b)

    def foo_bool(a: bool, b: bool = False):
        return ('bool', a, b)

    def foo_str(a: str, b: str = '0'):
        return ('str', a, b)

    def foo_list(a: typing.List[int], b: list = [0, 0, 0]):
        return ('list', a, b)

    def foo_tuple(a: typing.Tuple[int, float], b: tuple = (0, 0)):
        return ('tuple', a, b)

    def foo_dict(a: typing.Dict[str, int], b: typing.Dict[int, bool] = {0: False}):
        return ('dict', a, b)

    def foo_union(a: typing.Union[int, str], b: typing.Union[float, bool] = None):
        return('union', a, b)

    def foo_list_in_list(a: typing.List[typing.List[int]],
                         b: typing.List[typing.List[typing.List[float]]] = [[[0.0, 0.0]]]):
        return('list_in_list', a, b)

    def foo_tuple_in_tuple(a: typing.Tuple[typing.Tuple[int, int]],
                           b: typing.Tuple[typing.Tuple[typing.Tuple[float, float]]] = ((0.0, 0.0))):
        return('tuple_in_tuple', a, b)

    def foo_typevars_T_T(a: T, b: T):
        return('TypeVars_TT', a, b)

    def foo_typevars_T_K(a: T, b: K):
        return('TypeVars_TK', a, b)

    def foo_typevars_list_T(a: typing.List[T], b: T):
        return('TypeVars_ListT', a, b)

    def foo_typevars_list_dict(a: typing.List[T], b: typing.Dict[K, T]):
        return('TypeVars_List_Dict', a, b)

    def foo_typevars_list_dict_list(a: typing.List[T], b: typing.Dict[K, typing.List[T]]):
        return('TypeVars_List_Dict_List', a, b)

    return foo_int, foo_float, foo_bool, foo_str, foo_list, foo_tuple, foo_dict, foo_union,\
        foo_list_in_list, foo_tuple_in_tuple, foo_typevars_T_T, foo_typevars_list_T,\
        foo_typevars_list_dict, foo_typevars_list_dict_list, foo_typevars_T_K


@njit
def jit_func(a, b):
    return func(a, b)


class TestOverloadListDefault(unittest.TestCase):
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

    def test_myfunc_int_type_default(self):
        def foo(a, b=0):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo_int(a: int, b: int = 0):
                return ('int', a, b)

            return (foo_int,)

        @njit
        def jit_func(a):
            return foo(a)

        self.assertEqual(jit_func(1), ('int', 1, 0))

    def test_myfunc_float_type_default(self):
        def foo(a, b=0.0):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo_float(a: float, b: float = 0.0):
                return ('float', a, b)

            return (foo_float,)

        @njit
        def jit_func(a):
            return foo(a)

        self.assertEqual(jit_func(1.0), ('float', 1.0, 0.0))

    def test_myfunc_bool_type_default(self):
        def foo(a, b=False):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo_bool(a: bool, b: bool = False):
                return ('bool', a, b)

            return (foo_bool,)

        @njit
        def jit_func(a):
            return foo(a)

        self.assertEqual(jit_func(True), ('bool', True, False))

    def test_myfunc_str_type_default(self):
        def foo(a, b='0'):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo_str(a: str, b: str = '0'):
                return ('str', a, b)

            return (foo_str,)

        @njit
        def jit_func(a):
            return foo(a)

        self.assertEqual(jit_func('qwe'), ('str', 'qwe', '0'))

    def test_myfunc_tuple_type_default(self):
        def foo(a, b=(0, 0)):
            ...

        @overload_list.overload_list(foo)
        def foo_ovld_list():

            def foo_tuple(a: tuple, b: tuple = (0, 0)):
                return ('tuple', a, b)

            return (foo_tuple,)

        @njit
        def jit_func(a):
            return foo(a)

        self.assertEqual(jit_func((1, 2)), ('tuple', (1, 2), (0, 0)))

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


class TestOverloadList(unittest.TestCase):
    maxDiff = None

    def test_myfunc_int_type(self):
        self.assertEqual(jit_func(1, 2), ('int', 1, 2))

    def test_myfunc_float_type(self):
        self.assertEqual(jit_func(1.0, 2.0), ('float', 1.0, 2.0))

    def test_myfunc_bool_type(self):
        self.assertEqual(jit_func(True, True), ('bool', True, True))

    def test_myfunc_str_type(self):
        self.assertEqual(jit_func('qwe', 'qaz'), ('str', 'qwe', 'qaz'))

    def test_myfunc_list_type(self):
        self.assertEqual(jit_func([1, 2], [3, 4]), ('list', [1, 2], [3, 4]))

    def test_myfunc_List_typed(self):
        warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
        warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
        L = List([1, 2, 3])
        self.assertEqual(jit_func(L, [3, 4]), ('list', L, [3, 4]))

    def test_myfunc_tuple_type(self):
        self.assertEqual(jit_func((1, 2.0), ('3', False)), ('tuple', (1, 2.0), ('3', False)))

    def test_myfunc_dict_type(self):
        D = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        D_1 = Dict.empty(key_type=types.int64, value_type=types.boolean)
        D['qwe'] = 1
        D['qaz'] = 2
        D_1[1] = True
        D_1[0] = False
        self.assertEqual(jit_func(D, D_1), ('dict', D, D_1))

    def test_myfunc_union_typing_int_bool(self):
        self.assertEqual(jit_func(1, False), ('union', 1, False))

    def test_myfunc_union_typing_str_bool(self):
        self.assertEqual(jit_func('qwe', False), ('union', 'qwe', False))

    def test_myfunc_union_typing_int_float(self):
        self.assertEqual(jit_func(1, 2.0), ('union', 1, 2.0))

    def test_myfunc_union_typing_str_float(self):
        self.assertEqual(jit_func('qwe', 2.0), ('union', 'qwe', 2.0))

    def test_myfunc_list_in_list_type(self):
        L_int = List([List([1, 2])])
        L_float = List([List([List([3.0, 4.0])])])

        self.assertEqual(jit_func(L_int, L_float), ('list_in_list', L_int, L_float))

    def test_myfunc_tuple_in_tuple(self):
        self.assertEqual(jit_func(((1, 2),), (((3.0, 4.0),),)), ('tuple_in_tuple', ((1, 2),), (((3.0, 4.0),),)))

    def test_myfunc_typevar_T_T(self):
        self.assertEqual(jit_func(((1, 2),), ((3, 4),)), ('TypeVars_TT', ((1, 2),), ((3, 4),)))

    def test_myfunc_typevar_T_T_list(self):
        warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
        warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
        V = List([1.0, 2.0])
        self.assertEqual(jit_func(V, [3.0, 4.0]), ('TypeVars_TT', V, [3.0, 4.0]))

    def test_myfunc_typevar_T_K(self):
        self.assertEqual(jit_func(1.0, 2), ('TypeVars_TK', 1.0, 2))

    def test_myfunc_typevar_List_T(self):
        L_int = List([1, 2])
        self.assertEqual(jit_func(L_int, 2), ('TypeVars_ListT', L_int, 2))

    def test_myfunc_typevar_List_Dict(self):
        D = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        D['qwe'] = 0
        L_int = List([1, 2])
        self.assertEqual(jit_func(L_int, D), ('TypeVars_List_Dict', L_int, D))

    def test_myfunc_typevar_List_Dict_List(self):
        list_type = types.ListType(types.int64)
        D = Dict.empty(key_type=types.unicode_type, value_type=list_type)
        D['qwe'] = List([3, 4, 5])
        L_int = List([1, 2])

        self.assertEqual(jit_func(L_int, D), ('TypeVars_List_Dict_List', L_int, D))


if __name__ == "__main__":
    unittest.main()

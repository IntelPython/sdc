from typing import Any, Union, List, Tuple, Dict, TypeVar, Iterable, Generic, NewType
import typing


foo_int: int = 10
foo_str: str = 'ten'
foo_float: float = 10.5
foo_bool: bool = True
foo_bytes: bytes = b'\x00'
foo_global: int = 30

foo_list: List[str] = ['Hello', 'world']
foo_tuple: Tuple[int] = (1,)
foo_dict: Dict[str, int] = {'Hello': 1}

T = TypeVar('T')
S = TypeVar('S', int, str)
G = Generic[T]
UserId = NewType('UserId', int)


class A():
    ...


def qwe():
    qwe_int: int = 10
    qwe_list: List[int] = [1, 2, 3]
    qwe_tuple: Tuple[str, int] = ("qqq", 1)
    qwe_dict: Dict[str, str] = {'hello': 'world'}
    qwe_any: Any
    qwe_int_op: typing.Optional[int]
    qwe_int_float_un: Union[int, float]
    T_func: T
    S_func: S
    G_func: G
    User_func: UserId
    class_func: A

from typing import Optional, Any, Union, List, Tuple, Dict

foo_int: int = 10
foo_str: str = 'ten'
foo_float: float = 10.5
foo_bool: bool = True
foo_bytes: bytes = b'\x00'
foo_global: int = 30

foo_list: List[str] = ['Hello', 'world']
foo_tuple: Tuple[int] = (1,)
foo_dict: Dict[str, int] = {'Hello': 1}


def qwe():
    qwe_int: int = 10
    qwe_int_op_n: Optional[int]
    qwe_any: Any
    qwe_l: List[int] = [1, 2, 3]
    qwe_list: Optional[int] = [1, 2, 3]
    qwe_list_u: Union[int, float]
    qwe_tuple: Union[str, int] = ("qqq", 1)
    qwe_dict: Union[str, str] = {'hello': 'world'}
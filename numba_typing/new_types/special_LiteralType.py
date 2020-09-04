import typing
import number_types

T = typing.TypeVar('T')


class LiteralType(typing.Generic[T]):
    pass


class PreferLiteralType(typing.Generic[T]):
    pass


class PreferNonLiteralType(typing.Generic[T]):
    pass


L_int = LiteralType[int]
L_float = LiteralType[float]

TL_int = LiteralType[number_types.T_int]
TL_float = LiteralType[number_types.T_float]

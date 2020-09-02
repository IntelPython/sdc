import typing
import numpy as np
import pandas as pd
from abc import abstractmethod
import typing_extensions

T = typing.TypeVar('T')
S = typing.TypeVar('S')


class Array(typing.Generic[T, S]):
    '''Annotation for np.ndarray
    Use square brackets to indicate type and dimension
    For example: Array[int, typing_extensions.Literal[4]]'''
    __slots__ = ()

    @abstractmethod
    def __array__(self) -> np.ndarray:
        pass


class Series(typing.Generic[T, S]):
    '''Annotation for pandas.Series
    For expample: Series[int, str]
    int - type of index
    str - type of value'''
    __slots__ = ()

    @abstractmethod
    def __array__(self) -> pd.core.series.Series:
        pass


class int8(typing_extensions.Protocol):
    """Annotation for int8"""
    @abstractmethod
    def __int__(self) -> int:
        pass


class int16(typing_extensions.Protocol):
    """Annotation for int16"""
    @abstractmethod
    def __int__(self) -> int:
        pass


class int32(typing_extensions.Protocol):
    """Annotation for int32"""
    @abstractmethod
    def __int__(self) -> int:
        pass


class int64(typing_extensions.Protocol):
    """Annotation for int64"""
    @abstractmethod
    def __int__(self) -> int:
        pass


class unint8(typing_extensions.Protocol):
    """Annotation for unsigned int8"""
    @abstractmethod
    def __int__(self) -> int:
        pass


class unint16(typing_extensions.Protocol):
    """Annotation for unsigned int16"""
    @abstractmethod
    def __int__(self) -> int:
        pass


class unint32(typing_extensions.Protocol):
    """Annotation for unsigned int32"""
    @abstractmethod
    def __int__(self) -> int:
        pass


class unint64(typing_extensions.Protocol):
    """Annotation for unsigned int64"""
    @abstractmethod
    def __int__(self) -> int:
        pass


class float32(typing_extensions.Protocol):
    """Annotation for float32"""
    @abstractmethod
    def __float__(self) -> float:
        pass


class float64(typing_extensions.Protocol):
    """Annotation for float32"""
    @abstractmethod
    def __float__(self) -> float:
        pass


number = typing.Union[int, float]  # numba's number type


T_int = typing.TypeVar('T_int', int8, int16, int32, int64)
T_float = typing.TypeVar('T_float', float32, float64)


class LiteralType(typing.Generic[T]):
    pass


class PreferLiteralType(typing.Generic[T]):
    pass


class PreferNonLiteralType(typing.Generic[T]):
    pass


L_int = LiteralType[int]
L_float = LiteralType[float]

TL_int = LiteralType[T_int]
TL_float = LiteralType[T_float]

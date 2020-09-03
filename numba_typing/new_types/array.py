import typing
import numpy as np
from abc import abstractmethod

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

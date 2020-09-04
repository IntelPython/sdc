import typing
import pandas as pd
from abc import abstractmethod

T = typing.TypeVar('T')
S = typing.TypeVar('S')


class Series(typing.Generic[T, S]):
    '''Annotation for pandas.Series
    For expample: Series[int, str]
    int - type of index
    str - type of value'''
    __slots__ = ()

    @abstractmethod
    def __array__(self) -> pd.core.series.Series:
        pass

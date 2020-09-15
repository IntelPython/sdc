import typing
from abc import abstractmethod
import typing_extensions


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

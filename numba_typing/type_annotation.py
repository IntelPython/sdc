from inspect import signature, getfullargspec
from typing import get_type_hints


def get_func_annotations(func):
    """Get annotations and default values of the fuction parameters."""
    sig = signature(func)
    annotations = {}
    defaults = {}

    for name, param in sig.parameters.items():
        if (param.annotation == sig.empty):
            raise SyntaxError

        annotations[name] = param.annotation
        if (param.default != sig.empty):
            defaults[name] = param.default

    return annotations, defaults


def get_cls_annotations(cls):
    """Get annotations of class attributes."""
    return get_type_hints(cls)


if __name__ == '__main__':
    def foo(a: int, b: int = 3,):
        pass

    print(get_func_annotations(foo))

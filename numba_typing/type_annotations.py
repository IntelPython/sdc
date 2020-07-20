from inspect import signature
from typing import get_type_hints, Union


def get_func_annotations(func):
    """Get annotations and default values of the fuction parameters."""
    sig = signature(func)
    annotations = {}
    defaults = {}

    for name, param in sig.parameters.items():
        if param.annotation == sig.empty:
            raise SyntaxError(f'Not found annotation for parameter {name}')
        annot = param.annotation
        if get_args_union(annot):
            annotations[name] = get_args_union(annot)
        else:
            annotations[name] = annot
        if param.default != sig.empty:
            defaults[name] = param.default

    return annotations, defaults


def get_cls_annotations(cls):
    """Get annotations of class attributes."""
    annotations = get_type_hints(cls)
    for x in annotations:
        if get_args_union(annotations[x]):
            annotations[x] = get_args_union(annotations[x])
    return annotations, {}


def get_args_union(annot):
    try:
        annot.__origin__
    except:
        return None
    else:
        if annot.__origin__ is Union:
            return list(annot.__args__)
        else:
            return None

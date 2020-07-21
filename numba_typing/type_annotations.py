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

        annotations[name] = get_annotation_types(param.annotation)
        if param.default != sig.empty:
            defaults[name] = param.default

    return annotations, defaults


def get_cls_annotations(cls):
    """Get annotations of class attributes."""
    annotations = get_type_hints(cls)
    for x in annotations:
        annotations[x] = get_annotation_types(annotations[x])
    return annotations, {}


def get_annotation_types(annotation):
    """Get types of passed annotation."""
    try:
        if annotation.__origin__ is Union:
            return list(annotation.__args__)
    except AttributeError:
        pass

    return [annotation, ]

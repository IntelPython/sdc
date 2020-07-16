from inspect import signature, getfullargspec
from typing import get_type_hints


def get_arguments_annotation_default_func(method):
    sig = signature(method)
    argspec = getfullargspec(method)
    arg_index = 0
    annotations = {}
    defaults = {}

    for arg in argspec.args:
        if (sig.parameters[arg].annotation == sig.empty):
            raise SyntaxError
        else:
            annotations[arg] = sig.parameters[arg].annotation
    for arg in argspec.args:
        if (sig.parameters[arg].default != sig.empty):
            defaults[arg] = sig.parameters[arg].default
    return annotations, defaults
def get_arguments_annotation_class(cls):
    return get_type_hints(cls)
    
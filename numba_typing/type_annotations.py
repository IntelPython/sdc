from inspect import signature
from typing import get_type_hints, Union, TypeVar
from itertools import product
from copy import deepcopy


def get_func_annotations(func):
    """Get annotations and default values of the fuction parameters."""
    sig = signature(func)
    annotations = {}
    defaults = {}

    for name, param in sig.parameters.items():
        if param.annotation == sig.empty:
            raise SyntaxError(f'No annotation for parameter {name}')

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


def product_annotations(annotations):
    """Get all variants of annotations."""
    types, vals = annotations
    types_product = list(product(*types.values()))
    typevars_unique = {}
    count = 1
    for name, typs in types.items():
        for typ in typs:
            if not isinstance(typ, TypeVar) or not typ.__constraints__:
                continue

            if typ not in typevars_unique:
                typevars_unique[typ] = typ.__constraints__
                count *= len(typ.__constraints__)

    prod = list(product(*typevars_unique.values()))
    temp_res = []

    for typs in types_product:
        temp = []
        temp_dict = {}
        num = 0
        for attr in types:
            temp_dict[attr] = typs[num]
            num += 1
        temp.append(temp_dict)
        temp.append(vals)
        temp_res.append(temp)

    result = []
    for examp in temp_res:
        for i in range(count):
            result.append(deepcopy(examp))

    name_of_typevars = list(typevars_unique.keys())
    for k in range(len(result)):
        pos = k % count
        for x in result[k][0]:
            for i in range(len(prod[pos])):
                if result[k][0][x] == name_of_typevars[i]:
                    result[k][0][x] = prod[pos][i]

    return result

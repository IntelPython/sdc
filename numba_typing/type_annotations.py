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
    annot = annotations[0]
    vals = annotations[1]
    list_of_annot = list(product(*annot.values()))
    tvs = {}
    tvs_unique = {}
    count = 1
    for x in annot:
        for y in annot[x]:
            if isinstance(y, TypeVar) and y.__constraints__ != ():
                if x in tvs:
                    tvs[x].append(y)
                else:
                    tvs[x] = [y, ]
                if y not in tvs_unique.keys():
                    tvs_unique[y] = y.__constraints__
                    count *= len(y.__constraints__)

    prod = list(product(*tvs_unique.values()))
    temp_res = []

    for i in range(len(list_of_annot)):
        temp = []
        temp_dict = {}
        num = 0
        for attr in annot:
            temp_dict[attr] = list_of_annot[i][num]
            num += 1
        temp.append(temp_dict)
        temp.append(vals)
        temp_res.append(temp)

    result = []
    for examp in temp_res:
        for i in range(count):
            result.append(deepcopy(examp))

    types = list(tvs_unique.keys())
    for k in range(len(result)):
        pos = k % count
        for x in result[k][0]:
            for i in range(len(prod[pos])):
                if result[k][0][x] == types[i]:
                    result[k][0][x] = prod[pos][i]

    return result

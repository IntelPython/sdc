from inspect import signature
from typing import get_type_hints, Union, TypeVar, _GenericAlias
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
    '''Get all variants of annotations.'''
    types, vals = annotations
    list_of_sig = convert_to_sig_list(types)
    signature = []
    #unique_typevars = get_internal_typevars(list_of_sig)

    for sig in list_of_sig:
        signature.extend(get_internal_typevars(sig))

    return add_vals_to_signature(signature, vals)


def add_vals_to_signature(signature, vals):
    '''Add default values ​​to all signatures'''
    result = []
    for sig in signature:
        annotation = []
        annotation.append(sig)
        annotation.append(vals)
        result.append(annotation)
    return result


def convert_to_sig_list(types):
    '''Expands all Unions'''
    types_product = list(product(*types.values()))
    names = [name for name in types.keys()]
    result = []

    for sig in types_product:
        sig_result = {}
        for i in range(len(sig)):
            sig_result[names[i]] = sig[i]
        result.append(sig_result)

    return result


def get_internal_typevars(sig):
    '''Get unique typevars in signature'''
    unique_typevars = set()
    for typ in sig.values():
        unique_typevars.update(get_typevars(typ))

    if len(unique_typevars) == 0:
        return sig

    return expand_typevars(sig, unique_typevars)


def get_typevars(type):
    '''Get unique typevars in type (container)'''
    if isinstance(type, TypeVar) and type.__constraints__:
        return {type, }
    elif isinstance(type, _GenericAlias):
        result = set()
        for arg in type.__args__:
            result.update(get_typevars(arg))
        return result

    return set()


def expand_typevars(sig, unique_typevars):
    '''Exstend all Typevars in signature'''
    result = [sig]

    for typevar in unique_typevars:
        temp_result = []
        for temp_sig in result:
            temp_result.extend(update_sig(temp_sig, typevar))
        result = temp_result

    return result


def update_sig(temp_sig, typevar):
    '''Expand one typevar'''
    result = []
    for constr_type in typevar.__constraints__:
        sig = {}
        for name, typ in temp_sig.items():
            if True in exist_typevar(typ, typevar):
                sig[name] = replace_typevar(typ, typevar, constr_type)
            else:
                sig[name] = typ

        result.append(sig)

    return result


def exist_typevar(typ, typevar):
    '''Сheck if there is a typevar in type (container)'''
    if typ == typevar:
        return {True, }
    elif isinstance(typ, _GenericAlias):
        result = set()
        for arg in typ.__args__:
            result.update(exist_typevar(arg, typevar))
        return result

    return {False, }


def replace_typevar(typ, typevar, final_typ):
    '''Replace typevar with type in container
        For example:
        # typ = Dict[T, V]
        # typevar = T(int, str)
        # final_typ = int
    '''

    if typ == typevar:
        return (final_typ)
    elif isinstance(typ, _GenericAlias):
        result = list()
        for arg in typ.__args__:
            result.append(replace_typevar(arg, typevar, final_typ))
        result_type = typ.copy_with(tuple(result))
        return (result_type)

    return (typ)

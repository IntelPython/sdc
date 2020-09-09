import numba
from numba import types
from numba.extending import overload
from type_annotations import product_annotations, get_func_annotations
import typing
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba.typed import List, Dict
from inspect import getfullargspec


def overload_list(orig_func):
    def overload_inner(ovld_list):
        def wrapper(*args):
            func_list = ovld_list()
            sig_list = []
            for func in func_list:
                sig_list.append((product_annotations(
                    get_func_annotations(func)), func))
            args_orig_func = getfullargspec(orig_func)
            values_dict = {name: typ for name, typ in zip(args_orig_func.args, args)}
            defaults_dict = {}
            if args_orig_func.defaults:
                defaults_dict = {name: value for name, value in zip(
                    args_orig_func.args[::-1], args_orig_func.defaults[::-1])}
            result = choose_func_by_sig(sig_list, values_dict, defaults_dict)

            if result is None:
                raise numba.TypingError(f'Unsupported types a={a}, b={b}')

            return result

        return overload(orig_func, strict=False)(wrapper)

    return overload_inner


def check_int_type(n_type):
    return isinstance(n_type, types.Integer)


def check_float_type(n_type):
    return isinstance(n_type, types.Float)


def check_bool_type(n_type):
    return isinstance(n_type, types.Boolean)


def check_str_type(n_type):
    return isinstance(n_type, types.UnicodeType)


def check_list_type(self, p_type, n_type):
    res = isinstance(n_type, types.List) or isinstance(n_type, types.ListType)
    if isinstance(p_type, type):
        return res
    else:
        return res and self.match(p_type.__args__[0], n_type.dtype)


def check_tuple_type(self, p_type, n_type):
    res = False
    if isinstance(n_type, types.Tuple):
        res = True
        if isinstance(p_type, type):
            return res
        for p_val, n_val in zip(p_type.__args__, n_type.key):
            res = res and self.match(p_val, n_val)
    if isinstance(n_type, types.UniTuple):
        res = True
        if isinstance(p_type, type):
            return res
        for p_val in p_type.__args__:
            res = res and self.match(p_val, n_type.key[0])
    return res


def check_dict_type(self, p_type, n_type):
    res = False
    if isinstance(n_type, types.DictType):
        res = True
        if isinstance(p_type, type):
            return res
        for p_val, n_val in zip(p_type.__args__, n_type.keyvalue_type):
            res = res and self.match(p_val, n_val)
    return res


class TypeChecker:

    _types_dict = {int: check_int_type, float: check_float_type, bool: check_bool_type,
                   str: check_str_type, list: check_list_type,
                   tuple: check_tuple_type, dict: check_dict_type}

    def __init__(self):
        self._typevars_dict = {}

    def clear_typevars_dict(self):
        self._typevars_dict.clear()

    @classmethod
    def add_type_check(cls, type_check, func):
        cls._types_dict[type_check] = func

    @staticmethod
    def _is_generic(p_obj):
        if isinstance(p_obj, typing._GenericAlias):
            return True

        if isinstance(p_obj, typing._SpecialForm):
            return p_obj not in {typing.Any}

        return False

    @staticmethod
    def _get_origin(p_obj):
        return p_obj.__origin__

    def match(self, p_type, n_type):
        try:
            if p_type == typing.Any:
                return True
            elif self._is_generic(p_type):
                origin_type = self._get_origin(p_type)
                if origin_type == typing.Generic:
                    return self.match_generic(p_type, n_type)
                else:
                    return self._types_dict[origin_type](self, p_type, n_type)
            elif isinstance(p_type, typing.TypeVar):
                return self.match_typevar(p_type, n_type)
            else:
                if p_type in (list, tuple):
                    return self._types_dict[p_type](self, p_type, n_type)
                return self._types_dict[p_type](n_type)
        except KeyError:
            print((f'A check for the {p_type} was not found.'))
            return None

    def match_typevar(self, p_type, n_type):
        if not self._typevars_dict.get(p_type) and n_type not in self._typevars_dict.values():
            self._typevars_dict[p_type] = n_type
            return True
        return self._typevars_dict.get(p_type) == n_type

    def match_generic(self, p_type, n_type):
        res = True
        for arg in p_type.__args__:
            res = res and self.match(arg, n_type)
        return res


def choose_func_by_sig(sig_list, values_dict, defaults_dict):
    checker = TypeChecker()
    for sig, func in sig_list:  # sig = (Signature,func)
        for param in sig.parameters:  # param = {'a':int,'b':int}
            full_match = True
            for name, typ in values_dict.items():  # name,type = 'a',int64
                if isinstance(typ, types.Literal):

                    full_match = full_match and checker.match(
                        param[name], typ.literal_type)

                    if sig.defaults.get(name, False):
                        full_match = full_match and sig.defaults[name] == typ.literal_value
                else:
                    full_match = full_match and checker.match(param[name], typ)

                if not full_match:
                    break

            for name, val in defaults_dict.items():
                if not sig.defaults.get(name) is None:
                    full_match = full_match and sig.defaults[name] == val

            checker.clear_typevars_dict()
            if full_match:
                return func

    return None

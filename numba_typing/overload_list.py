import numpy
import numba
from numba import types
from numba import typeof
from numba.extending import overload
from type_annotations import product_annotations, get_func_annotations
from numba import njit
import typing
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba.typed import List, Dict
from inspect import getfullargspec


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def overload_list(orig_func):
    def overload_inner(ovld_list):
        def wrapper(*args):
            func_list = ovld_list()
            sig_list = []
            for func in func_list:
                sig_list.append((product_annotations(
                    get_func_annotations(func)), func))
            param = getfullargspec(orig_func).args
            kwargs = {name: typ for name, typ in zip(param, args)}
            result = choose_func_by_sig(sig_list, **kwargs)

            if result is None:
                raise numba.TypingError(f'Unsupported types a={a}, b={b}')

            return result

        return overload(orig_func, strict=False)(wrapper)

    return overload_inner


class TypeChecker:
    def __init__(self):
        self._types_dict = {int: check_int_type, float: check_float_type, bool: check_bool_type,
                            str: check_str_type, list: check_list_type,
                            tuple: check_tuple_type, dict: check_dict_type}
        self._typevars_dict = {}

    def add_type_check(self, type_check, func):
        self._types_dict[type_check] = func

    def _is_generic(self, p_obj):
        if isinstance(p_obj, typing._GenericAlias):
            return True

        if isinstance(p_obj, typing._SpecialForm):
            return p_obj not in {typing.Any}

        return False

    def _get_origin(self, p_obj):
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
            print((f'A check for the {p_type} was not found'))

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


def choose_func_by_sig(sig_list, **kwargs):
    for sig in sig_list:  # sig = (Signature,func)
        checker = TypeChecker()
        for param in sig[0].parameters:  # param = {'a':int,'b':int}
            full_match = True
            for name, typ in kwargs.items():  # name,type = 'a',int64
                if isinstance(typ, types.Literal):

                    full_match = full_match and checker.match(
                        param[name], typ.literal_type)

                    if sig[0].defaults.get(name, False):
                        full_match = full_match and sig[0].defaults[name] == typ.literal_value
                else:
                    full_match = full_match and checker.match(param[name], typ)

                if not full_match:
                    break
            if full_match:
                return sig[1]

    return None

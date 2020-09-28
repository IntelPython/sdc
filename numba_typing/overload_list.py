import numba
from numba import types
from numba.extending import overload
from type_annotations import product_annotations, get_func_annotations
import typing
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
            if valid_signature(sig_list, values_dict, defaults_dict):
                result = choose_func_by_sig(sig_list, values_dict)

            if result is None:
                raise TypeError(f'Unsupported types {args}')

            return result

        return overload(orig_func, strict=False)(wrapper)

    return overload_inner


def valid_signature(list_signature, values_dict, defaults_dict):
    def check_defaults(list_param, sig_def):
        for name, val in defaults_dict.items():
            if sig_def.get(name) is None:
                raise AttributeError(f'{name} does not match the signature of the function passed to overload_list')
            if sig_def[name] != val:
                raise ValueError(f'The default arguments are not equal: {name}: {val} != {sig_def[name]}')
            if type(sig_def[name]) != list_param[name]:
                raise TypeError(f'The default value does not match the type: {list_param[name]}')

    for sig, _ in list_signature:
        for param in sig.parameters:
            if len(param) != len(values_dict):
                check_defaults(param, sig.defaults)

    return True


def check_int_type(n_type):
    return isinstance(n_type, types.Integer)


def check_float_type(n_type):
    return isinstance(n_type, types.Float)


def check_bool_type(n_type):
    return isinstance(n_type, types.Boolean)


def check_str_type(n_type):
    return isinstance(n_type, types.UnicodeType)


def check_list_type(self, p_type, n_type):
    res = isinstance(n_type, (types.List, types.ListType))
    if p_type == list:
        return res
    else:
        return res and self.match(p_type.__args__[0], n_type.dtype)


def check_tuple_type(self, p_type, n_type):
    if not isinstance(n_type, (types.Tuple, types.UniTuple)):
        return False
    try:
        if len(p_type.__args__) != len(n_type.types):
            return False
    except AttributeError:  # if p_type == tuple
        return True

    for p_val, n_val in zip(p_type.__args__, n_type.types):
        if not self.match(p_val, n_val):
            return False

    return True


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

    _types_dict: dict = {}

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
        if p_type == typing.Any:
            return True
        try:
            if self._is_generic(p_type):
                origin_type = self._get_origin(p_type)
                if origin_type == typing.Generic:
                    return self.match_generic(p_type, n_type)

                return self._types_dict[origin_type](self, p_type, n_type)

            if isinstance(p_type, typing.TypeVar):
                return self.match_typevar(p_type, n_type)

            if p_type in (list, tuple, dict):
                return self._types_dict[p_type](self, p_type, n_type)

            return self._types_dict[p_type](n_type)

        except KeyError:
            raise TypeError(f'A check for the {p_type} was not found.')

    def match_typevar(self, p_type, n_type):
        if isinstance(n_type, types.List):
            n_type = types.ListType(n_type.dtype)
        if not self._typevars_dict.get(p_type):
            self._typevars_dict[p_type] = n_type
            return True
        return self._typevars_dict.get(p_type) == n_type

    def match_generic(self, p_type, n_type):
        raise SystemError


TypeChecker.add_type_check(int, check_int_type)
TypeChecker.add_type_check(float, check_float_type)
TypeChecker.add_type_check(str, check_str_type)
TypeChecker.add_type_check(bool, check_bool_type)
TypeChecker.add_type_check(list, check_list_type)
TypeChecker.add_type_check(tuple, check_tuple_type)
TypeChecker.add_type_check(dict, check_dict_type)


def choose_func_by_sig(sig_list, values_dict):
    def check_signature(sig_params, types_dict):
        checker = TypeChecker()
        for name, typ in types_dict.items():  # name,type = 'a',int64
            if isinstance(typ, types.Literal):
                typ = typ.literal_type
            if not checker.match(sig_params[name], typ):
                return False

        return True

    for sig, func in sig_list:  # sig = (Signature,func)
        for param in sig.parameters:  # param = {'a':int,'b':int}
            if check_signature(param, values_dict):
                return func

    return None

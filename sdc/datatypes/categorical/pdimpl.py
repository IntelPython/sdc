# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import pandas as pd

from numba.extending import overload
from numba.extending import intrinsic
from numba.extending import type_callable
from numba.extending import lower_builtin
from numba import types
from numba import typeof
from numba import objmode

from .types import (
    CategoricalDtypeType,
    Categorical,
)

from . import pandas_support


# Possible alternative implementations:
# 1. @overload + @intrinsic
# 2. @type_callable + @lower_builtin
# They are equivalent. Who is defined firts - has higher priority.


def _reconstruct_CategoricalDtype(dtype):
    if isinstance(dtype, types.Literal):
        return dtype.literal_value

    if isinstance(dtype, CategoricalDtypeType):
        return pandas_support.as_dtype(dtype)

    raise NotImplementedError()


@overload(pd.CategoricalDtype)
def _CategoricalDtype(categories=None, ordered=None):
    """
    Implementation of constructor for pandas CategoricalDtype.
    """
    if isinstance(ordered, types.Literal):
        ordered_const = ordered.literal_value
    else:
        ordered_const = ordered

    def impl(categories=None, ordered=None):
        return _CategoricalDtype_intrinsic(categories, ordered_const)
    return impl


@intrinsic
def _CategoricalDtype_intrinsic(typingctx, categories, ordered):
    """
    Creates CategoricalDtype object.

    Assertions:
        categories - Tuple of literal values or None
        ordered - literal Bool
    """
    if isinstance(categories, types.NoneType):
        categories_list = None
    if isinstance(categories, types.Tuple):
        categories_list = [c.literal_value for c in categories]

    if isinstance(ordered, types.NoneType):
        ordered_value = None
    if isinstance(ordered, types.Literal):
        ordered_value = ordered.literal_value

    return_type = CategoricalDtypeType(categories_list, ordered_value)
    sig = return_type(categories, ordered)

    def codegen(context, builder, signature, args):
        # All CategoricalDtype objects are dummy values in LLVM.
        # They only exist in the type level.
        return context.get_dummy_value()

    return sig, codegen


# TODO: move to tools
def is_categoricaldtype(dtype):
    if isinstance(dtype, types.Literal) and dtype.literal_value == 'category':
        return True

    if isinstance(dtype, CategoricalDtypeType):
        return True

    return False


# @type_callable(pd.CategoricalDtype)
# def type_CategoricalDtype_constructor(context):
#     def typer(categories, ordered):
#         # TODO: check all Literal in categories
#         if isinstance(categories, types.Tuple) and isinstance(ordered, types.Literal):
#             categories_list = [c.literal_value for c in categories]
#             return CategoricalDtypeType(categories_list, ordered.literal_value)

#     return typer


# @lower_builtin(pd.CategoricalDtype, types.Any, types.Any)
# def _CategoricalDtype_constructor(context, builder, sig, args):
#     # All CategoricalDtype objects are dummy values in LLVM.
#     # They only exist in the type level.
#     return context.get_dummy_value()


# @type_callable(pd.CategoricalDtype)
# def type_CategoricalDtype_constructor(context):
#     def typer(categories):
#         # TODO: check all Literal in categories
#         if isinstance(categories, types.Tuple):
#             categories_list = [c.literal_value for c in categories]
#             return CategoricalDtypeType(categories_list)

#     return typer


# @lower_builtin(pd.CategoricalDtype, types.Any)
# def _CategoricalDtype_constructor(context, builder, sig, args):
#     # All CategoricalDtype objects are dummy values in LLVM.
#     # They only exist in the type level.
#     return context.get_dummy_value()


# TODO: use dtype too
def _reconstruct_Categorical(values):
    values_list = [v.literal_value for v in values]
    return pd.Categorical(values=values_list)


@overload(pd.Categorical)
def _Categorical(values, categories=None, ordered=None, dtype=None, fastpath=False):
    """
    Implementation of constructor for pandas Categorical via objmode.
    """
    # TODO: support other parameters (only values now)

    ty = typeof(_reconstruct_Categorical(values))
    tyname = ty.name
    setattr(types, tyname, ty)

    from textwrap import dedent
    text = dedent(f"""
    def impl(values, categories=None, ordered=None, dtype=None, fastpath=False):
        with objmode(categorical='vars()["{tyname}"]'):
            categorical = pd.Categorical(values, categories, ordered, dtype, fastpath)
        return categorical
    """)
    globals, locals = {'objmode': objmode, 'pd': pd}, {}
    exec(text, globals, locals)
    impl = locals['impl']
    return impl


# @type_callable(pd.Categorical)
# def type_Categorical_constructor(context):
#     """
#     Similar to @infer_global(np.array).
#     """
#     def typer(values, categories=None, ordered=None, dtype=None, fastpath=False):
#         # from numba.typing import npydecl
#         # codes = npydecl.NpArray(context).generic()(values)
#         categorical = _reconstruct_Categorical(values)
#         return typeof(categorical)

#     return typer


# @lower_builtin(pd.Categorical, types.Any)
# # @lower_builtin(np.Categorical, types.Any, types.DTypeSpec)
# def pd_Categorical(context, builder, sig, args):
#     """
#     Similar to @lower_builtin(np.array, ...).
#     """
#     from numba.targets import arrayobj
#     codes = sig.return_type.codes
#     return arrayobj.np_array(context, builder, sig.replace(return_type=codes), args)


# via intrinsic
# @overload(pd.Categorical)
# def _Categorical(values, categories=None, ordered=None, dtype=None, fastpath=False):
#     """
#     Implementation of constructor for pandas Categorical.
#     """
#     def impl(values, categories=None, ordered=None, dtype=None, fastpath=False):
#         return _Categorical_intrinsic(values, categories, ordered, dtype, fastpath)
#     return impl


# @intrinsic
# def _Categorical_intrinsic(typingctx, values, categories, ordered, dtype, fastpath):
#     """
#     Creates Categorical object.
#     """
#     if isinstance(values, types.Tuple):
#         values_list = [v.literal_value for v in values]
#         categorical = pd.Categorical(values=values_list)
#         return_type = typeof(categorical)

#         def codegen(context, builder, signature, args):
#             [values] = args
#             # TODO: can not recall similar function
#             native_value = boxing.unbox_array(typ.codes, codes, c)
#             return native_value

#     sig = return_type(values, categories, ordered, dtype, fastpath)
#     return sig, codegen

#     # return_type = Categorical(dtype=CategoricalDtypeType(), codes=types.Array(types.int8, 1, 'C'))
#     # sig = return_type(values)

#     # def codegen(context, builder, signature, args):
#     #     return context.get_dummy_value()

#     # return sig, codegen

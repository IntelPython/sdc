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

from .types import CategoricalDtypeType


# Possible alternative implementations:
# 1. @overload + @intrinsic
# 2. @type_callable + @lower_builtin
# They are equivalent. Who is defined firts - has higher priority.


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

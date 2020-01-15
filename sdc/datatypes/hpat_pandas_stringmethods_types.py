# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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

"""
| :class:`pandas.core.strings.StringMethods` type implementation in HPAT
| Also, it contains related types and iterators for StringMethods type handling
"""


import pandas

from numba import types, cgutils
from numba.extending import (models, overload, register_model, make_attribute_wrapper, intrinsic)
from numba.datamodel import (register_default, StructModel)
from numba.typing.templates import signature
from sdc.hiframes.split_impl import SplitViewStringMethodsType, StringArraySplitViewType
from sdc.utils import sdc_overload


class StringMethodsType(types.IterableType):
    """
    Type definition for pandas.core.strings.StringMethods functions handling.

    Members
    ----------
    _data: :class:`SeriesType`
        input arg
    """

    def __init__(self, data):
        self.data = data
        name = 'StringMethodsType({})'.format(self.data)
        super(StringMethodsType, self).__init__(name)

    @property
    def iterator_type(self):
        return None


@register_model(StringMethodsType)
class StringMethodsTypeModel(StructModel):
    """
    Model for StringMethodsType type
    All members must be the same as main type for this model
    """

    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.data)
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(StringMethodsType, 'data', '_data')


def _gen_hpat_pandas_stringmethods_init(string_methods_type=None):
    string_methods_type = string_methods_type or StringMethodsType

    def _hpat_pandas_stringmethods_init(typingctx, data):
        """
        Internal Numba required function to register StringMethodsType and
        connect it with corresponding Python type mentioned in @overload(pandas.core.strings.StringMethods)
        """

        def _hpat_pandas_stringmethods_init_codegen(context, builder, signature, args):
            """
            It is looks like it creates StringMethodsModel structure

            - Fixed number of parameters. Must be 4
            - increase reference count for the data
            """

            [data_val] = args
            stringmethod = cgutils.create_struct_proxy(signature.return_type)(context, builder)
            stringmethod.data = data_val

            if context.enable_nrt:
                context.nrt.incref(builder, data, stringmethod.data)

            return stringmethod._getvalue()

        ret_typ = string_methods_type(data)
        sig = signature(ret_typ, data)
        """
        Construct signature of the Numba SeriesGroupByType::ctor()
        """

        return sig, _hpat_pandas_stringmethods_init_codegen

    return _hpat_pandas_stringmethods_init


_hpat_pandas_stringmethods_init = intrinsic(
    _gen_hpat_pandas_stringmethods_init(string_methods_type=StringMethodsType))
_hpat_pandas_split_view_stringmethods_init = intrinsic(
    _gen_hpat_pandas_stringmethods_init(string_methods_type=SplitViewStringMethodsType))


@sdc_overload(pandas.core.strings.StringMethods)
def hpat_pandas_stringmethods(obj):
    """
    Special Numba procedure to overload Python type pandas.core.strings.StringMethods::ctor()
    with Numba registered model
    """
    if isinstance(obj.data, StringArraySplitViewType):
        def hpat_pandas_split_view_stringmethods_impl(obj):
            return _hpat_pandas_split_view_stringmethods_init(obj)

        return hpat_pandas_split_view_stringmethods_impl

    def hpat_pandas_stringmethods_impl(obj):
        return _hpat_pandas_stringmethods_init(obj)

    return hpat_pandas_stringmethods_impl

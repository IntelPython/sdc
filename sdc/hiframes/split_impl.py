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


from numba import types

from sdc.str_ext import string_type

import llvmlite.binding as ll
from .. import hstr_ext
ll.add_symbol('array_setitem', hstr_ext.array_setitem)
ll.add_symbol('array_getptr1', hstr_ext.array_getptr1)

ll.add_symbol('dtor_str_arr_split_view', hstr_ext.dtor_str_arr_split_view)
ll.add_symbol('str_arr_split_view_impl', hstr_ext.str_arr_split_view_impl)
ll.add_symbol('str_arr_split_view_alloc', hstr_ext.str_arr_split_view_alloc)


# nested offset structure to represent S.str.split()
# data_offsets array includes offsets to character data array
# index_offsets array includes offsets to data_offsets array to identify lists
class StringArraySplitViewType(types.IterableType):
    def __init__(self):
        super(StringArraySplitViewType, self).__init__(
            name='StringArraySplitViewType()')

    @property
    def dtype(self):
        # TODO: optimized list type
        return types.List(string_type)

    # TODO
    @property
    def iterator_type(self):
        return  # StringArrayIterator()

    def copy(self):
        return StringArraySplitViewType()


class SplitViewStringMethodsType(types.IterableType):
    """
    Type definition for pandas.core.strings.StringMethods functions handling.

    Members
    ----------
    _data: :class:`SeriesType`
        input arg
    """

    def __init__(self, data):
        self.data = data
        name = 'SplitViewStringMethodsType({})'.format(self.data)
        super(SplitViewStringMethodsType, self).__init__(name)

    @property
    def iterator_type(self):
        return None

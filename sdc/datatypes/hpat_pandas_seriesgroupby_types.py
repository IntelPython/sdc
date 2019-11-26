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

| :class:`pandas.SeriesGroupBy` type implementation in SDC
| Also, it contains related types and iterators for SeriesGroupBy type handling

"""


import pandas

from numba import types, cgutils
from numba.extending import (models, overload, register_model, make_attribute_wrapper, intrinsic)
from numba.datamodel import (register_default, StructModel)
from numba.typing.templates import signature


class SeriesGroupByTypeIterator(types.SimpleIteratorType):
    """
    Iterator type for SeriesGroupByType type

    Members
    ----------
    _data: :class:`SeriesGroupByType`
        input arg
    """

    def __init__(self, data):
        self.data = data
        super(SeriesGroupByTypeIterator, self).__init__("SeriesGroupByTypeIterator", data)


@register_default(SeriesGroupByTypeIterator)
class SeriesGroupByTypeIteratorModel(StructModel):
    """
    Model for SeriesGroupByTypeIterator type
    All members must be the same as main type for this model

    Test:
    """

    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.data)
        ]
        super(SeriesGroupByTypeIteratorModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesGroupByTypeIterator, 'data', '_data')


class SeriesGroupByType(types.IterableType):
    """
    Type definition for SeriesGroupBy functions handling.

    Members
    ----------
    _data: :class:`SeriesType`
        input arg
    """

    def __init__(self, data):
        self.data = data
        super(SeriesGroupByType, self).__init__('SeriesGroupByType')

    @property
    def iterator_type(self):
        return SeriesGroupByTypeIterator(self)


@register_model(SeriesGroupByType)
class SeriesGroupByTypeModel(StructModel):
    """
    Model for SeriesGroupByType type
    All members must be the same as main type for this model

    Test: python -m sdc.runtests sdc.tests.test_series.TestSeries.test_series_groupby_count
    """

    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.data)
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(SeriesGroupByType, 'data', '_data')


@intrinsic
def _hpat_pandas_seriesgroupby_init(typingctx, data):
    """
    Internal Numba required function to register SeriesGroupByType and
    connect it with corresponding Python type mentioned in @overload(pandas.core.groupby.SeriesGroupBy)
    """

    def _hpat_pandas_seriesgroupby_init_codegen(context, builder, signature, args):
        """
        It is looks like it creates SeriesGroupByModel structure

        - Fixed number of parameters. Must be 4
        - increase reference counr for the data
        """

        [data_val] = args
        series = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        series.data = data_val

        if context.enable_nrt:
            context.nrt.incref(builder, data, series.data)

        return series._getvalue()

    ret_typ = SeriesGroupByType(data)
    sig = signature(ret_typ, data)
    """
    Construct signature of the Numba SeriesGroupByType::ctor()
    """

    return sig, _hpat_pandas_seriesgroupby_init_codegen


@overload(pandas.core.groupby.SeriesGroupBy)
def hpat_pandas_seriesgroupby(
        obj,
        keys=None,
        axis=0,
        level=None,
        grouper=None,
        exclusions=None,
        selection=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        observed=False):
    """
    Special Numba procedure to overload Python type SeriesGroupBy::ctor() with Numba registered model
    """

    def hpat_pandas_seriesgroupby_impl(
            obj,
            keys=None,
            axis=0,
            level=None,
            grouper=None,
            exclusions=None,
            selection=None,
            as_index=True,
            sort=True,
            group_keys=True,
            squeeze=False,
            observed=False):
        return _hpat_pandas_seriesgroupby_init(obj)

    return hpat_pandas_seriesgroupby_impl

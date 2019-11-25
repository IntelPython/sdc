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

| :class:`pandas.DataFrame` type implementation in Intel SDC
| Also, it contains related types and iterators for DataFrame type handling

"""


import operator
import pandas

from numba import types, cgutils
from numba.extending import (models, overload, register_model, make_attribute_wrapper, intrinsic, box, unbox)
from numba.datamodel import register_default, StructModel
from numba.typing.templates import signature, infer_global, AbstractTemplate

from sdc.config import config_pipeline_hpat_default


class DataFrameTypeIterator(types.SimpleIteratorType):
    """
    Iterator type for DataFrameType type

    Members
    ----------
    _data: :class:`DataFrameType`
        input arg
    """

    def __init__(self, data=None):
        self.data = data

        super(DataFrameTypeIterator, self).__init__("DataFrameTypeIterator(data={})".format(self.data), data)


@register_default(DataFrameTypeIterator)
class DataFrameTypeIteratorModel(StructModel):
    """
    Model for DataFrameTypeIterator type
    All members must be the same as main type for this model

    Test:
    """

    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.data),
        ]
        super(DataFrameTypeIteratorModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameTypeIterator, 'data', '_data')


class DataFrameType(types.IterableType):
    """
    Type definition for DataFrame functions handling.

    Members
    ----------
    data: Dictinary of :class:`SeriesType`
        input arg

    index: DataFrame index
        *unsupported*

        Dictinary looks a bit ambigous due to keys are column names which already presented in Series.
        This type selected due to pandas.DataFrame interprets input :class:`SeriesType` as rows instead
        expected columns if passed as a list.
        This data is interpreted as columns if passed as a dictinary only.

    Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_create
    """

    def __init__(self, data=None):

        self.data = data

        type_str = "DataFrameType(data={})".format(self.data)
        super(DataFrameType, self).__init__(type_str)

    @property
    def iterator_type(self):
        return DataFrameTypeIterator(self)


if config_pipeline_hpat_default is 0:
    @register_model(DataFrameType)
    class DataFrameTypeModel(StructModel):
        """
        Model for DataFrameType type
        All members must be the same as main type for this model

        Test: python -m sdc.runtests sdc.tests.test_dataframe.TestDataFrame.test_create_numeric_column
        """

        def __init__(self, dmm, fe_type):
            members = [
                ('data', fe_type.data)
            ]
            models.StructModel.__init__(self, dmm, fe_type, members)

    make_attribute_wrapper(DataFrameType, 'data', '_data')


@intrinsic
def _hpat_pandas_dataframe_init(typingctx, data=None):
    """
    Internal Numba required function to register DataFrameType and
    connect it with corresponding Python type mentioned in @overload(pandas.DataFrame)
    """

    def _hpat_pandas_dataframe_init_codegen(context, builder, signature, args):
        """
        It is looks like it creates DataFrameModel structure

        - Fixed number of parameters. Must be 4
        - increase reference counr for the data
        """

        [data_val] = args

        dataframe = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        dataframe.data = data_val

        if context.enable_nrt:
            context.nrt.incref(builder, data, dataframe.data)

        return dataframe._getvalue()

    ret_typ = DataFrameType(data)
    sig = signature(ret_typ, data)
    """
    Construct signature of the Numba DataFrameType::ctor()
    """

    return sig, _hpat_pandas_dataframe_init_codegen


if config_pipeline_hpat_default is 0:
    @overload(pandas.DataFrame)
    def hpat_pandas_dataframe(data=None, index=None, columns=None, dtype=None, copy=False):
        """
        Special Numba procedure to overload Python type pandas.DataFrame::ctor() with Numba registered model
        """
        print("====================@overload(pandas.DataFrame)=======================")
        print("data:", data)
        if isinstance(data, types.DictType):
            def hpat_pandas_dataframe_impl(data=None, index=None, columns=None, dtype=None, copy=False):
                series_dict = {}
                series_list = []

                for key, value in data.items():
                    """
                    Convert input dictionary with:
                        key - unicode string
                        value - array
                    into dictinary of pandas.Series with same names and values
                    """

                    series_item = pandas.Series(data=value, name=key)
                    series_dict[key] = series_item
                    series_list.append(series_item)

                # return _hpat_pandas_dataframe_init(series_dict)
                return _hpat_pandas_dataframe_init(series_list)

            return hpat_pandas_dataframe_impl

    @box(DataFrameType)
    def hpat_pandas_dataframe_box(typ, val, c):
        """
        This method is to copy data from JITted region data structure
        to new Python object data structure.
        Python object data structure has creating in this procedure.
        """

        dataframe = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

        ir_ptr_data = c.box(typ.data, dataframe.data)

        dataframe_ctor_args = c.pyapi.tuple_pack([ir_ptr_data, ])
        # dataframe_ctor_kwargs = c.pyapi.dict_pack([("data", ir_ptr_data), ])
        """
        It is better to use kwargs but it fails into SIGSEGV
        """

        dataframe_ctor_fn = c.pyapi.unserialize(c.pyapi.serialize_object(pandas.DataFrame))
        """
        Create a pandas.DataFrame ctor() function pointer
        """

        df_obj = c.pyapi.call(dataframe_ctor_fn, dataframe_ctor_args)  # kws=dataframe_ctor_kwargs)
        """
        Call pandas.DataFrame function pointer with parameters
        """

        c.pyapi.decref(ir_ptr_data)
        c.pyapi.decref(dataframe_ctor_args)
        c.pyapi.decref(dataframe_ctor_fn)

        return df_obj

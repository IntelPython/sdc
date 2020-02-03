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

import llvmlite.llvmpy.core as lc

import numpy as np

from numba import types, cgutils
from numba.numpy_support import from_dtype
from numba.extending import (models, register_model, make_attribute_wrapper, lower_builtin)
from numba.targets.imputils import (impl_ret_new_ref, iternext_impl, RefType)
from numba.targets.arrayobj import (make_array, _getitem_array1d)

from sdc.str_arr_ext import string_array_type
from sdc.str_ext import string_type, list_string_array_type
from sdc.hiframes.pd_categorical_ext import (PDCategoricalDtype, CategoricalArray)
from sdc.hiframes.pd_timestamp_ext import (pandas_timestamp_type, datetime_date_type)
from sdc.str_arr_ext import (string_array_type, iternext_str_array, StringArrayType)


class SeriesType(types.IterableType):
    """Temporary type class for Series objects.
    """

    def __init__(self, dtype, data=None, index=None, is_named=False):
        # keeping data array in type since operators can make changes such
        # as making array unaligned etc.
        data = _get_series_array_type(dtype) if data is None else data
        # convert Record to tuple (for tuple output of map)
        # TODO: handle actual Record objects in Series?
        self.dtype = (types.Tuple(list(dict(dtype.members).values()))
                      if isinstance(dtype, types.Record) else dtype)
        self.data = data
        if index is None:
            index = types.none
        self.index = index
        # keep is_named in type to enable boxing
        self.is_named = is_named
        super(SeriesType, self).__init__(
            name="series({}, {}, {}, {})".format(dtype, data, index, is_named))

    def copy(self, dtype=None):
        # XXX is copy necessary?
        index = types.none if self.index == types.none else self.index.copy()
        dtype = dtype if dtype is not None else self.dtype
        data = _get_series_array_type(dtype)
        return SeriesType(dtype, data, index)

    @property
    def key(self):
        # needed?
        return self.dtype, self.data, self.index, self.is_named

    @property
    def ndim(self):
        return self.data.ndim

    def unify(self, typingctx, other):
        if isinstance(other, SeriesType):
            new_index = types.none
            if self.index != types.none and other.index != types.none:
                new_index = self.index.unify(typingctx, other.index)
            elif other.index != types.none:
                new_index = other.index
            elif self.index != types.none:
                new_index = self.index

            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                return SeriesType(
                    self.dtype,
                    self.data.unify(typingctx, other.data),
                    new_index)

        # XXX: unify Series/Array as Array
        return super(SeriesType, self).unify(typingctx, other)

    def can_convert_to(self, typingctx, other):
        # same as types.Array
        if (isinstance(other, SeriesType) and other.dtype == self.dtype):
            # called for overload selection sometimes
            # TODO: fix index
            if self.index == types.none and other.index == types.none:
                return self.data.can_convert_to(typingctx, other.data)
            if self.index != types.none and other.index != types.none:
                return max(self.data.can_convert_to(typingctx, other.data),
                           self.index.can_convert_to(typingctx, other.index))

    def is_precise(self):
        # same as types.Array
        return self.dtype.is_precise()

    @property
    def iterator_type(self):
        # TODO: fix timestamp
        return SeriesIterator(self)


# register_model(SeriesType)(models.ArrayModel)
# need to define model since fix_df_array overload goes to native code
@register_model(SeriesType)
class SeriesModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        name_typ = string_type if fe_type.is_named else types.none
        members = [
            ('data', fe_type.data),
            ('index', fe_type.index),
            ('name', name_typ),
        ]
        super(SeriesModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesType, 'data', '_data')
make_attribute_wrapper(SeriesType, 'index', '_index')
make_attribute_wrapper(SeriesType, 'name', '_name')


class SeriesIterator(types.SimpleIteratorType):
    """
    Type class for iterator over dataframe series.
    """

    def __init__(self, series_type):
        self.series_type = series_type
        self.array_type = series_type.data

        name = f'iter({self.series_type.data})'
        yield_type = series_type.dtype
        super(SeriesIterator, self).__init__(name, yield_type)

    @property
    def _iternext(self):
        if isinstance(self.array_type, StringArrayType):
            return iternext_str_array
        elif isinstance(self.array_type, types.Array):
            return iternext_series_array


@register_model(SeriesIterator)
class SeriesIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('array', fe_type.series_type.data)]

        models.StructModel.__init__(self, dmm, fe_type, members)


@lower_builtin('getiter', SeriesType)
def getiter_series(context, builder, sig, args):
    """
    Getting iterator for the Series type

    :param context: context descriptor
    :param builder: llvmlite IR Builder
    :param sig: iterator signature
    :param args: tuple with iterator arguments, such as instruction, operands and types
    :param result: iternext result
    :return: reference to iterator
    """

    arraytype = sig.args[0].data

    # Create instruction to get array to iterate
    zero_member_pointer = context.get_constant(types.intp, 0)
    zero_member = context.get_constant(types.int32, 0)
    alloca = args[0].operands[0]
    gep_result = builder.gep(alloca, [zero_member_pointer, zero_member])
    array = builder.load(gep_result)

    # TODO: call numba getiter with gep_result for array
    iterobj = context.make_helper(builder, sig.return_type)
    zero_index = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero_index)

    iterobj.index = indexptr
    iterobj.array = array

    if context.enable_nrt:
        context.nrt.incref(builder, arraytype, array)

    result = iterobj._getvalue()
    # Note: a decref on the iterator will dereference all internal MemInfo*
    out = impl_ret_new_ref(context, builder, sig.return_type, result)
    return out


# TODO: call it from numba.targets.arrayobj, need separate function in numba
def iternext_series_array(context, builder, sig, args, result):
    """
    Implementation of iternext() for the ArrayIterator type

    :param context: context descriptor
    :param builder: llvmlite IR Builder
    :param sig: iterator signature
    :param args: tuple with iterator arguments, such as instruction, operands and types
    :param result: iternext result
    """

    [iterty] = sig.args
    [iter] = args
    arrayty = iterty.array_type

    if arrayty.ndim != 1:
        raise NotImplementedError("iterating over %dD array" % arrayty.ndim)

    iterobj = context.make_helper(builder, iterty, value=iter)
    ary = make_array(arrayty)(context, builder, value=iterobj.array)

    nitems, = cgutils.unpack_tuple(builder, ary.shape, count=1)

    index = builder.load(iterobj.index)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        value = _getitem_array1d(context, builder, arrayty, ary, index,
                                 wraparound=False)
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


@lower_builtin('iternext', SeriesIterator)
@iternext_impl(RefType.BORROWED)
def iternext_series(context, builder, sig, args, result):
    """
    Iternext implementation depending on Array type

    :param context: context descriptor
    :param builder: llvmlite IR Builder
    :param sig: iterator signature
    :param args: tuple with iterator arguments, such as instruction, operands and types
    :param result: iternext result
    """
    iternext_func = sig.args[0]._iternext
    iternext_func(context=context, builder=builder, sig=sig, args=args, result=result)


def _get_series_array_type(dtype):
    """get underlying default array type of series based on its dtype
    """
    # list(list(str))
    if dtype == types.List(string_type):
        # default data layout is list but split view is used if possible
        return list_string_array_type
    # string array
    elif dtype == string_type:
        return string_array_type

    # categorical
    if isinstance(dtype, PDCategoricalDtype):
        return CategoricalArray(dtype)

    # use recarray data layout for series of tuples
    if isinstance(dtype, types.BaseTuple):
        if any(not isinstance(t, types.Number) for t in dtype.types):
            # TODO: support more types. what types can be in recarrays?
            raise ValueError("series tuple dtype {} includes non-numerics".format(dtype))
        np_dtype = np.dtype(
            ','.join(str(t) for t in dtype.types), align=True)
        dtype = from_dtype(np_dtype)

    # TODO: other types?
    # regular numpy array
    return types.Array(dtype, 1, 'C')

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


import numba
from numba import types
from numba.extending import (typeof_impl, unbox, register_model, models,
                             NativeValue, box)
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import (impl_ret_new_ref, impl_ret_borrowed,
                                    lower_builtin)
from numba.typing import signature

import sdc
from sdc.hiframes.pd_timestamp_ext import (datetime_date_type,
                                            box_datetime_date_array)


_data_array_typ = types.Array(types.int64, 1, 'C')

# Array of datetime date objects, same as Array but knows how to box
# TODO: defer to Array for all operations


class ArrayDatetimeDate(types.Array):
    def __init__(self):
        super(ArrayDatetimeDate, self).__init__(
            datetime_date_type, 1, 'C', name='array_datetime_date')


array_datetime_date = ArrayDatetimeDate()


@register_model(ArrayDatetimeDate)
class ArrayDatetimeDateModel(models.ArrayModel):
    def __init__(self, dmm, fe_type):
        super(ArrayDatetimeDateModel, self).__init__(dmm, _data_array_typ)


@box(ArrayDatetimeDate)
def box_df_dummy(typ, val, c):
    return box_datetime_date_array(typ, val, c)


# dummy function use to change type of Array(datetime_date) to
# array_datetime_date
def np_arr_to_array_datetime_date(A):
    return A


@infer_global(np_arr_to_array_datetime_date)
class NpArrToArrDtType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(array_datetime_date, *args)


@lower_builtin(np_arr_to_array_datetime_date, types.Array(types.int64, 1, 'C'))
@lower_builtin(np_arr_to_array_datetime_date,
               types.Array(datetime_date_type, 1, 'C'))
def lower_np_arr_to_array_datetime_date(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])

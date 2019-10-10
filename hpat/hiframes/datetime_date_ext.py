import numba
from numba import types
from numba.extending import (typeof_impl, unbox, register_model, models,
                             NativeValue, box)
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import (impl_ret_new_ref, impl_ret_borrowed,
                                    lower_builtin)
from numba.typing import signature

import hpat
from hpat.hiframes.pd_timestamp_ext import (datetime_date_type,
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

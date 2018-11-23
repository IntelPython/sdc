from numba import types
from numba.extending import overload
from hpat.timsort import getitem_arr_tup

# returns scalar instead of tuple if only one array
def getitem_arr_tup_single(arrs, i):
    return arrs[0][i]

@overload(getitem_arr_tup_single)
def getitem_arr_tup_single_overload(arrs_t, i_t):
    if len(arrs_t.types) == 1:
        return lambda arrs, i: arrs[0][i]
    return lambda arrs, i: getitem_arr_tup(arrs, i)

def val_to_tup(val):
    return (val,)

@overload(val_to_tup)
def val_to_tup_overload(val_t):
    if isinstance(val_t, types.BaseTuple):
        return lambda a: a
    return lambda a: (a,)

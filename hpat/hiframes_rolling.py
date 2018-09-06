import hpat
import numba
from numba import types
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.ir_utils import guard, find_const



def rolling_fixed(arr, win):  # pragma: no cover
    return arr


@infer_global(rolling_fixed)
class RollingType(AbstractTemplate):
    def generic(self, args, kws):
        arr = args[0]  # array or series
        # result is always float64 in pandas
        # see _prep_values() in window.py
        return signature(arr.copy(dtype=types.float64), *args)




def get_rolling_setup_args(func_ir, rhs, get_consts=True):
    """
    Handle Series rolling calls like:
        r = df.column.rolling(3)
    """
    center = False
    kws = dict(rhs.kws)
    if rhs.args:
        window = rhs.args[0]
    elif 'window' in kws:
        window = kws['window']
    else:  # pragma: no cover
        raise ValueError("window argument to rolling() required")
    if get_consts:
        window_const = guard(find_const, func_ir, window)
        window = window_const if window_const is not None else window
    if 'center' in kws:
        # TODO: fix center
        center_const = guard(find_const, func_ir, kws['center'])
        center = center_const if center_const is not None else center
    return window, center

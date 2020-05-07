# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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
import sdc
import numba
from numba.ir_utils import guard, find_const


supported_rolling_funcs = ('sum', 'mean', 'var', 'std', 'count', 'median',
                           'min', 'max', 'cov', 'corr', 'apply')


def get_rolling_setup_args(func_ir, rhs, get_consts=True):
    """
    Handle Series rolling calls like:
        r = df.column.rolling(3)
    """
    center = False
    on = None
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
        center = kws['center']
        if get_consts:
            center_const = guard(find_const, func_ir, center)
            center = center_const if center_const is not None else center
    if 'on' in kws:
        on = guard(find_const, func_ir, kws['on'])
        if on is None:
            raise ValueError("'on' argument to rolling() should be constant string")
    # convert string offset window statically to nanos
    # TODO: support dynamic conversion
    # TODO: support other offsets types (time delta, etc.)
    if on is not None:
        window = guard(find_const, func_ir, window)
        if not isinstance(window, str):
            raise ValueError("window argument to rolling should be constant"
                             "string in the offset case (variable window)")
        window = pd.tseries.frequencies.to_offset(window).nanos
    return window, center, on

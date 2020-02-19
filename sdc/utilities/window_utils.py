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

import numpy as np
from numba import jitclass, njit, types


spec = [
    ('size', types.int64),
    ('minp', types.int64),
    ('_nfinite', types.int64),
    ('_nroll', types.int64),
    ('_result', types.float64)
]


@jitclass(spec)
class WindowSum:
    def __init__(self, size, minp):
        self.size = size
        self.minp = minp

        self._nfinite = 0
        self._nroll = 0
        self._result = 0.

    @property
    def result(self):
        """Get the latest result taking into account min periods."""
        if self._nfinite < self.minp:
            return np.nan

        return self._result

    def _put(self, value):
        """Calculate the window sum with new value."""
        if np.isfinite(value):
            self._nfinite += 1
            self._result += value

    def _pop(self, value):
        """Calculate the window sum without old value."""
        if np.isfinite(value):
            self._nfinite -= 1
            self._result -= value

    def _roll(self, data, idx):
        """Calculate the window sum based on the previous result."""
        if self._nroll >= self.size:
            value = data[idx - self.size]
            self._pop(value)

        value = data[idx]
        self._put(value)

        self._nroll += 1

    def roll(self, data, idx):
        """Calculate the window sum."""
        if self._nroll == 0:
            start = max(idx + 1 - self.size, 0)
            for i in range(start, idx):
                self._roll(data, i)

        self._roll(data, idx)

    def free(self):
        """Free the window."""
        self._nfinite = 0
        self._nroll = 0
        self._result = 0.


if __name__ == '__main__':
    @njit
    def sum():
        win_sum = WindowSum(3, 2)
        data = list(range(5))  # 0, 1, 2, 3, 4
        for i in data:
            win_sum.roll(data, i)
            print(win_sum.result)  # nan, 1.0, 3.0, 6.0, 9.0
    sum()

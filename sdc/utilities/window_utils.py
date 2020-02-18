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

    def roll(self, data, idx):
        """Calculate the window sum."""
        if self._nroll >= self.size:
            excluded_value = data[idx - self.size]
            if np.isfinite(excluded_value):
                self._nfinite -= 1
                self._result -= excluded_value

        value = data[idx]
        if np.isfinite(value):
            self._nfinite += 1
            self._result += value

        self._nroll += 1

    def free(self):
        """Free the window."""
        self._nfinite = 0
        self._nroll = 0
        self._result = 0.


if __name__ == '__main__':
    @njit
    def sum():
        win_sum = WindowSum(3, 2)
        data = list(range(5)) # 0, 1, 2, 3, 4
        for i in data:
            win_sum.roll(data, i)
            print(win_sum.result) # nan, 1.0, 3.0, 6.0, 9.0
    sum()

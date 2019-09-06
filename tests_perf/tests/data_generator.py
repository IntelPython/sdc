import string

from collections.abc import Iterable

import numpy as np
import pandas as pd

from numpy.random import randn
from pandas.util import testing as tm


class DataGenerator:
    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def randu(self, length):
        """Generate one random unicode string."""
        return tm.randu(length)

    def range(self, length, repeat=1):
        """Generate range with int32 values."""
        return np.arange(length).repeat(repeat)

    def randn(self, length, repeat=1):
        """Generate random array with float64 values."""
        return randn(length).repeat(repeat)

    def make_int_series(self, length, repeat=1, index=None, name=None):
        """Generate series with float range."""
        data = pd.Int64Index(self.range(length, repeat=repeat))
        return pd.Series(data, index=index, name=name)

    def make_uint_series(self, length, repeat=1, index=None, name=None):
        """Generate series with unsigned integers range."""
        data = pd.UInt64Index(self.range(length, repeat=repeat))
        return pd.Series(data, index=index, name=name)

    def make_float_series(self, length, repeat=1, index=None, name=None):
        """Generate series with random floats."""
        data = pd.Float64Index(self.randn(length, repeat=repeat))
        return pd.Series(data, index=index, name=name)

    def make_numeric_dataframe(self, length, repeat=1, index=None):
        """Generate simple numeric data frame with integers/floats ranges."""
        data = self.range(length, repeat=repeat)
        return pd.DataFrame({
            'A': data,
            'B': data + 1.0,
            'C': data + 2.0,
            'D': data + 3.0,
        }, index=index)


class StringSeriesGenerator(DataGenerator):
    NCHARS = [0, 1, 3, 5, 9, 17, 33, 61, 97]
    N = len(NCHARS) * (10 ** 5 + 513)
    RANDS_CHARS = np.array(list(string.ascii_letters + string.digits), dtype=(np.str_, 1))

    def __init__(self, size=None, nchars=None):
        self.size = size or self.N
        self.nchars = nchars or self.NCHARS

        if not isinstance(self.nchars, Iterable):
            raise TypeError(f'nchars={self.nchars} is not iterable, should be iterable')

        if self.size % len(self.nchars) != 0:
            raise ValueError(f'len(nchars)={len(self.nchars)} does NOT match size={size}, should be multiples')

    def generate(self):
        """Generate series of strings."""
        return pd.Series(pd.Index(self._rands_array))

    @property
    def _rands_array(self):
        """Generate an array of byte strings."""
        arrays = []
        for n in self.nchars:
            if n == 0:
                # generate array of empty strings
                arr = np.array(self.size * [''])
            else:
                # generate array of random n-size strings
                arr = np.random.choice(self.RANDS_CHARS, size=n * self.size).view((np.str_, n))
            arrays.append(arr)

        return np.concatenate(arrays)

import numpy as np
import pandas as pd

from numpy.random import randn
from pandas.util import testing as tm


class DataGenerator:
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

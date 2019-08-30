import numpy as np
import pandas as pd

from numpy.random import randn
from pandas.util import testing as tm


class DataGenerator:
    def __init__(self):
        pass

    def randu(self, length):
        """Generate one random unicode string."""
        return tm.randu(length)

    def make_int_series(self, length, repeat=1, index=None, name=None):
        """Generate series with float range."""
        data = pd.Int64Index(np.arange(length)).repeat(repeat)
        return pd.Series(data, index=index, name=name)

    def make_uint_series(self, length, repeat=1, index=None, name=None):
        """Generate series with unsigned integers range."""
        data = pd.UInt64Index(np.arange(length)).repeat(repeat)
        return pd.Series(data, index=index, name=name)

    def make_float_series(self, length, repeat=1, index=None, name=None):
        """Generate series with random integers."""
        data = pd.Float64Index(randn(length)).repeat(repeat)
        return pd.Series(data, index=index, name=name)

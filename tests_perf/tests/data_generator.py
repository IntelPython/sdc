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


"""
Input data generator for performance tests.
"""
import contextlib
import string

from collections.abc import Iterable

import numpy as np
import pandas as pd

from numpy.random import randn
from pandas.util import testing as tm


class DataGenerator:
    N = 10 ** 6 + 513
    SEED = 123

    def __init__(self, size=None, seed=None):
        self.size = size or self.N
        self.seed = seed or self.SEED

    @contextlib.contextmanager
    def set_seed(self):
        """Substitute random seed for context"""
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            yield
        finally:
            np.random.set_state(state)

    def generate(self, *args, **kwargs):
        """Generate data"""
        raise NotImplementedError

    @property
    def rand_array(self, *args, **kwargs):
        """Random array"""
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


class SeriesGenerator(DataGenerator):

    def generate(self):
        """Generate series"""
        return pd.Series(self.rand_array)


class StringSeriesGenerator(SeriesGenerator):
    NCHARS = [1, 3, 5, 9, 17, 33]
    N = 5 * 10 ** 6 + 513
    # RANDS_CHARS = [a-zA-Z] + [0-9] + [ \t\n\r\f\v]
    RANDS_CHARS = np.array(list(string.ascii_letters + string.digits + string.whitespace), dtype=(np.str_, 1))

    def __init__(self, size=None, nchars=None, seed=None):
        super().__init__(size=size, seed=seed)

        self.nchars = nchars or self.NCHARS
        if not isinstance(self.nchars, Iterable):
            self.nchars = [self.nchars]

        self.size = len(self.nchars) * self.size

    @property
    def rand_array(self):
        """Array of random strings"""
        arrays = []
        for n in self.nchars:
            arrays.append(self._rand_array(n))

        result_array = np.concatenate(arrays)
        # shuffle strings array
        with self.set_seed():
            np.random.shuffle(result_array)

        return result_array

    def _rand_array(self, n):
        """Generate an array of random n-size strings"""
        if n == 0:
            # generate array of empty strings
            return np.array(self.size * [''])

        # generate array of random n-size strings
        with self.set_seed():
            return np.random.choice(self.RANDS_CHARS, size=n * self.size).view((np.str_, n))


class WhiteSpaceStringSeriesGenerator(StringSeriesGenerator):
    def _rand_array(self, n):
        """Generate an array of random n-size strings which start and end with white space"""
        if n < 3:
            # generate array of white space strings
            return np.array(self.size * [' ' * n])

        # generate array of random n-size strings which start and end with white space
        with self.set_seed():
            arr = np.random.choice(self.RANDS_CHARS, size=(n - 2) * self.size).view((np.str_, n - 2))
            np.char.center(arr, n)
            return arr


class FloatSeriesGenerator(SeriesGenerator):
    N = 5 * 10 ** 6 + 513

    def __init__(self, size=None, seed=None):
        super().__init__(size=size, seed=seed)

    @property
    def rand_array(self):
        """Array of random floats"""
        with self.set_seed():
            return np.array(randn(self.size))


class FloatSeriesIndexGenerator(FloatSeriesGenerator):

    def generate(self):
        index = StringSeriesGenerator(self.size).generate()
        data = FloatSeriesGenerator(self.size).generate()
        return pd.Series(data=data, index=index)

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


import hpat

from .common import Implementation as Impl
from .data_generator import StringSeriesGenerator, WhiteSpaceStringSeriesGenerator
from .data_generator import FloatSeriesGenerator, FloatSeriesIndexGenerator


class String:
    params = [
        [StringSeriesGenerator.N],
        StringSeriesGenerator.NCHARS,
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'nchars', 'implementation']

    def setup(self, size, nchars, implementation):
        self.series = StringSeriesGenerator(size=size, nchars=nchars).generate()

    @staticmethod
    @hpat.jit
    def _len(series):
        return series.str.len()

    def time_len(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.len"""
        if implementation == Impl.compiled_python.value:
            return self._len(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.len()

    @staticmethod
    @hpat.jit
    def _capitalize(series):
        return series.str.capitalize()

    def time_capitalize(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.capitalize"""
        if implementation == Impl.compiled_python.value:
            return self._capitalize(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.capitalize()

    @staticmethod
    @hpat.jit
    def _lower(series):
        return series.str.lower()

    def time_lower(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.lower"""
        if implementation == Impl.compiled_python.value:
            return self._lower(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.lower()

    @staticmethod
    @hpat.jit
    def _swapcase(series):
        return series.str.swapcase()

    def time_swapcase(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.swapcase"""
        if implementation == Impl.compiled_python.value:
            return self._swapcase(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.swapcase()

    @staticmethod
    @hpat.jit
    def _title(series):
        return series.str.title()

    def time_title(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.title"""
        if implementation == Impl.compiled_python.value:
            return self._title(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.title()

    @staticmethod
    @hpat.jit
    def _upper(series):
        return series.str.upper()

    def time_upper(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.upper"""
        if implementation == Impl.compiled_python.value:
            return self._upper(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.upper()


class WhiteSpaceString:
    params = [
        [WhiteSpaceStringSeriesGenerator.N],
        WhiteSpaceStringSeriesGenerator.NCHARS,
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'nchars', 'implementation']

    def setup(self, size, nchars, implementation):
        self.series = WhiteSpaceStringSeriesGenerator(size=size, nchars=nchars).generate()

    @staticmethod
    @hpat.jit
    def _lstrip(series):
        return series.str.lstrip()

    def time_lstrip(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.lstrip"""
        if implementation == Impl.compiled_python.value:
            return self._lstrip(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.lstrip()

    @staticmethod
    @hpat.jit
    def _rstrip(series):
        return series.str.rstrip()

    def time_rstrip(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.rstrip"""
        if implementation == Impl.compiled_python.value:
            return self._rstrip(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.rstrip()

    @staticmethod
    @hpat.jit
    def _strip(series):
        return series.str.strip()

    def time_strip(self, size, nchars, implementation):
        """Time both interpreted and compiled Series.str.strip"""
        if implementation == Impl.compiled_python.value:
            return self._strip(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.str.strip()


class SortValues:
    params = [
        [2 * 10 ** 6 + 513],
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'implementation']

    def setup(self, size, implementation):
        self.series = FloatSeriesGenerator(size=size).generate()

    @staticmethod
    @hpat.jit
    def _sort_values(series):
        return series.sort_values()

    def time_sort_values(self, size, implementation):
        if implementation == Impl.compiled_python.value:
            return self._sort_values(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.sort_values()


class IdxMaxMin:
    params = [
        [10 ** 5],
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'implementation']

    def setup(self, size, implementation):
        self.series = FloatSeriesIndexGenerator(size=size).generate()

    @staticmethod
    @hpat.jit
    def _idxmax(series):
        return series.idxmax()

    def time_idxmax(self, size, implementation):
        if implementation == Impl.compiled_python.value:
            return self._idxmax(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.idxmax()

    @staticmethod
    @hpat.jit
    def _idxmin(series):
        return series.idxmin()

    def time_idxmin(self, size, implementation):
        if implementation == Impl.compiled_python.value:
            return self._idxmin(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.idxmin()

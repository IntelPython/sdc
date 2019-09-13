import hpat

from .common import Implementation as Impl
from .data_generator import StringSeriesGenerator


class Methods:
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

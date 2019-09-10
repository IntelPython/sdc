"""

"""
import hpat

from .common import Implementation as Impl
from .data_generator import StringSeriesGenerator


class Methods:
    params = [
        [StringSeriesGenerator.N],
        [StringSeriesGenerator.NCHARS],
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

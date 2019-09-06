import hpat

from .common import Implementation
from .data_generator import StringSeriesGenerator


class Methods:
    params = [
        [Implementation.native.value, Implementation.hpat.value]
    ]
    param_names = ['implementation']

    def setup(self, implementation):
        self.s = StringSeriesGenerator().generate()

    @staticmethod
    @hpat.jit
    def _len(s):
        return s.str.len()

    def time_len(self, implementation):
        if implementation == Implementation.hpat.value:
            return self._len(self.s)
        if implementation == Implementation.native.value:
            return self.s.str.len()

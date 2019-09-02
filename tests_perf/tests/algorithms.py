import hpat

from .common import Implementation
from .data_generator import DataGenerator


class Quantile():
    params = [[0.5],
              ['linear', 'nearest', 'lower', 'higher', 'midpoint'],
              ['float', 'int', 'uint'],
              [Implementation.native.value, Implementation.hpat.value]]
    pparam_names = ['quantile', 'interpolation', 'dtype', 'implementation']

    def setup(self, quantile, interpolation, dtype, implementation):
        N = 10 ** 7
        data_generator = DataGenerator()
        data = {
            'int': data_generator.make_int_series(N, repeat=5),
            'uint': data_generator.make_uint_series(N, repeat=5),
            'float': data_generator.make_float_series(N, repeat=5),
        }
        self.idx = data[dtype]

    @staticmethod
    @hpat.jit
    def _quantile(idx, quantile, interpolation):
        return idx.quantile(quantile, interpolation=interpolation)

    def time_quantile(self, quantile, interpolation, dtype, implementation):
        if implementation == Implementation.hpat.value:
            return self._quantile(self.idx, quantile, interpolation)
        if implementation == Implementation.native.value:
            return self.idx.quantile(quantile, interpolation)

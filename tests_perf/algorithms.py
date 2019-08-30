from .common import Implementation, ImplRunner
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

    def time_quantile(self, quantile, interpolation, dtype, implementation):
        def _quantile(idx, quantile, interpolation=interpolation):
            return idx.quantile(quantile, interpolation=interpolation)
        ImplRunner(implementation, _quantile).run(self.idx, quantile, interpolation=interpolation)

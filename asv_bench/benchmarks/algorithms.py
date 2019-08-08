import hpat

from .hpat_vb_common import *
from pd_benchmarks import algorithms


class Quantile(algorithms.Quantile):
    params = [[0, 0.5, 1],
              ['linear', 'nearest', 'lower', 'higher', 'midpoint'],
              ['float', 'int', 'uint'],
              [tool.value for tool in Tool]]
    param_names = ['quantile', 'interpolation', 'dtype', 'tool']

    def setup(self, quantile, interpolation, dtype, tool):
        super().setup(quantile, interpolation, dtype)

    @staticmethod
    @hpat.jit
    def _quantile(idx, quantile, interpolation=None):
        idx.quantile(quantile, interpolation=interpolation)

    def time_quantile(self, quantile, interpolation, dtype, tool):
        if tool == Tool.pandas.value:
            super().time_quantile(quantile, interpolation, dtype)
        elif tool == Tool.hpat.value:
            self._quantile(self.idx, quantile, interpolation=interpolation)

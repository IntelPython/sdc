import hpat

from .hpat_vb_common import *
from pd_benchmarks import algorithms


class Quantile(algorithms.Quantile):

    params = algorithms.Quantile.params + [[tool.value for tool in Tool]]
    param_names = algorithms.Quantile.param_names + ['tool']

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

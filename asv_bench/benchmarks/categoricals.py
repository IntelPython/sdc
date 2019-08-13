import pandas as pd

import hpat

from .hpat_vb_common import *
from pd_benchmarks import categoricals


class Indexing(categoricals.Indexing):

    params = [tool.value for tool in Tool]
    param_names = ['tool']

    def setup(self, tool):
        super().setup()

    def time_get_loc(self, tool):
        raise NotImplementedError

    def time_shape(self, tool):
        raise NotImplementedError

    def time_shallow_copy(self, tool):
        raise NotImplementedError

    @staticmethod
    @hpat.jit
    def _align(series):
        return pd.DataFrame({'a': series, 'b': series[:500]})

    def time_align(self, tool):
        if tool == Tool.pandas.value:
            super().time_align()
        elif tool == Tool.hpat.value:
            self._align(self.series)

    def time_intersection(self, tool):
        raise NotImplementedError

    def time_unique(self, tool):
        raise NotImplementedError

    def time_reindex(self, tool):
        raise NotImplementedError

    def time_reindex_missing(self, tool):
        raise NotImplementedError

    def time_sort_values(self, tool):
        raise NotImplementedError

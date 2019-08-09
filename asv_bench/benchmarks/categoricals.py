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
        if tool == Tool.pandas.value:
            super().time_get_loc()
        elif tool == Tool.hpat.value:
            raise NotImplementedError

    def time_shape(self, tool):
        if tool == Tool.pandas.value:
            super().time_shape()
        elif tool == Tool.hpat.value:
            raise NotImplementedError

    def time_shallow_copy(self, tool):
        if tool == Tool.pandas.value:
            super().time_shallow_copy()
        elif tool == Tool.hpat.value:
            raise NotImplementedError

    @staticmethod
    @hpat.jit
    def _align(series):
        pd.DataFrame({'a': series, 'b': series[:500]})

    def time_align(self, tool):
        if tool == Tool.pandas.value:
            super().time_align()
        elif tool == Tool.hpat.value:
            self._align(self.series)

    def time_intersection(self, tool):
        if tool == Tool.pandas.value:
            super().time_intersection()
        elif tool == Tool.hpat.value:
            raise NotImplementedError

    def time_unique(self, tool):
        if tool == Tool.pandas.value:
            super().time_unique()
        elif tool == Tool.hpat.value:
            raise NotImplementedError

    def time_reindex(self, tool):
        if tool == Tool.pandas.value:
            super().time_reindex()
        elif tool == Tool.hpat.value:
            raise NotImplementedError

    def time_reindex_missing(self, tool):
        if tool == Tool.pandas.value:
            super().time_reindex_missing()
        elif tool == Tool.hpat.value:
            raise NotImplementedError

    def time_sort_values(self, tool):
        if tool == Tool.pandas.value:
            super().time_sort_values()
        elif tool == Tool.hpat.value:
            raise NotImplementedError

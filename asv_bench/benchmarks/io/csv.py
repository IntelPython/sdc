import hpat

from ..hpat_vb_common import *
from pd_benchmarks.io import csv


class ToCSV(csv.ToCSV):

    params = [['long'] , [tool.value for tool in Tool]]
    param_names = csv.ToCSV.param_names + ['tool']

    def setup(self, kind, tool):
        super().setup(kind)

    @staticmethod
    @hpat.jit
    def _frame(df, fname):
        df.to_csv(fname)

    def time_frame(self, kind, tool):
        if tool == Tool.pandas.value:
            super().time_frame(kind)
        elif tool == Tool.hpat.value:
            self._frame(self.df, self.fname)

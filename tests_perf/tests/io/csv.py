import hpat

from ..common import BaseIO, Implementation as Impl
from ..data_generator import DataGenerator


class ToCSV(BaseIO):
    fname = '__test__.csv'
    params = [Impl.interpreted_python.value, Impl.compiled_python.value]
    pparam_names = ['implementation']

    def setup(self, implementation):
        N = 10 ** 4
        data_generator = DataGenerator()
        self.df = data_generator.make_numeric_dataframe(5*N)

    @staticmethod
    @hpat.jit
    def _frame(df, fname):
        df.to_csv(fname)

    def time_frame(self, implementation):
        """Time both interpreted and compiled DataFrame.to_csv"""
        if implementation == Impl.compiled_python.value:
            self._frame(self.df, self.fname)
        elif implementation == Impl.interpreted_python.value:
            self.df.to_csv(self.fname)

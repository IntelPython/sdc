import hpat

from ..common import BaseIO, Implementation
from ..data_generator import DataGenerator


class ToCSV(BaseIO):
    fname = '__test__.csv'
    params = [Implementation.native.value, Implementation.hpat.value]
    pparam_names = ['implementation']

    def setup(self, implementation):
        N = 10 ** 4
        data_generator = DataGenerator()
        self.df = data_generator.make_numeric_dataframe(5 * N)

    @staticmethod
    @hpat.jit
    def _frame(df, fname):
        df.to_csv(fname)

    def time_frame(self, implementation):
        if implementation == Implementation.hpat.value:
            self._frame(self.df, self.fname)
        elif implementation == Implementation.native.value:
            self.df.to_csv(self.fname)

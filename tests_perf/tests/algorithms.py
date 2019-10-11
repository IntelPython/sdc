import hpat

from .common import Implementation as Impl
from .data_generator import DataGenerator, FloatSeriesGenerator


class Quantile():
    params = [[0.5],
              ['linear', 'nearest', 'lower', 'higher', 'midpoint'],
              ['float', 'int', 'uint'],
              [Impl.interpreted_python.value, Impl.compiled_python.value]]
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
        if implementation == Impl.compiled_python.value:
            return self._quantile(self.idx, quantile, interpolation)
        if implementation == Impl.interpreted_python.value:
            return self.idx.quantile(quantile, interpolation)


class Absolute:
    params = [
        [3 * 10 ** 8 + 513],
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'implementation']

    def setup(self, size, implementation):
        self.series = FloatSeriesGenerator(size=size).generate()

    @staticmethod
    @hpat.jit
    def _abs(series):
        return series.abs()

    def time_abs(self, size, implementation):
        """Time both interpreted and compiled Series.abs"""
        if implementation == Impl.compiled_python.value:
            return self._abs(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.abs()


class ValueCounts:
    params = [
        [5 * 10 ** 6 + 513],
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'implementation']

    def setup(self, size, implementation):
        self.series = FloatSeriesGenerator(size).generate()

    @staticmethod
    @hpat.jit
    def _value_counts(series):
        return series.value_counts()

    def time_value_counts(self,  size, implementation):
        """Time both interpreted and compiled Series.value_counts"""
        if implementation == Impl.compiled_python.value:
            return self._value_counts(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.value_counts()


class MinMax:
    params = [
        [25 * 10 ** 7 + 513],
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'implementation']

    def setup(self, size, implementation):
        self.series = FloatSeriesGenerator(size=size).generate()

    @staticmethod
    @hpat.jit
    def _min(series):
        return series.min()

    def time_min(self, size, implementation):
        """Time both interpreted and compiled Series.min"""
        if implementation == Impl.compiled_python.value:
            return self._min(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.min()

    @staticmethod
    @hpat.jit
    def _max(series):
        return series.max()

    def time_max(self, size, implementation):
        """Time both interpreted and compiled Series.max"""
        if implementation == Impl.compiled_python.value:
            return self._max(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.max()


class Correlation:
    params = [
        [10 ** 8 + 513],
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'implementation']

    def setup(self, size, implementation):
        self.series = FloatSeriesGenerator(size).generate()
        self.series2 = FloatSeriesGenerator(size).generate()

    @staticmethod
    @hpat.jit
    def _cov(series, series2):
        return series.cov(series2)

    def time_cov(self, size, implementation):
        """Time both interpreted and compiled Series.cov"""
        if implementation == Impl.compiled_python.value:
            return self._cov(self.series, self.series2)
        if implementation == Impl.interpreted_python.value:
            return self.series.cov(self.series2)

    @staticmethod
    @hpat.jit
    def _corr(series, series2):
        return series.corr(series2)

    def time_corr(self, size, implementation):
        """Time both interpreted and compiled Series.cov"""
        if implementation == Impl.compiled_python.value:
            return self._corr(self.series, self.series2)
        if implementation == Impl.interpreted_python.value:
            return self.series.corr(self.series2)


class Sum:
    params = [
        [10 ** 8 + 513],
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['size', 'implementation']

    def setup(self, size, implementation):
        self.series = FloatSeriesGenerator(size=size).generate()

    @staticmethod
    @hpat.jit
    def _sum(series):
        return series.sum()

    def time_sum(self, size, implementation):
        """Time both interpreted and compiled Series.min"""
        if implementation == Impl.compiled_python.value:
            return self._sum(self.series)
        if implementation == Impl.interpreted_python.value:
            return self.series.sum()

    class Count:
        params = [
            [5 * 10 ** 8 + 513],
            [Impl.interpreted_python.value, Impl.compiled_python.value]
        ]
        param_names = ['size', 'implementation']

        def setup(self, size, implementation):
            self.series = FloatSeriesGenerator(size).generate()

        @staticmethod
        @hpat.jit
        def _count(series):
            return series.count()

        def time_count(self, size, implementation):
            """Time both interpreted and compiled Series.value_counts"""
            if implementation == Impl.compiled_python.value:
                return self._count(self.series)
            if implementation == Impl.interpreted_python.value:
                return self.series.count()


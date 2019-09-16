import numba

from .common import Implementation as Impl
from .data_generator import DataGenerator


class Methods:
    params = [
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    param_names = ['implementation']

    def setup(self, implementation):
        N = 10 ** 4
        data_generator = DataGenerator()
        self.s = data_generator.randu(N)


class WidthMethods(Methods):
    def setup(self, implementation):
        super().setup(implementation)
        self.width = 10 ** 8

    @staticmethod
    @numba.njit
    def _center(s, width):
        return s.center(width)

    def time_center(self, implementation):
        if implementation == Impl.compiled_python.value:
            return self._center(self.s, self.width)
        if implementation == Impl.interpreted_python.value:
            return self.s.center(self.width)

    @staticmethod
    @numba.njit
    def _ljust(s, width):
        return s.ljust(width)

    def time_ljust(self, implementation):
        if implementation == Impl.compiled_python.value:
            return self._rjust(self.s, self.width)
        if implementation == Impl.interpreted_python.value:
            return self.s.rjust(self.width)

    @staticmethod
    @numba.njit
    def _rjust(s, width):
        return s.rjust(width)

    def time_rjust(self, implementation):
        if implementation == Impl.compiled_python.value:
            return self._rjust(self.s, self.width)
        if implementation == Impl.interpreted_python.value:
            return self.s.rjust(self.width)

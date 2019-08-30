from .common import Implementation, ImplRunner
from .data_generator import DataGenerator


class Methods:
    params = [Implementation.native.value, Implementation.njit.value]
    param_names = ['implementation']

    def setup(self, implementation):
        N = 10 ** 4
        data_generator = DataGenerator()
        self.s = data_generator.randu(N)


class WidthMethods(Methods):
    def setup(self, implementation):
        super().setup(implementation)
        self.width = 10 ** 8

    def time_center(self, implementation):
        def _center(s, width):
            return s.center(width)
        ImplRunner(implementation, _center).run(self.s, self.width)

    def time_ljust(self, implementation):
        def _ljust(s, width):
            return s.ljust(width)
        ImplRunner(implementation, _ljust).run(self.s, self.width)

    def time_rjust(self, implementation):
        def _rjust(s, width):
            return s.ljust(width)
        ImplRunner(implementation, _rjust).run(self.s, self.width)

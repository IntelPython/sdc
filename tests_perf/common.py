from enum import Enum

import numba

import hpat


class Implementation(Enum):
    native = 'native'
    njit = 'njit'
    hpat = 'hpat'


class ImplRunner:
    def __init__(self, implementation, func):
        self.implementation = implementation
        self.func = func

    @property
    def runner(self):
        if self.implementation == Implementation.hpat.value:
            return hpat.jit(self.func)
        elif self.implementation == Implementation.njit.value:
            return numba.njit(self.func)
        elif self.implementation == Implementation.native.value:
            return self.func

        raise ValueError(f'Unknown implementation: {self.implementation}')

    def run(self, *args, **kwargs):
        return self.runner(*args, **kwargs)

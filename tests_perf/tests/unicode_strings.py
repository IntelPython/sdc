# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


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

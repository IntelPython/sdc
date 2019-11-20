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


import sdc

from ..common import BaseIO, Implementation as Impl
from ..data_generator import DataGenerator


class ToCSV(BaseIO):
    fname = '__test__.csv'
    params = [
        [Impl.interpreted_python.value, Impl.compiled_python.value]
    ]
    pparam_names = ['implementation']

    def setup(self, implementation):
        N = 10 ** 4
        data_generator = DataGenerator()
        self.df = data_generator.make_numeric_dataframe(5 * N)

    @staticmethod
    @sdc.jit
    def _to_csv(df, fname):
        return df.to_csv(fname)

    def time_to_csv(self, implementation):
        """Time both interpreted and compiled DataFrame.to_csv"""
        if implementation == Impl.compiled_python.value:
            return self._to_csv(self.df, self.fname)
        if implementation == Impl.interpreted_python.value:
            return self.df.to_csv(self.fname)

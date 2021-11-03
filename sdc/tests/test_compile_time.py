# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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
import numpy as np
import pandas as pd
import re
import unittest

from contextlib import redirect_stdout
from io import StringIO
from sdc.tests.test_base import TestCase
from sdc.decorators import debug_compile_time


# regexp patterns for lines in @debug_compile_time output log
line_heading = r'\*+\s+COMPILE STATS\s+\*+\n'
line_function = r'Function: [^\s]+\n'
line_args = r'\s+Args:.*\n'
line_pipeline = r'\s+Pipeline: \w+\n'
line_passes = r'(\s+\w+\s+[\d.]+\n)+'
line_time = r'\s+Time: [\d.]+\n'
line_ending = r'\*+\n'


class TestCompileTime(TestCase):

    @staticmethod
    def _gen_usecase_data():
        n = 11
        S1 = pd.Series(np.ones(n))
        S2 = pd.Series(2 ** np.arange(n))
        return S1, S2

    def test_log_format_summary(self):
        """ Verifies shortened log format when only summary info is printed """

        @debug_compile_time(level=0)
        @self.jit
        def test_impl(S1, S2):
            return S1 + S2

        buffer = StringIO()
        with redirect_stdout(buffer):
            S1, S2 = self._gen_usecase_data()
            test_impl(S1, S2)

        entry_format = fr'{line_function}{line_pipeline}{line_time}\n'
        log_format = fr'{line_heading}({entry_format})+{line_ending}$'
        self.assertRegex(buffer.getvalue(), log_format)

    def test_log_format_detailed(self):
        """ Verifies detailed log format with passes and args information """

        @debug_compile_time()
        @self.jit
        def test_impl(S1, S2):
            return S1 + S2

        buffer = StringIO()
        with redirect_stdout(buffer):
            S1, S2 = self._gen_usecase_data()
            test_impl(S1, S2)

        entry_format = fr'{line_function}{line_args}{line_pipeline}{line_passes}{line_time}\n'
        log_format = fr'{line_heading}({entry_format})+{line_ending}'
        self.assertRegex(buffer.getvalue(), log_format)

    def test_func_names_filter(self):
        """ Verifies filtering log entries via func_names paramter """
        searched_name = 'add'

        @debug_compile_time(func_names=[searched_name])
        @self.jit
        def test_impl(S1, S2):
            return S1 + S2

        buffer = StringIO()
        with redirect_stdout(buffer):
            S1, S2 = self._gen_usecase_data()
            test_impl(S1, S2)

        line_function = r'Function: ([^\s]+)\n'
        match_iter = re.finditer(line_function, buffer.getvalue())
        next(match_iter)    # skip entry for top-level func
        for m in match_iter:
            self.assertIn(searched_name, m.group(1))


if __name__ == "__main__":
    unittest.main()

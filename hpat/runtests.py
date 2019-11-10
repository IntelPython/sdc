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


import os
import unittest
import hpat.tests
from hpat.tests.test_basic import get_rank

"""
    Every test in suite can be executed specified times using
    desired value for SDC_REPEAT_TEST_NUMBER environment variable.
    This can be used to locate scpecific failures occured
    on next execution of affected test.

    loadTestsFromModule returns TestSuite obj with _tests member
    which contains further TestSuite instanses for each found testCase:
    hpat_tests = TestSuite(hpat.tests)
    TestSuite(hpat.tests)._tests = [TestSuite(hpat.tests.TestBasic), TestSuite(hpat.tests.TestDataFrame), ...]
    TestSuite(hpat.tests.TestBasic)._tests = [TestBasic testMethod=test_array_reduce, ...]
"""


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    hpat_tests = loader.loadTestsFromModule(hpat.tests)
    repeat_test_number = int(os.getenv('SDC_REPEAT_TEST_NUMBER', '1'))

    if repeat_test_number > 1:
        for i, test_case in enumerate(hpat_tests):
            extended_tests = []
            for test in test_case:
                for _ in range(repeat_test_number):
                    extended_tests.append(test)
            hpat_tests._tests[i]._tests = extended_tests

    suite.addTests(hpat_tests)
    return suite


if __name__ == '__main__':
    if hpat.config.config_pipeline_hpat_default:
        # initialize MPI
        get_rank()

    unittest.main()

import os
import unittest
import hpat.tests

"""
    Every test in suite can be executed specified times using
    desired value for HPAT_REPEAT_TEST_NUMBER environment variable.
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
    repeat_test_number = int(os.getenv('HPAT_REPEAT_TEST_NUMBER', '1'))

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
    unittest.main()

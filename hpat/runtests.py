import os
import unittest
import hpat.tests

from hpat.tests.tests_config import tests_to_repeat

repeat_test_number = 2
if 'REPEAT_TEST_NUMBER' in os.environ:
    repeat_test_number = int(os.environ['REPEAT_TEST_NUMBER'])

""" Decorator to repeat test execution """
def repeat_test(test):
    def repeat_test_wrapper(*args, **kwargs):
        for _ in range(repeat_test_number):
            test(*args, **kwargs)

    return repeat_test_wrapper

"""
    Wrap found tests to execute it multiple times using repeat_test decorator

    loadTestsFromModule returns suiteClass with _tests member
    which contains further suiteClass instanses:
    hpat_tests = suiteClass(hpat.tests)
    suiteClass(hpat.tests)._tests = [suiteClass(hpat.tests.TestBasic), suiteClass(hpat.tests.TestDataFrame), ...]
    suiteClass(hpat.tests)._tests[0] = suiteClass(hpat.tests.TestBasic)
    suiteClass(hpat.tests.TestBasic)._tests = [TestBasic testMethod=test_array_reduce, ...]
    test.id() returns the string like hpat.tests.test_basic.TestBasic.test_array_reduce
"""
def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    hpat_tests = loader.loadTestsFromModule(hpat.tests)

    print('Notice: {} tests will be executed {} times'.format(len(tests_to_repeat), repeat_test_number))
    for i in range(len(hpat_tests._tests)):
        for j in range(len(hpat_tests._tests[i]._tests)):
            if hpat_tests._tests[i]._tests[j].id().split('.')[-1] in tests_to_repeat:
                hpat_tests._tests[i]._tests[j] = repeat_test(hpat_tests._tests[i]._tests[j])

    suite.addTests(hpat_tests)
    return suite


if __name__ == '__main__':
    unittest.main()

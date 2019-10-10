import unittest

import hpat.tests_perf
from hpat.tests.test_basic import get_rank


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    hpat_tests_perf = loader.loadTestsFromModule(hpat.tests_perf)

    suite.addTests(hpat_tests_perf)
    return suite


if __name__ == '__main__':
    # initialize MPI
    get_rank()
    unittest.main()

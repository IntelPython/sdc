import os
import unittest

from sdc.tests.tests_perf.test_perf_utils import *


class TestBase(unittest.TestCase):
    iter_number = 5
    test_results_class = TestResults

    @classmethod
    def create_test_results(cls):
        drivers = []
        if is_true(os.environ.get('SDC_TEST_PERF_EXCEL', True)):
            drivers.append(ExcelResultsDriver('perf_results.xlsx'))
        if is_true(os.environ.get('SDC_TEST_PERF_CSV', False)):
            drivers.append(CSVResultsDriver('perf_results.csv'))

        results = cls.test_results_class(drivers)

        if is_true(os.environ.get('LOAD_PREV_RESULTS')):
            results.load()

        return results

    @classmethod
    def setUpClass(cls):
        cls.test_results = cls.create_test_results()

        cls.total_data_length = []
        cls.num_threads = int(os.environ.get('NUMBA_NUM_THREADS', config.NUMBA_NUM_THREADS))
        cls.threading_layer = os.environ.get('NUMBA_THREADING_LAYER', config.THREADING_LAYER)

    @classmethod
    def tearDownClass(cls):
        cls.test_results.print()
        cls.test_results.dump()

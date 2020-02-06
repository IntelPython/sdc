import os
import unittest
import numba

from sdc.tests.tests_perf.test_perf_utils import *


class TestBase(unittest.TestCase):
    iter_number = 5
    results_class = TestResults

    @classmethod
    def create_test_results(cls):
        drivers = []
        if is_true(os.environ.get('SDC_TEST_PERF_EXCEL', True)):
            drivers.append(ExcelResultsDriver('perf_results.xlsx'))
        if is_true(os.environ.get('SDC_TEST_PERF_CSV', False)):
            drivers.append(CSVResultsDriver('perf_results.csv'))

        results = cls.results_class(drivers)

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
        # TODO: https://jira.devtools.intel.com/browse/SAT-2371
        cls.test_results.print()
        cls.test_results.dump()

    def _test_jitted(self, pyfunc, record, *args, **kwargs):
        # compilation time
        record["compile_results"] = calc_compilation(pyfunc, *args, **kwargs)

        cfunc = numba.njit(pyfunc)

        # execution and boxing time
        record["test_results"], record["boxing_results"] = \
            get_times(cfunc, *args, **kwargs)

    def _test_python(self, pyfunc, record, *args, **kwargs):
        record["test_results"], _ = \
            get_times(pyfunc, *args, **kwargs)

    def test_jit(self, pyfunc, base, *args):
        record = base.copy()
        record["test_type"] = 'SDC'
        self._test_jitted(pyfunc, record, *args)
        self.test_results.add(**record)

    def test_py(self, pyfunc, base, *args):
        record = base.copy()
        record["test_type"] = 'Python'
        self._test_python(pyfunc, record, *args)
        self.test_results.add(**record)

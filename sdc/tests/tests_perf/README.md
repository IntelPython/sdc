### Performance testing
based on Python unit testing framework where typical test suite looks like:
```
class TestSuite(unittest.TestCase):
    # how many times function will be executed for more accurate measurements
    iter_number = 5

    @classmethod
    def setUpClass(cls):
        """
        1. Initalize object `TestResults` to work with results
        2. Define some testing attributes, e.g. list of data length
        """
        cls.test_results = TestResults()
        cls.total_data_length = [10**5, 10**6]

    @classmethod
    def tearDownClass(cls):
        """Manipulate result through object `TestResults`"""
        cls.test_results.print()

    def test_series_smth(self):
        """Test series.smth"""
        pyfunc = series_smth
        hpat_func = sdc.jit(pyfunc)
        for data_length in self.total_data_length:
            data = gen_some_data(data_length)
            test_data = pd.Series(data)

            # calculate compilation time of `hpat_func` based in `pyfunc`
            compile_results = calc_compilation(pyfunc, test_data, iter_number=self.iter_number)
            # Warming up
            hpat_func(test_data)

            # calculate execution and boxing/unboxing times of `hpat_func`
            exec_times, boxing_times = get_times(hpat_func, test_data, iter_number=self.iter_number)

            # add these times to the results for further processing
            self.test_results.add('test_series_smth', 'JIT', test_data.size, exec_times,
                                  boxing_times, compile_results=compile_results)

            # calculate execution times of `pyfunc`
            exec_times, _ = get_times(pyfunc, test_data, iter_number=self.iter_number)

            # add these times to the results for further processing
            self.test_results.add('test_series_smth', 'Reference', test_data.size, exec_times)
```

##### Extras:
1. `test_perf_utils.py` contains utils for the development of the performance tests,
which can be extended if it is required. The utils use extra Python modules `xlrd` and `openpyxl`
which should be installed for correct work.
2. `__init__.py` defines all the test suites.

##### How to run performance testing:
all:<br>
`python -m sdc.runtests sdc.tests.tests_perf`<br>
a single one:<br>
`python -m sdc.runtests sdc.tests.tests_perf.test_perf_series_str.TestSeriesStringMethods.test_series_str_len`

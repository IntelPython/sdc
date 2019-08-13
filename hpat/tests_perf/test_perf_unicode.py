# -*- coding: utf-8 -*-
import unittest

from numba import njit, types

import hpat

STRING_CASES = [
    'ascii',
    '1234567890',
    '¡Y tú quién te crees?',
    '🐍⚡',
    '大处着眼，小处着手。',
]


def center_usecase(x, y):
    return x.center(y)


def center_usecase_fillchar(x, y, fillchar):
    return x.center(y, fillchar)


def ljust_usecase(x, y):
    return x.ljust(y)


def ljust_usecase_fillchar(x, y, fillchar):
    return x.ljust(y, fillchar)


def rjust_usecase(x, y):
    return x.rjust(y)


def rjust_usecase_fillchar(x, y, fillchar):
    return x.rjust(y, fillchar)


class TestUnicodeStrings(unittest.TestCase):
    """
    Test unicode strings operations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._test_iteration = 10

    def test_justification(self):
        for pyfunc, case_name in [(center_usecase, 'center'),
                                  (ljust_usecase, 'ljust'),
                                  (rjust_usecase, 'rjust')]:

            # compilation time is also needs to be measured
            cfunc = njit(pyfunc)

            for iter in range(self._test_iteration):
                for s in STRING_CASES:
                    for width in range(-3, 20):
                        # these function calls needs to be benchmarked
                        pyfunc_result = pyfunc(s, width)
                        cfunc_result = cfunc(s, width)

                        self.assertEqual(pyfunc_result, cfunc_result,
                                         "'%s'.%s(%d)?" % (s, case_name, width))

    def test_justification_fillchar(self):
        for pyfunc, case_name in [(center_usecase_fillchar, 'center'),
                                  (ljust_usecase_fillchar, 'ljust'),
                                  (rjust_usecase_fillchar, 'rjust')]:

            # compilation time is also needs to be measured
            cfunc = njit(pyfunc)

            for iter in range(self._test_iteration):
                # allowed fillchar cases
                for fillchar in [' ', '+', 'ú', '处']:
                    for s in STRING_CASES:
                        for width in range(-3, 20):
                            # these function calls needs to be benchmarked
                            pyfunc_result = pyfunc(s, width, fillchar)
                            cfunc_result = cfunc(s, width, fillchar)

                            self.assertEqual(pyfunc_result, cfunc_result,
                                             "'%s'.%s(%d, '%s')?" % (s, case_name, width, fillchar))


if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
import numpy as np
from math import sqrt
import numba
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs,
                            count_parfor_OneD_Vars, count_array_OneD_Vars,
                            dist_IR_contains)
from datetime import datetime
import random


class TestDate(unittest.TestCase):
    def test_datetime_index(self):
        def test_impl(df):
            df['hpat'] = pd.DatetimeIndex(df['orig'])

        hpat_func = hpat.jit(test_impl)
        rows = 10
        data = []
        for row in range(rows):
            data.append(datetime(2017, random.randint(1,12), random.randint(1,28)).isoformat())
        df = pd.DataFrame({'orig' : data})
        hpat_func(df)
        df['std'] = pd.DatetimeIndex(df['orig'])
        allequal = (df['std'].equals(df['hpat']))
        self.assertTrue(allequal)

    def test_extract(self):
        def test_impl(s):
            return s.month

        hpat_func = hpat.jit(test_impl)
        ts = pd.Timestamp(datetime(2017, 4, 26).isoformat())
        month = hpat_func(ts)
        self.assertEqual(month, 4)

if __name__ == "__main__":
    unittest.main()

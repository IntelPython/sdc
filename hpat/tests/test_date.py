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
            return pd.DatetimeIndex(df['str_date']).values

        hpat_func = hpat.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_datetime_index_set(self):
        def test_impl(df):
            df['hpat'] = pd.DatetimeIndex(df['str_date']).values

        hpat_func = hpat.jit(test_impl)
        df = self._gen_str_date_df()
        hpat_func(df)
        df['std'] = pd.DatetimeIndex(df['str_date'])
        allequal = (df['std'].equals(df['hpat']))
        self.assertTrue(allequal)

    def test_extract(self):
        def test_impl(s):
            return s.month

        hpat_func = hpat.jit(test_impl)
        ts = pd.Timestamp(datetime(2017, 4, 26).isoformat())
        month = hpat_func(ts)
        self.assertEqual(month, 4)

    def test_datetimeindex_str_comp(self):
        def test_impl(df):
            return (df.A >= '2011-10-23').values

        df = pd.DataFrame({'A': pd.DatetimeIndex(['2015-01-03', '2010-10-11'])})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_datetimeindex_str_comp2(self):
        def test_impl(df):
            return ('2011-10-23' <= df.A).values

        df = pd.DataFrame({'A': pd.DatetimeIndex(['2015-01-03', '2010-10-11'])})
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def _gen_str_date_df(self):
        rows = 10
        data = []
        for row in range(rows):
            data.append(datetime(2017, random.randint(1,12), random.randint(1,28)).isoformat())
        return pd.DataFrame({'str_date' : data})

if __name__ == "__main__":
    unittest.main()

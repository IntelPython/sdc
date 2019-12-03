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


import unittest
import pandas as pd
import numpy as np
from math import sqrt
import numba
import sdc
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                   count_parfor_OneDs, count_array_OneDs,
                                   count_parfor_OneD_Vars, count_array_OneD_Vars,
                                   dist_IR_contains,
                                   skip_numba_jit)
from datetime import datetime
import random


class TestDate(TestCase):
    @unittest.skip("needs support for boxing/unboxing DatetimeIndex")
    def test_datetime_index_in(self):
        def test_impl(dti):
            return dti

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        dti = pd.DatetimeIndex(df['str_date'])
        np.testing.assert_array_equal(hpat_func(dti).values, test_impl(dti).values)

    @skip_numba_jit
    def test_datetime_index(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).values

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_kw(self):
        def test_impl(df):
            return pd.DatetimeIndex(data=df['str_date']).values

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    def test_datetime_arg(self):
        def test_impl(A):
            return A

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    @skip_numba_jit
    def test_datetime_getitem(self):
        def test_impl(A):
            return A[0]

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        self.assertEqual(hpat_func(A), test_impl(A))

    @skip_numba_jit
    def test_ts_map(self):
        def test_impl(A):
            return A.map(lambda x: x.hour)

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    @skip_numba_jit
    def test_ts_map_date(self):
        def test_impl(A):
            return A.map(lambda x: x.date())[0]

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        np.testing.assert_array_equal(hpat_func(A), test_impl(A))

    @skip_numba_jit
    def test_ts_map_date2(self):
        def test_impl(df):
            return df.apply(lambda row: row.dt_ind.date(), axis=1)[0]

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        df['dt_ind'] = pd.DatetimeIndex(df['str_date'])
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_ts_map_date_set(self):
        def test_impl(df):
            df['hpat_date'] = df.dt_ind.map(lambda x: x.date())

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        df['dt_ind'] = pd.DatetimeIndex(df['str_date'])
        hpat_func(df)
        df['pd_date'] = df.dt_ind.map(lambda x: x.date())
        np.testing.assert_array_equal(df['hpat_date'], df['pd_date'])

    @skip_numba_jit
    def test_date_series_unbox(self):
        def test_impl(A):
            return A[0]

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series().map(lambda x: x.date())
        self.assertEqual(hpat_func(A), test_impl(A))

    @skip_numba_jit
    def test_date_series_unbox2(self):
        def test_impl(A):
            return A[0]

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).map(lambda x: x.date())
        self.assertEqual(hpat_func(A), test_impl(A))

    @skip_numba_jit
    def test_datetime_index_set(self):
        def test_impl(df):
            df['sdc'] = pd.DatetimeIndex(df['str_date']).values

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        hpat_func(df)
        df['std'] = pd.DatetimeIndex(df['str_date'])
        allequal = (df['std'].equals(df['sdc']))
        self.assertTrue(allequal)

    def test_timestamp(self):
        def test_impl():
            dt = datetime(2017, 4, 26)
            ts = pd.Timestamp(dt)
            return ts.day + ts.hour + ts.microsecond + ts.month + ts.nanosecond + ts.second + ts.year

        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_extract(self):
        def test_impl(s):
            return s.month

        hpat_func = self.jit(test_impl)
        ts = pd.Timestamp(datetime(2017, 4, 26).isoformat())
        month = hpat_func(ts)
        self.assertEqual(month, 4)

    def test_timestamp_date(self):
        def test_impl(s):
            return s.date()

        hpat_func = self.jit(test_impl)
        ts = pd.Timestamp(datetime(2017, 4, 26).isoformat())
        self.assertEqual(hpat_func(ts), test_impl(ts))

    @skip_numba_jit
    def test_datetimeindex_str_comp(self):
        def test_impl(df):
            return (df.A >= '2011-10-23').values

        df = pd.DataFrame({'A': pd.DatetimeIndex(['2015-01-03', '2010-10-11'])})
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetimeindex_str_comp2(self):
        def test_impl(df):
            return ('2011-10-23' <= df.A).values

        df = pd.DataFrame({'A': pd.DatetimeIndex(['2015-01-03', '2010-10-11'])})
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_df(self):
        def test_impl(df):
            df = pd.DataFrame({'A': pd.DatetimeIndex(df['str_date'])})
            return df.A

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_date(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).date

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_max(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).max()

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        self.assertEqual(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_min(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).min()

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        self.assertEqual(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_timedelta_days(self):
        def test_impl(df):
            s = pd.DatetimeIndex(df['str_date'])
            t = s - s.min()
            return t.days

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_timedelta_seconds(self):
        def test_impl(df):
            s = pd.DatetimeIndex(df['str_date'])
            t = s - s.min()
            return t.seconds

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_timedelta_microseconds(self):
        def test_impl(df):
            s = pd.DatetimeIndex(df['str_date'])
            t = s - s.min()
            return t.microseconds

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_timedelta_nanoseconds(self):
        def test_impl(df):
            s = pd.DatetimeIndex(df['str_date'])
            t = s - s.min()
            return t.nanoseconds

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_ret(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date'])

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        pd.testing.assert_index_equal(hpat_func(df), test_impl(df),
                                      check_names=False)

    @skip_numba_jit
    def test_datetime_index_year(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).year

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_month(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).month

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_day(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).day

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_hour(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).hour

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_minute(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).minute

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_second(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).second

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_microsecond(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).microsecond

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_index_nanosecond(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).nanosecond

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(hpat_func(df), test_impl(df))

    @skip_numba_jit
    def test_datetime_series_dt_date(self):
        def test_impl(A):
            return A.dt.date

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        # TODO: fix index and name
        pd.testing.assert_series_equal(
            hpat_func(A), test_impl(A).reset_index(drop=True),
            check_names=False)

    @skip_numba_jit
    def test_datetime_series_dt_year(self):
        def test_impl(A):
            return A.dt.year

        hpat_func = self.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        # TODO: fix index and name
        pd.testing.assert_series_equal(
            hpat_func(A), test_impl(A).reset_index(drop=True),
            check_names=False)

    def _gen_str_date_df(self):
        rows = 10
        data = []
        for row in range(rows):
            data.append(datetime(2017, random.randint(1, 12), random.randint(1, 28)).isoformat())
        return pd.DataFrame({'str_date': data})


if __name__ == "__main__":
    unittest.main()

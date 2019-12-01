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


import os
import unittest
import platform
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import pyarrow.parquet as pq
import sdc
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                                   count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_rank,
                                   get_start_end,
                                   skip_numba_jit)
from numba.config import IS_32BITS

from sdc.io.csv_ext import pandas_read_csv as pd_read_csv


kde_file = 'kde.parquet'


class TestIO(TestCase):

    def setUp(self):
        if get_rank() == 0:
            # test_csv_cat1
            data = ("2,B,SA\n"
                    "3,A,SBC\n"
                    "4,C,S123\n"
                    "5,B,BCD\n")

            with open("csv_data_cat1.csv", "w") as f:
                f.write(data)

            # test_csv_single_dtype1
            data = ("2,4.1\n"
                    "3,3.4\n"
                    "4,1.3\n"
                    "5,1.1\n")

            with open("csv_data_dtype1.csv", "w") as f:
                f.write(data)

            # test_np_io1
            n = 111
            A = np.random.ranf(n)
            A.tofile("np_file1.dat")

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_read(self):
        def test_impl():
            t = pq.read_table('kde.parquet')
            df = t.to_pandas()
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_read_global_str1(self):
        def test_impl():
            df = pd.read_parquet(kde_file)
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_read_freevar_str1(self):
        kde_file2 = 'kde.parquet'

        def test_impl():
            df = pd.read_parquet(kde_file2)
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pd_read_parquet(self):
        def test_impl():
            df = pd.read_parquet('kde.parquet')
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_str(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.two.values == 'foo'
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_str_with_nan_seq(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values == 'foo'
            return A

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_str_with_nan_par(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values == 'foo'
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_str_with_nan_par_multigroup(self):
        def test_impl():
            df = pq.read_table('example2.parquet').to_pandas()
            A = df.five.values == 'foo'
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_bool(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.three.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.one.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip('Error - fix needed\n'
                   'NUMA_PES=3 build')
    def test_pq_float_no_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.four.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_pandas_date(self):
        def test_impl():
            df = pd.read_parquet('pandas_dt.pq')
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('Error: Attribute "dtype" are different\n'
                   '[left]:  datetime64[ns]\n'
                   '[right]: object')
    def test_pq_spark_date(self):
        def test_impl():
            df = pd.read_parquet('sdf_dt.pq')
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_pyarrow(self):
        tests = [
            "csv1",
            "csv_keys1",
            "csv_const_dtype1",
            "csv_infer1",
            "csv_infer_parallel1",
            "csv_skip1",
            "csv_infer_skip1",
            "csv_infer_skip_parallel1",
            "csv_rm_dead1",
            "csv_date1",
            "csv_str1",
            "csv_parallel1",
            "csv_str_parallel1",
            "csv_usecols1",
            "csv_cat1",
            "csv_cat2",
            "csv_single_dtype1",
        ]
        for test in tests:
            with self.subTest(test=test):
                pa_val = getattr(self, f"pa_{test}")()()
                pd_val = getattr(self, f"pd_{test}")()()
                if isinstance(pd_val, pd.Series):
                    pd.testing.assert_series_equal(pa_val, pd_val,
                        check_categorical=False
                    )
                elif isinstance(pd_val, pd.DataFrame):
                    pd.testing.assert_frame_equal(pa_val, pd_val,
                        check_categorical=False
                    )
                elif isinstance(pd_val, np.ndarray):
                    np.testing.assert_array_equal(pa_val, pd_val)
                elif isinstance(pd_val, tuple):
                    self.assertEqual(pa_val, pd_val)
                else:
                    self.fail(f"Unknown Pandas type: {type(pd_val)}")

    def pd_csv1(self):
        # TODO: w/a for Numba issue with int typing rules infering intp for integers literals
        # unlike NumPy which uses int32 by default - causes dtype mismatch on Windows 64 bit
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                return pd.read_csv("csv_data1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.intp, 'B': np.float, 'C': np.float, 'D': np.intp},
                                   )
        else:
            def test_impl():
                return pd.read_csv("csv_data1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int},
                                   )
        return test_impl

    def pa_csv1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                return pd_read_csv("csv_data1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.intp, 'B': np.float, 'C': np.float, 'D': np.intp},
                                   )
        else:
            def test_impl():
                return pd_read_csv("csv_data1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int},
                                   )
        return test_impl

    @skip_numba_jit
    def test_csv1(self):
        test_impl = self.pd_csv1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_keys1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                dtype = {'A': np.intp, 'B': np.float, 'C': np.float, 'D': np.intp}
                return pd.read_csv("csv_data1.csv",
                                   names=dtype.keys(),
                                   dtype=dtype,
                                   )
        else:
            def test_impl():
                dtype = {'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int}
                return pd.read_csv("csv_data1.csv",
                                   names=dtype.keys(),
                                   dtype=dtype,
                                   )
        return test_impl

    def pa_csv_keys1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                dtype = {'A': np.intp, 'B': np.float, 'C': np.float, 'D': np.intp}
                return pd_read_csv("csv_data1.csv",
                                   names=dtype.keys(),
                                   dtype=dtype,
                                   )
        else:
            def test_impl():
                dtype = {'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int}
                return pd_read_csv("csv_data1.csv",
                                   names=dtype.keys(),
                                   dtype=dtype,
                                   )
        return test_impl

    @skip_numba_jit
    def test_csv_keys1(self):
        test_impl = self.pd_csv_keys1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_const_dtype1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                dtype = {'A': 'int64', 'B': 'float64', 'C': 'float', 'D': 'int64'}
                return pd.read_csv("csv_data1.csv",
                                   names=dtype.keys(),
                                   dtype=dtype,
                                   )
        else:
            def test_impl():
                dtype = {'A': 'int', 'B': 'float64', 'C': 'float', 'D': 'int64'}
                return pd.read_csv("csv_data1.csv",
                                   names=dtype.keys(),
                                   dtype=dtype,
                                   )
        return test_impl

    def pa_csv_const_dtype1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                dtype = {'A': 'int64', 'B': 'float64', 'C': 'float', 'D': 'int64'}
                return pd_read_csv("csv_data1.csv",
                                   names=dtype.keys(),
                                   dtype=dtype,
                                   )
        else:
            def test_impl():
                dtype = {'A': 'int', 'B': 'float64', 'C': 'float', 'D': 'int64'}
                return pd_read_csv("csv_data1.csv",
                                   names=dtype.keys(),
                                   dtype=dtype,
                                   )
        return test_impl

    @skip_numba_jit
    def test_csv_const_dtype1(self):
        test_impl = self.pd_csv_const_dtype1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_infer1(self):
        def test_impl():
            return pd.read_csv("csv_data_infer1.csv")
        return test_impl

    def pa_csv_infer1(self):
        def test_impl():
            return pd_read_csv("csv_data_infer1.csv")
        return test_impl

    @skip_numba_jit
    def test_csv_infer1(self):
        test_impl = self.pd_csv_infer1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_infer_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_infer1.csv")
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()
        return test_impl

    def pa_csv_infer_parallel1(self):
        def test_impl():
            df = pd_read_csv("csv_data_infer1.csv")
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()
        return test_impl

    @skip_numba_jit
    def test_csv_infer_parallel1(self):
        test_impl = self.pd_csv_infer_parallel1()
        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def pd_csv_skip1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                return pd.read_csv("csv_data1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int64, 'B': np.float, 'C': np.float, 'D': np.int64},
                                   skiprows=2,
                                   )
        else:
            def test_impl():
                return pd.read_csv("csv_data1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int},
                                   skiprows=2,
                                   )
        return test_impl

    def pa_csv_skip1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                return pd_read_csv("csv_data1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int64, 'B': np.float, 'C': np.float, 'D': np.int64},
                                   skiprows=2,
                                   )
        else:
            def test_impl():
                return pd_read_csv("csv_data1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int},
                                   skiprows=2,
                                   )
        return test_impl

    @skip_numba_jit
    def test_csv_skip1(self):
        test_impl = self.pd_csv_skip1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_infer_skip1(self):
        def test_impl():
            return pd.read_csv("csv_data_infer1.csv", skiprows=2)
        return test_impl

    def pa_csv_infer_skip1(self):
        def test_impl():
            return pd_read_csv("csv_data_infer1.csv", skiprows=2)
        return test_impl

    @skip_numba_jit
    def test_csv_infer_skip1(self):
        test_impl = self.pd_csv_infer_skip1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_infer_skip_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_infer1.csv", skiprows=2,
                             names=['A', 'B', 'C', 'D'])
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()
        return test_impl

    def pa_csv_infer_skip_parallel1(self):
        def test_impl():
            df = pd_read_csv("csv_data_infer1.csv", skiprows=2,
                             names=['A', 'B', 'C', 'D'])
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()
        return test_impl

    @skip_numba_jit
    def test_csv_infer_skip_parallel1(self):
        test_impl = self.pd_csv_infer_skip_parallel1()
        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def pd_csv_rm_dead1(self):
        def test_impl():
            df = pd.read_csv("csv_data1.csv",
                             names=['A', 'B', 'C', 'D'],
                             dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int},)
            return df.B.values
        return test_impl

    def pa_csv_rm_dead1(self):
        def test_impl():
            df = pd_read_csv("csv_data1.csv",
                             names=['A', 'B', 'C', 'D'],
                             dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int},)
            return df.B.values
        return test_impl

    @skip_numba_jit
    def test_csv_rm_dead1(self):
        test_impl = self.pd_csv_rm_dead1()
        hpat_func = self.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def pd_csv_date1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                return pd.read_csv("csv_data_date1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int64, 'B': np.float, 'C': str, 'D': np.int64},
                                   parse_dates=[2])
        else:
            def test_impl():
                return pd.read_csv("csv_data_date1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int},
                                   parse_dates=[2])
        return test_impl

    def pa_csv_date1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                return pd_read_csv("csv_data_date1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int64, 'B': np.float, 'C': str, 'D': np.int64},
                                   parse_dates=[2])
        else:
            def test_impl():
                return pd_read_csv("csv_data_date1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int},
                                   parse_dates=[2])
        return test_impl

    @skip_numba_jit
    def test_csv_date1(self):
        test_impl = self.pd_csv_date1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_str1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                return pd.read_csv("csv_data_date1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int64, 'B': np.float, 'C': str, 'D': np.int64})
        else:
            def test_impl():
                return pd.read_csv("csv_data_date1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int})
        return test_impl

    def pa_csv_str1(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                return pd_read_csv("csv_data_date1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int64, 'B': np.float, 'C': str, 'D': np.int64})
        else:
            def test_impl():
                return pd_read_csv("csv_data_date1.csv",
                                   names=['A', 'B', 'C', 'D'],
                                   dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int})
        return test_impl

    @skip_numba_jit
    def test_csv_str1(self):
        test_impl = self.pd_csv_str1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data1.csv",
                             names=['A', 'B', 'C', 'D'],
                             dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int})
            return (df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum())
        return test_impl

    def pa_csv_parallel1(self):
        def test_impl():
            df = pd_read_csv("csv_data1.csv",
                             names=['A', 'B', 'C', 'D'],
                             dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': np.int})
            return (df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum())
        return test_impl

    @skip_numba_jit
    def test_csv_parallel1(self):
        test_impl = self.pd_csv_parallel1()
        hpat_func = self.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def pd_csv_str_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_date1.csv",
                             names=['A', 'B', 'C', 'D'],
                             dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int})
            return (df.A.sum(), df.B.sum(), (df.C == '1966-11-13').sum(),
                    df.D.sum())
        return test_impl

    def pa_csv_str_parallel1(self):
        def test_impl():
            df = pd_read_csv("csv_data_date1.csv",
                             names=['A', 'B', 'C', 'D'],
                             dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int})
            return (df.A.sum(), df.B.sum(), (df.C == '1966-11-13').sum(),
                    df.D.sum())
        return test_impl

    @skip_numba_jit
    def test_csv_str_parallel1(self):
        test_impl = self.pd_csv_str_parallel1()
        hpat_func = self.jit(locals={'df:return': 'distributed'})(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def pd_csv_usecols1(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                               names=['C'],
                               dtype={'C': np.float},
                               usecols=[2],
                               )
        return test_impl

    def pa_csv_usecols1(self):
        def test_impl():
            return pd_read_csv("csv_data1.csv",
                               names=['C'],
                               dtype={'C': np.float},
                               usecols=[2],
                               )
        return test_impl

    @skip_numba_jit
    def test_csv_usecols1(self):
        test_impl = self.pd_csv_usecols1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_cat1(self):
        def test_impl():
            # names = ['C1', 'C2', 'C3']
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1': np.int, 'C2': ct_dtype, 'C3': str}
            df = pd.read_csv("csv_data_cat1.csv",
                # names=names,  # Error: names should be constant list
                names=['C1', 'C2', 'C3'],
                dtype=dtypes
            )
            return df.C2
        return test_impl

    def pa_csv_cat1(self):
        def test_impl():
            # names = ['C1', 'C2', 'C3']
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1': np.int, 'C2': ct_dtype, 'C3': str}
            df = pd_read_csv("csv_data_cat1.csv",
                # names=names,  # Error: names should be constant list
                names=['C1', 'C2', 'C3'],
                dtype=dtypes
            )
            return df.C2
        return test_impl

    @skip_numba_jit
    def test_csv_cat1(self):
        test_impl = self.pd_csv_cat1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(), test_impl(), check_names=False)

    def pd_csv_cat2(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
                df = pd.read_csv("csv_data_cat1.csv",
                                 names=['C1', 'C2', 'C3'],
                                 dtype={'C1': np.int64, 'C2': ct_dtype, 'C3': str},
                                 )
                return df
        else:
            def test_impl():
                ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
                df = pd.read_csv("csv_data_cat1.csv",
                                 names=['C1', 'C2', 'C3'],
                                 dtype={'C1': np.int, 'C2': ct_dtype, 'C3': str},
                                 )
                return df
        return test_impl

    def pa_csv_cat2(self):
        if platform.system() == 'Windows' and not IS_32BITS:
            def test_impl():
                ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
                df = pd_read_csv("csv_data_cat1.csv",
                                 names=['C1', 'C2', 'C3'],
                                 dtype={'C1': np.int64, 'C2': ct_dtype, 'C3': str},
                                 )
                return df
        else:
            def test_impl():
                ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
                df = pd_read_csv("csv_data_cat1.csv",
                                 names=['C1', 'C2', 'C3'],
                                 dtype={'C1': np.int, 'C2': ct_dtype, 'C3': str},
                                 )
                return df
        return test_impl

    @skip_numba_jit
    def test_csv_cat2(self):
        test_impl = self.pd_csv_cat2()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def pd_csv_single_dtype1(self):
        def test_impl():
            df = pd.read_csv("csv_data_dtype1.csv",
                             names=['C1', 'C2'],
                             dtype=np.float64,
                             )
            return df
        return test_impl

    def pa_csv_single_dtype1(self):
        def test_impl():
            df = pd_read_csv("csv_data_dtype1.csv",
                             names=['C1', 'C2'],
                             dtype=np.float64,
                             )
            return df
        return test_impl

    @skip_numba_jit
    def test_csv_single_dtype1(self):
        test_impl = self.pd_csv_single_dtype1()
        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    @unittest.skip('TypeError: to_csv() takes from 1 to 20 positional arguments but 21 were given)\n'
                   'Notice: Not seen with Pandas 0.24.2')
    def test_write_csv1(self):
        def test_impl(df, fname):
            df.to_csv(fname)

        hpat_func = self.jit(test_impl)
        n = 111
        df = pd.DataFrame({'A': np.arange(n)})
        hp_fname = 'test_write_csv1_sdc.csv'
        pd_fname = 'test_write_csv1_pd.csv'
        hpat_func(df, hp_fname)
        test_impl(df, pd_fname)
        # TODO: delete files
        pd.testing.assert_frame_equal(pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    @unittest.skip('AttributeError: Failed in hpat mode pipeline (step: convert to distributed)\n'
                   'module \'sdc.hio\' has no attribute \'file_write_parallel\'')
    def test_write_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({'A': np.arange(n)})
            df.to_csv(fname)

        hpat_func = self.jit(test_impl)
        n = 111
        hp_fname = 'test_write_csv1_hpat_par.csv'
        pd_fname = 'test_write_csv1_pd_par.csv'
        hpat_func(n, hp_fname)
        test_impl(n, pd_fname)
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        # TODO: delete files
        if get_rank() == 0:
            pd.testing.assert_frame_equal(
                pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    @skip_numba_jit
    def test_np_io1(self):
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    @skip_numba_jit
    def test_np_io2(self):
        # parallel version
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_np_io3(self):
        def test_impl(A):
            if get_rank() == 0:
                A.tofile("np_file_3.dat")

        hpat_func = self.jit(test_impl)
        n = 111
        A = np.random.ranf(n)
        hpat_func(A)
        if get_rank() == 0:
            B = np.fromfile("np_file_3.dat", np.float64)
            np.testing.assert_almost_equal(A, B)

    def test_np_io4(self):
        # parallel version
        def test_impl(n):
            A = np.arange(n)
            A.tofile("np_file_3.dat")

        hpat_func = self.jit(test_impl)
        n = 111
        A = np.arange(n)
        hpat_func(n)
        B = np.fromfile("np_file_3.dat", np.int64)
        np.testing.assert_almost_equal(A, B)


if __name__ == "__main__":
    unittest.main()

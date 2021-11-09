# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
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

import numpy as np
import os
import pandas as pd
import platform
import pyarrow.parquet as pq
import unittest
import numba
from numba.core.config import IS_32BITS
from numba.core.errors import TypingError
from pandas import CategoricalDtype

import sdc
from sdc.io.csv_ext import pandas_read_csv as pd_read_csv
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import (count_array_OneDs,
                                  count_array_REPs,
                                  count_parfor_OneDs,
                                  count_parfor_REPs,
                                  dist_IR_contains,
                                  get_rank,
                                  get_start_end,
                                  skip_numba_jit)


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
        super(TestIO, self).setUp()


class TestCSVNewOnlyInferParams(TestIO):

    @unittest.skip
    def test_pyarrow(self):
        tests = [
            "csv_keys1",
            "csv_const_dtype1",
            "csv_infer_file_default",
            "csv_infer_file_sep",
            "csv_infer_file_delimiter",
            "csv_infer_file_names",
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
                test = getattr(self, f"pd_{test}")
                pd_val = test(use_pyarrow=False)()
                pa_val = test(use_pyarrow=True)()
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

    def _int_type(self):
        # TODO: w/a for Numba issue with int typing rules infering intp for integers literals
        # unlike NumPy which uses int32 by default - causes dtype mismatch on Windows 64 bit
        if platform.system() == 'Windows' and not IS_32BITS:
            return np.intp
        else:
            return np.int

    def _int_type_str(self):
        return np.dtype(self._int_type()).name

    def _read_csv(self, use_pyarrow=False):
        return pd_read_csv if use_pyarrow else pd.read_csv

    # inference errors

    ### FIXME: this test should actually fail with literally block that will literal fname!
    @unittest.skip
    def test_csv_infer_error(self):

        def test_impl(fname):
            return pd.read_csv(fname)
        sdc_func = self.jit(test_impl)

        with self.assertRaises(TypingError) as raises:
            sdc_func("csv_data1.csv")

        self.assertIn("Cannot infer resulting DataFrame", raises.exception.msg)

    # inference from parameters

    def test_csv_infer_params_default(self):
        """Test verifies DF type inference from parameters when names and dtype params are used"""

        int_type = self._int_type()

        def test_impl(fname):
            return pd.read_csv(fname,
                            names=['A', 'B', 'C', 'D'],
                            dtype={'A': int_type, 'B': np.float, 'C': 'float', 'D': str})       ### FIXME: int_type??
        sdc_func = self.jit(test_impl)

        for fname in ["csv_data1.csv", "csv_data2.csv"]:
            with self.subTest(fname=fname):
                result = sdc_func(fname)
                result_ref = test_impl(fname)
                pd.testing.assert_frame_equal(result, result_ref)

#     def test_csv_infer_params_usecols_names(self):
#         """Test verifies DF type inference from parameters when both names and usecols are used"""
#         int_type = self._int_type()
# 
#         def test_impl(fname):
#             return pd.read_csv(fname,
#                                names=['A', 'B', 'C', 'D'],
#                                dtype={'A': int_type, 'B': np.float, 'C': np.float, 'D': str},
#                                usecols=['B', 'D'])
#         sdc_func = self.jit(test_impl)
# 
#         fname = "csv_data1.csv"
#         pd.testing.assert_frame_equal(sdc_func(fname), test_impl(fname))

#     def test_csv_infer_params_usecols_no_names(self):
#         """Test verifies DF type inference from parameters when only usecols is used and names are default"""
# 
#         def test_impl(fname):
#             return pd.read_csv(fname, dtype={'B': np.float, 'D': str}, usecols=['B', 'D'])
# 
#         fname = "csv_data_infer1.csv"
#         sdc_func = self.jit(test_impl)
#         pd.testing.assert_frame_equal(sdc_func(fname), test_impl(fname))



class TestCSVNewInferFileOnly(TestIO):

    def _int_type(self):
        # TODO: w/a for Numba issue with int typing rules infering intp for integers literals
        # unlike NumPy which uses int32 by default - causes dtype mismatch on Windows 64 bit
        if platform.system() == 'Windows' and not IS_32BITS:
            return np.intp
        else:
            return np.int

    def _int_type_str(self):
        return np.dtype(self._int_type()).name

    def get_int_converter(self):

        @self.jit
        def to_int_and_inc(x):
            return int(x) + 1
        return to_int_and_inc

    def get_float_converter(self):

        @self.jit
        def to_float_and_dec(x):
            return float(x) - 1
        return to_float_and_dec

    def get_str_converter(self):

        @self.jit
        def str_len(x):
            return len(x)
        return str_len


    def test_csv_infer_file_default(self):
        """Test verifies basic usage of pandas read_csv when DF type is inferred from const filename"""

        def test_impl(file_name):
            return pd.read_csv("csv_data_infer1.csv")   ### FIXME: rollback
        sdc_func = self.jit(test_impl)

        for file_name in ["csv_data_infer1.csv", "csv_data_infer_no_column_name.csv"]:
            with self.subTest(file_name=file_name):
                pd.testing.assert_frame_equal(sdc_func(file_name), test_impl(file_name))

    def test_csv_infer_file_param_sep(self):
        """Test verifies pandas read_csv impl supports parameter sep (DF type inference from const file name)"""

        def test_impl():
            return pd.read_csv("csv_data_infer_sep.csv", sep=';')
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_delimiter(self):
        """Test verifies pandas read_csv impl supports parameter delimiter
           (DF type inference from const file name)"""

        def test_impl():
            return pd.read_csv("csv_data_infer_sep.csv", delimiter=';')
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_names(self):
        """Test verifies pandas read_csv impl supports parameter names when DF type inferred from const filename"""

        def test_impl(file_name):
            return pd.read_csv(file_name, names=['A', 'B', 'C', 'D'])
        sdc_func = self.jit(test_impl)

        for file_name in ["csv_data1.csv", "csv_data_infer1.csv"]:
            with self.subTest(file_name=file_name):
                pd.testing.assert_frame_equal(sdc_func(file_name), test_impl(file_name))

    def test_csv_infer_file_param_usecols_no_names(self):
        """Test verifies pandas read_csv impl supports parameter usecols when no column names are
           specified and they are inferred during compilation from const filename"""

        def test_impl(file_name):
            return pd.read_csv(file_name, usecols=['B', 'D'])
        sdc_func = self.jit(test_impl)

        for file_name in ["csv_data_infer1.csv"]:
            with self.subTest(file_name=file_name):
                pd.testing.assert_frame_equal(sdc_func(file_name), test_impl(file_name))

    def test_csv_infer_file_param_usecols_and_dtype_no_names(self):
        """Test verifies pandas read_csv impl supports parameter usecols when no column names are
           specified and dtype conversion is done (DF type is inferred from const filename)"""

        def test_impl(fname):
            return pd.read_csv(fname, dtype={'A': np.float, 'D': str}, usecols=['A', 'D'])

        fname = "csv_data_infer1.csv"
        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(fname), test_impl(fname))

    def test_csv_infer_file_param_usecols_with_names(self):
        """Test verifies pandas read_csv impl supports combination of parameters usecols and names
           when DF type is inferred from const filename"""

        def test_impl(file_name):
            return pd.read_csv(file_name, names=['A', 'B', 'C', 'D'], usecols=['B', 'D'])
        sdc_func = self.jit(test_impl)

        for file_name in ["csv_data1.csv", "csv_data_infer1.csv"]:
            with self.subTest(file_name=file_name):
                pd.testing.assert_frame_equal(sdc_func(file_name), test_impl(file_name))

    def test_csv_infer_file_param_usecols_dtype_names(self):
        """Test verifies pandas read_csv impl supports combination of parameters names, dtype and usecols
           when DF type is inferred from const filename"""

        int_type = self._int_type()

        def test_impl(fname):
            return pd.read_csv(fname,
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': int_type, 'B': np.float, 'C': np.float, 'D': str},
                               usecols=['B', 'D'])
        sdc_func = self.jit(test_impl)

        fname = "csv_data1.csv"
        pd.testing.assert_frame_equal(sdc_func(fname), test_impl(fname))

    def test_csv_infer_file_param_usecols_as_numbers(self):
        """Test verifies pandas read_csv impl supports parameter usecols with numbers of columns to
           include in the resulting DF and inference of DF type from const filename"""

        def test_impl():
            return pd.read_csv("csv_data1.csv",
                               names=['C'],
                               dtype={'C': np.float},
                               usecols=[2],)
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_str1(self):
        """Test verifies pandas read_csv impl supports dtype conversion of data column"""

        int_type = self._int_type()

        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': int_type, 'B': np.float, 'C': str, 'D': np.int64})

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_converters_1(self):
        """Test verifies pandas read_csv impl supports conversion of all columns using converters parameter"""

        int_converter = self.get_int_converter()
        float_converter = self.get_float_converter()
        str_converter = self.get_str_converter()

        def test_impl():
            return pd.read_csv("csv_data1.csv",
                               names=['A', 'B', 'C', 'D'],
                               converters={'A': int_converter,
                                           'B': float_converter,
                                           'C': float_converter,
                                           'D': str_converter
                                           })

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_converters_2(self):
        """Test verifies pandas read_csv impl supports conversion of some of columns with converters parameter"""

        float_converter = self.get_float_converter()
        str_converter = self.get_str_converter()

        def test_impl():
            return pd.read_csv("csv_data1.csv",
                            names=['A', 'B', 'C', 'D'],
                            converters={'B': float_converter,
                                        'D': str_converter})

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_converters_3(self):
        """Test verifies pandas read_csv impl supports conversion of some of columns with other columns
           converterted via dtype parameter"""

        float_converter = self.get_float_converter()
        str_converter = self.get_str_converter()
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                            names=['A', 'B', 'C', 'D'],
                            dtype={'A': np.float64, 'C': np.float32},
                            converters={'B': float_converter,
                                        'D': str_converter})

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_skiprows_1(self):
        int_type = self._int_type()

        def test_impl():
            return pd.read_csv("csv_data1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': int_type, 'B': np.float, 'C': np.float, 'D': str},
                               skiprows=2,
                               )
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_skiprows_2(self):

        def test_impl():
            return pd.read_csv("csv_data_infer1.csv", skiprows=2)
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_parse_dates(self):
        int_type = self._int_type()

        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': int_type, 'B': np.float, 'C': str, 'D': np.int64},
                               parse_dates=[2, ])
        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_dtype_common(self):
        def test_impl():
            df = pd.read_csv("csv_data_dtype1.csv",
                             names=['C1', 'C2'],
                             dtype=np.float64,
                             )
            return df
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    ### FIXME: check what it tested before!
    def test_csv_const_dtype1(self):
        int_type = self._int_type_str()

        def test_impl():
            col_names = ['A', 'B', 'C', 'D']
            dtype = {'A': int_type, 'B': 'float64', 'C': 'float', 'D': 'str'}
            return pd.read_csv("csv_data1.csv",
                               names=col_names,
                               dtype=dtype,
                               )
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_cat1(self):

        def test_impl():
            names = ['C1', 'C2', 'C3']
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1': np.int, 'C2': ct_dtype, 'C3': str}
            df = pd.read_csv("csv_data_cat1.csv", names=names, dtype=dtypes)
            return df
        sdc_func = self.jit(test_impl)

        result = sdc_func()
        result_ref = test_impl()
        pd.testing.assert_frame_equal(result, result_ref, check_names=False)

    def test_csv_cat2(self):
        int_type = self._int_type()

        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
            df = pd.read_csv("csv_data_cat1.csv",
                          names=['C1', 'C2', 'C3'],
                          dtype={'C1': int_type, 'C2': ct_dtype, 'C3': str},
                          )
            return df
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    # not supported

    @skip_numba_jit
    def test_csv_not_supported_dict_use_1(self):

        int_type = self._int_type()

        def test_impl():
            dtype = {'A': int_type, 'B': np.float, 'C': np.float, 'D': str}
            return pd.read_csv("csv_data1.csv",
                            names=dtype.keys(),
                            dtype=dtype,)
        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    @skip_numba_jit
    def test_write_csv1(self):
        def test_impl(df, fname):
            df.to_csv(fname)

        sdc_func = self.jit(test_impl)
        n = 111
        df = pd.DataFrame({'A': np.arange(n)})
        hp_fname = 'test_write_csv1_sdc.csv'
        pd_fname = 'test_write_csv1_pd.csv'
        sdc_func(df, hp_fname)
        test_impl(df, pd_fname)
        # TODO: delete files
        pd.testing.assert_frame_equal(pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    # not supported parallelization checking tests

    @skip_numba_jit("Compiles and executes, but fusing is not checked")
    def test_csv_infer_file_parallel(self):
        def test_impl():
            df = pd.read_csv("csv_data_infer1.csv")
            return df.A.sum(), df.B.sum(), df.C.sum()
        sdc_func = self.jit(test_impl)

        self.assertEqual(sdc_func(), test_impl())

    @skip_numba_jit("Compiles and executes, but fusing is not checked")
    def test_csv_infer_file_parallel_skiprows(self):

        def test_impl():
            df = pd.read_csv("csv_data_infer1.csv",
                             skiprows=2,
                             names=['A', 'B', 'C', 'D'])
            return df.A.sum(), df.B.sum(), df.C.sum()
        sdc_func = self.jit(test_impl)

        self.assertEqual(sdc_func(), test_impl())

    @skip_numba_jit
    def test_csv_rm_dead1(self):

        def test_impl():
            df = pd.read_csv("csv_data1.csv",
                          names=['A', 'B', 'C', 'D'],
                          dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': str},)
            return df.B.values
        sdc_func = self.jit(test_impl)

        np.testing.assert_array_equal(sdc_func(), test_impl())

    @skip_numba_jit
    def test_csv_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data1.csv",
                          names=['A', 'B', 'C', 'D'],
                          dtype={'A': np.int, 'B': np.float, 'C': np.float, 'D': str})
            return (df.A.sum(), df.B.sum(), df.C.sum())
        sdc_func = self.jit(test_impl)
        self.assertEqual(sdc_func(), test_impl())

    @skip_numba_jit
    def test_csv_str_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_date1.csv",
                          names=['A', 'B', 'C', 'D'],
                          dtype={'A': np.int, 'B': np.float, 'C': str, 'D': np.int})
            return (df.A.sum(), df.B.sum(), (df.C == '1966-11-13').sum(), df.D.sum())
        sdc_func = self.jit(locals={'df:return': 'distributed'})(test_impl)

        self.assertEqual(sdc_func(), test_impl())

    @skip_numba_jit
    def test_write_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({'A': np.arange(n)})
            df.to_csv(fname)

        sdc_func = self.jit(test_impl)
        n = 111
        hp_fname = 'test_write_csv1_hpat_par.csv'
        pd_fname = 'test_write_csv1_pd_par.csv'
        sdc_func(n, hp_fname)
        test_impl(n, pd_fname)
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        # TODO: delete files
        if get_rank() == 0:
            pd.testing.assert_frame_equal(
                pd.read_csv(hp_fname), pd.read_csv(pd_fname))


if __name__ == "__main__":
    unittest.main()

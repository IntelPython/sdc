# *****************************************************************************
# Copyright (c) 2019-2020, Intel Corporation All rights reserved.
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
from pandas import CategoricalDtype

import sdc
from sdc.tests.test_base import TestCase
from sdc.tests.test_utils import skip_numba_jit


kde_file = 'kde.parquet'


class TestIO(TestCase):

    def setUp(self):
        super(TestIO, self).setUp()


class TestParquet(TestIO):

    @skip_numba_jit
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

    @skip_numba_jit
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

    @skip_numba_jit
    def test_pd_read_parquet(self):
        def test_impl():
            df = pd.read_parquet('kde.parquet')
            X = df['points']
            return X.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_str(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.two.values == 'foo'
            return A.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_str_with_nan_seq(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values == 'foo'
            return A

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    @skip_numba_jit
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

    @skip_numba_jit
    def test_pq_bool(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.three.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
    def test_pq_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.one.sum()

        hpat_func = self.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @skip_numba_jit
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

    @skip_numba_jit
    def test_pq_spark_date(self):
        def test_impl():
            df = pd.read_parquet('sdf_dt.pq')
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())


class TestCSV(TestIO):

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
            return pd.read_csv(file_name)
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

    @skip_numba_jit
    def test_csv_infer_file_param_usecols_and_dtype_unsupported(self):
        """Test verifies pandas read_csv impl supports combination of parameters usecols and names
           when DF type is inferred from const filename"""

        def test_impl():
            return pd.read_csv("csv_data_infer1.csv",
                               usecols=[2, 3],
                               dtype={'A': 'int',
                                      'B': 'float',
                                      'C': 'float',
                                      'D': 'str'})
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

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

    def test_csv_infer_file_param_usecols_reorder(self):
        """Test verifies pandas read_csv impl handles case when usecols parameter
           specifies names in different order than columns in a CSV """

        def test_impl(file_name):
            return pd.read_csv(file_name, usecols=['D', 'B'])
        sdc_func = self.jit(test_impl)

        for file_name in ["csv_data_infer1.csv"]:
            with self.subTest(file_name=file_name):
                pd.testing.assert_frame_equal(sdc_func(file_name), test_impl(file_name))

    @unittest.expectedFailure
    def test_csv_infer_file_read_unsupported_as_str(self):
        """Test verifies that CSV columns that would be of pyobject dtype in pandas,
           would be read as strings in SDC """

        def test_impl():
            return pd.read_csv("csv_data_pyobject.csv", names=['A', 'B', 'C'])
        sdc_func = self.jit(test_impl)

        result = sdc_func()
        result_ref = test_impl()
        self.assertEqual(str(result), str(result_ref))          # passed
        self.assertIsInstance(result['B'][1], str)              # passed
        self.assertIsInstance(result_ref['B'][1], bool)         # passed

        # fails since in pandas df.B is mixed NaN-s with Booleans (such array cannot be unboxed)
        # so current SDC implementation reads them as strings, but fails the next check
        pd.testing.assert_frame_equal(result, result_ref)

    def test_csv_infer_file_param_names_and_dtype(self):
        """Test verifies pandas read_csv impl supports combination of parameters names and dtype """

        int_type = self._int_type()

        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'B': np.float, 'A': int_type, 'D': np.int64, 'C': str})

        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_names_no_rewrite(self):
        """Test verifies pandas read_csv impl supports names parameters as tuple of columns
           both captured as global or local variable """

        def test_impl_1():
            const_names = ('A', 'B', 'C', 'D')
            return pd.read_csv("csv_data_date1.csv",
                               names=const_names)

        global_csv_names = ('A', 'B', 'C', 'D')

        def test_impl_2():
            return pd.read_csv("csv_data_date1.csv",
                               names=global_csv_names)

        for test_impl in [test_impl_1, test_impl_2]:
            with self.subTest(tested_func_name=test_impl.__name__):
                sdc_func = self.jit(test_impl)
                pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_converters_1(self):
        """Test verifies pandas read_csv impl supports conversion of all columns using converters parameter"""

        def test_impl(int_fn, float_fn, str_fn):
            return pd.read_csv("csv_data1.csv",
                               names=['A', 'B', 'C', 'D'],
                               converters={'A': int_fn,
                                           'B': float_fn,
                                           'C': float_fn,
                                           'D': str_fn})
        sdc_func = self.jit(test_impl)

        fn_list = [
            self.get_int_converter(),
            self.get_float_converter(),
            self.get_str_converter(),
        ]
        py_fn_list = [x.py_func for x in fn_list]

        pd.testing.assert_frame_equal(
            sdc_func(*fn_list),
            test_impl(*py_fn_list)
        )

    def test_csv_infer_file_param_converters_2(self):
        """Test verifies pandas read_csv impl supports conversion of some of columns with converters parameter"""

        def test_impl(float_fn, str_fn):
            return pd.read_csv("csv_data1.csv",
                               names=['A', 'B', 'C', 'D'],
                               converters={'B': float_fn,
                                           'D': str_fn})
        sdc_func = self.jit(test_impl)

        float_fn = self.get_float_converter()
        str_fn = self.get_str_converter()
        pd.testing.assert_frame_equal(
            sdc_func(float_fn, str_fn),
            test_impl(float_fn.py_func, str_fn.py_func)
        )

    def test_csv_infer_file_param_converters_3(self):
        """Test verifies pandas read_csv impl supports conversion of some of columns with other columns
           converterted via dtype parameter"""

        def test_impl(float_fn, str_fn):
            return pd.read_csv("csv_data1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': np.float64, 'C': np.float32},
                               converters={'B': float_fn,
                                           'D': str_fn})
        sdc_func = self.jit(test_impl)

        float_fn = self.get_float_converter()
        str_fn = self.get_str_converter()
        pd.testing.assert_frame_equal(
            sdc_func(float_fn, str_fn),
            test_impl(float_fn.py_func, str_fn.py_func)
        )

    def test_csv_infer_file_param_converters_4(self):
        """Test verifies pandas read_csv impl supports conversion of columns
           using converters parameter together with usecols parameter """

        # float_fn not used (needed for dict to be LiteralStrKeyDict
        def test_impl(int_fn, float_fn):
            return pd.read_csv("csv_data_infer1.csv",
                               converters={'A': int_fn, 'B': float_fn},
                               usecols=['A', 'C'])
        sdc_func = self.jit(test_impl)

        int_fn = self.get_int_converter()
        float_fn = self.get_float_converter()
        pd.testing.assert_frame_equal(sdc_func(int_fn, float_fn),
                                      test_impl(int_fn.py_func, float_fn.py_func))

    @unittest.expectedFailure
    def test_csv_infer_file_param_converters_unsupported(self):
        """Test checks read_csv usecase that is not supported due to Numba limitation
           on creating non-literal dict for a const dict with single element """

        def test_impl(int_fn):
            # workaround: use dict(zip(('A', ), (int_fn, )))
            return pd.read_csv("csv_data_infer1.csv",
                               converters={'A': int_fn})
        sdc_func = self.jit(test_impl)

        int_fn = self.get_int_converter()
        pd.testing.assert_frame_equal(sdc_func(int_fn),
                                      test_impl(int_fn.py_func))

    def test_csv_infer_file_param_skiprows_1(self):
        """Test verifies pandas read_csv impl supports parameter skiprows with explicit names and dtypes """
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
        """Test verifies pandas read_csv impl supports parameter skiprows with omitted names and dtypes """

        def test_impl():
            return pd.read_csv("csv_data_infer1.csv", skiprows=2)
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_parse_dates(self):
        """Test verifies pandas read_csv impl supports parsing string data as datetime
           using parse_dates parameter """

        int_type = self._int_type()

        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                               names=['A', 'B', 'C', 'D'],
                               dtype={'A': int_type, 'B': np.float, 'C': str, 'D': np.int64},
                               parse_dates=[2, ])
        sdc_func = self.jit(test_impl)
        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_param_dtype_common(self):
        """Test verifies pandas read_csv impl supports single dtype for dtype parameter """

        def test_impl():
            df = pd.read_csv("csv_data_dtype1.csv",
                             names=['C1', 'C2'],
                             dtype=np.float64,
                             )
            return df
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_const_dtype1(self):
        """Test verifies dtype dict can use const strings as column dtypes """

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

    def test_csv_infer_file_dtype_as_literal_dict(self):
        """Test verifies dtype parameter can be a const dict produced from tuples
           of column names and column types """

        def test_impl():

            # this relies on dict and zip overloads to build LiteralStrKeyDict from tuples
            col_names = ('A', 'B', 'C', 'D')
            col_types = (np.intp, 'float64', np.float, str)
            dtype = dict(zip(col_names, col_types))

            return pd.read_csv("csv_data1.csv",
                               names=col_names,
                               dtype=dtype,
                               )
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    def test_csv_infer_file_cat1(self):
        """Test verifies pandas read_csv impl supports reading categorical columns via dtype paramter """

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

    def test_csv_infer_file_cat2(self):
        """Test verifies reading categorical columns preserves unused categories during conversion
           from pyarrow to pandas DF """

        int_type = self._int_type()

        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
            df = pd.read_csv("csv_data_cat1.csv",
                             names=['C1', 'C2', 'C3'],
                             dtype={'C1': int_type, 'C2': ct_dtype, 'C3': str})
            return df
        sdc_func = self.jit(test_impl)

        pd.testing.assert_frame_equal(sdc_func(), test_impl())

    # not supported

    @skip_numba_jit
    def test_csv_not_supported_dict_use_1(self):

        int_type = self._int_type()

        def test_impl():
            dtype = {'A': int_type, 'B': np.float, 'C': np.float, 'D': str}
            # XXX: there are two reasons this fails:
            # 1. dtype doesn't exist as dict after RewriteReadCsv
            # 2. dict keys would be of DictKeys type, not tuple
            return pd.read_csv("csv_data1.csv",
                               names=dtype.keys(),
                               dtype=dtype)
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


class TestNumpy(TestIO):

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
            A.tofile("np_file_3.dat")

        hpat_func = self.jit(test_impl)
        n = 111
        A = np.random.ranf(n)
        hpat_func(A)
        B = np.fromfile("np_file_3.dat", np.float64)
        np.testing.assert_almost_equal(A, B)

    @skip_numba_jit("AssertionError: Failed in nopython mode pipeline (step: Preprocessing for parfors)")
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

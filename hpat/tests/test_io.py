import unittest
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import h5py
import pyarrow.parquet as pq
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_rank,
    get_start_end)


class TestIO(unittest.TestCase):

    def setUp(self):
        if get_rank() == 0:
            # h5 filter test
            n = 11
            size = (n, 13, 21, 3)
            A = np.random.randint(0, 120, size, np.uint8)
            f = h5py.File('h5_test_filter.h5', "w")
            f.create_dataset('test', data=A)
            f.close()

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

    def test_h5_read_seq(self):
        def test_impl():
            f = h5py.File("lr.hdf5", "r")
            X = f['points'][:]
            f.close()
            return X

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_allclose(hpat_func(), test_impl())

    def test_h5_read_const_infer_seq(self):
        def test_impl():
            p = 'lr'
            f = h5py.File(p + ".hdf5", "r")
            s = 'po'
            X = f[s + 'ints'][:]
            f.close()
            return X

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_allclose(hpat_func(), test_impl())

    def test_h5_read_parallel(self):
        def test_impl():
            f = h5py.File("lr.hdf5", "r")
            X = f['points'][:]
            Y = f['responses'][:]
            f.close()
            return X.sum() + Y.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl(), decimal=2)
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    @unittest.skip("fix collective create dataset")
    def test_h5_write_parallel(self):
        def test_impl(N, D):
            points = np.ones((N,D))
            responses = np.arange(N)+1.0
            f = h5py.File("lr_w.hdf5", "w")
            dset1 = f.create_dataset("points", (N,D), dtype='f8')
            dset1[:] = points
            dset2 = f.create_dataset("responses", (N,), dtype='f8')
            dset2[:] = responses
            f.close()

        N = 101
        D = 10
        hpat_func = hpat.jit(test_impl)
        hpat_func(N, D)
        f = h5py.File("lr_w.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]
        f.close()
        np.testing.assert_almost_equal(X, np.ones((N,D)))
        np.testing.assert_almost_equal(Y, np.arange(N)+1.0)

    @unittest.skip("fix collective create dataset and group")
    def test_h5_write_group(self):
        def test_impl(n, fname):
            arr = np.arange(n)
            n = len(arr)
            f = h5py.File(fname, "w")
            g1 = f.create_group("G")
            dset1 = g1.create_dataset("data", (n,), dtype='i8')
            dset1[:] = arr
            f.close()

        n = 101
        arr = np.arange(n)
        fname = "test_group.hdf5"
        hpat_func = hpat.jit(test_impl)
        hpat_func(n, fname)
        f = h5py.File(fname, "r")
        X = f['G']['data'][:]
        f.close()
        np.testing.assert_almost_equal(X, arr)

    def test_h5_read_group(self):
        def test_impl():
            f = h5py.File("test_group_read.hdf5", "r")
            g1 = f['G']
            X = g1['data'][:]
            f.close()
            return X.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_h5_file_keys(self):
        def test_impl():
            f = h5py.File("test_group_read.hdf5", "r")
            s = 0
            for gname in f.keys():
                X = f[gname]['data'][:]
                s += X.sum()
            f.close()
            return s

        hpat_func = hpat.jit(test_impl, h5_types={'X': hpat.int64[:]})
        self.assertEqual(hpat_func(), test_impl())
        # test using locals for typing
        hpat_func = hpat.jit(test_impl, locals={'X': hpat.int64[:]})
        self.assertEqual(hpat_func(), test_impl())

    def test_h5_group_keys(self):
        def test_impl():
            f = h5py.File("test_group_read.hdf5", "r")
            g1 = f['G']
            s = 0
            for dname in g1.keys():
                X = g1[dname][:]
                s += X.sum()
            f.close()
            return s

        hpat_func = hpat.jit(test_impl, h5_types={'X': hpat.int64[:]})
        self.assertEqual(hpat_func(), test_impl())

    def test_h5_filter(self):
        def test_impl():
            f = h5py.File("h5_test_filter.h5", "r")
            b = np.arange(11) % 3 == 0
            X = f['test'][b,:,:,:]
            f.close()
            return X

        hpat_func = hpat.jit(locals={'X:return': 'distributed'})(test_impl)
        n = 4  # len(test_impl())
        start, end = get_start_end(n)
        np.testing.assert_allclose(hpat_func(), test_impl()[start:end])

    def test_pq_read(self):
        def test_impl():
            t = pq.read_table('kde.parquet')
            df = t.to_pandas()
            X = df['points']
            return X.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pd_read_parquet(self):
        def test_impl():
            df = pd.read_parquet('kde.parquet')
            X = df['points']
            return X.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_str(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.two.values=='foo'
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_str_with_nan_seq(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values=='foo'
            return A

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    def test_pq_str_with_nan_par(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values=='foo'
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_str_with_nan_par_multigroup(self):
        def test_impl():
            df = pq.read_table('example2.parquet').to_pandas()
            A = df.five.values=='foo'
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_bool(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.three.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.one.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_float_no_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.four.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_pandas_date(self):
        def test_impl():
            df = pd.read_parquet('pandas_dt.pq')
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_pq_spark_date(self):
        def test_impl():
            df = pd.read_parquet('sdf_dt.pq')
            return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv1(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},
            )
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_infer1(self):
        def test_impl():
            return pd.read_csv("csv_data_infer1.csv")

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_infer_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_infer1.csv")
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_csv_skip1(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},
                skiprows=2,
            )
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_infer_skip1(self):
        def test_impl():
            return pd.read_csv("csv_data_infer1.csv", skiprows=2)

        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_infer_skip_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_infer1.csv", skiprows=2,
                names=['A', 'B', 'C', 'D'])
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_csv_rm_dead1(self):
        def test_impl():
            df = pd.read_csv("csv_data1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},)
            return df.B.values
        hpat_func = hpat.jit(test_impl)
        np.testing.assert_array_equal(hpat_func(), test_impl())

    def test_csv_date1(self):
        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int},
                parse_dates=[2])
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_str1(self):
        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int})
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int})
            return (df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum())
        hpat_func = hpat.jit(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_csv_str_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_date1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int})
            return (df.A.sum(), df.B.sum(), (df.C == '1966-11-13').sum(),
                    df.D.sum())
        hpat_func = hpat.jit(locals={'df:return': 'distributed'})(test_impl)
        self.assertEqual(hpat_func(), test_impl())

    def test_csv_usecols1(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                names=['C'],
                dtype={'C':np.float},
                usecols=[2],
            )
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_cat1(self):
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1':np.int, 'C2': ct_dtype, 'C3':str}
            df = pd.read_csv("csv_data_cat1.csv",
                names=['C1', 'C2', 'C3'],
                dtype=dtypes,
            )
            return df.C2
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_series_equal(
            hpat_func(), test_impl(), check_names=False)

    def test_csv_cat2(self):
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
            df = pd.read_csv("csv_data_cat1.csv",
                names=['C1', 'C2', 'C3'],
                dtype={'C1':np.int, 'C2': ct_dtype, 'C3':str},
            )
            return df
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_csv_single_dtype1(self):
        def test_impl():
            df = pd.read_csv("csv_data_dtype1.csv",
                names=['C1', 'C2'],
                dtype=np.float64,
            )
            return df
        hpat_func = hpat.jit(test_impl)
        pd.testing.assert_frame_equal(hpat_func(), test_impl())

    def test_write_csv1(self):
        def test_impl(df, fname):
            df.to_csv(fname)

        hpat_func = hpat.jit(test_impl)
        n = 111
        df = pd.DataFrame({'A': np.arange(n)})
        hp_fname = 'test_write_csv1_hpat.csv'
        pd_fname = 'test_write_csv1_pd.csv'
        hpat_func(df, hp_fname)
        test_impl(df, pd_fname)
        # TODO: delete files
        pd.testing.assert_frame_equal(pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    def test_write_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({'A': np.arange(n)})
            df.to_csv(fname)

        hpat_func = hpat.jit(test_impl)
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

    def test_np_io1(self):
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())

    def test_np_io2(self):
        # parallel version
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A.sum()

        hpat_func = hpat.jit(test_impl)
        np.testing.assert_almost_equal(hpat_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_np_io3(self):
        def test_impl(A):
            if get_rank() == 0:
                A.tofile("np_file_3.dat")

        hpat_func = hpat.jit(test_impl)
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

        hpat_func = hpat.jit(test_impl)
        n = 111
        A = np.arange(n)
        hpat_func(n)
        B = np.fromfile("np_file_3.dat", np.int64)
        np.testing.assert_almost_equal(A, B)


if __name__ == "__main__":
    unittest.main()

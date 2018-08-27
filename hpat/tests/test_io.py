import unittest
import pandas as pd
import numpy as np
import h5py
import pyarrow.parquet as pq
import hpat
from hpat.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                        count_parfor_OneDs, count_array_OneDs, dist_IR_contains)


class TestIO(unittest.TestCase):
    def test_h5_read_seq(self):
        def test_impl():
            f = h5py.File("lr.hdf5", "r")
            X = f['points'][:]
            #f.close()
            return X

        hpat_func = hpat.jit(test_impl, h5_types={'X': hpat.float64[:]})
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
        f.close()

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


if __name__ == "__main__":
    unittest.main()

import h5py
import numpy as np
import argparse
import time

def gen_lir(N, D, file_name):
    np.random.seed(0)
    points = np.random.random((N,D))
    responses = np.random.random(N)
    f = h5py.File(file_name, "w")
    dset1 = f.create_dataset("points", (N,D), dtype='f8')
    dset1[:] = points
    dset2 = f.create_dataset("responses", (N,), dtype='f8')
    dset2[:] = responses
    f.close()

def main():
    parser = argparse.ArgumentParser(description='Gen Logistic Regression.')
    parser.add_argument('--samples', dest='samples', type=int, default=2000)
    parser.add_argument('--features', dest='features', type=int, default=10)
    parser.add_argument('--file', dest='file', type=str, default="lr.hdf5")
    args = parser.parse_args()
    N = args.samples
    D = args.features
    file_name = args.file

    gen_lir(N, D, file_name)

if __name__ == '__main__':
    main()

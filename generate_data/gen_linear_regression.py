import h5py
import numpy as np
import argparse
import time

def gen_lir(N, D, p, file_name):
    np.random.seed(0)
    points = np.random.random((N,D))
    responses = np.random.random((N,p))
    f = h5py.File(file_name, "w")
    dset1 = f.create_dataset("points", (N,D), dtype='f8')
    dset1[:] = points
    dset2 = f.create_dataset("responses", (N,p), dtype='f8')
    dset2[:] = responses
    f.close()

def main():
    parser = argparse.ArgumentParser(description='Gen Linear Regression.')
    parser.add_argument('--samples', dest='samples', type=int, default=2000)
    parser.add_argument('--features', dest='features', type=int, default=10)
    parser.add_argument('--functions', dest='functions', type=int, default=4)
    parser.add_argument('--file', dest='file', type=str, default="lir.hdf5")
    args = parser.parse_args()
    N = args.samples
    D = args.features
    p = args.functions
    file_name = args.file

    gen_lir(N, D, p, file_name)

if __name__ == '__main__':
    main()

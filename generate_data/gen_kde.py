import h5py
import numpy as np
import argparse
import time

def gen_kde(N, file_name):
    np.random.seed(0)
    points = np.random.random(N)
    f = h5py.File(file_name, "w")
    dset1 = f.create_dataset("points", (N,), dtype='f8')
    dset1[:] = points
    f.close()


def main():
    parser = argparse.ArgumentParser(description='Gen KDE.')
    parser.add_argument('--size', dest='size', type=int, default=2000)
    parser.add_argument('--file', dest='file', type=str, default="kde.hdf5")
    args = parser.parse_args()
    N = args.size
    file_name = args.file

    gen_kde(N, file_name)


if __name__ == '__main__':
    main()

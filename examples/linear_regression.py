import hpat
import numpy as np
import h5py
import argparse
import time

@hpat.jit
def linear_regression(iterations):
    f = h5py.File("lir.hdf5", "r")
    X = f['points'][:]
    Y = f['responses'][:]
    N,D = X.shape
    p = Y.shape[1]
    alphaN = 0.01/N
    w = np.zeros((D,p))
    t1 = time.time()
    for i in range(iterations):
        w -= alphaN * np.dot(X.T, np.dot(X,w)-Y)
    t2 = time.time()
    print("Execution time:", t2-t1, "\nresult:", w)
    return w

def main():
    parser = argparse.ArgumentParser(description='Linear Regression.')
    parser.add_argument('--file', dest='file', type=str, default="lr.hdf5")
    parser.add_argument('--iterations', dest='iterations', type=int, default=30)
    args = parser.parse_args()

    file_name = args.file
    iterations = args.iterations

    w = linear_regression(iterations)

if __name__ == '__main__':
    main()

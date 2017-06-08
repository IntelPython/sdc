import hpat
import numpy as np
import h5py
import argparse
import time

@hpat.jit(locals={'X':hpat.float64[:,:], 'Y':hpat.float64[:]})
def logistic_regression(iterations):
    f = h5py.File("lr.hdf5", "r")
    X = f['points'][:]
    Y = f['responses'][:]
    D = X.shape[1]
    w = 2.0*np.ones(D)-1.3
    t1 = time.time()
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X,w))) - 1.0) * Y),X)
    t2 = time.time()
    print("Execution time:", t2-t1, "\nresult:", w)
    return w

def main():
    parser = argparse.ArgumentParser(description='Logistic Regression.')
    parser.add_argument('--file', dest='file', type=str, default="lr.hdf5")
    parser.add_argument('--iterations', dest='iterations', type=int, default=20)
    args = parser.parse_args()

    file_name = args.file
    iterations = args.iterations

    w = logistic_regression(iterations)

if __name__ == '__main__':
    main()

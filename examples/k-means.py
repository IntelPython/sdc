import numpy as np
from math import sqrt
import argparse
import time
import h5py
import hpat

@hpat.jit
def kmeans(numCenter, numIter):
    f = h5py.File("lr.hdf5", "r")
    A = f['points'][:]
    N, D = A.shape

    centroids = np.random.ranf((numCenter, D))
    t1 = time.time()

    for l in range(numIter):
        dist = np.array([[sqrt(np.sum((A[i,:]-centroids[j,:])**2))
                                for j in range(numCenter)] for i in range(N)])
        labels = np.array([dist[i,:].argmin() for i in range(N)])

        centroids = np.array([[np.sum(A[labels==i, j])/np.sum(labels==i)
                                 for j in range(D)] for i in range(numCenter)])

    t2 = time.time()
    print("Execution time:", t2-t1, "\nresult:", centroids)
    return centroids

def main():
    parser = argparse.ArgumentParser(description='K-Means')
    # parser.add_argument('--file', dest='file', type=str, default="lr.hdf5")
    parser.add_argument('--centers', dest='centers', type=int, default=3)
    parser.add_argument('--iterations', dest='iterations', type=int, default=20)
    args = parser.parse_args()
    centers = args.centers
    iterations = args.iterations

    #D = 10
    #np.random.seed(0)
    #init_centroids = np.random.ranf((centers, D))
    res = kmeans(centers, iterations)

if __name__ == '__main__':
    main()

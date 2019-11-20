# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
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
from math import sqrt
import argparse
import time
import h5py
import sdc


@sdc.jit
def kmeans(numCenter, numIter):
    f = h5py.File("lr.hdf5", "r")
    A = f['points'][:]
    f.close()
    N, D = A.shape

    centroids = np.random.ranf((numCenter, D))
    t1 = time.time()

    for l in range(numIter):
        dist = np.array([[sqrt(np.sum((A[i, :] - centroids[j, :])**2))
                          for j in range(numCenter)] for i in range(N)])
        labels = np.array([dist[i, :].argmin() for i in range(N)])

        centroids = np.array([[np.sum(A[labels == i, j]) / np.sum(labels == i)
                               for j in range(D)] for i in range(numCenter)])

    t2 = time.time()
    print("Execution time:", t2 - t1, "\nresult:", centroids)
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
    # np.random.seed(0)
    #init_centroids = np.random.ranf((centers, D))
    res = kmeans(centers, iterations)


if __name__ == '__main__':
    main()

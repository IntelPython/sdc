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
    f.close()
    N, D = X.shape
    p = Y.shape[1]
    alphaN = 0.01 / N
    w = np.zeros((D, p))
    t1 = time.time()
    for i in range(iterations):
        w -= alphaN * np.dot(X.T, np.dot(X, w) - Y)
    t2 = time.time()
    print("Execution time:", t2 - t1, "\nresult:", w)
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

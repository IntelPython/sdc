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


import h5py
import numpy as np
import argparse
import time
import hpat


@hpat.jit
def gen_lir(N, D, p, file_name):
    # np.random.seed(0)
    points = np.random.random((N, D))
    responses = np.random.random((N, p))
    f = h5py.File(file_name, "w")
    dset1 = f.create_dataset("points", (N, D), dtype='f8')
    dset1[:] = points
    dset2 = f.create_dataset("responses", (N, p), dtype='f8')
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

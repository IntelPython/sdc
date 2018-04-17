#!/bin/bash

source activate $CONDA_ENV

# # install Numba in a directory to avoid import conflict
# mkdir req_install
# pushd req_install
# git clone https://github.com/IntelLabs/numba
# pushd numba
# git checkout hpat_req
# python setup.py install
# popd
# popd

pushd parquet_reader
mkdir build
pushd build
cmake -DCMAKE_BUILD_TYPE=release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_INSTALL_LIBDIR=$CONDA_PREFIX/lib -DPQ_PREFIX=$CONDA_PREFIX ..
make VERBOSE=1
make install
popd
popd

# build HPAT
DAALROOT=$CONDA_PREFIX HDF5_DIR=$CONDA_PREFIX python setup.py install

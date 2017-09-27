.. _install:

Installing HPAT
===============

We recommend `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python 3.6 for setting up HPAT. These commands install HPAT and its dependencies
such as Numba and LLVM on Ubuntu Linux::

    sudo apt install llvm-4.0 make libc6-dev gcc-4.8
    conda create -n HPAT
    source activate HPAT
    conda install numpy scipy pandas gcc mpich2 llvmlite
    git clone https://github.com/IntelLabs/numba.git
    cd numba
    git checkout hpat_req
    python setup.py install
    cd ..
    git clone https://github.com/IntelLabs/hpat.git
    cd hpat
    LDSHARED="mpicxx -shared" CXX=mpicxx LD=mpicxx \
        CC="mpicxx -std=c++11" python setup.py install

A command line for running the Pi example on 4 cores::

    mpirun -n 4 python examples/pi.py

HDF5 Support
------------

HPAT supports reading and writing HDF5 files in parallel. The instructions below
describe building and setting up HDF5 from its
`source code <https://www.hdfgroup.org/downloads/hdf5/source-code/>`_::

    # download hdf5-1.10.1.tar.gz
    tar xzf hdf5-1.10.1.tar.gz
    cd hdf5-1.10.1/
    CC=mpicc CXX=mpicxx ./configure --enable-parallel
    make; make install
    cd ..
    export HDF5_DIR=/home/user/hdf5-1.10.1/hdf5/
    export C_INCLUDE_PATH=$C_INCLUDE_PATH:$HDF5_DIR/include
    export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$HDF5_DIR/include
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF5_DIR/lib
    export LIBRARY_PATH=$LIBRARY_PATH:$HDF5_DIR/lib
    git clone https://github.com/h5py/h5py.git
    cd h5py
    python setup.py configure --hdf5=$HDF5_DIR
    LDSHARED="mpicc -shared" CXX=mpicxx LD=mpicc CC="mpicc" \
        python setup.py install

Commands for generating HDF5 data and running the logistic regression example::

    python generate_data/gen_logistic_regression.py
    mpirun -n 4 python examples/logistic_regression.py

.. _install:

Installing HPAT
===============

HPAT can be installed in `Anaconda <https://www.anaconda.com/download/>`_ environment easily::

    conda install numpy scipy pandas llvmlite=0.20 python=3.6
    conda install pyarrow mpich -c conda-forge
    conda install hpat -c ehsantn

Building HPAT from Source
-------------------------

We use `Anaconda 4.4 <https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh>`_ distribution of
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

    mpiexec -n 4 python examples/pi.py

Building HDF5 Support
---------------------

HPAT supports reading and writing HDF5 files in parallel. The instructions below
describe building and setting up HDF5 from its
`source code (v1.8.19) <https://support.hdfgroup.org/ftp/HDF5/current18/src/hdf5-1.8.19.tar.gz>`_::

    # download hdf5-1.8.19.tar.gz
    tar xzf hdf5-1.8.19.tar.gz
    cd hdf5-1.8.19/
    CC=mpicc CXX=mpicxx ./configure --enable-parallel
    make; make install
    cd ..
    export HDF5_DIR=$HOME/hdf5-1.8.19/hdf5/
    git clone https://github.com/h5py/h5py.git
    cd h5py
    python setup.py configure --hdf5=$HDF5_DIR
    LDSHARED="mpicc -shared" CXX=mpicxx LD=mpicc CC="mpicc" \
        python setup.py install

HPAT needs to be rebuilt after setting up HDF5. We use HDF5 v1.8.x since the
latest versions (v1.10.x) have an issue with LLVM which is under investigation.
Commands for generating HDF5 data and running the logistic regression example::

    python generate_data/gen_logistic_regression.py
    mpiexec -n 4 python examples/logistic_regression.py

Parquet Support
---------------

HPAT uses the `pyarrow` package to provide Parquet support::

    conda install pyarrow -c conda-forge

HPAT needs to be rebuilt after setting up pyarrow.

Building from Source on Windows
-------------------------------

Building HPAT on Windows requires Build Tools for Visual Studio 2017 (14.0) and Intel MPI:

* Install `Build Tools for Visual Studio 2017 (14.0) <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>`_.
* Setup the environment by running ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat amd64``.
* Install `Intel MPI <https://software.intel.com/en-us/intel-mpi-library>`_.
* Setup the environment by following
  `Intel MPI installation instructions <https://software.intel.com/en-us/articles/intel-mpi-library-for-windows-installation-instructions>`_.
* Install `Anaconda 4.4 for Windows <https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe>`_.
* Setup the Conda environment in Anaconda Prompt::

    conda create -n HPAT
    activate HPAT
    conda install numpy scipy pandas llvmlite
    git clone https://github.com/IntelLabs/numba.git
    cd numba
    git checkout hpat_req
    python setup.py install
    cd ..
    conda install pyarrow -c conda-forge
    git clone https://github.com/IntelLabs/hpat.git
    cd hpat
    set INCLUDE=%INCLUDE%;%CONDA_PREFIX%\Library\include
    set LIB=%LIB%;%CONDA_PREFIX%\Library\lib
    python setup.py install


Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If the ``cl`` compiler throws the error fatal ``error LNK1158: cannot run ‘rc.exe’``,
  add Windows Kits to your PATH (e.g. ``C:\Program Files (x86)\Windows Kits\8.0\bin\x86``).
* Some errors can be mitigated by ``set DISTUTILS_USE_SDK=1``.
* For setting up Visual Studio, one might need go to registry at
  ``HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7``,
  and add a string value named ``14.0`` whose data is ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\``.

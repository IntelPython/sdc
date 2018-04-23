.. _install:

Installing HPAT
===============

HPAT can be installed in `Anaconda <https://www.anaconda.com/download/>`_ environment
easily. On Linux/Mac::

    conda create -n HPAT python=3.6
    source activate HPAT
    conda install pandas
    conda install numba -c numba
    conda install pyarrow mpich -c conda-forge
    conda install hpat -c ehsantn

On Windows::

    conda create -n HPAT python=3.6
    activate HPAT
    conda install pandas
    conda install numba -c numba
    conda install pyarrow -c conda-forge
    conda install hpat -c ehsantn

Building HPAT from Source
-------------------------

We use `Anaconda 5.1.0 <https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh>`_ distribution of
Python 3.6 for setting up HPAT. These commands install HPAT and its dependencies
such as Numba and LLVM on Ubuntu Linux::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n HPAT -q -y python=3.6 numpy scipy pandas boost cmake
    source activate HPAT
    conda install -c numba numba=0.38.0rc1
    conda install pyarrow=0.9.* mpich -c conda-forge
    conda install h5py -c ehsantn
    conda install gcc_linux-64 gxx_linux-64 gfortran_linux-64
    git clone https://github.com/IntelLabs/hpat
    cd hpat
    pushd parquet_reader
    mkdir build
    pushd build
    cmake -DCMAKE_BUILD_TYPE=release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
        -DCMAKE_INSTALL_LIBDIR=$CONDA_PREFIX/lib -DPQ_PREFIX=$CONDA_PREFIX ..
    make
    make install
    popd
    popd
    # build HPAT
    HDF5_DIR=$CONDA_PREFIX python setup.py develop


A command line for running the Pi example on 4 cores::

    mpiexec -n 4 python examples/pi.py

Running unit tests::

    python hpat/tests/gen_test_data.py
    python -m unittest

In case of issues, reinstalling in a new conda environment is recommended.
Also, a common issue is ``hdf5`` package reverting to default instead of the
parallel version installed from ``ehsantn`` channel. Use ``conda list``
to check the channel of ``hdf5`` package.

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
    conda install pyarrow=0.8.* -c conda-forge
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

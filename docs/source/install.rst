.. _install:

Installing HPAT
===============

HPAT can be installed in `Anaconda <https://www.anaconda.com/download/>`_ environment
easily. On Linux/Mac/Windows::

    conda create -n HPAT -c ehsantn -c numba -c anaconda -c conda-forge hpat

Windows installaton requires
`Intel MPI <https://software.intel.com/en-us/intel-mpi-library>`_ to be
installed.

Building HPAT from Source
-------------------------

We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python 3.6 for setting up HPAT. These commands install HPAT and its dependencies
such as Numba on Ubuntu Linux::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n HPAT -q -y python=3.6 numpy scipy pandas boost cmake
    source activate HPAT
    conda install -c numba numba
    conda install mpich -c conda-forge
    conda install pyarrow
    conda install h5py -c ehsantn
    conda install gcc_linux-64 gxx_linux-64 gfortran_linux-64
    git clone https://github.com/IntelLabs/hpat
    cd hpat
    # build HPAT
    HDF5_DIR=$CONDA_PREFIX python setup.py develop


A command line for running the Pi example on 4 cores::

    mpiexec -n 4 python examples/pi.py

Running unit tests::

    conda install pyspark
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
* Install `Intel MPI <https://software.intel.com/en-us/intel-mpi-library>`_.
* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda prompt'
* Setup the Conda environment in Anaconda Prompt::

    conda create -n HPAT -c ehsantn -c numba -c anaconda -c conda-forge python=3.6 pandas pyarrow h5py numba scipy boost libboost tbb-devel mkl-devel
    activate HPAT
    git clone https://github.com/IntelLabs/hpat.git
    cd hpat
    set INCLUDE=%INCLUDE%;%CONDA_PREFIX%\Library\include
    set LIB=%LIB%;%CONDA_PREFIX%\Library\lib
    "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat amd64"
    "<mpi-install-root>\intel64\bin\mpivars.bat"
    python setup.py install

Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If the ``cl`` compiler throws the error fatal ``error LNK1158: cannot run ‘rc.exe’``,
  add Windows Kits to your PATH (e.g. ``C:\Program Files (x86)\Windows Kits\8.0\bin\x86``).
* Some errors can be mitigated by ``set DISTUTILS_USE_SDK=1``.
* For setting up Visual Studio, one might need go to registry at
  ``HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7``,
  and add a string value named ``14.0`` whose data is ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\``.

.. _install:

Installing HPAT
===============

HPAT can be installed in `Anaconda <https://www.anaconda.com/download/>`_ environment
easily. On Linux/Mac/Windows::

    conda create -n HPAT -c ehsantn -c anaconda -c conda-forge hpat

.. used if master of Numba is needed for latest hpat package
.. conda create -n HPAT -c ehsantn -c numba/label/dev -c anaconda -c conda-forge hpat

Building from Source on Linux
-----------------------------

We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up HPAT.

Miniconda3 is required for build::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH

It is possible to build HPAT via conda-build or setuptools. Follow one of the cases below to install HPAT and its dependencies
such as Numba on Ubuntu Linux.

Build with conda-build:
~~~~~~~~~~~~~~~~~~~~~~~
::

    conda create -n HPAT python=<3.7 or 3.6>
    source activate HPAT
    conda install conda-build
    git clone https://github.com/IntelPython/hpat
    # build HPAT
    conda build --python <3.6 or 3.7> -c numba -c conda-forge -c defaults hpat/buildscripts/hpat-conda-recipe/

Build with setuptools:
~~~~~~~~~~~~~~~~~~~~~~
::

    conda create -n HPAT -q -y numpy scipy pandas boost cmake python=<3.6 or 3.7>
    source activate HPAT
    conda install -c numba/label/dev numba
    conda install mpich mpi -c conda-forge
    conda install pyarrow
    conda install h5py -c ehsantn
    conda install gcc_linux-64 gxx_linux-64 gfortran_linux-64
    git clone https://github.com/IntelPython/hpat
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

Building HPAT on Windows requires Build Tools for Visual Studio 2017 (14.0):

* Install `Build Tools for Visual Studio 2017 (14.0) <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>`_.
* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda prompt'
* Setup the Conda environment in Anaconda Prompt::

It is possible to build HPAT via conda-build or setuptools. Follow one of the cases below to install HPAT and its dependencies on Windows.

Build with conda-build:
~~~~~~~~~~~~~~~~~~~~~~~
::

    conda create -n HPAT python=<3.7 or 3.6>
    activate HPAT
    conda install vc vs2015_runtime vs2015_win-64
    git clone https://github.com/IntelPython/hpat.git
    conda build --python <3.6 or 3.7> -c numba -c conda-forge -c defaults -c intel hpat/buildscripts/hpat-conda-recipe/

Build with setuptools:
~~~~~~~~~~~~~~~~~~~~~~
::

    conda create -n HPAT -c ehsantn -c numba/label/dev -c anaconda -c conda-forge -c intel python=<3.6 or 3.7> pandas pyarrow h5py numba scipy boost libboost tbb-devel mkl-devel impi-devel impi_rt
    activate HPAT
    conda install vc vs2015_runtime vs2015_win-64
    git clone https://github.com/IntelPython/hpat.git
    cd hpat
    set INCLUDE=%INCLUDE%;%CONDA_PREFIX%\Library\include
    set LIB=%LIB%;%CONDA_PREFIX%\Library\lib
    %CONDA_PREFIX%\Library\bin\mpivars.bat quiet
    set HDF5_DIR=%CONDA_PREFIX%\Library
    python setup.py develop

.. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If the ``cl`` compiler throws the error fatal ``error LNK1158: cannot run ‘rc.exe’``,
  add Windows Kits to your PATH (e.g. ``C:\Program Files (x86)\Windows Kits\8.0\bin\x86``).
* Some errors can be mitigated by ``set DISTUTILS_USE_SDK=1``.
* For setting up Visual Studio, one might need go to registry at
  ``HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7``,
  and add a string value named ``14.0`` whose data is ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\``.
 
AWS Setup
---------

This page describes a simple setup process for HPAT on Amazon EC2 instances. You need to have an account on Amazon Web Services (AWS)
and be familiar with the general AWS EC2 instance launch interface. The process below is for demonstration purposes only and is not
recommended for production usage due to security, performance and other considerations.

1. Launch instances:
    a. Select a Linux instance type (e.g. Ubuntu Server 18.04, c5n types for high network bandwidth).
    b. Select number of instances (e.g. 4).
    c. Select placement group option for better network performance (check "add instance to placement group").
    d. Enable all ports in security group configuration to simplify MPI setup (add a new rule with "All traffic" Type and "Anywhere" Source).

2. Setup password-less ssh between instances:
    a. Copy your key from your client to all instances. For example, on a Linux clients run this for all instances (find public host names from AWS portal)::

        scp -i "user.pem" user.pem ubuntu@ec2-11-111-11-111.us-east-2.compute.amazonaws.com:~/.ssh/id_rsa

    b. Disable ssh host key check by running this command on all instances::

        echo -e "Host *\n    StrictHostKeyChecking no" > .ssh/config

    c. Create a host file with list of private hostnames of instances on home directory of all instances::

        echo -e "ip-11-11-11-11.us-east-2.compute.internal\nip-11-11-11-12.us-east-2.compute.internal\n" > hosts

3. Install Anaconda Python distribution and HPAT on all instances::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n HPAT -c ehsantn -c anaconda -c conda-forge hpat
    source activate HPAT

4. Copy the `Pi example <https://github.com/IntelLabs/hpat#example>`_ to a file called pi.py in the home directory of all instances and run it with and without MPI and see execution times.
   You should see speed up when running on more cores ("-n 2" and "-n 4" cases)::

    python pi.py  # Execution time: 2.119
    mpiexec -f hosts -n 2 python pi.py  # Execution time: 1.0569
    mpiexec -f hosts -n 4 python pi.py  # Execution time: 0.5286


Possible next experiments from here are running a more complex example like the
`logistic regression example <https://github.com/IntelLabs/hpat/blob/master/examples/logistic_regression_rand.py>`_.
Furthermore, attaching a shared EFS storage volume and experimenting with parallel I/O in HPAT is recommended.

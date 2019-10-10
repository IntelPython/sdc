*****
HPAT
*****

.. image:: https://travis-ci.com/IntelPython/hpat.svg?branch=master
    :target: https://travis-ci.com/IntelPython/hpat

.. image:: https://coveralls.io/repos/github/IntelPython/hpat/badge.svg?branch=master
    :target: https://coveralls.io/github/IntelPython/hpat?branch=master

A compiler-based framework for big data in Python
#################################################

High Performance Analytics Toolkit (HPAT) scales analytics/ML codes in Python
to bare-metal cluster/cloud performance automatically.
It compiles a subset of Python (Pandas/Numpy) to efficient parallel binaries
with MPI, requiring only minimal code changes.
HPAT is orders of magnitude faster than
alternatives like `Apache Spark <http://spark.apache.org/>`_.

HPAT's documentation can be found `here <https://intellabs.github.io/hpat-doc/>`_.

Installing Binary Packages (conda)
----------------------------------
::

   conda install -c intel -c intel/label/test hpat


Example
#######

Here is a Pi calculation example in HPAT:

.. code:: python

    import hpat
    import numpy as np
    import time

    @hpat.jit
    def calc_pi(n):
        t1 = time.time()
        x = 2 * np.random.ranf(n) - 1
        y = 2 * np.random.ranf(n) - 1
        pi = 4 * np.sum(x**2 + y**2 < 1) / n
        print("Execution time:", time.time()-t1, "\nresult:", pi)
        return pi

    calc_pi(2 * 10**8)

Save this in a file named `pi.py` and run (on 8 cores)::

    mpiexec -n 8 python pi.py

This should demonstrate about 100x speedup compared to regular Python version
without `@hpat.jit` and `mpiexec`.


References
##########

These academic papers describe the underlying methods in HPAT:

- `HPAT paper at ICS'17 <http://dl.acm.org/citation.cfm?id=3079099>`_
- `HPAT at HotOS'17 <http://dl.acm.org/citation.cfm?id=3103004>`_
- `HiFrames on arxiv <https://arxiv.org/abs/1704.02341>`_


Building HPAT from Source on Linux
----------------------------------

We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up HPAT build environment.

If you do not have conda, we recommend using Miniconda3::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH

It is possible to build HPAT via conda-build or setuptools. Follow one of the
cases below to install HPAT and its dependencies on Linux.

Building on Linux with conda-build
~~~~~~~~~~~~~~~~~~~~~~~~~
::

    PYVER=<3.6 or 3.7>
    conda create -n CBLD python=$PYVER conda-build
    source activate CBLD
    git clone https://github.com/IntelPython/hpat
    cd hpat
    # build HPAT
    conda build --python $PYVER --override-channels -c numba -c conda-forge -c defaults buildscripts/hpat-conda-recipe

Building on Linux with setuptools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    PYVER=<3.6 or 3.7>
    conda create -n HPAT -q -y -c numba -c conda-forge -c defaults numba mpich pyarrow=0.14.1 arrow-cpp=0.14.1 gcc_linux-64 gxx_linux-64 gfortran_linux-64 scipy pandas boost python=$PYVER
    source activate HPAT
    git clone https://github.com/IntelPython/hpat
    cd hpat
    # build HPAT
    python setup.py install

In case of issues, reinstalling in a new conda environment is recommended.

Building HPAT from Source on Windows
------------------------------------

Building HPAT on Windows requires Build Tools for Visual Studio 2019 (with component MSVC v140 - VS 2015 C++ build tools (v14.00)):

* Install `Build Tools for Visual Studio 2019 (with component MSVC v140 - VS 2015 C++ build tools (v14.00)) <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_.
* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda prompt'

It is possible to build HPAT via conda-build or setuptools. Follow one of the
cases below to install HPAT and its dependencies on Windows.

Building on Windows with conda-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    set PYVER=<3.6 or 3.7>
    conda create -n CBLD -q -y python=%PYVER% conda-build conda-verify vc vs2015_runtime vs2015_win-64
    conda activate CBLD
    git clone https://github.com/IntelPython/hpat.git
    cd hpat
    conda build --python %PYVER% --override-channels -c numba -c defaults -c intel buildscripts\hpat-conda-recipe

Building on Windows with setuptools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    conda create -n HPAT -c numba -c defaults -c intel python=<3.6 or 3.7> numba impi-devel pyarrow=0.14.1 arrow-cpp=0.14.1 scipy pandas boost
    conda activate HPAT
    git clone https://github.com/IntelPython/hpat.git
    cd hpat
    set INCLUDE=%INCLUDE%;%CONDA_PREFIX%\Library\include
    set LIB=%LIB%;%CONDA_PREFIX%\Library\lib
    %CONDA_PREFIX%\Library\bin\mpivars.bat quiet
    python setup.py install

.. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If the ``cl`` compiler throws the error fatal ``error LNK1158: cannot run 'rc.exe'``,
  add Windows Kits to your PATH (e.g. ``C:\Program Files (x86)\Windows Kits\8.0\bin\x86``).
* Some errors can be mitigated by ``set DISTUTILS_USE_SDK=1``.
* For setting up Visual Studio, one might need go to registry at
  ``HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7``,
  and add a string value named ``14.0`` whose data is ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\``.
* Sometimes if the conda version or visual studio version being used are not latest then building HPAT can throw some vague error about a keyword used in a file. So make sure you are using the latest versions.

Running unit tests
------------------
::

    conda install h5py
    python hpat/tests/gen_test_data.py
    python -m unittest

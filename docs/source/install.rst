.. _install:

Installing HPAT
===============

HPAT can be installed in `Anaconda <https://www.anaconda.com/download/>`_ environment
easily. On Linux/Mac/Windows::

    conda create -n HPAT -c ehsantn -c anaconda -c conda-forge hpat

.. used if master of Numba is needed for latest hpat package
.. conda create -n HPAT -c ehsantn -c numba/label/dev -c anaconda -c conda-forge hpat

Building HPAT from Source
-------------------------

We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up HPAT. These commands install HPAT and its dependencies
such as Numba on Ubuntu Linux::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n HPAT python=<3.7 or 3.6>
    source activate HPAT
    conda install conda-build
    git clone https://github.com/IntelPython/hpat
    # build HPAT
    conda build --python <3.6 or 3.7> -c numba -c conda-forge -c defaults hpat/buildscripts/hpat-conda-recipe/


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

    conda create -n HPAT python=<3.7 or 3.6>
    activate HPAT
    conda install vc vs2015_runtime vs2015_win-64
    git clone https://github.com/IntelPython/hpat.git
    conda build --python <3.6 or 3.7> -c numba -c conda-forge -c defaults -c intel hpat/buildscripts/hpat-conda-recipe/

.. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If the ``cl`` compiler throws the error fatal ``error LNK1158: cannot run ‘rc.exe’``,
  add Windows Kits to your PATH (e.g. ``C:\Program Files (x86)\Windows Kits\8.0\bin\x86``).
* Some errors can be mitigated by ``set DISTUTILS_USE_SDK=1``.
* For setting up Visual Studio, one might need go to registry at
  ``HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7``,
  and add a string value named ``14.0`` whose data is ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\``.

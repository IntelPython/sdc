*****
Sdc
*****

Intel® Scalable Dataframe Compiler
###################################################

.. image:: https://travis-ci.com/IntelPython/sdc.svg?branch=master
    :target: https://travis-ci.com/IntelPython/sdc
    :alt: Travis CI

.. image:: https://dev.azure.com/IntelPython/HPAT/_apis/build/status/IntelPython.sdc?branchName=master
    :target: https://dev.azure.com/IntelPython/HPAT/_build/latest?definitionId=2&branchName=master
    :alt: Azure Pipelines

.. _Numba*: https://numba.pydata.org/
.. _Pandas*: https://pandas.pydata.org/
.. _Sphinx*: https://www.sphinx-doc.org/

Numba* Extension For Pandas* Operations Compilation
###################################################

Intel® Scalable Dataframe Compiler (Intel® SDC) is an extension of `Numba*`_
that enables compilation of `Pandas*`_ operations. It automatically vectorizes and parallelizes
the code by leveraging modern hardware instructions and by utilizing all available cores.

Intel® SDC documentation can be found `here <https://intelpython.github.io/sdc-doc/>`__.

.. note::
    For maximum performance and stability, please use numba from ``intel/label/beta`` channel.

Installing Binary Packages (conda and wheel)
--------------------------------------------

Intel® SDC is available on the Anaconda Cloud ``intel/label/beta`` channel.
Distribution includes Intel® SDC for Python 3.6 and Python 3.7 for Windows and Linux platforms.

Intel® SDC conda package can be installed using the steps below::

    > conda create -n sdc-env python=<3.7 or 3.6> -c anaconda -c conda-forge
    > conda activate sdc-env
    > conda install sdc -c intel/label/beta -c intel -c defaults -c conda-forge --override-channels

Intel® SDC wheel package can be installed using the steps below::

    > conda create -n sdc-env python=<3.7 or 3.6> pip -c anaconda -c conda-forge
    > conda activate sdc-env
    > pip install --index-url https://pypi.anaconda.org/intel/label/beta/simple --extra-index-url https://pypi.anaconda.org/intel/simple --extra-index-url https://pypi.org/simple sdc


Building Intel® SDC from Source on Linux
----------------------------------------

We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up Intel® SDC build environment.

If you do not have conda, we recommend using Miniconda3::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH

.. note::
    For maximum performance and stability, please use numba from ``intel/label/beta`` channel.

It is possible to build Intel® SDC via conda-build or setuptools. Follow one of the
cases below to install Intel® SDC and its dependencies on Linux.

Building on Linux with conda-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    PYVER=<3.6 or 3.7>
    NUMPYVER=<1.16 or 1.17>
    conda create -n conda-build-env python=$PYVER conda-build
    source activate conda-build-env
    git clone https://github.com/IntelPython/sdc.git
    cd sdc
    conda build --python $PYVER --numpy $NUMPYVER --output-folder=<output_folder> -c intel/label/beta -c defaults -c intel -c conda-forge --override-channels conda-recipe

Building on Linux with setuptools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    export PYVER=<3.6 or 3.7>
    export NUMPYVER=<1.16 or 1.17>
    conda create -n sdc-env -q -y -c intel/label/beta -c defaults -c intel -c conda-forge python=$PYVER numpy=$NUMPYVER tbb-devel tbb4py numba=0.54.1 pandas=1.3.4 pyarrow=4.0.1 gcc_linux-64 gxx_linux-64
    source activate sdc-env
    git clone https://github.com/IntelPython/sdc.git
    cd sdc
    python setup.py install

In case of issues, reinstalling in a new conda environment is recommended.

Building Intel® SDC from Source on Windows
------------------------------------------

Building Intel® SDC on Windows requires Build Tools for Visual Studio 2019 (with component MSVC v140 - VS 2015 C++ build tools (v14.00)):

* Install `Build Tools for Visual Studio 2019 (with component MSVC v140 - VS 2015 C++ build tools (v14.00)) <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_.
* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda prompt'

It is possible to build Intel® SDC via conda-build or setuptools. Follow one of the
cases below to install Intel® SDC and its dependencies on Windows.

Building on Windows with conda-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    set PYVER=<3.6 or 3.7>
    set NUMPYVER=<1.16 or 1.17>
    conda create -n conda-build-env -q -y python=%PYVER% conda-build conda-verify vc vs2015_runtime vs2015_win-64
    conda activate conda-build-env
    git clone https://github.com/IntelPython/sdc.git
    cd sdc
    conda build --python %PYVER% --numpy %NUMPYVER% --output-folder=<output_folder> -c intel/label/beta -c defaults -c intel -c conda-forge --override-channels conda-recipe

Building on Windows with setuptools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    set PYVER=<3.6 or 3.7>
    set NUMPYVER=<1.16 or 1.17>
    conda create -n sdc-env -c intel/label/beta -c defaults -c intel -c conda-forge python=%PYVER% numpy=%NUMPYVER% tbb-devel tbb4py numba=0.54.1 pandas=1.3.4 pyarrow=4.0.1
    conda activate sdc-env
    set INCLUDE=%INCLUDE%;%CONDA_PREFIX%\Library\include
    set LIB=%LIB%;%CONDA_PREFIX%\Library\lib
    git clone https://github.com/IntelPython/sdc.git
    cd sdc
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
* Sometimes if the conda version or visual studio version being used are not latest then
  building Intel® SDC can throw some vague error about a keyword used in a file.
  So make sure you are using the latest versions.

Building documentation
----------------------

Building Intel® SDC User's Guide documentation requires pre-installed Intel® SDC package
along with compatible `Pandas*`_ version as well as `Sphinx*`_ 2.2.1 or later.

Intel® SDC documentation includes Intel® SDC examples output which is pasted to functions description in the API Reference.

Use ``pip`` to install `Sphinx*`_ and extensions:
::

    pip install sphinx sphinxcontrib-programoutput

Currently the build precedure is based on ``make`` located at ``./sdc/docs/`` folder.
While it is not generally required we recommended that you clean up the system from previous documentaiton build by running:
::

    make clean

To build HTML documentation you will need to run:
::

    make html

The built documentation will be located in the ``./sdc/docs/build/html`` directory.
To preview the documentation open ``index.html`` file.


More information about building and adding documentation can be found `here <docs/README.rst>`__.


Running unit tests
------------------
::

    python sdc/tests/gen_test_data.py
    python -m unittest

References
##########

Intel® SDC follows ideas and initial code base of High-Performance Analytics Toolkit (HPAT). These academic papers describe ideas and methods behind HPAT:

- `HPAT paper at ICS'17 <http://dl.acm.org/citation.cfm?id=3079099>`_
- `HPAT at HotOS'17 <http://dl.acm.org/citation.cfm?id=3103004>`_
- `HiFrames on arxiv <https://arxiv.org/abs/1704.02341>`_

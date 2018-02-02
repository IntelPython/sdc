*****
HPAT
*****

.. image:: https://travis-ci.org/IntelLabs/hpat.svg?branch=master
    :target: https://travis-ci.org/IntelLabs/hpat

.. image:: https://coveralls.io/repos/github/IntelLabs/hpat/badge.svg?branch=master
    :target: https://coveralls.io/github/IntelLabs/hpat?branch=master

A compiler-based framework for big data in Python
#################################################

High Performance Analytics Toolkit (HPAT) scales analytics/ML codes in Python
to bare-metal cluster/cloud performance automatically.
It compiles a subset of Python (Pandas/Numpy) to efficient parallel binaries
with MPI, requiring only minimal code changes.
HPAT is orders of magnitude faster than
alternatives like `Apache Spark <http://spark.apache.org/>`_.

HPAT's documentation can be found `here <https://intellabs.github.io/hpat/>`_.

Installation
############

HPAT can be installed in `Anaconda <https://www.anaconda.com/download/>`_
environment easily (Linux/Mac)::

    conda create -n HPAT python=3.6
    source activate HPAT
    conda install numpy scipy pandas
    conda install pyarrow=0.8.* mpich -c conda-forge
    conda install hpat -c ehsantn

Windows installaton requires
`Intel MPI <https://software.intel.com/en-us/intel-mpi-library>`_ to be
installed separately instead of `mpich`. The rest is the same::

    conda create -n HPAT python=3.6
    activate HPAT
    conda install numpy scipy pandas
    conda install pyarrow=0.8.* -c conda-forge
    conda install hpat -c ehsantn

Docker Container
----------------

An HPAT docker image is also available for running containers. For example::

    docker run -it ehsantn/hpat bash

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

    calc_pi(200000000)

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

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

Installation
############

HPAT can be installed in `Anaconda <https://www.anaconda.com/download/>`_
environment easily (Linux/Mac/Windows)::

    conda create -n HPAT -c ehsantn -c anaconda -c conda-forge hpat

.. used if master of Numba is needed for latest hpat package
.. conda create -n HPAT -c ehsantn -c numba/label/dev -c anaconda -c conda-forge hpat

Docker Container
----------------

An HPAT docker image is also available for running containers. For example::

    docker run -it ehsantn/hpat bash

Building HPAT from Source
-------------------------

To build HPAT from Source, please refer to the following `instrunction <docs/source/install.rst>`_

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

Jupyter Notebook
################

To use HPAT with Jupyter Notebook, install jupyter, ipython, and ipyparallel.
Then, create a mpi profile for ipython::

    ipython profile create --parallel --profile=mpi

Next, edit the ipcluster_config.py file.  This file will be in your ipython
directory in the profile_mpi directory.  Your ipython directory is in your
IPYTHONDIR environment variable if you have one defined and ~/.ipython if you
don't have this variable defined.  To the ipcluster_config.py file, add the
following line::

    c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'

Then, start the Jupyter notebook and click on IPython Clusters, select the
number of engines (i.e., cores) you'd like to use and click Start next to the
mpi profile.  You can now run an HPAT function and the work will be distributed
across the number of cores you selected on the current node.

If you wish to run across multiple nodes, you can add the following to
ipcluster_config.py::

    c.MPILauncher.mpi_args = ["-machinefile", "path_to_file/machinefile"]

This machinefile option is forwarded to mpi and the specified machine file in
the second argument contains a list of machine names across which to distribute work.
More information about the -machinefile option can be found 
`here <https://www.open-mpi.org/faq/?category=running#mpirun-hostfile>`_.

References
##########

These academic papers describe the underlying methods in HPAT:

- `HPAT paper at ICS'17 <http://dl.acm.org/citation.cfm?id=3079099>`_
- `HPAT at HotOS'17 <http://dl.acm.org/citation.cfm?id=3103004>`_
- `HiFrames on arxiv <https://arxiv.org/abs/1704.02341>`_

.. _overview:

Quick Introduction to HPAT
==========================

.. todo:: Change content here considering Developer's perspective

High Performance Analytics Toolkit (HPAT) is a big data analytics and machine
learning framework that provides Python's ease of use but is extremely fast.

HPAT scales analytics programs in python to cluster/cloud environments
automatically, requiring only minimal code changes. Here is a logistic
regression program using HPAT::

    @hpat.jit
    def logistic_regression(iterations):
        f = h5py.File("lr.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]
        D = X.shape[1]
        w = np.random.ranf(D)
        t1 = time.time()
        for i in range(iterations):
            z = ((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y)
            w -= np.dot(z, X)
        return w

This code runs on cluster and cloud environments using a simple command like::

    mpiexec -n 1024 python logistic_regression.py

HPAT compiles the source code to efficient native parallel code
(with `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_).
This is in contrast to other frameworks such as Apache Spark which are
master-executor libraries. Hence, HPAT is typically 100x or more faster.
HPAT is built on top of `Numba <https://github.com/numba/numba>`_
and `LLVM <https://llvm.org/>`_ compilers.

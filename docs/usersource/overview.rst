.. _overview:

What is HPAT?
==========================

HPAT is the extension of `Numba <http://numba.pydata.org/numba-doc/latest/user/overview.html>`_ that allows a just-in-time compilation of Python codes, which are the mix of `Pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ , `NumPy Array <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_ , and other numerical functions. 
 
Being the `Numba <http://numba.pydata.org/numba-doc/latest/user/overview.html>`_ extension, with the :func:`@jit <numba.jit>` and respective compilation options HPAT generates machine code using the `LLVM Compiler <http://llvm.org/docs/>`_ as well as can auto-parallelize the code.

On a single machine HPAT parallelism can be either multi-threading (based on TBB or `OpenMP <https://openmp.org>`_ ) or multi-processing. In addition HPAT can seamlessly scale to many nodes with `MPI <https://www.open-mpi.org/doc/>`_ , which allows to implement big data analysis workflows using familiar Python APIs such as `Pandas <http://pandas.pydata.org/>`_ and `Numpy <http://www.numpy.org/>`_ and distributed numerical and machine learning libraries, such as `daal4py <https://intelpython.github.io/daal4py/index.html>`_ . 


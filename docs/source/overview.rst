.. _overview:
.. include:: ./ext_links.txt

What is Intel® Scalable Dataframe Compiler?
===========================================

Intel® Scalable Dataframe Compiler (Intel® SDC) is an extension of
`Numba*`_ that allows just-in-time and ahead-of-time
compilation of Python codes, which are the mix of `Pandas*`_,
`NumPy*`_, and other numerical functions.
 
Being the Numba extension, with the ``@njit`` decorator and respective compilation options
Intel SDC generates machine code using the `LLVM* Compiler`_:

.. literalinclude:: ../../examples/basic_workflow.py
   :language: python
   :emphasize-lines: 9-10
   :lines: 27-
   :caption: Example 1: Compiling Basic Pandas* Workflow
   :name: ex_basic_workflow

On a single machine Intel SDC uses multi-threading (based on `Intel® TBB`_ or `OpenMP*`_ )
to parallelize `Pandas*`_ and `Numpy*`_ operations. To turn on the multi-threading you just need to add
``parallel=True`` option to ``@njit`` decorator:

.. literalinclude:: ../../examples/basic_workflow_parallel.py
   :language: python
   :emphasize-lines: 9-10
   :lines: 27-
   :caption: Example 2: Parallelizing `Pandas*`_ Workflow
   :name: ex_basic_workflow_parallel

.. note::
    Using the same ``@njit`` decorator the Intel SDC is designed to scale to many nodes automatically
    without the need to use frameworks like `Dask*`_ , `Ray*`_, and `Spark*`_.

    This feature is in active development, and will become available in a future Intel SDC release.

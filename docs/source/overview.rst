.. _overview:
.. include:: ./ext_links.txt

What is Intel® SDC?
===================

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
to parallelize `Pandas*`_ and `Numpy*`_ operations. Most of these operations are parallelized on function-level,
so that no extra action is required from users in most cases.

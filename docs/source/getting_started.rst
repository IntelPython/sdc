.. _getting_started:
.. include:: ./ext_links.txt

Getting Started
===============

Intel® Scalable Dataframe Compiler (Intel® SDC) extends capabilities of `Numba*`_ to compile a subset
of `Pandas*`_ into native code. Being integral part of `Numba*`_ it allows to combine regular `NumPy*`_ codes
with `Pandas*`_ operations.

Like in `Numba*`_ the compilation is controlled by a regular ``@njit`` decorator and respective compilation
directives which control its behavior.

The code below illustrates a typical workflow that Intel® SDC is intended to compile:

.. literalinclude:: ../../examples/basic_workflow.py
   :language: python
   :lines: 27-
   :caption: Example 1: Compiling Basic Pandas* Workflow
   :name: ex_getting_started_basic_workflow

The workflow typically starts with reading data from a file (or multiple files) into a dataframe (or multiple
dataframes) followed by data transformations of dataframes and/or individual columns, cleaning the data, grouping and
binning, and finally by feeding the cleaned data into machine learning algorithm for training or inference.

.. image:: ./_images/workflow.png
    :width: 526px
    :align: center
    :alt: Data analytics workflows

We also recommend to read `A ~5 minute guide to Numba <https://numba.pydata.org/numba-doc/dev/user/5minguide.html>`_
for getting started with `Numba*`_.

Installation
#############
You can use conda and pip package managers to install Intel® SDC into your `Python*`_ environment.

Intel SDC is available on the Anaconda Cloud intel/label/beta channel.
Distribution includes Intel SDC for Python 3.6 and 3.7 for Windows and Linux platforms.

Intel SDC conda package can be installed using the steps below:
::

    > conda create -n sdc_env python=<3.7 or 3.6>
    > conda activate sdc_env
    > conda install sdc -c intel/label/beta -c intel -c defaults -c conda-forge --override-channels

Intel SDC wheel package can be installed using the steps below:
::

    > conda create -n sdc_env python=<3.7 or 3.6> pip
    > conda activate sdc_env
    > pip install --index-url https://pypi.anaconda.org/intel/label/beta/simple --extra-index-url https://pypi.anaconda.org/intel/simple --extra-index-url https://pypi.org/simple sdc


Experienced users can also build Intel SDC from sources
`for Linux* <https://github.com/IntelPython/sdc#building-intel-sdc-from-source-on-linux>`_ and
`for Windows* <https://github.com/IntelPython/sdc#building-intel-sdc-from-source-on-windows>`_.
 
Basic Usage
###########
The code below illustrates a typical ML workflow that consists of data pre-processing and predicting stages.
Intel® SDC is intended to compile pre-processing stage that includes
reading dataset from a csv file, filtering data and performing Pearson correlation operation.
The prediction based on gradient boosting regression module is made using scikit-learn module.

.. literalinclude:: ../../examples/basic_usage_nyse_predict.py
   :language: python
   :lines: 27-
   :caption: Typical usage of Intel® SDC in combination with scikit-learn
   :name: ex_getting_started_basic_usage_nyse_predict

What If I Get A Compilation Error
#################################

.. todo::
   Need to give basic information that hpat and numba do not support full set of Pandas and Numpy APIs, provide the link to the API Reference section for Intel® SDC, relevant reference to Numba documentation.
 
Also give very short introduction to what kind of code Numba/Intel® SDC can compile and what cannot, i.e. type stability etc. Provide the links to relevant sections in Intel® SDC and Numba documentations focusing on compilation issues/limitations
  
Measuring Performance
#####################

.. 1. Short intro how to measure performance.

Lets consider we want to measure performance of Series.max() method.

.. code::

   from numba import njit

   @njit
   def series_max(s):
      return s.max()

.. 2. Compilation time and run time.

First, recall that Intel® SDC is based on Numba. Therefore, execution time may consist of the following:
   1. Numba has to *compile* your function for the first time, this takes time.
   2. *Boxing* and *unboxing* convert Python objects into native values, and vice-versa. They occur at the boundaries of calling a `Numba*`_ function from the Python interpreter. E.g. boxing and unboxing apply to `Pandas*`_ types like :ref:`Series <pandas.Series>` and :ref:`DataFrame <pandas.DataFrame>`.
   3. The execution of the *function itself*.

A really common mistake when measuring performance is to not account for the above behaviour and
to time code once with a simple timer that includes the time taken to compile your function in the execution time.

A good way to measure the impact Numba JIT has on your code is to time execution using
the `timeit <https://docs.python.org/3/library/timeit.html>`_ module functions.

Intel® SDC also recommends eliminate the impact of compilation and boxing/unboxing by measuring the time inside Numba JIT code.

.. 3. Illustrate by example.

Example of measuring performance:

.. code::

   import time
   import numpy as np
   import pandas as pd
   from numba import njit

   @njit
   def perf_series_max(s):                  # <-- unboxing
      start_time = time.time()              # <-- time inside Numba JIT code
      res = s.max()
      finish_time = time.time()             # <-- time inside Numba JIT code
      return finish_time - start_time, res  # <-- boxing

   s = pd.Series(np.random.ranf(size=100000))
   exec_time, res = perf_series_max(s)
   print("Execution time in JIT code: ", exec_time)

.. 4. Reference to relevant discussion in Numba documentation.

See also `Numba*`_ documentation `How to measure the performance of Numba? <http://numba.pydata.org/numba-doc/latest/user/5minguide.html#how-to-measure-the-performance-of-numba>`_

.. 5. Link to performance tests.

See also Intel® SDC repository `performance tests <https://github.com/IntelPython/sdc/tree/master/sdc/tests/tests_perf>`_.

What If I Get Poor Performance?
###############################

.. 1. Short introduction why performance may be slower than expected.
.. 2. GIL, Object mode and nopython mode.
.. 3. Overheads related to boxing and unboxing Python objects.
.. 4. Reference to relevant sections of Intel® SDC and Numba documentation for detailed discussion

If you get poor performance you need to consider several reasons, among which
compilation overheads, overheads related to converting Python objects to native structures and back,
amount of parallelism in compiled code, to what extent the code is “static” and many other factors.
See more details in Intel® SDC documentation :ref:`Getting Performance With Intel® SDC <performance>`.

Also you need to consider limitations of particular function.
See more details in Intel® SDC documentation for particular function :ref:`apireference`.

See also `Numba*`_ documentation `Performance Tips <http://numba.pydata.org/numba-doc/latest/user/performance-tips.html>`_
and `The compiled code is too slow <http://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#the-compiled-code-is-too-slow>`_.

Build Instructions
##################

Build instructions for Linux*: https://github.com/IntelPython/sdc#building-intel-sdc-from-source-on-linux
Build instructions for Windows*: https://github.com/IntelPython/sdc#building-intel-sdc-from-source-on-windows

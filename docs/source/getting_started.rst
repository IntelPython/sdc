.. _getting_started:
.. include:: ./ext_links.txt

Getting Started
===============

Intel® Scalable Dataframe Compiler (Intel® SDC) extends capabilities of Numba* to compile a subset
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
You can use conda and pip package managers to install Intel® SDC into your Python* environment.

.. todo::
    Provide installation instructions for public packages on Anaconda.org and PyPI

Experienced users can also buildIntel SDC from sources
`for Linux*<https://github.com/IntelPython/sdc#building-intel-sdc-from-source-on-linux>`_ and
`for Windows*<https://github.com/IntelPython/sdc#building-intel-sdc-from-source-on-windows>`_.
 
Basic Usage
###########
.. todo::
   Provide a few code snapshots illustrating typical usages of Intel® SDC:
    •	Reading a file
    •	Working with a column - a few basic ops, e.g. aggregation or sorting + UDF
    •	Working with a dataframe
    •	Working with a machine learning library, e.g. scikit-learn, xgboost, daal
    
    Each snapshot can have two flavors - serial and parallel to illustrate easiness of getting parallel performance.
Each code snapshot provides the link to full examples located at GitHub repo>

Here's an example which describes reading data from a csv file and performing basic operation like finding mean and
sorting values of a specific column:

What If I Get A Compilation Error
#################################

.. todo::
   Need to give basic information that hpat and numba do not support full set of Pandas and Numpy APIs, provide the link to the API Reference section for Intel® SDC, relevant reference to Numba documentation.
 
Also give very short introduction to what kind of code Numba/Intel® SDC can compile and what cannot, i.e. type stability etc. Provide the links to relevant sections in Intel® SDC and Numba documentations focusing on compilation issues/limitations
  
Measuring Performance
#####################

.. todo::
   Short intro how to measure performance. Compilation time and run time. Illustrate by example. Reference to relevant discussion in Numba documentation
 
What If I Get Poor Performance?
###############################

.. todo::
   Short introduction why performance may be slower than expected. GIL, Object mode and nopython mode. Overheads related to boxing and unboxing Python objects.
   Reference to relevant sections of Intel® SDC and Numba documentation for detailed discussion

Build Instructions
##################

.. todo::
   Provide step-by-step build instructions for Linux*, Windows*, and Mac*.

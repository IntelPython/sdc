.. _GettingStarted:

Getting Started with HPAT
~~~~~~~~~~~~~~~~~~~~~~~~~

HPAT is useful to accelerate a subset of `Python <https://docs.python.org/3/>`_ operations working with  `Pandas Series <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_ and `Dataframes <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ as well as with `Numpy Arrays <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_ . Being the just-in-time compiler built on top of `Numba <http://numba.pydata.org/numba-doc/latest/index.html>`_ HPAT will compile a subset of Pandas and Numpy codes into the native code. The compilation is controlled by a set of `Numba decorators <http://numba.pydata.org/numba-doc/0.8/modules/decorators.html>`_ and **HPAT decorators** that can be applied to a function.
 
The code below illustrates a typical workflow that HPAT is intended to compile

.. todo::
    Short code illustrating how hpat can compile read_csv and compute aggregators over columns
 
We also recommend to read `A ~5 minute guide to Numba <https://numba.pydata.org/numba-doc/dev/user/5minguide.html>`_ .
 
Installing HPAT
===============

.. todo::
         
   instructions how to install hpat using 1) conda, 2) pip
 
Experienced users can also compile HPAT  from sources<link to github build instructions for hpat>
 
How to use HPAT
================

.. todo::
   Provide a few code snapshots illustrating typical usages of HPAT:
    •	Reading a file
    •	Working with a column - a few basic ops, e.g. aggregation or sorting + UDF
    •	Working with a dataframe
    •	Working with a machine learning library, e.g. scikit-learn, xgboost, daal
    
    Each snapshot can have two flavors - serial and parallel to illustrate easiness of getting parallel performance. Each code snapshot provides the link to full examples located at GitHub repo>

Here's an example which describes reading data from a csv file and performing basic operation like finding mean and sorting values of a specific column:

.. literalinclude:: ../../examples/series_basic.py
   :language: python
   :linenos:
   :caption: series_basic
   :name: series_basic
   
Here's another simple example which uses merge and concat operations for Pandas Dataframes:

.. literalinclude:: ../../examples/Basic_DataFrame.py
   :language: python
   :linenos:
   :caption: merge_concat
   :name: merge_concat

   
What If I Get A Compilation Error
=================================

.. todo::
   Need to give basic information that hpat and numba do not support full set of Pandas and Numpy APIs, provide the link to the API Reference section for HPAT, relevant reference to Numba documentation.
 
Also give very short introduction to what kind of code Numba/HPAT can compile and what cannot, i.e. type stability etc. Provide the links to relevant sections in HPAT and Numba documentations focusing on compilation issues/limitations
  
Measuring HPAT performance
===========================

.. todo::
   Short intro how to measure performance. Compilation time and run time. Illustrate by example. Reference to relevant discussion in Numba documentation
 
What If I Get Poor Performance?
===============================

.. todo::
   Short introduction why performance may be slower than expected. GIL, Object mode and nopython mode. Overheads related to boxing and unboxing Python objects.
   Reference to relevant sections of HPAT and Numba documentation for detailed discussion

.. _compilation:

Compiling With Intel(R) SDC
~~~~~~~~~~~~~~~~~~~

.. todo::
     Basic compilation controls. What can be compiled and what cannot. How to work around compilation issues. References to relevant discussion in Numba. Specifics for Series, Dataframes, and other hpat specific data structures 
 
What if I get a compilation error
===================================

There are a few reasons why Intel(R) SDC cannot compile your code out-of-the-box. 
 
1.	Intel(R) SDC does support only a subset of Pandas APIs. 
2.	Intel(R) SDC and `Numba <http://numba.pydata.org/numba-doc/latest/index.html>`_ can compile only a subset of Python data types.
3.	Intel(R) SDC cannot infer the type of a variable at compile time.

Unsupported APIs
-----------------

Intel(R) SDC is able to compile variety of the most typical workflows that involve Pandas operations but not all. Sometimes it means that your code cannot be compiled out-of-the-box.
 
.. todo:: 
    Give an example here of unsupported Pandas API that cannot be compiled as is, e.g. pd.read_excel
 
You can work this around by <give the list of recommendations how to work around such a situation, e.g. getting the function out of jitted region, compilation with nopython=False, using alternative APIs in Pandas or NumPy. Each alternative needs to be illustrated by a code snippet>
 
<Provide the link to the API Reference section with the list of supported APIs and arguments>
 
Unsupported Data Types
------------------------

The other common reason why Intel(R) SDC or `Numba <http://numba.pydata.org/numba-doc/latest/index.html>`_ cannot compile the code is because it does not support a certain data type. <Any example?> You can work this around by using an alternative data type.

.. todo::
    Give examples with dictionaries or datetime, show how one type can be replaced with another
 
Type Inference And Type Stability
----------------------------------

The last but certainly not the least why Intel(R) SDC cannot compile your code is because it cannot infer the type at the time of compilation. The most frequent cause for that is the type instability. 
 
The static compilation is a powerful technology to obtain high efficiency of a code but the flip side is the compiler should be able to infer all variable types at the time of compilation and these types remain stable within the region being compiled.
 
The following is an example of the type-unstable variable a, and hence this code cannot be compiled by Intel(R) SDC::
   
   if flag:
       a = 1.0
   else:
       a = np.ones(10)

.. todo::
    Discuss the workaround, show the modified code
 
The use of isinstance operator often means type instability and is not supported. Similarly, function calls should also be deterministic. The below example is not supported since function f is not known in advance::

    if flag:
        f = np.zeros
    else:
        f = np.random.ranf
    A = f(10)

.. todo::
     Discuss the workaround, show the modified code
 
Discuss other typical scenarios when Numba or hpat cannot perform type inference
 
Dealing With Integer NaN Values
=================================

`Pandas Series <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_ are built upon `Numpy Arrays <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_ , which do not support NaN values for integers. For that reason Pandas dynamically converts integer columns to floating point ones when NaN values are needed. Intel(R) SDC can perform such a conversion only if enough information about NaN values is available at compilation time. When it is impossible the user is responsible for manual conversion of integer data to floating point data.
 
.. todo::
    Show example when hpat can infer NaNs in integer Series. Also show example where information about NaNs cannot be known at compile time and show how it can be worked around
 
Type Inference In I/O Operations
=================================

If the filename is constant, the Intel(R) SDC may be able to determine the file schema at compilation time. It will allow to perform type inference of columns in respective Pandas dataframe.
 
.. todo::
    Show example with reading file into dataframe when hpat can do type inferencing at compile time
 
If Intel(R) SDC  fails to infer types from the file, the schema must be manually specified.

.. todo::
    Show example how to manually specify the schema
 
Alternatively you can take file reading out of the compiled region, or you can try

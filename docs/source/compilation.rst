.. _compilation:
.. include:: ./ext_links.txt

Compiling With Intel® SDC
=========================

.. todo::
     Basic compilation controls. What can be compiled and what cannot. How to work around compilation issues.
     References to relevant discussion in `Numba*`_. Specifics for Series, Dataframes, and other hpat specific
     data structures
 
What if I get a compilation error
=================================

There are a few reasons why Intel SDC cannot compile your code out-of-the-box.
 
1.	Intel SDC does support only a subset of `Pandas*`_ APIs.
2.	Intel SDC and `Numba*`_ can compile only a subset of Python data types.
3.	Intel SDC cannot infer the type of a variable at compile time.

Unsupported APIs
-----------------

Intel® SDC is able to compile variety of the most typical workflows that involve `Pandas*`_ operations but not all.
Sometimes it means that your code cannot be compiled out-of-the-box.
 
.. todo:: 
    Give an example here of unsupported `Pandas*`_ API that cannot be compiled as is, e.g. pd.read_excel
 
.. todo::
    Give the list of recommendations how to work around such a situation,
    e.g. getting the function out of jitted region, compilation with nopython=False,
    using alternative APIs in `Pandas*`_ or `NumPy*`_. Each alternative needs to be illustrated by a code snippet
 
.. todo::
    Provide the link to the API Reference section with the list of supported APIs and arguments
 
Unsupported Data Types
------------------------

The other common reason why Intel® SDC or `Numba*`_ cannot compile the code is because it does not support
a certain data type. You can work this around by using an alternative data type.

.. todo::
    Give examples with dictionaries or datetime, show how one type can be replaced with another
 
Type Inference And Type Stability
----------------------------------

The last but certainly not the least why Intel® SDC cannot compile your code is because it cannot infer the type
at the time of compilation. The most frequent cause for that is the type instability.
 
The static compilation is a powerful technology to obtain high efficiency of a code but the flip side is the
compiler should be able to infer all variable types at the time of compilation and these types remain stable
within the region being compiled.
 
The following is an example of the type-unstable variable ``a``, and hence this code cannot
be compiled by `Numba*`_

.. code-block::
    :emphasize-lines: 2, 4

    if flag:
       a = 1.0
    else:
       a = np.ones(10)

.. todo::
    Discuss the workaround, show the modified code
 
The use of :func:`isinstance` function often means type instability and is not supported. Similarly, function calls
should also be deterministic. The below example is not supported since the function :func:`f` is not known in advance:

.. code-block::
    :emphasize-lines: 2, 4

    if flag:
        f = np.zeros
    else:
        f = np.random.ranf
    a = f(10)

.. todo::
     Discuss the workaround, show the modified code
     Discuss other typical scenarios when Numba or hpat cannot perform type inference
 
Dealing With Integer NaN Values
=================================

The :py:class:`pandas.Series` are built upon :py:class:`numpy.array`, which does not support
``NaN`` values for integers. For that reason `Pandas*`_ dynamically converts integer columns to floating point ones
when ``NaN`` values are needed. Intel SDC can perform such a conversion only if enough information about
``NaN`` values is available at compilation time. When it is impossible the user is responsible for manual
conversion of integer data to floating point data.
 
.. todo::
    Show example when Intel SDC can infer ``Nan`` in integer Series. Also show example where information about
    ``NaN`` cannot be known at compile time and show how it can be worked around
 
Type Inference In I/O Operations
=================================

If the filename is constant, the Intel SDC may be able to determine file schema at compilation time. It will allow
to perform type inference of columns in respective `Pandas*`_ dataframe.
 
.. todo::
    Show example with reading file into dataframe when Intel SDC can do type inferencing at compile time
 
If Intel SDC fails to infer types from the file, the schema must be manually specified.

.. todo::
    Show example how to manually specify the schema
 
Alternatively you can take file reading out of the compiled region.

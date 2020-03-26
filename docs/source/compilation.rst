.. _compilation:
.. include:: ./ext_links.txt

Compiling With Intel® SDC
=========================

What if I get a compilation error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a few reasons why Intel SDC cannot compile your code out-of-the-box.

1.	Intel SDC does support only a subset of `Pandas*`_ APIs.
2.	Intel SDC and `Numba*`_ can compile only a subset of Python data types.
3.	Intel SDC cannot infer the type of a variable at compile time.

Unsupported APIs
-----------------

Intel® SDC is able to compile variety of the most typical workflows that involve `Pandas*`_ operations but not all.
Sometimes it means that your code cannot be compiled out-of-the-box:

.. code-block::

    import numba
    import pandas

    @numba.njit
    def read_df(filename):
        return pandas.read_excel(filename)

    read_df("data.xlsx")

Output:
::

    Traceback (most recent call last):
    ...
    numba.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
    Unknown attribute 'read_excel' of type Module(<module 'pandas' from ...)


In such case you have the following options:

* Replace unsupported function with similar ones which are supported
  (e.g. use :ref:`pandas.read_csv <pandas.read_csv>` instead of :ref:`pandas.read_excel <pandas.read_csv>`):

.. code-block::

    import numba
    import pandas

    @numba.njit
    def read_df():
        return pandas.read_csv("data.csv")

    read_df()

* Use `Numba*`_ `objmode <https://numba.pydata.org/numba-doc/latest/user/withobjmode.html>`_:

.. code-block::

    import numba
    import pandas

    @numba.njit
    def cummax():
        s = pandas.Series([0, 1, 0, 2, 0, 3, 0, 4])

        with numba.objmode(r='intp[:]'):
            r = s.cummax().values

        return pandas.Series(r)

Please note, that an array is returned from objmode. Returning Series or DataFrame from objmode is not a trivial task.

* Exclude such calls from jit region:

.. code-block::

    import numba
    import pandas

    def cummax():
        @numba.njit
        def create_series():
            return pandas.Series([0, 1, 0, 2, 0, 3, 0, 4])

        s = create_series()

        return s.cummax()


Please note that last two options would result in performing boxing/unboxing which could significantly affect performance.

For more details on performance see :ref:`Getting Performance With Intel® SDC <performance>`

For list of supported functions see :ref:`API Reference <apireference>`

Unsupported Data Types
------------------------

The other common reason why Intel® SDC or `Numba*`_ cannot compile the code is because it does not support
a certain data type. e.g. `Numba*`_ doesn't support heterogeneous lists and dicts:

.. code-block::

    a = [0, 2, 5, "a", "b"]

Literal heterogeneous lists usually could be replaced with tuples:

.. code-block::

    a = (0, 2, 5, "a", "b")

While heterogeneous dicts are not supported, it could be passed as parameter to :ref:`pandas.DataFrame <pandas.dataframe>`
or :ref:`pandas.read_csv <pandas.read_csv>`:

.. code-block::

    data = {'A': np.ranf(10), 'B': np.ones(10)}
    df = pandas.DataFrame(data=data)


Intel® SDC supports :ref:`pandas.Series <pandas.series>` only of boolean, integer, float and string types.
Other types like Series of datetime or categorical are not supported.


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

The use of :func:`isinstance` function often means type instability and is not supported. Similarly, function calls
should also be deterministic. The below example is not supported since the function :func:`f` is not known in advance:

.. code-block::
    :emphasize-lines: 2, 4

    if flag:
        f = np.zeros
    else:
        f = np.random.ranf
    a = f(10)

Dealing With Integer NaN Values
-------------------------------

The :py:class:`pandas.Series` are built upon :py:class:`numpy.ndarray`, which does not support
``NaN`` values for integers and booleans. For that reason `Pandas*`_ dynamically converts integer columns to floating point ones
when ``NaN`` values are needed. Intel SDC doesn't perform such conversion and it is user responsibility to manually
convert from integer data to floating point data.


Type Inference In I/O Operations
--------------------------------

If the filename is constant, the Intel SDC may be able to determine file schema at compilation time. It will allow
to perform type inference of columns in respective `Pandas*`_ dataframe.

.. code-block::

    df = pandas.read_csv("data.csv")

If Intel SDC fails to infer types from the file, the schema must be manually specified.

.. code-block::

    names = ['A', 'B']
    usecols = ['A']
    dtypes={'A': np.float64}
    pd.read_csv(file_name, names=names, usecols=usecols, dtype=dtypes)

Alternatively you can take file reading out of the compiled region.

Note: if data file contains integer data with empty positions (Nans) it is highly recommended to manually specify column type to float.

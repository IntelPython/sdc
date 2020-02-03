.. _pandas.Series.apply:

:orphan:

pandas.Series.apply
*******************

Invoke function on values of Series.

Can be ufunc (a NumPy function that applies to the entire Series)
or a Python function that only works on single values.

:param func:
    function
        Python function or NumPy ufunc to apply.

:param convert_dtype:
    bool, default True
        Try to find better dtype for elementwise function results. If
        False, leave as dtype=object.

:param args:
    tuple
        Positional arguments passed to func after the series value.
        \*\*kwds
        Additional keyword arguments passed to func.

:return: Series or DataFrame
    If func returns a Series object the result will be a DataFrame.

Limitations
-----------
`convert_dtype`, `args` and `\*\*kwds` are currently unsupported by Intel Scalable Dataframe Compiler.

Examples
--------
.. literalinclude:: ../../../examples/series/series_apply.py
    :language: python
    :lines: 33-
    :caption: Square the values by defining a function and passing it as an argument to `apply()`.
    :name: ex_series_apply

.. command-output:: python ./series/series_apply.py
    :cwd: ../../../examples

.. literalinclude:: ../../../examples/series/series_apply_lambda.py
    :language: python
    :lines: 33-
    :caption: Square the values by passing an anonymous function as an argument to `apply()`.
    :name: ex_series_apply_lambda

.. command-output:: python ./series/series_apply_lambda.py
    :cwd: ../../../examples

.. literalinclude:: ../../../examples/series/series_apply_log.py
    :language: python
    :lines: 33-
    :caption: Use a function from the Numpy library.
    :name: ex_series_apply_log

.. command-output:: python ./series/series_apply_log.py
    :cwd: ../../../examples

.. seealso::

    :ref:`Series.map <pandas.Series.map>`
        For element-wise operations.
    :ref:`Series.agg <pandas.Series.agg>`
        Only perform aggregating type operations.
    :ref:`Series.transform <pandas.transform>`
        Only perform transforming type operations.


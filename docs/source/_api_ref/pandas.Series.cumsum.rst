.. _pandas.Series.cumsum:

:orphan:

pandas.Series.cumsum
********************

Return cumulative sum over a DataFrame or Series axis.

Returns a DataFrame or Series of the same size containing the cumulative
sum.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The index or the name of the axis. 0 is equivalent to None or 'index'.

:param skipna:
    boolean, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA.
        \*args, \*\*kwargs :
        Additional keywords have no effect but might be accepted for
        compatibility with NumPy.

:return: scalar or Series

Examples
--------
.. literalinclude:: ../../../examples/series/series_cumsum.py
   :language: python
   :lines: 27-
   :caption: Return cumulative sum over a DataFrame or Series axis.
   :name: ex_series_cumsum

.. command-output:: python ./series/series_cumsum.py
   :cwd: ../../../examples

.. note::
    Parameter axis is currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Expanding.sum.html#pandas.core.window.Expanding.sum>`_
        Similar functionality but ignores NaN values.

    :ref:`Series.sum <pandas.Series.sum>`
        Return the sum over Series axis.

    :ref:`Series.cummax <pandas.Series.cummax>`
        Return cumulative maximum over Series axis.

    :ref:`Series.cummin <pandas.Series.cummin>`
        Return cumulative minimum over Series axis.

    :ref:`Series.cumprod <pandas.Series.cumprod>`
        Return cumulative product over Series axis.


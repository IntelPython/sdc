.. _pandas.Series.mean:

:orphan:

pandas.Series.mean
******************

Return the mean of the values for the requested axis.

:param axis:
    {index (0)}
        Axis for the function to be applied on.

:param skipna:
    bool, default True
        Exclude NA/null values when computing the result.

:param level:
    int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.

:param numeric_only:
    bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
        \*\*kwargs
        Additional keyword arguments to be passed to the function.

:return: scalar or Series (if level specified)

Limitations
-----------
- Parameters level, numeric_only are currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_mean.py
   :language: python
   :lines: 27-
   :caption: Return the mean of the values for the requested axis.
   :name: ex_series_mean

.. command-output:: python ./series/series_mean.py
   :cwd: ../../../examples

.. note::
    Parameter axis is currently unsupported by Intel Scalable Dataframe Compiler


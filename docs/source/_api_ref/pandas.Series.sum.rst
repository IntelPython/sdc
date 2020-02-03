.. _pandas.Series.sum:

:orphan:

pandas.Series.sum
*****************

Return the sum of the values for the requested axis.

            This is equivalent to the method ``numpy.sum``.

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

:param min_count:
    int, default 0
        The required number of valid values to perform the operation. If fewer than
        ``min_count`` non-NA values are present the result will be NA.

        .. versionadded :: 0.22.0

        Added with the default being 0. This means the sum of an all-NA
        or empty Series is 0, and the product of an all-NA or empty
        Series is 1.
        \*\*kwargs
        Additional keyword arguments to be passed to the function.

:return: scalar or Series (if level specified)

Limitations
-----------
- Parameters level, numeric_only, min_count are currently unsupported by Intel Scalable Dataframe Compiler

Examples
--------
.. literalinclude:: ../../../examples/series/series_sum.py
   :language: python
   :lines: 27-
   :caption: Return the sum of the values for the requested axis.
   :name: ex_series_sum

.. command-output:: python ./series/series_sum.py
   :cwd: ../../../examples

.. note::
    Parameter axis is currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    :ref:`Series.sum <pandas.Series.sum>`
        Return the sum.

    :ref:`Series.min <pandas.Series.min>`
        Return the minimum.

    :ref:`Series.max <pandas.Series.max>`
        Return the maximum.

    :ref:`DataFrame.idxmin <pandas.DataFrame.idxmin>`
        Return the index of the minimum over the requested axis.

    :ref:`DataFrame.idxmax <pandas.DataFrame.idxmax>`
        Return index of first occurrence of maximum over requested axis.


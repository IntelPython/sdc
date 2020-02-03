.. _pandas.Series.min:

:orphan:

pandas.Series.min
*****************

Return the minimum of the values for the requested axis.

            If you want the *index* of the minimum, use ``idxmin``. This is
            the equivalent of the ``numpy.ndarray`` method ``argmin``.

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

Examples
--------
.. literalinclude:: ../../../examples/series/series_min.py
   :language: python
   :lines: 27-
   :caption: Getting the minimum value of Series elements
   :name: ex_series_min

.. command-output:: python ./series/series_min.py
   :cwd: ../../../examples

.. note::
    Parameters axis, level, numeric_only are currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    :ref:`Series.sum <pandas.Series.sum>`
        Return the sum.

    :ref:`Series.min <pandas.Series.min>`
        Return the minimum.

    :ref:`Series.max <pandas.Series.max>`
        Return the maximum.

    :ref:`Series.idxmin <pandas.Series.idxmin>`
        Return the index of the minimum.

    :ref:`Series.idxmax <pandas.Series.idxmax>`
        Return the index of the maximum.


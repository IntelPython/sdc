.. _pandas.Series.idxmax:

:orphan:

pandas.Series.idxmax
********************

Return the row label of the maximum value.

If multiple values equal the maximum, the first row label with that
value is returned.

:param skipna:
    bool, default True
        Exclude NA/null values. If the entire Series is NA, the result
        will be NA.

:param axis:
    int, default 0
        For compatibility with DataFrame.idxmax. Redundant for application
        on Series.
        \*args, \*\*kwargs
        Additional keywords have no effect but might be accepted
        for compatibility with NumPy.

:return: Index
    Label of the maximum value.

:raises:
    ValueError
        If the Series is empty.

Examples
--------
.. literalinclude:: ../../../examples/series/series_idxmax.py
   :language: python
   :lines: 27-
   :caption: Getting the row label of the maximum value.
   :name: ex_series_idxmax

.. command-output:: python ./series/series_idxmax.py
   :cwd: ../../../examples

.. note::
    Parameter axis is currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    :ref:`Series.idxmin <pandas.Series.idxmin>`
        Return index label of the first occurrence of minimum of values.

    `numpy.absolute <https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html#numpy.argmax>`_
        Return indices of the maximum values along the given axis.

    :ref:`DataFrame.idxmax <pandas.DataFrame.idxmax>`
        Return index of first occurrence of maximum over requested axis.


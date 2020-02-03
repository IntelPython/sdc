.. _pandas.Series.idxmin:

:orphan:

pandas.Series.idxmin
********************

Return the row label of the minimum value.

If multiple values equal the minimum, the first row label with that
value is returned.

:param skipna:
    bool, default True
        Exclude NA/null values. If the entire Series is NA, the result
        will be NA.

:param axis:
    int, default 0
        For compatibility with DataFrame.idxmin. Redundant for application
        on Series.
        \*args, \*\*kwargs
        Additional keywords have no effect but might be accepted
        for compatibility with NumPy.

:return: Index
    Label of the minimum value.

:raises:
    ValueError
        If the Series is empty.

Examples
--------
.. literalinclude:: ../../../examples/series/series_idxmin.py
   :language: python
   :lines: 27-
   :caption: Getting the row label of the minimum value.
   :name: ex_series_idxmin

.. command-output:: python ./series/series_idxmin.py
   :cwd: ../../../examples

.. note::
    Parameter axis is currently unsupported by Intel Scalable Dataframe Compiler

.. seealso::

    :ref:`Series.idxmax <pandas.Series.idxmax>`
        Return index label of the first occurrence of maximum of values.

    `numpy.absolute <https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html#numpy.argmin>`_
        Return indices of the minimum values along the given axis.

    :ref:`DataFrame.idxmin <pandas.DataFrame.idxmin>`
        Return index of first occurrence of minimum over requested axis.


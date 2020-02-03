.. _pandas.Series.astype:

:orphan:

pandas.Series.astype
********************

Cast a pandas object to a specified dtype ``dtype``.

:param dtype:
    data type, or dict of column name -> data type
        Use a numpy.dtype or Python type to cast entire pandas object to
        the same type. Alternatively, use {col: dtype, ...}, where col is a
        column label and dtype is a numpy.dtype or Python type to cast one
        or more of the DataFrame's columns to column-specific types.

:param copy:
    bool, default True
        Return a copy when ``copy=True`` (be very careful setting
        ``copy=False`` as changes to values then may propagate to other
        pandas objects).

:param errors:
    {'raise', 'ignore'}, default 'raise'
        Control raising of exceptions on invalid data for provided dtype.

:param - ``raise``:
    allow exceptions to be raised

:param - ``ignore``:
    suppress exceptions. On error return original object

        .. versionadded:: 0.20.0

:param kwargs:
    keyword arguments to pass on to the constructor

:return: casted : same type as caller

Examples
--------
.. literalinclude:: ../../../examples/series/series_astype.py
   :language: python
   :lines: 27-
   :caption: Cast a pandas object to a specified dtype dtype.
   :name: ex_series_astype

.. command-output:: python ./series/series_astype.py
   :cwd: ../../../examples

.. seealso::

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html#pandas.to_datetime>`_
        Convert argument to datetime.

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html#pandas.to_timedelta>`_
        Convert argument to timedelta.

    `pandas.absolute
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html#pandas.to_numeric>`_
        Convert argument to a numeric type.

    `numpy.absolute
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype>`_
        Copy of the array, cast to a specified type.


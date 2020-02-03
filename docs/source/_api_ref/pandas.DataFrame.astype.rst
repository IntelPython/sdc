.. _pandas.DataFrame.astype:

:orphan:

pandas.DataFrame.astype
***********************

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



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.Series.argmin:

:orphan:

pandas.Series.argmin
********************

Return the row label of the minimum value.

.. deprecated:: 0.21.0

The current behaviour of 'Series.argmin' is deprecated, use 'idxmin'
instead.
The behavior of 'argmin' will be corrected to return the positional
minimum in the future. For now, use 'series.values.argmin' or
'np.argmin(np.array(values))' to get the position of the minimum
row.

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



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


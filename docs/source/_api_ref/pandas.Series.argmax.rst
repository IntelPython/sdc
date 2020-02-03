.. _pandas.Series.argmax:

:orphan:

pandas.Series.argmax
********************

Return the row label of the maximum value.

.. deprecated:: 0.21.0

The current behaviour of 'Series.argmax' is deprecated, use 'idxmax'
instead.
The behavior of 'argmax' will be corrected to return the positional
maximum in the future. For now, use 'series.values.argmax' or
'np.argmax(np.array(values))' to get the position of the maximum
row.

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



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.DataFrame.idxmin:

:orphan:

pandas.DataFrame.idxmin
***********************

Return index of first occurrence of minimum over requested axis.
NA/null values are excluded.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        0 or 'index' for row-wise, 1 or 'columns' for column-wise

:param skipna:
    boolean, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA.

:return: Series
    Indexes of minima along the specified axis.

:raises:
    ValueError
        - If the row/column is empty



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.Series.cummin:

:orphan:

pandas.Series.cummin
********************

Return cumulative minimum over a DataFrame or Series axis.

Returns a DataFrame or Series of the same size containing the cumulative
minimum.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The index or the name of the axis. 0 is equivalent to None or 'index'.

:param skipna:
    boolean, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA.
        \*args, \*\*kwargs :
        Additional keywords have no effect but might be accepted for
        compatibility with NumPy.

:return: scalar or Series



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


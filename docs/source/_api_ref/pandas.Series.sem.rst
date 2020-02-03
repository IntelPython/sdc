.. _pandas.Series.sem:

:orphan:

pandas.Series.sem
*****************

Return unbiased standard error of the mean over requested axis.

Normalized by N-1 by default. This can be changed using the ddof argument

:param axis:
    {index (0)}

:param skipna:
    bool, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA

:param level:
    int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar

:param ddof:
    int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.

:param numeric_only:
    bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.

:return: scalar or Series (if level specified)



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


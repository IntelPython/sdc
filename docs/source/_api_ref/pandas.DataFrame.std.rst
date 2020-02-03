.. _pandas.DataFrame.std:

:orphan:

pandas.DataFrame.std
********************

Return sample standard deviation over requested axis.

Normalized by N-1 by default. This can be changed using the ddof argument

:param axis:
    {index (0), columns (1)}

:param skipna:
    bool, default True
        Exclude NA/null values. If an entire row/column is NA, the result
        will be NA

:param level:
    int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a Series

:param ddof:
    int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.

:param numeric_only:
    bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.

:return: Series or DataFrame (if level specified)



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


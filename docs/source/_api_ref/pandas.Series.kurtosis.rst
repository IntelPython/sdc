.. _pandas.Series.kurtosis:

:orphan:

pandas.Series.kurtosis
**********************

Return unbiased kurtosis over requested axis using Fisher's definition of
kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

:param axis:
    {index (0)}
        Axis for the function to be applied on.

:param skipna:
    bool, default True
        Exclude NA/null values when computing the result.

:param level:
    int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.

:param numeric_only:
    bool, default None
        Include only float, int, boolean columns. If None, will attempt to use
        everything, then use only numeric data. Not implemented for Series.
        \*\*kwargs
        Additional keyword arguments to be passed to the function.

:return: scalar or Series (if level specified)



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


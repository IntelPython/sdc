.. _pandas.DataFrame.corrwith:

:orphan:

pandas.DataFrame.corrwith
*************************

Compute pairwise correlation between rows or columns of DataFrame
with rows or columns of Series or DataFrame.  DataFrames are first
aligned along both axes before computing the correlations.

:param other:
    DataFrame, Series
        Object with which to compute correlations.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        0 or 'index' to compute column-wise, 1 or 'columns' for row-wise.

:param drop:
    bool, default False
        Drop missing indices from result.

:param method:
    {'pearson', 'kendall', 'spearman'} or callable

:param \* pearson:
    standard correlation coefficient

:param \* kendall:
    Kendall Tau correlation coefficient

:param \* spearman:
    Spearman rank correlation
        - callable: callable with input two 1d ndarrays
            and returning a float

        .. versionadded:: 0.24.0

:return: Series
    Pairwise correlations.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


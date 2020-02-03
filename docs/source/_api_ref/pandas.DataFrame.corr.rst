.. _pandas.DataFrame.corr:

:orphan:

pandas.DataFrame.corr
*********************

Compute pairwise correlation of columns, excluding NA/null values.

:param method:
    {'pearson', 'kendall', 'spearman'} or callable

:param \* pearson:
    standard correlation coefficient

:param \* kendall:
    Kendall Tau correlation coefficient

:param \* spearman:
    Spearman rank correlation
        - callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior
            .. versionadded:: 0.24.0

:param min_periods:
    int, optional
        Minimum number of observations required per pair of columns
        to have a valid result. Currently only available for Pearson
        and Spearman correlation.

:return: DataFrame
    Correlation matrix.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.Series.cov:

:orphan:

pandas.Series.cov
*****************

Compute covariance with Series, excluding missing values.

:param other:
    Series
        Series with which to compute the covariance.

:param min_periods:
    int, optional
        Minimum number of observations needed to have a valid result.

:return: float
    Covariance between Series and other normalized by N-1
    (unbiased estimator).



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


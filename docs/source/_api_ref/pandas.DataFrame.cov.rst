.. _pandas.DataFrame.cov:

:orphan:

pandas.DataFrame.cov
********************

Compute pairwise covariance of columns, excluding NA/null values.

Compute the pairwise covariance among the series of a DataFrame.
The returned data frame is the `covariance matrix
<https://en.wikipedia.org/wiki/Covariance_matrix>`__ of the columns
of the DataFrame.

Both NA and null values are automatically excluded from the
calculation. (See the note below about bias from missing values.)
A threshold can be set for the minimum number of
observations for each value created. Comparisons with observations
below this threshold will be returned as ``NaN``.

This method is generally used for the analysis of time series data to
understand the relationship between different measures
across time.

:param min_periods:
    int, optional
        Minimum number of observations required per pair of columns
        to have a valid result.

:return: DataFrame
    The covariance matrix of the series of the DataFrame.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


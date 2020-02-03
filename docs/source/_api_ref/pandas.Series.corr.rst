.. _pandas.Series.corr:

:orphan:

pandas.Series.corr
******************

Compute correlation with `other` Series, excluding missing values.

:param other:
    Series
        Series with which to compute the correlation.

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
        Minimum number of observations needed to have a valid result.

:return: float
    Correlation with other.

Limitations
-----------
- 'method' parameter accepts only 'pearson' value. Other values are not supported
- Unsupported mixed numeric and string data

Examples
--------
.. literalinclude:: ../../../examples/series/series_corr.py
   :language: python
   :lines: 27-
   :caption: Compute correlation with other Series, excluding missing values.
   :name: ex_series_copy

.. command-output:: python ./series/series_corr.py
   :cwd: ../../../examples


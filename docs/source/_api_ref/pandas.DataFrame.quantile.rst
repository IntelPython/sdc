.. _pandas.DataFrame.quantile:

:orphan:

pandas.DataFrame.quantile
*************************

Return values at the given quantile over requested axis.

:param q:
    float or array-like, default 0.5 (50% quantile)
        Value between 0 <= q <= 1, the quantile(s) to compute.

:param axis:
    {0, 1, 'index', 'columns'} (default 0)
        Equals 0 or 'index' for row-wise, 1 or 'columns' for column-wise.

:param numeric_only:
    bool, default True
        If False, the quantile of datetime and timedelta data will be
        computed as well.

:param interpolation:
    {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

        - linear: `i + (j - i) \* fraction`, where `fraction` is the
            fractional part of the index surrounded by `i` and `j`.
        - lower: `i`.
        - higher: `j`.
        - nearest: `i` or `j` whichever is nearest.
        - midpoint: (`i` + `j`) / 2.

        .. versionadded:: 0.18.0

:return: Series or DataFrame

    If ``q`` is an array, a DataFrame will be returned where the
    index is ``q``, the columns are the columns of self, and the
    values are the quantiles.
    If ``q`` is a float, a Series will be returned where the
    index is the columns of self and the values are the quantiles.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


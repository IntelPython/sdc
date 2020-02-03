.. _pandas.DataFrame.reindex:

:orphan:

pandas.DataFrame.reindex
************************

Conform DataFrame to new index with optional filling logic, placing
NA/NaN in locations having no value in the previous index. A new object
is produced unless the new index is equivalent to the current one and
``copy=False``.

:param labels:
    array-like, optional
        New labels / index to conform the axis specified by 'axis' to.

:param index, columns:
    array-like, optional
        New labels / index to conform to, should be specified using
        keywords. Preferably an Index object to avoid duplicating data

:param axis:
    int or str, optional
        Axis to target. Can be either the axis name ('index', 'columns')
        or number (0, 1).

:param method:
    {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}
        Method to use for filling holes in reindexed DataFrame.
        Please note: this is only applicable to DataFrames/Series with a
        monotonically increasing/decreasing index.

        - None (default): don't fill gaps
        - pad / ffill: propagate last valid observation forward to next
            valid
        - backfill / bfill: use next valid observation to fill gap
        - nearest: use nearest valid observations to fill gap

:param copy:
    bool, default True
        Return a new object, even if the passed indexes are the same.

:param level:
    int or name
        Broadcast across a level, matching Index values on the
        passed MultiIndex level.

:param fill_value:
    scalar, default np.NaN
        Value to use for missing values. Defaults to NaN, but can be any
        "compatible" value.

:param limit:
    int, default None
        Maximum number of consecutive elements to forward or backward fill.

:param tolerance:
    optional
        Maximum distance between original and new labels for inexact
        matches. The values of the index at the matching locations most
        satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

        Tolerance may be a scalar value, which applies the same tolerance
        to all values, or list-like, which applies variable tolerance per
        element. List-like includes list, tuple, array, Series, and must be
        the same size as the index and its dtype must exactly match the
        index's type.

        .. versionadded:: 0.21.0 (list-like tolerance)

:return: DataFrame with changed index.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


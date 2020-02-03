.. _pandas.Series.reindex_like:

:orphan:

pandas.Series.reindex_like
**************************

Return an object with matching indices as other object.

Conform the object to the same index on all axes. Optional
filling logic, placing NaN in locations having no value
in the previous index. A new object is produced unless the
new index is equivalent to the current one and copy=False.

:param other:
    Object of the same data type
        Its row and column indices are used to define the new indices
        of this object.

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

:param limit:
    int, default None
        Maximum number of consecutive labels to fill for inexact matches.

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

:return: Series or DataFrame
    Same type as caller, but with changed indices on each axis.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.Series.drop:

:orphan:

pandas.Series.drop
******************

Return Series with specified index labels removed.

Remove elements of a Series based on specifying the index labels.
When using a multi-index, labels on different levels can be removed
by specifying the level.

:param labels:
    single label or list-like
        Index labels to drop.

:param axis:
    0, default 0
        Redundant for application on Series.

:param index, columns:
    None
        Redundant for application on Series, but index can be used instead
        of labels.

        .. versionadded:: 0.21.0

:param level:
    int or level name, optional
        For MultiIndex, level for which the labels will be removed.

:param inplace:
    bool, default False
        If True, do operation inplace and return None.

:param errors:
    {'ignore', 'raise'}, default 'raise'
        If 'ignore', suppress error and only existing labels are dropped.

:return: Series
    Series with specified index labels removed.

:raises:
    KeyError
        If none of the labels are found in the index.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


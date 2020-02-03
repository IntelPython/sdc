.. _pandas.Series.cat.rename_categories:

:orphan:

pandas.Series.cat.rename_categories
***********************************

Rename categories.

:param new_categories:
    list-like, dict-like or callable

        - list-like: all items must be unique and the number of items in
            the new categories must match the existing number of categories.

        - dict-like: specifies a mapping from
            old categories to new. Categories not contained in the mapping
            are passed through and extra categories in the mapping are
            ignored.

        .. versionadded:: 0.21.0

:param \* callable:
    a callable that is called on all items in the old
        categories and whose return values comprise the new categories.

        .. versionadded:: 0.23.0

:param inplace:
    bool, default False
        Whether or not to rename the categories inplace or return a copy of
        this categorical with renamed categories.

:return: cat : Categorical or None
    With ``inplace=False``, the new categorical is returned.
    With ``inplace=True``, there is no return value.

:raises:
    ValueError
        If new categories are list-like and do not have the same number of
        items than the current categories or do not validate as categories



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


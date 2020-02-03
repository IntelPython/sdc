.. _pandas.Series.cat.remove_categories:

:orphan:

pandas.Series.cat.remove_categories
***********************************

Remove the specified categories.

`removals` must be included in the old categories. Values which were in
the removed categories will be set to NaN

:param removals:
    category or list of categories
        The categories which should be removed.

:param inplace:
    bool, default False
        Whether or not to remove the categories inplace or return a copy of
        this categorical with removed categories.

:return: cat : Categorical with removed categories or None if inplace.

:raises:
    ValueError
        If the removals are not contained in the categories



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


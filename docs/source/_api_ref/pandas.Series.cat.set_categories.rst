.. _pandas.Series.cat.set_categories:

:orphan:

pandas.Series.cat.set_categories
********************************

Set the categories to the specified new_categories.

`new_categories` can include new categories (which will result in
unused categories) or remove old categories (which results in values
set to NaN). If `rename==True`, the categories will simple be renamed
(less or more items than in old categories will result in values set to
NaN or in unused categories respectively).

This method can be used to perform more than one action of adding,
removing, and reordering simultaneously and is therefore faster than
performing the individual steps via the more specialised methods.

On the other hand this methods does not do checks (e.g., whether the
old categories are included in the new categories on a reorder), which
can result in surprising changes, for example when using special string
dtypes on python3, which does not considers a S1 string equal to a
single char python string.

:param new_categories:
    Index-like
        The categories in new order.

:param ordered:
    bool, default False
        Whether or not the categorical is treated as a ordered categorical.
        If not given, do not change the ordered information.

:param rename:
    bool, default False
        Whether or not the new_categories should be considered as a rename
        of the old categories or as reordered categories.

:param inplace:
    bool, default False
        Whether or not to reorder the categories in-place or return a copy
        of this categorical with reordered categories.

:return: Categorical with reordered categories or None if inplace.

:raises:
    ValueError
        If new_categories does not validate as categories



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


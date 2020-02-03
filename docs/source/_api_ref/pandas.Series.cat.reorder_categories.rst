.. _pandas.Series.cat.reorder_categories:

:orphan:

pandas.Series.cat.reorder_categories
************************************

Reorder categories as specified in new_categories.

`new_categories` need to include all old categories and no new category
items.

:param new_categories:
    Index-like
        The categories in new order.

:param ordered:
    bool, optional
        Whether or not the categorical is treated as a ordered categorical.
        If not given, do not change the ordered information.

:param inplace:
    bool, default False
        Whether or not to reorder the categories inplace or return a copy of
        this categorical with reordered categories.

:return: cat : Categorical with reordered categories or None if inplace.

:raises:
    ValueError
        If the new categories do not contain all old category items or any
        new ones



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


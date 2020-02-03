.. _pandas.Series.cat.add_categories:

:orphan:

pandas.Series.cat.add_categories
********************************

Add new categories.

`new_categories` will be included at the last/highest place in the
categories and will be unused directly after this call.

:param new_categories:
    category or list-like of category
        The new categories to be included.

:param inplace:
    bool, default False
        Whether or not to add the categories inplace or return a copy of
        this categorical with added categories.

:return: cat : Categorical with new categories added or None if inplace.

:raises:
    ValueError
        If the new categories include old categories or do not validate as
        categories



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


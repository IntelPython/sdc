.. _pandas.Series.cat.categories:

:orphan:

pandas.Series.cat.categories
****************************

The categories of this categorical.

Setting assigns new values to each category (effectively a rename of
each individual category).

The assigned value has to be a list-like object. All items must be
unique and the number of items in the new categories must be the same
as the number of items in the old categories.

Assigning to `categories` is a inplace operation!

:raises:
    ValueError
        If the new categories do not validate as categories or if the
        number of new categories is unequal the number of old categories



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


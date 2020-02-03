.. _pandas.DataFrame.reorder_levels:

:orphan:

pandas.DataFrame.reorder_levels
*******************************

Rearrange index levels using input order. May not drop or
duplicate levels.

:param order:
    list of int or list of str
        List representing new level order. Reference level by number
        (position) or by key (label).

:param axis:
    int
        Where to reorder levels.

:return: type of caller (new object)



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


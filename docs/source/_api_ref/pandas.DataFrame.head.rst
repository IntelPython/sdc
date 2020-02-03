.. _pandas.DataFrame.head:

:orphan:

pandas.DataFrame.head
*********************

Return the first `n` rows.

This function returns the first `n` rows for the object based
on position. It is useful for quickly testing if your object
has the right type of data in it.

:param n:
    int, default 5
        Number of rows to select.

:return: obj_head : same type as caller
    The first `n` rows of the caller object.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


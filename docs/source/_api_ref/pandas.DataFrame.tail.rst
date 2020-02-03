.. _pandas.DataFrame.tail:

:orphan:

pandas.DataFrame.tail
*********************

Return the last `n` rows.

This function returns last `n` rows from the object based on
position. It is useful for quickly verifying data, for example,
after sorting or appending rows.

:param n:
    int, default 5
        Number of rows to select.

:return: type of caller
    The last `n` rows of the caller object.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


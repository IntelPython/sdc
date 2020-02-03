.. _pandas.Series.drop_duplicates:

:orphan:

pandas.Series.drop_duplicates
*****************************

Return Series with duplicate values removed.

:param keep:
    {'first', 'last', ``False``}, default 'first'

:param - 'first':
    Drop duplicates except for the first occurrence.

:param - 'last':
    Drop duplicates except for the last occurrence.

:param - ``False``:
    Drop all duplicates.

:param inplace:
    bool, default ``False``
        If ``True``, performs operation inplace and returns None.

:return: Series
    Series with duplicates dropped.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


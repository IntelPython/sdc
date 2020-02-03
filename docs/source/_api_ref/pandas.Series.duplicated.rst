.. _pandas.Series.duplicated:

:orphan:

pandas.Series.duplicated
************************

Indicate duplicate Series values.

Duplicated values are indicated as ``True`` values in the resulting
Series. Either all duplicates, all except the first or all except the
last occurrence of duplicates can be indicated.

:param keep:
    {'first', 'last', False}, default 'first'

:param - 'first':
    Mark duplicates as ``True`` except for the first
        occurrence.

:param - 'last':
    Mark duplicates as ``True`` except for the last
        occurrence.

:param - ``False``:
    Mark all duplicates as ``True``.

:return: Series
    Series indicating whether each value has occurred in the
    preceding values.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


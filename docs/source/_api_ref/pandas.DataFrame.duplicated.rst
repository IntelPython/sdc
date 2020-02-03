.. _pandas.DataFrame.duplicated:

:orphan:

pandas.DataFrame.duplicated
***************************

Return boolean Series denoting duplicate rows, optionally only
considering certain columns.

:param subset:
    column label or sequence of labels, optional
        Only consider certain columns for identifying duplicates, by
        default use all of the columns

:param keep:
    {'first', 'last', False}, default 'first'

:param - ``first``:
    Mark duplicates as ``True`` except for the
        first occurrence.

:param - ``last``:
    Mark duplicates as ``True`` except for the
        last occurrence.

:param - False:
    Mark all duplicates as ``True``.

:return: Series



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


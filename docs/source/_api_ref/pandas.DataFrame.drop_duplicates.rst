.. _pandas.DataFrame.drop_duplicates:

:orphan:

pandas.DataFrame.drop_duplicates
********************************

Return DataFrame with duplicate rows removed, optionally only
considering certain columns. Indexes, including time indexes
are ignored.

:param subset:
    column label or sequence of labels, optional
        Only consider certain columns for identifying duplicates, by
        default use all of the columns

:param keep:
    {'first', 'last', False}, default 'first'

:param - ``first``:
    Drop duplicates except for the first occurrence.

:param - ``last``:
    Drop duplicates except for the last occurrence.

:param - False:
    Drop all duplicates.

:param inplace:
    boolean, default False
        Whether to drop duplicates in place or to return a copy

:return: DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


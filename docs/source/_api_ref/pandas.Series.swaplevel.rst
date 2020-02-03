.. _pandas.Series.swaplevel:

:orphan:

pandas.Series.swaplevel
***********************

Swap levels i and j in a MultiIndex.

:param i, j:
    int, str (can be mixed)
        Level of index to be swapped. Can pass level name as string.

:return: Series
    Series with levels swapped in MultiIndex.

    .. versionchanged:: 0.18.1

    The indexes ``i`` and ``j`` are now optional, and default to
    the two innermost levels of the index.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


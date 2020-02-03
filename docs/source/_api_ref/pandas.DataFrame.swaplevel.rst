.. _pandas.DataFrame.swaplevel:

:orphan:

pandas.DataFrame.swaplevel
**************************

Swap levels i and j in a MultiIndex on a particular axis.

:param i, j:
    int, string (can be mixed)
        Level of index to be swapped. Can pass level name as string.

:return: DataFrame

    .. versionchanged:: 0.18.1

    The indexes ``i`` and ``j`` are now optional, and default to
    the two innermost levels of the index.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


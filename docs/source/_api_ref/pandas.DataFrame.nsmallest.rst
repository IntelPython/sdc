.. _pandas.DataFrame.nsmallest:

:orphan:

pandas.DataFrame.nsmallest
**************************

Return the first `n` rows ordered by `columns` in ascending order.

Return the first `n` rows with the smallest values in `columns`, in
ascending order. The columns that are not specified are returned as
well, but not used for ordering.

This method is equivalent to
``df.sort_values(columns, ascending=True).head(n)``, but more
performant.

:param n:
    int
        Number of items to retrieve.

:param columns:
    list or str
        Column name or names to order by.

:param keep:
    {'first', 'last', 'all'}, default 'first'
        Where there are duplicate values:

:param - ``first``:
    take the first occurrence.

:param - ``last``:
    take the last occurrence.

:param - ``all``:
    do not drop any duplicates, even it means
        selecting more than `n` items.

        .. versionadded:: 0.24.0

:return: DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


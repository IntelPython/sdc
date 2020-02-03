.. _pandas.DataFrame.nlargest:

:orphan:

pandas.DataFrame.nlargest
*************************

Return the first `n` rows ordered by `columns` in descending order.

Return the first `n` rows with the largest values in `columns`, in
descending order. The columns that are not specified are returned as
well, but not used for ordering.

This method is equivalent to
``df.sort_values(columns, ascending=False).head(n)``, but more
performant.

:param n:
    int
        Number of rows to return.

:param columns:
    label or list of labels
        Column label(s) to order by.

:param keep:
    {'first', 'last', 'all'}, default 'first'
        Where there are duplicate values:

:param - `first`:
    prioritize the first occurrence(s)

:param - `last`:
    prioritize the last occurrence(s)

:param - ``all``:
    do not drop any duplicates, even it means
        selecting more than `n` items.

        .. versionadded:: 0.24.0

:return: DataFrame
    The first `n` rows ordered by the given columns in descending
    order.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


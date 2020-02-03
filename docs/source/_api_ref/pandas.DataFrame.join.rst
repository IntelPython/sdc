.. _pandas.DataFrame.join:

:orphan:

pandas.DataFrame.join
*********************

Join columns of another DataFrame.

Join columns with `other` DataFrame either on index or on a key
column. Efficiently join multiple DataFrame objects by index at once by
passing a list.

:param other:
    DataFrame, Series, or list of DataFrame
        Index should be similar to one of the columns in this one. If a
        Series is passed, its name attribute must be set, and that will be
        used as the column name in the resulting joined DataFrame.

:param on:
    str, list of str, or array-like, optional
        Column or index level name(s) in the caller to join on the index
        in `other`, otherwise joins index-on-index. If multiple
        values given, the `other` DataFrame must have a MultiIndex. Can
        pass an array as the join key if it is not already contained in
        the calling DataFrame. Like an Excel VLOOKUP operation.

:param how:
    {'left', 'right', 'outer', 'inner'}, default 'left'
        How to handle the operation of the two objects.

        - left: use calling frame's index (or column if on is specified)
        - right: use `other`'s index.
        - outer: form union of calling frame's index (or column if on is
            specified) with `other`'s index, and sort it.
            lexicographically.
        - inner: form intersection of calling frame's index (or column if
            on is specified) with `other`'s index, preserving the order
            of the calling's one.

:param lsuffix:
    str, default ''
        Suffix to use from left frame's overlapping columns.

:param rsuffix:
    str, default ''
        Suffix to use from right frame's overlapping columns.

:param sort:
    bool, default False
        Order result DataFrame lexicographically by the join key. If False,
        the order of the join key depends on the join type (how keyword).

:return: DataFrame
    A dataframe containing columns from both the caller and `other`.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.DataFrame.merge:

:orphan:

pandas.DataFrame.merge
**********************

Merge DataFrame or named Series objects with a database-style join.

The join is done on columns or indexes. If joining columns on
columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
on indexes or indexes on a column or columns, the index will be passed on.

:param right:
    DataFrame or named Series
        Object to merge with.

:param how:
    {'left', 'right', 'outer', 'inner'}, default 'inner'
        Type of merge to be performed.

        - left: use only keys from left frame, similar to a SQL left outer join;
            preserve key order.
        - right: use only keys from right frame, similar to a SQL right outer join;
            preserve key order.
        - outer: use union of keys from both frames, similar to a SQL full outer
            join; sort keys lexicographically.
        - inner: use intersection of keys from both frames, similar to a SQL inner
            join; preserve the order of the left keys.

:param on:
    label or list
        Column or index level names to join on. These must be found in both
        DataFrames. If `on` is None and not merging on indexes then this defaults
        to the intersection of the columns in both DataFrames.

:param left_on:
    label or list, or array-like
        Column or index level names to join on in the left DataFrame. Can also
        be an array or list of arrays of the length of the left DataFrame.
        These arrays are treated as if they are columns.

:param right_on:
    label or list, or array-like
        Column or index level names to join on in the right DataFrame. Can also
        be an array or list of arrays of the length of the right DataFrame.
        These arrays are treated as if they are columns.

:param left_index:
    bool, default False
        Use the index from the left DataFrame as the join key(s). If it is a
        MultiIndex, the number of keys in the other DataFrame (either the index
        or a number of columns) must match the number of levels.

:param right_index:
    bool, default False
        Use the index from the right DataFrame as the join key. Same caveats as
        left_index.

:param sort:
    bool, default False
        Sort the join keys lexicographically in the result DataFrame. If False,
        the order of the join keys depends on the join type (how keyword).

:param suffixes:
    tuple of (str, str), default ('_x', '_y')
        Suffix to apply to overlapping column names in the left and right
        side, respectively. To raise an exception on overlapping columns use
        (False, False).

:param copy:
    bool, default True
        If False, avoid copy if possible.

:param indicator:
    bool or str, default False
        If True, adds a column to output DataFrame called "_merge" with
        information on the source of each row.
        If string, column with information on source of each row will be added to
        output DataFrame, and column will be named value of string.
        Information column is Categorical-type and takes on a value of "left_only"
        for observations whose merge key only appears in 'left' DataFrame,
        "right_only" for observations whose merge key only appears in 'right'
        DataFrame, and "both" if the observation's merge key is found in both.

:param validate:
    str, optional
        If specified, checks if merge is of specified type.

        - "one_to_one" or "1:1": check if merge keys are unique in both
            left and right datasets.
        - "one_to_many" or "1:m": check if merge keys are unique in left
            dataset.
        - "many_to_one" or "m:1": check if merge keys are unique in right
            dataset.
        - "many_to_many" or "m:m": allowed, but does not result in checks.

        .. versionadded:: 0.21.0

:return: DataFrame
    A DataFrame of the two merged objects.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


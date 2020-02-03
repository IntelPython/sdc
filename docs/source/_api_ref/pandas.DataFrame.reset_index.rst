.. _pandas.DataFrame.reset_index:

:orphan:

pandas.DataFrame.reset_index
****************************

Reset the index, or a level of it.

Reset the index of the DataFrame, and use the default one instead.
If the DataFrame has a MultiIndex, this method can remove one or more
levels.

:param level:
    int, str, tuple, or list, default None
        Only remove the given levels from the index. Removes all levels by
        default.

:param drop:
    bool, default False
        Do not try to insert index into dataframe columns. This resets
        the index to the default integer index.

:param inplace:
    bool, default False
        Modify the DataFrame in place (do not create a new object).

:param col_level:
    int or str, default 0
        If the columns have multiple levels, determines which level the
        labels are inserted into. By default it is inserted into the first
        level.

:param col_fill:
    object, default ''
        If the columns have multiple levels, determines how the other
        levels are named. If None then the index name is repeated.

:return: DataFrame
    DataFrame with the new index.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.Series.reset_index:

:orphan:

pandas.Series.reset_index
*************************

Generate a new DataFrame or Series with the index reset.

This is useful when the index needs to be treated as a column, or
when the index is meaningless and needs to be reset to the default
before another operation.

:param level:
    int, str, tuple, or list, default optional
        For a Series with a MultiIndex, only remove the specified levels
        from the index. Removes all levels by default.

:param drop:
    bool, default False
        Just reset the index, without inserting it as a column in
        the new DataFrame.

:param name:
    object, optional
        The name to use for the column containing the original Series
        values. Uses ``self.name`` by default. This argument is ignored
        when `drop` is True.

:param inplace:
    bool, default False
        Modify the Series in place (do not create a new object).

:return: Series or DataFrame
    When `drop` is False (the default), a DataFrame is returned.
    The newly created columns will come first in the DataFrame,
    followed by the original Series values.
    When `drop` is True, a `Series` is returned.
    In either case, if ``inplace=True``, no value is returned.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


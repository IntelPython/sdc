.. _pandas.DataFrame.pivot_table:

:orphan:

pandas.DataFrame.pivot_table
****************************

Create a spreadsheet-style pivot table as a DataFrame. The levels in
the pivot table will be stored in MultiIndex objects (hierarchical
indexes) on the index and columns of the result DataFrame.

:param values:
    column to aggregate, optional

:param index:
    column, Grouper, array, or list of the previous
        If an array is passed, it must be the same length as the data. The
        list can contain any of the other types (except list).
        Keys to group by on the pivot table index.  If an array is passed,
        it is being used as the same manner as column values.

:param columns:
    column, Grouper, array, or list of the previous
        If an array is passed, it must be the same length as the data. The
        list can contain any of the other types (except list).
        Keys to group by on the pivot table column.  If an array is passed,
        it is being used as the same manner as column values.

:param aggfunc:
    function, list of functions, dict, default numpy.mean
        If list of functions passed, the resulting pivot table will have
        hierarchical columns whose top level are the function names
        (inferred from the function objects themselves)
        If dict is passed, the key is column to aggregate and value
        is function or list of functions

:param fill_value:
    scalar, default None
        Value to replace missing values with

:param margins:
    boolean, default False
        Add all row / columns (e.g. for subtotal / grand totals)

:param dropna:
    boolean, default True
        Do not include columns whose entries are all NaN

:param margins_name:
    string, default 'All'
        Name of the row / column that will contain the totals
        when margins is True.

:param observed:
    boolean, default False
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.

        .. versionchanged :: 0.25.0

:return: DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


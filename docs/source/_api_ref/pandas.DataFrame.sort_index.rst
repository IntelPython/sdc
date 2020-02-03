.. _pandas.DataFrame.sort_index:

:orphan:

pandas.DataFrame.sort_index
***************************

Sort object by labels (along an axis).

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The axis along which to sort.  The value 0 identifies the rows,
        and 1 identifies the columns.

:param level:
    int or level name or list of ints or list of level names
        If not None, sort on values in specified index level(s).

:param ascending:
    bool, default True
        Sort ascending vs. descending.

:param inplace:
    bool, default False
        If True, perform operation in-place.

:param kind:
    {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
        Choice of sorting algorithm. See also ndarray.np.sort for more
        information.  `mergesort` is the only stable algorithm. For
        DataFrames, this option is only applied when sorting on a single
        column or label.

:param na_position:
    {'first', 'last'}, default 'last'
        Puts NaNs at the beginning if `first`; `last` puts NaNs at the end.
        Not implemented for MultiIndex.

:param sort_remaining:
    bool, default True
        If True and sorting by level and index is multilevel, sort by other
        levels too (in order) after sorting by specified level.

:return: sorted_obj : DataFrame or None
    DataFrame with sorted index if inplace=False, None otherwise.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


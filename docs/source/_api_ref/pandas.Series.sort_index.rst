.. _pandas.Series.sort_index:

:orphan:

pandas.Series.sort_index
************************

Sort Series by index labels.

Returns a new Series sorted by label if `inplace` argument is
``False``, otherwise updates the original series and returns None.

:param axis:
    int, default 0
        Axis to direct sorting. This can only be 0 for Series.

:param level:
    int, optional
        If not None, sort on values in specified index level(s).

:param ascending:
    bool, default true
        Sort ascending vs. descending.

:param inplace:
    bool, default False
        If True, perform operation in-place.

:param kind:
    {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
        Choice of sorting algorithm. See also :func:`numpy.sort` for more
        information.  'mergesort' is the only stable algorithm. For
        DataFrames, this option is only applied when sorting on a single
        column or label.

:param na_position:
    {'first', 'last'}, default 'last'
        If 'first' puts NaNs at the beginning, 'last' puts NaNs at the end.
        Not implemented for MultiIndex.

:param sort_remaining:
    bool, default True
        If True and sorting by level and index is multilevel, sort by other
        levels too (in order) after sorting by specified level.

:return: Series
    The original Series sorted by the labels.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


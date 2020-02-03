.. _pandas.DataFrame.sort_values:

:orphan:

pandas.DataFrame.sort_values
****************************

Sort by the values along either axis.

:param by:
            str or list of str
                        Name or list of names to sort by.

                        - if `axis` is 0 or `'index'` then `by` may contain index
                            levels and/or column labels
                        - if `axis` is 1 or `'columns'` then `by` may contain column
                            levels and/or index labels

                        .. versionchanged:: 0.23.0

:param axis:
            {0 or 'index', 1 or 'columns'}, default 0
                        Axis to be sorted.

:param ascending:
            bool or list of bool, default True
                        Sort ascending vs. descending. Specify list for multiple sort
                        orders.  If this is a list of bools, must match the length of
                        the by.

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
                        Puts NaNs at the beginning if `first`; `last` puts NaNs at the
                        end.

:return: sorted_obj : DataFrame or None
    DataFrame with sorted values if inplace=False, None otherwise.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


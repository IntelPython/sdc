.. _pandas.Series.any:

:orphan:

pandas.Series.any
*****************

Return whether any element is True, potentially over an axis.

Returns False unless there at least one element within a series or
along a Dataframe axis that is True or equivalent (e.g. non-zero or
non-empty).

:param axis:
    {0 or 'index', 1 or 'columns', None}, default 0
        Indicate which axis or axes should be reduced.

:param \* 0 / 'index':
    reduce the index, return a Series whose index is the
        original column labels.

:param \* 1 / 'columns':
    reduce the columns, return a Series whose index is the
        original index.

:param \* None:
    reduce all axes, return a scalar.

:param bool_only:
    bool, default None
        Include only boolean columns. If None, will attempt to use everything,
        then use only boolean data. Not implemented for Series.

:param skipna:
    bool, default True
        Exclude NA/null values. If the entire row/column is NA and skipna is
        True, then the result will be False, as for an empty row/column.
        If skipna is False, then NA are treated as True, because these are not
        equal to zero.

:param level:
    int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a
        particular level, collapsing into a scalar.

:param \*\*kwargs:
    any, default None
        Additional keywords have no effect but might be accepted for
        compatibility with NumPy.

:return: scalar or Series
    If level is specified, then, Series is returned; otherwise, scalar
    is returned.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.DataFrame.groupby:

:orphan:

pandas.DataFrame.groupby
************************

Group DataFrame or Series using a mapper or by a Series of columns.

A groupby operation involves some combination of splitting the
object, applying a function, and combining the results. This can be
used to group large amounts of data and compute operations on these
groups.

:param by:
    mapping, function, label, or list of labels
        Used to determine the groups for the groupby.
        If ``by`` is a function, it's called on each value of the object's
        index. If a dict or Series is passed, the Series or dict VALUES
        will be used to determine the groups (the Series' values are first
        aligned; see ``.align()`` method). If an ndarray is passed, the
        values are used as-is determine the groups. A label or list of
        labels may be passed to group by the columns in ``self``. Notice
        that a tuple is interpreted as a (single) key.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Split along rows (0) or columns (1).

:param level:
    int, level name, or sequence of such, default None
        If the axis is a MultiIndex (hierarchical), group by a particular
        level or levels.

:param as_index:
    bool, default True
        For aggregated output, return object with group labels as the
        index. Only relevant for DataFrame input. as_index=False is
        effectively "SQL-style" grouped output.

:param sort:
    bool, default True
        Sort group keys. Get better performance by turning this off.
        Note this does not influence the order of observations within each
        group. Groupby preserves the order of rows within each group.

:param group_keys:
    bool, default True
        When calling apply, add group keys to index to identify pieces.

:param squeeze:
    bool, default False
        Reduce the dimensionality of the return type if possible,
        otherwise return a consistent type.

:param observed:
    bool, default False
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.

        .. versionadded:: 0.23.0

        \*\*kwargs
        Optional, only accepts keyword argument 'mutated' and is passed
        to groupby.

:return: DataFrameGroupBy or SeriesGroupBy
    Depends on the calling object and returns groupby object that
    contains information about the groups.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


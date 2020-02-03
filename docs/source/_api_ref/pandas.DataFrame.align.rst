.. _pandas.DataFrame.align:

:orphan:

pandas.DataFrame.align
**********************

Align two objects on their axes with the
specified join method for each axis Index.

:param other:
    DataFrame or Series

:param join:
    {'outer', 'inner', 'left', 'right'}, default 'outer'

:param axis:
    allowed axis of the other object, default None
        Align on index (0), columns (1), or both (None)

:param level:
    int or level name, default None
        Broadcast across a level, matching Index values on the
        passed MultiIndex level

:param copy:
    boolean, default True
        Always returns new objects. If copy=False and no reindexing is
        required then original objects are returned.

:param fill_value:
    scalar, default np.NaN
        Value to use for missing values. Defaults to NaN, but can be any
        "compatible" value

:param method:
    {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed Series
        pad / ffill: propagate last valid observation forward to next valid
        backfill / bfill: use NEXT valid observation to fill gap

:param limit:
    int, default None
        If method is specified, this is the maximum number of consecutive
        NaN values to forward/backward fill. In other words, if there is
        a gap with more than this number of consecutive NaNs, it will only
        be partially filled. If method is not specified, this is the
        maximum number of entries along the entire axis where NaNs will be
        filled. Must be greater than 0 if not None.

:param fill_axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Filling axis, method and limit

:param broadcast_axis:
    {0 or 'index', 1 or 'columns'}, default None
        Broadcast values along this axis, if aligning two objects of
        different dimensions

:return: (left, right) : (DataFrame, type of other)
    Aligned objects.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


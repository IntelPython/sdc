.. _pandas.DataFrame.count:

:orphan:

pandas.DataFrame.count
**********************

Count non-NA cells for each column or row.

The values `None`, `NaN`, `NaT`, and optionally `numpy.inf` (depending
on `pandas.options.mode.use_inf_as_na`) are considered NA.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        If 0 or 'index' counts are generated for each column.
        If 1 or 'columns' counts are generated for each **row**.

:param level:
    int or str, optional
        If the axis is a `MultiIndex` (hierarchical), count along a
        particular `level`, collapsing into a `DataFrame`.
        A `str` specifies the level name.

:param numeric_only:
    bool, default False
        Include only `float`, `int` or `boolean` data.

:return: Series or DataFrame
    For each column/row the number of non-NA/null entries.
    If `level` is specified returns a `DataFrame`.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


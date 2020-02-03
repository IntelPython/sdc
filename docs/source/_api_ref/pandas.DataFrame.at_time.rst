.. _pandas.DataFrame.at_time:

:orphan:

pandas.DataFrame.at_time
************************

Select values at particular time of day (e.g. 9:30AM).

:param time:
    datetime.time or str

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0

        .. versionadded:: 0.24.0

:return: Series or DataFrame

:raises:
    TypeError
        If the index is not  a :class:`DatetimeIndex`



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


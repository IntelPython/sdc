.. _pandas.DataFrame.between_time:

:orphan:

pandas.DataFrame.between_time
*****************************

Select values between particular times of the day (e.g., 9:00-9:30 AM).

By setting ``start_time`` to be later than ``end_time``,
you can get the times that are *not* between the two times.

:param start_time:
    datetime.time or str

:param end_time:
    datetime.time or str

:param include_start:
    bool, default True

:param include_end:
    bool, default True

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0

        .. versionadded:: 0.24.0

:return: Series or DataFrame

:raises:
    TypeError
        If the index is not  a :class:`DatetimeIndex`



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


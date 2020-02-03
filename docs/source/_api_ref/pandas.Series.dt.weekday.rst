.. _pandas.Series.dt.weekday:

:orphan:

pandas.Series.dt.weekday
************************

The day of the week with Monday=0, Sunday=6.

Return the day of the week. It is assumed the week starts on
Monday, which is denoted by 0 and ends on Sunday which is denoted
by 6. This method is available on both Series with datetime
values (using the `dt` accessor) or DatetimeIndex.

:return: Series or Index
    Containing integers indicating the day number.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.Series.dt.normalize:

:orphan:

pandas.Series.dt.normalize
**************************

Convert times to midnight.

The time component of the date-time is converted to midnight i.e.
00:00:00. This is useful in cases, when the time does not matter.
Length is unaltered. The timezones are unaffected.

This method is available on Series with datetime values under
the ``.dt`` accessor, and directly on Datetime Array/Index.

:return: DatetimeArray, DatetimeIndex or Series
    The same type as the original data. Series will have the same
    name and index. DatetimeIndex will have the same name.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


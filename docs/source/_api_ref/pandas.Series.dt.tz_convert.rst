.. _pandas.Series.dt.tz_convert:

:orphan:

pandas.Series.dt.tz_convert
***************************

Convert tz-aware Datetime Array/Index from one time zone to another.

:param tz:
    str, pytz.timezone, dateutil.tz.tzfile or None
        Time zone for time. Corresponding timestamps would be converted
        to this time zone of the Datetime Array/Index. A `tz` of None will
        convert to UTC and remove the timezone information.

:return: Array or Index

:raises:
    TypeError
        If Datetime Array/Index is tz-naive.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


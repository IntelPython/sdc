.. _pandas.Series.tz_convert:

:orphan:

pandas.Series.tz_convert
************************

Convert tz-aware axis to target time zone.

:param tz:
    string or pytz.timezone object

:param axis:
    the axis to convert

:param level:
    int, str, default None
        If axis ia a MultiIndex, convert a specific level. Otherwise
        must be None

:param copy:
    boolean, default True
        Also make a copy of the underlying data

:return: %(klass)s
    Object with time zone converted axis.

:raises:
    TypeError
        If the axis is tz-naive.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


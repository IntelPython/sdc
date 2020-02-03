.. _pandas.Series.to_timestamp:

:orphan:

pandas.Series.to_timestamp
**************************

Cast to DatetimeIndex of Timestamps, at *beginning* of period.

:param freq:
    str, default frequency of PeriodIndex
        Desired frequency.

:param how:
    {'s', 'e', 'start', 'end'}
        Convention for converting period to timestamp; start of period
        vs. end.

:param copy:
    bool, default True
        Whether or not to return a copy.

:return: Series with DatetimeIndex



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


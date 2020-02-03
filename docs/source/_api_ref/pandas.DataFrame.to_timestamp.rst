.. _pandas.DataFrame.to_timestamp:

:orphan:

pandas.DataFrame.to_timestamp
*****************************

Cast to DatetimeIndex of timestamps, at *beginning* of period.

:param freq:
    str, default frequency of PeriodIndex
        Desired frequency.

:param how:
    {'s', 'e', 'start', 'end'}
        Convention for converting period to timestamp; start of period
        vs. end.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The axis to convert (the index by default).

:param copy:
    bool, default True
        If False then underlying input data is not copied.

:return: DataFrame with DatetimeIndex



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


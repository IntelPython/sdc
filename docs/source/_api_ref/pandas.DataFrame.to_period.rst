.. _pandas.DataFrame.to_period:

:orphan:

pandas.DataFrame.to_period
**************************

Convert DataFrame from DatetimeIndex to PeriodIndex with desired
frequency (inferred from index if not passed).

:param freq:
    str, default
        Frequency of the PeriodIndex.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The axis to convert (the index by default).

:param copy:
    bool, default True
        If False then underlying input data is not copied.

:return: TimeSeries with PeriodIndex



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


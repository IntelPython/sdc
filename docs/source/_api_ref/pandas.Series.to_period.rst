.. _pandas.Series.to_period:

:orphan:

pandas.Series.to_period
***********************

Convert Series from DatetimeIndex to PeriodIndex with desired
frequency (inferred from index if not passed).

:param freq:
    str, default None
        Frequency associated with the PeriodIndex.

:param copy:
    bool, default True
        Whether or not to return a copy.

:return: Series
    Series with index converted to PeriodIndex.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.Series.first:

:orphan:

pandas.Series.first
*******************

Convenience method for subsetting initial periods of time series data
based on a date offset.

:param offset:
    string, DateOffset, dateutil.relativedelta

:return: subset : same type as caller

:raises:
    TypeError
        If the index is not  a :class:`DatetimeIndex`



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


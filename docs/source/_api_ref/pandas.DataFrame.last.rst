.. _pandas.DataFrame.last:

:orphan:

pandas.DataFrame.last
*********************

Convenience method for subsetting final periods of time series data
based on a date offset.

:param offset:
    string, DateOffset, dateutil.relativedelta

:return: subset : same type as caller

:raises:
    TypeError
        If the index is not  a :class:`DatetimeIndex`



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


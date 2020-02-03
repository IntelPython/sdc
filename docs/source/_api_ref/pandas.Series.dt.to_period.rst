.. _pandas.Series.dt.to_period:

:orphan:

pandas.Series.dt.to_period
**************************

Cast to PeriodArray/Index at a particular frequency.

Converts DatetimeArray/Index to PeriodArray/Index.

:param freq:
    str or Offset, optional
        One of pandas' :ref:`offset strings <timeseries.offset_aliases>`
        or an Offset object. Will be inferred by default.

:return: PeriodArray/Index

:raises:
    ValueError
        When converting a DatetimeArray/Index with non-regular values,
        so that a frequency cannot be inferred.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


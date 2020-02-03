.. _pandas.Series.asfreq:

:orphan:

pandas.Series.asfreq
********************

Convert TimeSeries to specified frequency.

Optionally provide filling method to pad/backfill missing values.

Returns the original data conformed to a new index with the specified
frequency. ``resample`` is more appropriate if an operation, such as
summarization, is necessary to represent the data at the new frequency.

:param freq:
    DateOffset object, or string

:param method:
    {'backfill'/'bfill', 'pad'/'ffill'}, default None
        Method to use for filling holes in reindexed Series (note this
        does not fill NaNs that already were present):

        - 'pad' / 'ffill': propagate last valid observation forward to next
            valid
        - 'backfill' / 'bfill': use NEXT valid observation to fill

:param how:
    {'start', 'end'}, default end
        For PeriodIndex only, see PeriodIndex.asfreq

:param normalize:
    bool, default False
        Whether to reset output index to midnight

:param fill_value:
    scalar, optional
        Value to use for missing values, applied during upsampling (note
        this does not fill NaNs that already were present).

        .. versionadded:: 0.20.0

:return: converted : same type as caller



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.DataFrame.resample:

:orphan:

pandas.DataFrame.resample
*************************

Resample time-series data.

Convenience method for frequency conversion and resampling of time
series. Object must have a datetime-like index (`DatetimeIndex`,
`PeriodIndex`, or `TimedeltaIndex`), or pass datetime-like values
to the `on` or `level` keyword.

:param rule:
    DateOffset, Timedelta or str
        The offset string or object representing target conversion.

:param how:
    str
        Method for down/re-sampling, default to 'mean' for downsampling.

        .. deprecated:: 0.18.0

        ``.resample(...).apply(<func>)``

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Which axis to use for up- or down-sampling. For `Series` this
        will default to 0, i.e. along the rows. Must be
        `DatetimeIndex`, `TimedeltaIndex` or `PeriodIndex`.

:param fill_method:
    str, default None
        Filling method for upsampling.

        .. deprecated:: 0.18.0

        e.g. ``.resample(...).pad()``

:param closed:
    {'right', 'left'}, default None
        Which side of bin interval is closed. The default is 'left'
        for all frequency offsets except for 'M', 'A', 'Q', 'BM',
        'BA', 'BQ', and 'W' which all have a default of 'right'.

:param label:
    {'right', 'left'}, default None
        Which bin edge label to label bucket with. The default is 'left'
        for all frequency offsets except for 'M', 'A', 'Q', 'BM',
        'BA', 'BQ', and 'W' which all have a default of 'right'.

:param convention:
    {'start', 'end', 's', 'e'}, default 'start'
        For `PeriodIndex` only, controls whether to use the start or
        end of `rule`.

:param kind:
    {'timestamp', 'period'}, optional, default None
        Pass 'timestamp' to convert the resulting index to a
        `DateTimeIndex` or 'period' to convert it to a `PeriodIndex`.
        By default the input representation is retained.

:param loffset:
    timedelta, default None
        Adjust the resampled time labels.

:param limit:
    int, default None
        Maximum size gap when reindexing with `fill_method`.

        .. deprecated:: 0.18.0

:param base:
    int, default 0
        For frequencies that evenly subdivide 1 day, the "origin" of the
        aggregated intervals. For example, for '5min' frequency, base could
        range from 0 through 4. Defaults to 0.

:param on:
    str, optional
        For a DataFrame, column to use instead of index for resampling.
        Column must be datetime-like.

        .. versionadded:: 0.19.0

:param level:
    str or int, optional
        For a MultiIndex, level (name or number) to use for
        resampling. `level` must be datetime-like.

        .. versionadded:: 0.19.0

:return: Resampler object



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


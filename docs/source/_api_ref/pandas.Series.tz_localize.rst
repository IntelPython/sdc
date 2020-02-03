.. _pandas.Series.tz_localize:

:orphan:

pandas.Series.tz_localize
*************************

Localize tz-naive index of a Series or DataFrame to target time zone.

This operation localizes the Index. To localize the values in a
timezone-naive Series, use :meth:`Series.dt.tz_localize`.

:param tz:
    string or pytz.timezone object

:param axis:
    the axis to localize

:param level:
    int, str, default None
        If axis ia a MultiIndex, localize a specific level. Otherwise
        must be None

:param copy:
    boolean, default True
        Also make a copy of the underlying data

:param ambiguous:
    'infer', bool-ndarray, 'NaT', default 'raise'
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from
        03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
        00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
        `ambiguous` parameter dictates how ambiguous times should be
        handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
            order
        - bool-ndarray where True signifies a DST time, False designates
            a non-DST time (note that this flag is only applicable for
            ambiguous times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous
            times

:param nonexistent:
    str, default 'raise'
        A nonexistent time does not exist in a particular timezone
        where clocks moved forward due to DST. Valid values are:

        - 'shift_forward' will shift the nonexistent time forward to the
            closest existing time
        - 'shift_backward' will shift the nonexistent time backward to the
            closest existing time
        - 'NaT' will return NaT where there are nonexistent times
        - timedelta objects will shift nonexistent times by the timedelta
        - 'raise' will raise an NonExistentTimeError if there are
            nonexistent times

        .. versionadded:: 0.24.0

:return: Series or DataFrame
    Same type as the input.

:raises:
    TypeError
        If the TimeSeries is tz-aware and tz is not None.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


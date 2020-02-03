.. _pandas.Series.dt.tz_localize:

:orphan:

pandas.Series.dt.tz_localize
****************************

Localize tz-naive Datetime Array/Index to tz-aware
Datetime Array/Index.

This method takes a time zone (tz) naive Datetime Array/Index object
and makes this time zone aware. It does not move the time to another
time zone.
Time zone localization helps to switch from time zone aware to time
zone unaware objects.

:param tz:
    str, pytz.timezone, dateutil.tz.tzfile or None
        Time zone to convert timestamps to. Passing ``None`` will
        remove the time zone information preserving local time.

:param ambiguous:
    'infer', 'NaT', bool array, default 'raise'
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from
        03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
        00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
        `ambiguous` parameter dictates how ambiguous times should be
        handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
            order
        - bool-ndarray where True signifies a DST time, False signifies a
            non-DST time (note that this flag is only applicable for
            ambiguous times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous
            times

:param nonexistent:
    'shift_forward', 'shift_backward, 'NaT', timedelta, default 'raise'
        A nonexistent time does not exist in a particular timezone
        where clocks moved forward due to DST.

        - 'shift_forward' will shift the nonexistent time forward to the
            closest existing time
        - 'shift_backward' will shift the nonexistent time backward to the
            closest existing time
        - 'NaT' will return NaT where there are nonexistent times
        - timedelta objects will shift nonexistent times by the timedelta
        - 'raise' will raise an NonExistentTimeError if there are
            nonexistent times

        .. versionadded:: 0.24.0

:param errors:
    {'raise', 'coerce'}, default None

        - 'raise' will raise a NonExistentTimeError if a timestamp is not
            valid in the specified time zone (e.g. due to a transition from
            or to DST time). Use ``nonexistent='raise'`` instead.
        - 'coerce' will return NaT if the timestamp can not be converted
            to the specified time zone. Use ``nonexistent='NaT'`` instead.

        .. deprecated:: 0.24.0

:return: Same type as self
    Array/Index converted to the specified time zone.

:raises:
    TypeError
        If the Datetime Array/Index is tz-aware and tz is not None.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


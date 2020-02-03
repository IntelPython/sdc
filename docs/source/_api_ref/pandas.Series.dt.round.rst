.. _pandas.Series.dt.round:

:orphan:

pandas.Series.dt.round
**********************

Perform round operation on the data to the specified `freq`.

:param freq:
    str or Offset
        The frequency level to round the index to. Must be a fixed
        frequency like 'S' (second) not 'ME' (month end). See
        :ref:`frequency aliases <timeseries.offset_aliases>` for
        a list of possible `freq` values.

:param ambiguous:
    'infer', bool-ndarray, 'NaT', default 'raise'
        Only relevant for DatetimeIndex:

        - 'infer' will attempt to infer fall dst-transition hours based on
            order
        - bool-ndarray where True signifies a DST time, False designates
            a non-DST time (note that this flag is only applicable for
            ambiguous times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous
            times

        .. versionadded:: 0.24.0

:param nonexistent:
    'shift_forward', 'shift_backward', 'NaT', timedelta, default 'raise'
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

:return: DatetimeIndex, TimedeltaIndex, or Series
    Index of the same type for a DatetimeIndex or TimedeltaIndex,
    or a Series with the same index for a Series.

:raises:
    ValueError if the `freq` cannot be converted.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


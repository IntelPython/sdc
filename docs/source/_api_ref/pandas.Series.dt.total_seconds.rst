.. _pandas.Series.dt.total_seconds:

:orphan:

pandas.Series.dt.total_seconds
******************************

Return total duration of each element expressed in seconds.

This method is available directly on TimedeltaArray, TimedeltaIndex
and on Series containing timedelta values under the ``.dt`` namespace.

:return: seconds : [ndarray, Float64Index, Series]
    When the calling object is a TimedeltaArray, the return type
    is ndarray.  When the calling object is a TimedeltaIndex,
    the return type is a Float64Index. When the calling object
    is a Series, the return type is Series of type `float64` whose
    index is the same as the original.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


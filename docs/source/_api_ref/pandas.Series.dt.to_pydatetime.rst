.. _pandas.Series.dt.to_pydatetime:

:orphan:

pandas.Series.dt.to_pydatetime
******************************

Return the data as an array of native Python datetime objects.

Timezone information is retained if present.

.. warning::
   Python's datetime uses microsecond resolution, which is lower than
   pandas (nanosecond). The values are truncated.

:return: numpy.ndarray
    Object dtype array containing native Python datetime objects.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


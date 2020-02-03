.. _pandas.Series.dt.to_pytimedelta:

:orphan:

pandas.Series.dt.to_pytimedelta
*******************************

Return an array of native `datetime.timedelta` objects.

Python's standard `datetime` library uses a different representation
timedelta's. This method converts a Series of pandas Timedeltas
to `datetime.timedelta` format with the same length as the original
Series.

:return: a : numpy.ndarray
    Array of 1D containing data with `datetime.timedelta` type.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


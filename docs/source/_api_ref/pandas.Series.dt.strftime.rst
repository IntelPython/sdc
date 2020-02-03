.. _pandas.Series.dt.strftime:

:orphan:

pandas.Series.dt.strftime
*************************

Convert to Index using specified date_format.

Return an Index of formatted strings specified by date_format, which
supports the same string format as the python standard library. Details
of the string format can be found in `python string format
doc <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`__.

:param date_format:
    str
        Date format string (e.g. "%Y-%m-%d").

:return: Index
    Index of formatted strings.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


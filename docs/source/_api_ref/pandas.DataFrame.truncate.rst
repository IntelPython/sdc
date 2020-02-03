.. _pandas.DataFrame.truncate:

:orphan:

pandas.DataFrame.truncate
*************************

Truncate a Series or DataFrame before and after some index value.

This is a useful shorthand for boolean indexing based on index
values above or below certain thresholds.

:param before:
    date, string, int
        Truncate all rows before this index value.

:param after:
    date, string, int
        Truncate all rows after this index value.

:param axis:
    {0 or 'index', 1 or 'columns'}, optional
        Axis to truncate. Truncates the index (rows) by default.

:param copy:
    boolean, default is True,
        Return a copy of the truncated section.

:return: type of caller
    The truncated Series or DataFrame.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


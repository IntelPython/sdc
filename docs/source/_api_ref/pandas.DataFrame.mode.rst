.. _pandas.DataFrame.mode:

:orphan:

pandas.DataFrame.mode
*********************

Get the mode(s) of each element along the selected axis.

The mode of a set of values is the value that appears most often.
It can be multiple values.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The axis to iterate over while searching for the mode:

:param \* 0 or 'index':
    get mode of each column

:param \* 1 or 'columns':
    get mode of each row

:param numeric_only:
    bool, default False
        If True, only apply to numeric columns.

:param dropna:
    bool, default True
        Don't consider counts of NaN/NaT.

        .. versionadded:: 0.24.0

:return: DataFrame
    The modes of each column or row.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


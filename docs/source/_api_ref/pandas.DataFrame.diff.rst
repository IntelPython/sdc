.. _pandas.DataFrame.diff:

:orphan:

pandas.DataFrame.diff
*********************

First discrete difference of element.

Calculates the difference of a DataFrame element compared with another
element in the DataFrame (default is the element in the same column
of the previous row).

:param periods:
    int, default 1
        Periods to shift for calculating difference, accepts negative
        values.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Take difference over rows (0) or columns (1).

        .. versionadded:: 0.16.1.

:return: DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


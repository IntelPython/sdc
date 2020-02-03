.. _pandas.Series.unstack:

:orphan:

pandas.Series.unstack
*********************

Unstack, a.k.a. pivot, Series with MultiIndex to produce DataFrame.
The level involved will automatically get sorted.

:param level:
    int, str, or list of these, default last level
        Level(s) to unstack, can pass level name.

:param fill_value:
    scalar value, default None
        Value to use when replacing NaN values.

        .. versionadded:: 0.18.0

:return: DataFrame
    Unstacked Series.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


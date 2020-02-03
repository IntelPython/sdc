.. _pandas.Series.diff:

:orphan:

pandas.Series.diff
******************

First discrete difference of element.

Calculates the difference of a Series element compared with another
element in the Series (default is element in previous row).

:param periods:
    int, default 1
        Periods to shift for calculating difference, accepts negative
        values.

:return: Series
    First differences of the Series.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.DataFrame.pct_change:

:orphan:

pandas.DataFrame.pct_change
***************************

Percentage change between the current and a prior element.

Computes the percentage change from the immediately previous row by
default. This is useful in comparing the percentage of change in a time
series of elements.

:param periods:
    int, default 1
        Periods to shift for forming percent change.

:param fill_method:
    str, default 'pad'
        How to handle NAs before computing percent changes.

:param limit:
    int, default None
        The number of consecutive NAs to fill before stopping.

:param freq:
    DateOffset, timedelta, or offset alias string, optional
        Increment to use from time series API (e.g. 'M' or BDay()).
        \*\*kwargs
        Additional keyword arguments are passed into
        `DataFrame.shift` or `Series.shift`.

:return: chg : Series or DataFrame
    The same type as the calling object.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


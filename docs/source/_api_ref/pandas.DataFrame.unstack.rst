.. _pandas.DataFrame.unstack:

:orphan:

pandas.DataFrame.unstack
************************

Pivot a level of the (necessarily hierarchical) index labels, returning
a DataFrame having a new level of column labels whose inner-most level
consists of the pivoted index labels.

If the index is not a MultiIndex, the output will be a Series
(the analogue of stack when the columns are not a MultiIndex).

The level involved will automatically get sorted.

:param level:
    int, string, or list of these, default -1 (last level)
        Level(s) of index to unstack, can pass level name

:param fill_value:
    replace NaN with this value if the unstack produces
        missing values

        .. versionadded:: 0.18.0

:return: Series or DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


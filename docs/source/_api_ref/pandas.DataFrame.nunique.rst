.. _pandas.DataFrame.nunique:

:orphan:

pandas.DataFrame.nunique
************************

Count distinct observations over requested axis.

Return Series with number of distinct observations. Can ignore NaN
values.

.. versionadded:: 0.20.0

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
        column-wise.

:param dropna:
    bool, default True
        Don't include NaN in the counts.

:return: Series



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


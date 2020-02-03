.. _pandas.Series.droplevel:

:orphan:

pandas.Series.droplevel
***********************

Return DataFrame with requested index / column level(s) removed.

.. versionadded:: 0.24.0

:param level:
    int, str, or list-like
        If a string is given, must be the name of a level
        If list-like, elements must be names or positional indexes
        of levels.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0

:return: DataFrame.droplevel()



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


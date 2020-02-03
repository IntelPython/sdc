.. _pandas.DataFrame.melt:

:orphan:

pandas.DataFrame.melt
*********************

Unpivot a DataFrame from wide format to long format, optionally
leaving identifier variables set.

This function is useful to massage a DataFrame into a format where one
or more columns are identifier variables (`id_vars`), while all other
columns, considered measured variables (`value_vars`), are "unpivoted" to
the row axis, leaving just two non-identifier columns, 'variable' and
'value'.
.. versionadded:: 0.20.0

:param frame:
    DataFrame

:param id_vars:
    tuple, list, or ndarray, optional
        Column(s) to use as identifier variables.

:param value_vars:
    tuple, list, or ndarray, optional
        Column(s) to unpivot. If not specified, uses all columns that
        are not set as `id_vars`.

:param var_name:
    scalar
        Name to use for the 'variable' column. If None it uses
        ``frame.columns.name`` or 'variable'.

:param value_name:
    scalar, default 'value'
        Name to use for the 'value' column.

:param col_level:
    int or string, optional
        If columns are a MultiIndex then use this level to melt.

:return: DataFrame
    Unpivoted DataFrame.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


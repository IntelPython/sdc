.. _pandas.Series.str.extract:

:orphan:

pandas.Series.str.extract
*************************

Extract capture groups in the regex `pat` as columns in a DataFrame.

For each subject string in the Series, extract groups from the
first match of regular expression `pat`.

:param pat:
    str
        Regular expression pattern with capturing groups.

:param flags:
    int, default 0 (no flags)
        Flags from the ``re`` module, e.g. ``re.IGNORECASE``, that
        modify regular expression matching for things like case,
        spaces, etc. For more details, see :mod:`re`.

:param expand:
    bool, default True
        If True, return DataFrame with one column per capture group.
        If False, return a Series/Index if there is one capture group
        or DataFrame if there are multiple capture groups.

        .. versionadded:: 0.18.0

:return: DataFrame or Series or Index
    A DataFrame with one row for each subject string, and one
    column for each group. Any capture group names in regular
    expression pat will be used for column names; otherwise
    capture group numbers will be used. The dtype of each result
    column is always object, even when no match is found. If
    ``expand=False`` and pat has only one capture group, then
    return a Series (if subject is a Series) or Index (if subject
    is an Index).



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


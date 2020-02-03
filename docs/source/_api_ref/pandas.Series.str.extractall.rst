.. _pandas.Series.str.extractall:

:orphan:

pandas.Series.str.extractall
****************************

For each subject string in the Series, extract groups from all
matches of regular expression pat. When each subject string in the
Series has exactly one match, extractall(pat).xs(0, level='match')
is the same as extract(pat).

.. versionadded:: 0.18.0

:param pat:
    str
        Regular expression pattern with capturing groups.

:param flags:
    int, default 0 (no flags)
        A ``re`` module flag, for example ``re.IGNORECASE``. These allow
        to modify regular expression matching for things like case, spaces,
        etc. Multiple flags can be combined with the bitwise OR operator,
        for example ``re.IGNORECASE | re.MULTILINE``.

:return: DataFrame
    A ``DataFrame`` with one row for each match, and one column for each
    group. Its rows have a ``MultiIndex`` with first levels that come from
    the subject ``Series``. The last level is named 'match' and indexes the
    matches in each item of the ``Series``. Any capture group names in
    regular expression pat will be used for column names; otherwise capture
    group numbers will be used.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


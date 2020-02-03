.. _pandas.Series.str.contains:

:orphan:

pandas.Series.str.contains
**************************

Test if pattern or regex is contained within a string of a Series or Index.

Return boolean Series or Index based on whether a given pattern or regex is
contained within a string of a Series or Index.

:param pat:
    str
        Character sequence or regular expression.

:param case:
    bool, default True
        If True, case sensitive.

:param flags:
    int, default 0 (no flags)
        Flags to pass through to the re module, e.g. re.IGNORECASE.

:param na:
    default NaN
        Fill value for missing values.

:param regex:
    bool, default True
        If True, assumes the pat is a regular expression.

        If False, treats the pat as a literal string.

:return: Series or Index of boolean values
    A Series or Index of boolean values indicating whether the
    given pattern is contained within the string of each element
    of the Series or Index.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


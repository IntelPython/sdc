.. _pandas.Series.str.findall:

:orphan:

pandas.Series.str.findall
*************************

Find all occurrences of pattern or regular expression in the Series/Index.

Equivalent to applying :func:`re.findall` to all the elements in the
Series/Index.

:param pat:
    str
        Pattern or regular expression.

:param flags:
    int, default 0
        Flags from ``re`` module, e.g. `re.IGNORECASE` (default is 0, which
        means no flags).

:return: Series/Index of lists of strings
    All non-overlapping matches of pattern or regular expression in each
    string of this Series/Index.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


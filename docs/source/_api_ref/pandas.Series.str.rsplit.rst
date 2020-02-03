.. _pandas.Series.str.rsplit:

:orphan:

pandas.Series.str.rsplit
************************

Split strings around given separator/delimiter.

Splits the string in the Series/Index from the end,
at the specified delimiter string. Equivalent to :meth:`str.rsplit`.

:param pat:
    str, optional
        String or regular expression to split on.
        If not specified, split on whitespace.

:param n:
    int, default -1 (all)
        Limit number of splits in output.
        ``None``, 0 and -1 will be interpreted as return all splits.

:param expand:
    bool, default False
        Expand the splitted strings into separate columns.

        - If ``True``, return DataFrame/MultiIndex expanding dimensionality.
        - If ``False``, return Series/Index, containing lists of strings.

:return: Series, Index, DataFrame or MultiIndex
    Type matches caller unless ``expand=True`` (see Notes).



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


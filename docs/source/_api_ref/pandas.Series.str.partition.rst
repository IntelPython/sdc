.. _pandas.Series.str.partition:

:orphan:

pandas.Series.str.partition
***************************

Split the string at the first occurrence of `sep`.

This method splits the string at the first occurrence of `sep`,
and returns 3 elements containing the part before the separator,
the separator itself, and the part after the separator.
If the separator is not found, return 3 elements containing the string itself, followed by two empty strings.

:param sep:
    str, default whitespace
        String to split on.

:param pat:
    str, default whitespace
        .. deprecated:: 0.24.0

:param expand:
    bool, default True
        If True, return DataFrame/MultiIndex expanding dimensionality.
        If False, return Series/Index.

:return: DataFrame/MultiIndex or Series/Index of objects



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


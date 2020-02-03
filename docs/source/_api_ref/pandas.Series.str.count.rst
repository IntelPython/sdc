.. _pandas.Series.str.count:

:orphan:

pandas.Series.str.count
***********************

Count occurrences of pattern in each string of the Series/Index.

This function is used to count the number of times a particular regex
pattern is repeated in each of the string elements of the
:class:`~pandas.Series`.

:param pat:
    str
        Valid regular expression.

:param flags:
    int, default 0, meaning no flags
        Flags for the `re` module. For a complete list, `see here
        <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
        \*\*kwargs
        For compatibility with other string methods. Not used.

:return: Series or Index
    Same type as the calling object containing the integer counts.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


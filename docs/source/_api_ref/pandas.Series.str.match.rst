.. _pandas.Series.str.match:

:orphan:

pandas.Series.str.match
***********************

Determine if each string matches a regular expression.

:param pat:
    str
        Character sequence or regular expression.

:param case:
    bool, default True
        If True, case sensitive.

:param flags:
    int, default 0 (no flags)
        re module flags, e.g. re.IGNORECASE.

:param na:
    default NaN
        Fill value for missing values.

:return: Series/array of boolean values



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


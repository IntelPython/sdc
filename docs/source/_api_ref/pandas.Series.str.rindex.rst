.. _pandas.Series.str.rindex:

:orphan:

pandas.Series.str.rindex
************************

Return highest indexes in each strings where the substring is
fully contained between [start:end]. This is the same as
``str.rfind`` except instead of returning -1, it raises a ValueError
when the substring is not found. Equivalent to standard ``str.rindex``.

:param sub:
    str
        Substring being searched

:param start:
    int
        Left edge index

:param end:
    int
        Right edge index

:return: found : Series/Index of objects



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


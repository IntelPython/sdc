.. _pandas.Series.str.rfind:

:orphan:

pandas.Series.str.rfind
***********************

Return highest indexes in each strings in the Series/Index
where the substring is fully contained between [start:end].
Return -1 on failure. Equivalent to standard :meth:`str.rfind`.

:param sub:
    str
        Substring being searched

:param start:
    int
        Left edge index

:param end:
    int
        Right edge index

:return: found : Series/Index of integer values



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


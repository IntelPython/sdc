.. _pandas.Series.str.pad:

:orphan:

pandas.Series.str.pad
*********************

Pad strings in the Series/Index up to width.

:param width:
    int
        Minimum width of resulting string; additional characters will be filled
        with character defined in `fillchar`.

:param side:
    {'left', 'right', 'both'}, default 'left'
        Side from which to fill resulting string.

:param fillchar:
    str, default ' '
        Additional character for filling, default is whitespace.

:return: Series or Index of object
    Returns Series or Index with minimum number of char in object.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


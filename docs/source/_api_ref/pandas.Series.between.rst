.. _pandas.Series.between:

:orphan:

pandas.Series.between
*********************

Return boolean Series equivalent to left <= series <= right.

This function returns a boolean vector containing `True` wherever the
corresponding Series element is between the boundary values `left` and
`right`. NA values are treated as `False`.

:param left:
    scalar
        Left boundary.

:param right:
    scalar
        Right boundary.

:param inclusive:
    bool, default True
        Include boundaries.

:return: Series
    Series representing whether each element is between left and
    right (inclusive).



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


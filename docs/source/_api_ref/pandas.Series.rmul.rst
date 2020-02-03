.. _pandas.Series.rmul:

:orphan:

pandas.Series.rmul
******************

Return Multiplication of series and other, element-wise (binary operator `rmul`).

Equivalent to ``other \* series``, but with support to substitute a fill_value for
missing data in one of the inputs.

:param other:
    Series or scalar value

:param fill_value:
    None or float value, default None (NaN)
        Fill existing missing (NaN) values, and any new element needed for
        successful Series alignment, with this value before computation.
        If data in both corresponding Series locations is missing
        the result will be missing.

:param level:
    int or name
        Broadcast across a level, matching Index values on the
        passed MultiIndex level.

:return: Series
    The result of the operation.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


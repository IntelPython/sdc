.. _pandas.DataFrame.mod:

:orphan:

pandas.DataFrame.mod
********************

Get Modulo of dataframe and other, element-wise (binary operator `mod`).

Equivalent to ``dataframe % other``, but with support to substitute a fill_value
for missing data in one of the inputs. With reverse version, `rmod`.

Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
arithmetic operators: `+`, `-`, `\*`, `/`, `//`, `%`, `\*\*`.

:param other:
    scalar, sequence, Series, or DataFrame
        Any single or multiple element data structure, or list-like object.

:param axis:
    {0 or 'index', 1 or 'columns'}
       Whether to compare by the index (0 or 'index') or columns
       (1 or 'columns'). For Series input, axis to match Series index on.

:param level:
    int or label
        Broadcast across a level, matching Index values on the
        passed MultiIndex level.

:param fill_value:
    float or None, default None
        Fill existing missing (NaN) values, and any new element needed for
        successful DataFrame alignment, with this value before computation.
        If data in both corresponding DataFrame locations is missing
        the result will be missing.

:return: DataFrame
    Result of the arithmetic operation.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


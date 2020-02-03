.. _pandas.DataFrame.le:

:orphan:

pandas.DataFrame.le
*******************

Get Less than or equal to of dataframe and other, element-wise (binary operator `le`).

Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
operators.

Equivalent to `==`, `=!`, `<=`, `<`, `>=`, `>` with support to choose axis
(rows or columns) and level for comparison.

:param other:
    scalar, sequence, Series, or DataFrame
        Any single or multiple element data structure, or list-like object.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 'columns'
       Whether to compare by the index (0 or 'index') or columns
       (1 or 'columns').

:param level:
    int or label
        Broadcast across a level, matching Index values on the passed
        MultiIndex level.

:return: DataFrame of bool
    Result of the comparison.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


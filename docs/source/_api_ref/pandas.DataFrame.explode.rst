.. _pandas.DataFrame.explode:

:orphan:

pandas.DataFrame.explode
************************

Transform each element of a list-like to a row, replicating the
index values.

.. versionadded:: 0.25.0

:param column:
    str or tuple

:return: DataFrame
    Exploded lists to rows of the subset columns;
    index will be duplicated for these rows.

:raises:
    ValueError :
        if columns of the frame are not unique.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


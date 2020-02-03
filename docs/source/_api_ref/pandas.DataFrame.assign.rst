.. _pandas.DataFrame.assign:

:orphan:

pandas.DataFrame.assign
***********************

Assign new columns to a DataFrame.

Returns a new object with all original columns in addition to new ones.
Existing columns that are re-assigned will be overwritten.

:param \*\*kwargs:
    dict of {str: callable or Series}
        The column names are keywords. If the values are
        callable, they are computed on the DataFrame and
        assigned to the new columns. The callable must not
        change input DataFrame (though pandas doesn't check it).
        If the values are not callable, (e.g. a Series, scalar, or array),
        they are simply assigned.

:return: DataFrame
    A new DataFrame with the new columns in addition to
    all the existing columns.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


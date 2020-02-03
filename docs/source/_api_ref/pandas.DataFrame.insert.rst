.. _pandas.DataFrame.insert:

:orphan:

pandas.DataFrame.insert
***********************

Insert column into DataFrame at specified location.

Raises a ValueError if `column` is already contained in the DataFrame,
unless `allow_duplicates` is set to True.

:param loc:
    int
        Insertion index. Must verify 0 <= loc <= len(columns)

:param column:
    string, number, or hashable object
        label of the inserted column

:param value:
    int, Series, or array-like

:param allow_duplicates:
    bool, optional



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


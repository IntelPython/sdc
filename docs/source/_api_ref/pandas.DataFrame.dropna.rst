.. _pandas.DataFrame.dropna:

:orphan:

pandas.DataFrame.dropna
***********************

Remove missing values.

See the :ref:`User Guide <missing_data>` for more on which values are
considered missing, and how to work with missing data.

:param axis:
    {0 or 'index', 1 or 'columns'}, default 0
        Determine if rows or columns which contain missing values are
        removed.

:param \* 0, or 'index':
    Drop rows which contain missing values.

:param \* 1, or 'columns':
    Drop columns which contain missing value.

        .. deprecated:: 0.23.0

        Pass tuple or list to drop on multiple axes.
        Only a single axis is allowed.

:param how:
    {'any', 'all'}, default 'any'
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.

:param \* 'any':
    If any NA values are present, drop that row or column.

:param \* 'all':
    If all values are NA, drop that row or column.

:param thresh:
    int, optional
        Require that many non-NA values.

:param subset:
    array-like, optional
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include.

:param inplace:
    bool, default False
        If True, do operation inplace and return None.

:return: DataFrame
    DataFrame with NA entries dropped from it.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


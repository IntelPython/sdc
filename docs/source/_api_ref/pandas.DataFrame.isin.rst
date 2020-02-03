.. _pandas.DataFrame.isin:

:orphan:

pandas.DataFrame.isin
*********************

Whether each element in the DataFrame is contained in values.

:param values:
    iterable, Series, DataFrame or dict
        The result will only be true at a location if all the
        labels match. If `values` is a Series, that's the index. If
        `values` is a dict, the keys must be the column names,
        which must match. If `values` is a DataFrame,
        then both the index and column labels must match.

:return: DataFrame
    DataFrame of booleans showing whether each element in the DataFrame
    is contained in values.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


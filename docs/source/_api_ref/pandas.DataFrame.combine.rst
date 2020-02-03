.. _pandas.DataFrame.combine:

:orphan:

pandas.DataFrame.combine
************************

Perform column-wise combine with another DataFrame.

Combines a DataFrame with `other` DataFrame using `func`
to element-wise combine columns. The row and column indexes of the
resulting DataFrame will be the union of the two.

:param other:
    DataFrame
        The DataFrame to merge column-wise.

:param func:
    function
        Function that takes two series as inputs and return a Series or a
        scalar. Used to merge the two dataframes column by columns.

:param fill_value:
    scalar value, default None
        The value to fill NaNs with prior to passing any column to the
        merge func.

:param overwrite:
    bool, default True
        If True, columns in `self` that do not exist in `other` will be
        overwritten with NaNs.

:return: DataFrame
    Combination of the provided DataFrames.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


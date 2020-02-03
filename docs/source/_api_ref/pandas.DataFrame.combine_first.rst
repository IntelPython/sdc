.. _pandas.DataFrame.combine_first:

:orphan:

pandas.DataFrame.combine_first
******************************

Update null elements with value in the same location in `other`.

Combine two DataFrame objects by filling null values in one DataFrame
with non-null values from other DataFrame. The row and column indexes
of the resulting DataFrame will be the union of the two.

:param other:
    DataFrame
        Provided DataFrame to use to fill null values.

:return: DataFrame



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


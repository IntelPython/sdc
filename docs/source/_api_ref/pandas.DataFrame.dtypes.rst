.. _pandas.DataFrame.dtypes:

:orphan:

pandas.DataFrame.dtypes
***********************

Return the dtypes in the DataFrame.

This returns a Series with the data type of each column.
The result's index is the original DataFrame's columns. Columns
with mixed types are stored with the ``object`` dtype. See
:ref:`the User Guide <basics.dtypes>` for more.

:return: pandas.Series
    The data type of each column.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.Series.view:

:orphan:

pandas.Series.view
******************

Create a new view of the Series.

This function will return a new Series with a view of the same
underlying values in memory, optionally reinterpreted with a new data
type. The new data type must preserve the same size in bytes as to not
cause index misalignment.

:param dtype:
    data type
        Data type object or one of their string representations.

:return: Series
    A new Series object as a view of the same data in memory.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


.. _pandas.DataFrame.iteritems:

:orphan:

pandas.DataFrame.iteritems
**************************

Iterator over (column name, Series) pairs.

Iterates over the DataFrame columns, returning a tuple with
the column name and the content as a Series.

:return: label : object
    The column names for the DataFrame being iterated over.
    content : Series
    The column entries belonging to each label, as a Series.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


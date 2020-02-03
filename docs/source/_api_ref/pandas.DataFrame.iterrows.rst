.. _pandas.DataFrame.iterrows:

:orphan:

pandas.DataFrame.iterrows
*************************

Iterate over DataFrame rows as (index, Series) pairs.

Yields
------

index : label or tuple of label
    The index of the row. A tuple for a `MultiIndex`.
data : Series
    The data of the row as a Series.

it : generator
    A generator that iterates over the rows of the frame.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


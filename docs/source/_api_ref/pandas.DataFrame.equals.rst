.. _pandas.DataFrame.equals:

:orphan:

pandas.DataFrame.equals
***********************

Test whether two objects contain the same elements.

This function allows two Series or DataFrames to be compared against
each other to see if they have the same shape and elements. NaNs in
the same location are considered equal. The column headers do not
need to have the same type, but the elements within the columns must
be the same dtype.

:param other:
    Series or DataFrame
        The other Series or DataFrame to be compared with the first.

:return: bool
    True if all elements are the same in both objects, False
    otherwise.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler


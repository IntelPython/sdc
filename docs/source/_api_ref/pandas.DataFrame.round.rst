.. _pandas.DataFrame.round:

:orphan:

pandas.DataFrame.round
**********************

Round a DataFrame to a variable number of decimal places.

:param decimals:
    int, dict, Series
        Number of decimal places to round each column to. If an int is
        given, round each column to the same number of places.
        Otherwise dict and Series round to variable numbers of places.
        Column names should be in the keys if `decimals` is a
        dict-like, or in the index if `decimals` is a Series. Any
        columns not included in `decimals` will be left as is. Elements
        of `decimals` which are not columns of the input will be
        ignored.
        \*args
        Additional keywords have no effect but might be accepted for
        compatibility with numpy.
        \*\*kwargs
        Additional keywords have no effect but might be accepted for
        compatibility with numpy.

:return: DataFrame
    A DataFrame with the affected columns rounded to the specified
    number of decimal places.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

